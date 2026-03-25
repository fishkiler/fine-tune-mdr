#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Evaluation Script
============================================================================
Runs inference on the test set, parses structured output, and computes
accuracy / precision / recall / F1 metrics.

Expected output format from model:
    T[ID] | [Tactic] | [Confidence]% | [Explanation]
    Example: T1059 | Execution | 87% | PowerShell command obfuscation detected

Usage:
    python eval.py
    python eval.py --config config.yaml --model-dir outputs
============================================================================
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Config
# ============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Output Parsing
# ============================================================================

# Matches: T1059 | Execution | 87% | some explanation
RESPONSE_PATTERN = re.compile(
    r"T(\d{4}(?:\.\d{3})?)\s*\|\s*(\w[\w\s]*?)\s*\|\s*(\d+)%?\s*\|\s*(.+)",
    re.IGNORECASE,
)


def parse_response(text: str) -> dict | None:
    """Parse structured model output into components."""
    for line in text.strip().split("\n"):
        m = RESPONSE_PATTERN.search(line)
        if m:
            return {
                "technique_id": f"T{m.group(1)}",
                "tactic": m.group(2).strip(),
                "confidence": int(m.group(3)),
                "explanation": m.group(4).strip(),
            }
    return None


def extract_ground_truth(text: str) -> dict | None:
    """Extract ground truth from the assistant turn in formatted text."""
    # Look for the assistant's response in the chat-formatted text
    # Try to find content after assistant header tokens
    assistant_markers = [
        "<|assistant|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "### Assistant:",
    ]
    for marker in assistant_markers:
        idx = text.rfind(marker)
        if idx != -1:
            response = text[idx + len(marker):].strip()
            # Remove any trailing special tokens
            for token in ["<|eot_id|>", "</s>", "<|end_of_text|>"]:
                response = response.replace(token, "")
            parsed = parse_response(response.strip())
            if parsed:
                return parsed
    return None


# ============================================================================
# Evaluation
# ============================================================================

ARTIFACT_PREFIX_RE = re.compile(r"^(?:[>*]{1,2}\s*)+")


def postprocess_response(text: str) -> str:
    """Strip artifact prefixes (>, **, *) from model output."""
    lines = text.split("\n")
    cleaned = [ARTIFACT_PREFIX_RE.sub("", line) for line in lines]
    return "\n".join(cleaned).strip()


def run_evaluation(model, tokenizer, test_ds, cfg: dict) -> dict:
    """Run inference on test set and compute metrics."""
    eval_cfg = cfg["evaluation"]
    model_cfg = cfg["model"]
    batch_size = eval_cfg["batch_size"]
    max_new_tokens = eval_cfg["max_new_tokens"]
    repetition_penalty = eval_cfg.get("repetition_penalty", 1.0)

    results = []
    parse_failures = 0
    total = len(test_ds)

    log.info(f"Evaluating {total} examples (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch = test_ds[i : i + batch_size]
        texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]

        for text in texts:
            # Extract ground truth from the full formatted example
            gt = extract_ground_truth(text)

            # Build the prompt (everything up to the assistant response)
            # Find where the assistant should respond
            prompt = text
            for marker in [
                "<|assistant|>",
                "<|start_header_id|>assistant<|end_header_id|>",
                "### Assistant:",
            ]:
                idx = text.rfind(marker)
                if idx != -1:
                    prompt = text[: idx + len(marker)]
                    break

            # Generate
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_cfg["max_seq_length"],
            ).to(model.device)

            with torch.no_grad():
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy for determinism
                    temperature=1.0,
                )
                if repetition_penalty != 1.0:
                    gen_kwargs["repetition_penalty"] = repetition_penalty
                output_ids = model.generate(**gen_kwargs)

            # Decode only the new tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            response = postprocess_response(response)

            pred = parse_response(response)
            if pred is None:
                parse_failures += 1

            results.append({
                "ground_truth": gt,
                "prediction": pred,
                "raw_response": response,
            })

        # Progress
        done = min(i + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            log.info(f"  Progress: {done}/{total} ({done/total*100:.0f}%)")

    # --- Compute metrics ---
    log.info("Computing metrics...")
    metrics = compute_metrics(results, parse_failures, total)
    return metrics


def compute_metrics(results: list[dict], parse_failures: int, total: int) -> dict:
    """Compute accuracy, P/R/F1 from parsed results."""
    exact_matches = 0
    tactic_matches = 0
    technique_matches = 0
    valid_comparisons = 0

    # Per-technique tracking
    technique_tp = defaultdict(int)
    technique_fp = defaultdict(int)
    technique_fn = defaultdict(int)

    for r in results:
        gt = r["ground_truth"]
        pred = r["prediction"]

        if gt is None or pred is None:
            if gt and pred is None:
                technique_fn[gt["technique_id"]] += 1
            continue

        valid_comparisons += 1

        # Exact match (technique + tactic)
        if gt["technique_id"] == pred["technique_id"] and gt["tactic"].lower() == pred["tactic"].lower():
            exact_matches += 1

        # Technique match
        if gt["technique_id"] == pred["technique_id"]:
            technique_matches += 1
            technique_tp[gt["technique_id"]] += 1
        else:
            technique_fn[gt["technique_id"]] += 1
            technique_fp[pred["technique_id"]] += 1

        # Tactic match
        if gt["tactic"].lower() == pred["tactic"].lower():
            tactic_matches += 1

    # Per-technique P/R/F1
    per_technique = {}
    all_techniques = set(technique_tp) | set(technique_fp) | set(technique_fn)
    for tid in sorted(all_techniques):
        tp = technique_tp[tid]
        fp = technique_fp[tid]
        fn = technique_fn[tid]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_technique[tid] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    # Macro F1
    f1_scores = [v["f1"] for v in per_technique.values()]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return {
        "total_examples": total,
        "valid_comparisons": valid_comparisons,
        "parse_failure_rate": parse_failures / total if total > 0 else 0,
        "exact_match_accuracy": exact_matches / valid_comparisons if valid_comparisons > 0 else 0,
        "technique_accuracy": technique_matches / valid_comparisons if valid_comparisons > 0 else 0,
        "tactic_accuracy": tactic_matches / valid_comparisons if valid_comparisons > 0 else 0,
        "macro_f1": macro_f1,
        "num_techniques": len(all_techniques),
        "per_technique": per_technique,
        "parse_failures": parse_failures,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default=None, help="Path to model (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = args.model_dir or cfg["training"]["output_dir"]
    eval_cfg = cfg["evaluation"]

    # --- Load model ---
    log.info(f"Loading model from {model_dir}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.to("cuda")
    model.eval()
    log.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    # --- Load test set ---
    test_path = Path(cfg["dataset"]["local_dir"]) / "test"
    if test_path.exists():
        test_ds = load_from_disk(str(test_path))
        log.info(f"Loaded test set: {len(test_ds)} examples")
    else:
        log.warning("No prepared test set found. Run scripts/prepare_data.py first.")
        log.info("Falling back to HuggingFace dataset split...")
        raw = load_dataset(cfg["dataset"]["name"], split="train", token=os.environ.get("HF_TOKEN"))
        split = raw.train_test_split(test_size=0.05, seed=cfg["dataset"]["seed"])
        test_ds = split["test"]
        log.info(f"Using {len(test_ds)} examples from random split")

    # Apply chat template if dataset has 'messages' but no 'text'
    if "text" not in test_ds.column_names and "messages" in test_ds.column_names:
        log.info("Applying chat template to test set...")
        test_ds = test_ds.map(
            lambda ex: {"text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )},
            remove_columns=[c for c in test_ds.column_names if c != "text"],
            desc="Formatting test set",
        )
        log.info("Chat template applied.")

    # --- Run evaluation ---
    metrics = run_evaluation(model, tokenizer, test_ds, cfg)

    # --- Report ---
    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Total examples:       {metrics['total_examples']}")
    log.info(f"  Valid comparisons:    {metrics['valid_comparisons']}")
    log.info(f"  Parse failure rate:   {metrics['parse_failure_rate']:.2%}")
    log.info(f"  Exact match accuracy: {metrics['exact_match_accuracy']:.2%}")
    log.info(f"  Technique accuracy:   {metrics['technique_accuracy']:.2%}")
    log.info(f"  Tactic accuracy:      {metrics['tactic_accuracy']:.2%}")
    log.info(f"  Macro F1:             {metrics['macro_f1']:.4f}")
    log.info(f"  Unique techniques:    {metrics['num_techniques']}")
    log.info("=" * 60)

    # --- Save ---
    results_dir = Path(eval_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"{timestamp}.json"

    # Convert per_technique for JSON serialization
    report = {
        **{k: v for k, v in metrics.items() if k != "per_technique"},
        "per_technique": metrics["per_technique"],
        "model_dir": model_dir,
        "timestamp": timestamp,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Results saved to {out_path}")

    # --- Baseline comparison ---
    if eval_cfg.get("run_baseline"):
        log.info("Running baseline comparison with base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["name"],
            torch_dtype=torch.bfloat16,
            attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        ).to("cuda")
        base_model.eval()
        base_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        base_metrics = run_evaluation(base_model, base_tokenizer, test_ds, cfg)

        log.info("BASELINE RESULTS")
        log.info(f"  Exact match: {base_metrics['exact_match_accuracy']:.2%}")
        log.info(f"  Macro F1:    {base_metrics['macro_f1']:.4f}")
        log.info(f"  Improvement: {metrics['macro_f1'] - base_metrics['macro_f1']:+.4f} F1")

        base_path = results_dir / f"{timestamp}_baseline.json"
        with open(base_path, "w") as f:
            json.dump({**base_metrics, "model": cfg["model"]["name"], "timestamp": timestamp}, f, indent=2)


if __name__ == "__main__":
    main()
