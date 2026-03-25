#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Game Adapter Evaluation (Offline Mode A)
============================================================================
Evaluates a trained game adapter by running inference on held-out test frames
and comparing predicted actions to ground truth labels.

Metrics reported:
  - Overall action accuracy
  - Per-action accuracy (confusion matrix)
  - Action diversity (entropy of predicted distribution)
  - Decision point accuracy (frames where action != NONE)

Usage:
    python eval_game_adapter.py --game pacman
    python eval_game_adapter.py --game pacman --adapter-dir adapters/games/pacman/v1
    python eval_game_adapter.py --game pacman --max-samples 500
============================================================================
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from math import log2
from pathlib import Path

import torch
import yaml
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_action(text: str, valid_actions: list[str]) -> str | None:
    """Extract the action from model output text.

    Looks for 'Action: <ACTION>' pattern first, then falls back to
    checking if any valid action appears as a standalone word.
    """
    # Primary: "Action: RIGHT" pattern
    match = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
    if match:
        action = match.group(1).upper()
        if action in valid_actions:
            return action

    # Fallback: last valid action word in the text
    upper = text.upper()
    for action in reversed(valid_actions):
        if re.search(r'\b' + action + r'\b', upper):
            return action

    return None


def entropy(counts: Counter) -> float:
    """Shannon entropy of a distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * log2(p) for p in probs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate game adapter (offline action accuracy)")
    parser.add_argument("--game", required=True, help="Game name (e.g. pacman)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--adapter-dir", default=None, help="Override adapter directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Max test samples (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ga_cfg = cfg.get("game_adapters", {})
    game_cfg = ga_cfg.get("games", {}).get(args.game)

    if not game_cfg:
        log.error(f"Game '{args.game}' not found in config.yaml")
        sys.exit(1)

    base_model_name = ga_cfg["base_model"]
    valid_actions = game_cfg["actions"]
    dataset_dir = Path(game_cfg["dataset_dir"])
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else Path(game_cfg["adapter_dir"]) / "v1"
    test_jsonl = dataset_dir / "training" / "test.jsonl"
    frames_base = PROJECT_ROOT / "data" / "games" / args.game

    if not test_jsonl.exists():
        log.error(f"Test data not found: {test_jsonl}")
        sys.exit(1)

    if not (adapter_dir / "adapter_model.safetensors").exists():
        log.error(f"Adapter not found: {adapter_dir}")
        sys.exit(1)

    # --- Load test data ---
    log.info(f"Loading test data from {test_jsonl}...")
    examples = []
    with open(test_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_samples > 0:
        examples = examples[:args.max_samples]
    log.info(f"Evaluating on {len(examples)} test examples")

    # --- CUDA setup ---
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model + adapter ---
    precision = ga_cfg.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    attn_impl = ga_cfg.get("attention", "sdpa")

    image_cfg = ga_cfg.get("image", {})
    proc_kwargs = {}
    if image_cfg.get("min_pixels"):
        proc_kwargs["min_pixels"] = image_cfg["min_pixels"]
    if image_cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = image_cfg["max_pixels"]

    log.info(f"Loading {base_model_name}...")
    processor = AutoProcessor.from_pretrained(base_model_name, **proc_kwargs)
    if proc_kwargs:
        log.info(f"Processor pixel limits: {proc_kwargs}")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model = model.to(device)

    log.info(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    log.info("Model + adapter loaded.")

    # --- Run inference ---
    correct = 0
    total = 0
    decision_correct = 0
    decision_total = 0
    pred_counts = Counter()
    true_counts = Counter()
    confusion = Counter()  # (true, pred) pairs
    errors = []

    log.info("Running evaluation...")
    t0 = time.time()

    for i, example in enumerate(examples):
        messages = example["messages"]

        # Extract ground truth action from assistant message
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        true_action = extract_action(assistant_msg, valid_actions)
        if true_action is None:
            continue

        # Extract image path from user message
        image_path = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        image_path = part["image"]

        if not image_path:
            continue

        # Load image
        full_path = frames_base / image_path
        if not full_path.exists():
            continue
        image = Image.open(full_path).convert("RGB")

        # Build inference prompt (system + user, no assistant)
        # Use multimodal content format for proper image token insertion
        infer_messages = []
        for msg in messages:
            if msg["role"] == "system":
                infer_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    content_parts = []
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            content_parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part["text"]})
                    infer_messages.append({"role": "user", "content": content_parts})
                else:
                    infer_messages.append(msg)

        text = processor.apply_chat_template(
            infer_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = processor(
            text=text,
            images=[image],
            return_tensors="pt",
        )
        gen_inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **gen_inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

        pred_action = extract_action(generated, valid_actions)
        if pred_action is None:
            pred_action = "UNKNOWN"

        # Score
        total += 1
        true_counts[true_action] += 1
        pred_counts[pred_action] += 1
        confusion[(true_action, pred_action)] += 1

        if pred_action == true_action:
            correct += 1
        else:
            if len(errors) < 20:
                errors.append({
                    "idx": i,
                    "true": true_action,
                    "pred": pred_action,
                    "output": generated[:200],
                })

        # Decision point accuracy (non-NONE actions)
        if true_action != "NONE":
            decision_total += 1
            if pred_action == true_action:
                decision_correct += 1

        # Progress
        if (i + 1) % 100 == 0:
            acc = correct / total * 100 if total > 0 else 0
            elapsed = time.time() - t0
            rate = total / elapsed
            log.info(f"  [{i+1}/{len(examples)}] acc={acc:.1f}% ({rate:.1f} samples/sec)")

    elapsed = time.time() - t0

    # --- Results ---
    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS — {args.game.upper()}")
    print("=" * 60)

    if total == 0:
        print("No valid samples evaluated.")
        sys.exit(1)

    overall_acc = correct / total * 100
    decision_acc = decision_correct / decision_total * 100 if decision_total > 0 else 0
    pred_entropy = entropy(pred_counts)
    max_entropy = log2(len(valid_actions)) if len(valid_actions) > 1 else 1

    print(f"\n  Overall Accuracy:    {correct}/{total} = {overall_acc:.1f}%")
    print(f"  Decision Accuracy:   {decision_correct}/{decision_total} = {decision_acc:.1f}%")
    print(f"  (non-NONE frames only)")
    print(f"\n  Action Diversity:    {pred_entropy:.2f} / {max_entropy:.2f} bits")
    print(f"  Eval Time:           {elapsed:.1f}s ({total/elapsed:.1f} samples/sec)")
    print(f"  Adapter:             {adapter_dir}")

    # Per-action breakdown
    print(f"\n  {'ACTION':<10} {'TRUE':>6} {'PRED':>6} {'ACC':>8}")
    print("  " + "-" * 34)
    for action in valid_actions:
        true_n = true_counts[action]
        correct_n = confusion[(action, action)]
        acc = correct_n / true_n * 100 if true_n > 0 else 0
        pred_n = pred_counts[action]
        print(f"  {action:<10} {true_n:>6} {pred_n:>6} {acc:>7.1f}%")
    if pred_counts.get("UNKNOWN", 0) > 0:
        print(f"  {'UNKNOWN':<10} {'':>6} {pred_counts['UNKNOWN']:>6} {'':>8}")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + " " * 10 + "".join(f"{a:>8}" for a in valid_actions)
    print(header)
    for true_a in valid_actions:
        row = f"  {true_a:<10}"
        for pred_a in valid_actions:
            row += f"{confusion[(true_a, pred_a)]:>8}"
        print(row)

    # Sample errors
    if errors:
        print(f"\n  Sample Errors (first {len(errors)}):")
        for e in errors[:10]:
            print(f"    #{e['idx']}: true={e['true']} pred={e['pred']}")
            print(f"      output: {e['output'][:100]}...")

    print("\n" + "=" * 60)

    # Save results
    results = {
        "game": args.game,
        "adapter_dir": str(adapter_dir),
        "test_samples": total,
        "overall_accuracy": round(overall_acc, 2),
        "decision_accuracy": round(decision_acc, 2),
        "action_diversity_bits": round(pred_entropy, 3),
        "eval_time_seconds": round(elapsed, 1),
        "per_action": {
            action: {
                "true_count": true_counts[action],
                "pred_count": pred_counts[action],
                "accuracy": round(confusion[(action, action)] / true_counts[action] * 100, 1) if true_counts[action] > 0 else 0,
            }
            for action in valid_actions
        },
    }

    results_path = adapter_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
