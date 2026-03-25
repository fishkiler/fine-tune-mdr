#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Confidence Calibration
============================================================================
Fits a temperature scaling parameter on the validation set to calibrate
model confidence scores. Generates a reliability diagram.

Usage:
    python calibrate.py
    python calibrate.py --config config.yaml --model-dir outputs
============================================================================
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
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


def save_config(cfg: dict, path: str = "config.yaml"):
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Temperature Scaling
# ============================================================================

class TemperatureScaler:
    """Learns a single temperature parameter to calibrate logits."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fit temperature using L-BFGS to minimize NLL.

        Args:
            logits: Pre-softmax logits, shape (N, C)
            labels: True class indices, shape (N,)
        Returns:
            Optimal temperature
        """
        from scipy.optimize import minimize

        def nll_with_temp(temp):
            temp = max(temp[0], 0.01)  # prevent division by zero
            scaled = logits / temp
            # Log-sum-exp for numerical stability
            max_logits = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(scaled - max_logits), axis=1)) + max_logits.squeeze()
            log_probs = scaled[np.arange(len(labels)), labels] - log_sum_exp
            return -np.mean(log_probs)

        result = minimize(nll_with_temp, [1.0], method="L-BFGS-B", bounds=[(0.01, 10.0)])
        self.temperature = float(result.x[0])
        log.info(f"Optimal temperature: {self.temperature:.4f}")
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities."""
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


# ============================================================================
# Reliability Diagram
# ============================================================================

def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int,
    save_path: str,
    title: str = "Reliability Diagram",
):
    """Plot and save a reliability diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(num_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    # ECE
    total = confidences.shape[0]
    ece = np.sum(bin_counts / total * np.abs(bin_accs - bin_confs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0a0a0f")

    # Reliability diagram
    ax1.set_facecolor("#12121a")
    ax1.bar(bin_centers, bin_accs, width=1 / num_bins, alpha=0.7, color="#00ff88", edgecolor="#0a0a0f", label="Model")
    ax1.plot([0, 1], [0, 1], "--", color="#ff4466", linewidth=1.5, label="Perfect calibration")
    ax1.set_xlabel("Confidence", color="#8888aa", fontsize=11)
    ax1.set_ylabel("Accuracy", color="#8888aa", fontsize=11)
    ax1.set_title(f"{title}  (ECE={ece:.4f})", color="#e0e0e0", fontsize=13)
    ax1.legend(facecolor="#12121a", edgecolor="#1e1e2e", labelcolor="#e0e0e0")
    ax1.tick_params(colors="#555577")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    for spine in ax1.spines.values():
        spine.set_color("#1e1e2e")

    # Count histogram
    ax2.set_facecolor("#12121a")
    ax2.bar(bin_centers, bin_counts, width=1 / num_bins, alpha=0.7, color="#00aaff", edgecolor="#0a0a0f")
    ax2.set_xlabel("Confidence", color="#8888aa", fontsize=11)
    ax2.set_ylabel("Count", color="#8888aa", fontsize=11)
    ax2.tick_params(colors="#555577")
    for spine in ax2.spines.values():
        spine.set_color("#1e1e2e")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor="#0a0a0f", bbox_inches="tight")
    plt.close()
    log.info(f"Reliability diagram saved to {save_path}")
    return ece


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrate model confidence")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_path = args.config
    model_dir = args.model_dir or cfg["training"]["output_dir"]
    cal_cfg = cfg["calibration"]
    model_cfg = cfg["model"]

    # --- Load model ---
    log.info(f"Loading model from {model_dir}...")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg["chat_template"])
    FastLanguageModel.for_inference(model)

    # --- Load validation set ---
    val_path = Path(cfg["dataset"]["local_dir"]) / "val"
    if not val_path.exists():
        log.error("No validation set found. Run scripts/prepare_data.py first.")
        return
    val_ds = load_from_disk(str(val_path))
    log.info(f"Loaded validation set: {len(val_ds)} examples")

    # --- Collect logits ---
    log.info("Collecting logits on validation set...")
    all_logits = []
    all_correct = []
    vocab_size = model.config.vocab_size

    from eval import parse_response, extract_ground_truth

    for i, example in enumerate(val_ds):
        text = example["text"]
        gt = extract_ground_truth(text)
        if gt is None:
            continue

        # Build prompt (up to assistant response)
        prompt = text
        for marker in [
            "<|start_header_id|>assistant<|end_header_id|>",
            "### Assistant:",
        ]:
            idx = text.rfind(marker)
            if idx != -1:
                prompt = text[: idx + len(marker)]
                break

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=model_cfg["max_seq_length"],
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg["evaluation"]["max_new_tokens"],
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Get logits for generated tokens
        if outputs.scores:
            # Stack scores: list of (1, vocab_size) -> (num_tokens, vocab_size)
            token_logits = torch.cat([s.squeeze(0).unsqueeze(0) for s in outputs.scores], dim=0)
            # Use mean logit confidence across generated tokens
            probs = torch.softmax(token_logits, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean().item()
        else:
            confidence = 0.5

        # Parse prediction
        new_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = parse_response(response)

        if pred is not None:
            correct = 1.0 if pred["technique_id"] == gt["technique_id"] else 0.0
            all_logits.append(confidence)
            all_correct.append(correct)

        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i + 1}/{len(val_ds)}")

    if len(all_logits) < 10:
        log.error(f"Only {len(all_logits)} valid samples — too few for calibration.")
        return

    confidences = np.array(all_logits)
    accuracies = np.array(all_correct)
    log.info(f"Collected {len(confidences)} confidence/accuracy pairs")

    # --- Pre-calibration reliability diagram ---
    log.info("Generating pre-calibration reliability diagram...")
    pre_ece = plot_reliability_diagram(
        confidences, accuracies, cal_cfg["num_bins"],
        cal_cfg["plot_path"].replace(".png", "_before.png"),
        title="Before Calibration",
    )
    log.info(f"Pre-calibration ECE: {pre_ece:.4f}")

    # --- Fit temperature ---
    # For temperature scaling we need proper logit vectors;
    # since we're working with max-prob confidence, we use a simple
    # Platt-scaling approach: fit sigmoid(logit/T) where logit = log(p/(1-p))
    log.info("Fitting temperature parameter...")

    # Convert confidence to logit space
    eps = 1e-7
    logit_values = np.log(np.clip(confidences, eps, 1 - eps) / np.clip(1 - confidences, eps, 1 - eps))

    from scipy.optimize import minimize_scalar

    def calibration_loss(temp):
        scaled_conf = 1 / (1 + np.exp(-logit_values / max(temp, 0.01)))
        # Binary cross-entropy
        bce = -(accuracies * np.log(np.clip(scaled_conf, eps, 1)) +
                (1 - accuracies) * np.log(np.clip(1 - scaled_conf, eps, 1)))
        return np.mean(bce)

    result = minimize_scalar(calibration_loss, bounds=(0.1, 10.0), method="bounded")
    optimal_temp = float(result.x)
    log.info(f"Optimal temperature: {optimal_temp:.4f}")

    # --- Post-calibration ---
    calibrated_conf = 1 / (1 + np.exp(-logit_values / optimal_temp))
    post_ece = plot_reliability_diagram(
        calibrated_conf, accuracies, cal_cfg["num_bins"],
        cal_cfg["plot_path"],
        title="After Calibration",
    )
    log.info(f"Post-calibration ECE: {post_ece:.4f}")
    log.info(f"ECE improvement: {pre_ece - post_ece:.4f}")

    # --- Update config ---
    log.info(f"Updating config.yaml with temperature={optimal_temp:.4f}...")
    cfg["calibration"]["temperature"] = round(optimal_temp, 4)
    cfg["inference"]["temperature"] = round(optimal_temp, 4)
    save_config(cfg, config_path)
    log.info("Config updated.")

    # --- Save calibration report ---
    report = {
        "optimal_temperature": optimal_temp,
        "pre_calibration_ece": float(pre_ece),
        "post_calibration_ece": float(post_ece),
        "num_samples": len(confidences),
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(accuracies.mean()),
    }
    report_path = Path(cfg["evaluation"]["results_dir"]) / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Calibration report saved to {report_path}")
    log.info("Calibration complete.")


if __name__ == "__main__":
    main()
