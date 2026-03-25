#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Model Export Utility
============================================================================
Re-exports the fine-tuned model to GGUF or merged safetensors format.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --format gguf --quantization q4_k_m
    python scripts/export_model.py --format merged_16bit
============================================================================
"""

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def export_model(
    model_dir: str,
    output_dir: str,
    export_format: str,
    quantization: str,
    max_seq_length: int,
):
    from unsloth import FastLanguageModel

    log.info(f"Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if export_format == "gguf":
        log.info(f"Exporting to GGUF with quantization={quantization}...")
        try:
            model.save_pretrained_gguf(
                str(out / "model"),
                tokenizer,
                quantization_method=quantization,
            )
            log.info(f"GGUF export saved to {out / 'model'}")
        except Exception as e:
            log.warning(f"GGUF export failed (may not be supported on this arch): {e}")
            log.info("Falling back to safetensors export...")
            model.save_pretrained_merged(
                str(out / "model-merged"),
                tokenizer,
                save_method="merged_16bit",
            )
            log.info(f"Safetensors fallback saved to {out / 'model-merged'}")

    elif export_format == "merged_16bit":
        log.info("Merging LoRA into base model (16-bit)...")
        model.save_pretrained_merged(
            str(out / "model-merged"),
            tokenizer,
            save_method="merged_16bit",
        )
        log.info(f"Merged model saved to {out / 'model-merged'}")

    elif export_format == "merged_4bit":
        log.info("Merging LoRA into base model (4-bit)...")
        model.save_pretrained_merged(
            str(out / "model-merged-4bit"),
            tokenizer,
            save_method="merged_4bit_forced",
        )
        log.info(f"Merged 4-bit model saved to {out / 'model-merged-4bit'}")

    elif export_format == "lora_only":
        log.info("Saving LoRA adapters only...")
        model.save_pretrained(str(out / "lora-adapters"))
        tokenizer.save_pretrained(str(out / "lora-adapters"))
        log.info(f"LoRA adapters saved to {out / 'lora-adapters'}")

    else:
        raise ValueError(f"Unknown export format: {export_format}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to fine-tuned model (default: from config)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["gguf", "merged_16bit", "merged_4bit", "lora_only"],
        help="Export format (default: from config)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantization method (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_dir = args.model_dir or cfg["training"]["output_dir"]
    output_dir = args.output_dir or cfg["export"]["output_dir"]
    export_format = args.format or ("gguf" if cfg["export"]["save_gguf"] else "merged_16bit")
    quantization = args.quantization or cfg["export"]["gguf_quantization"]
    max_seq_length = cfg["model"]["max_seq_length"]

    export_model(model_dir, output_dir, export_format, quantization, max_seq_length)

    # If config says save both formats, do the second one too
    if args.format is None and cfg["export"]["save_gguf"] and cfg["export"]["save_merged_16bit"]:
        log.info("Config requests both GGUF and merged_16bit — exporting merged...")
        export_model(model_dir, output_dir, "merged_16bit", quantization, max_seq_length)

    log.info("Export complete.")


if __name__ == "__main__":
    main()
