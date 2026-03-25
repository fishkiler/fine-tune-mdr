#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Fast Iteration Trainer (SmolLM2-135M)
============================================================================
Rapid data quality and prompt format validation using SmolLM2-135M (135M
params). Trains in ~30 minutes — use this before committing to a full
4-12 hour Foundation-Sec-8B run.

Workflow:
    1. Change domain weights or add new data
    2. python -m scripts.train_fast           (30 min)
    3. python eval.py --model outputs/smollm2_fast/adapter
    4. If accuracy improves -> promote to full Foundation-Sec-8B run

Usage:
    python -m scripts.train_fast
    python -m scripts.train_fast --data data/export/chatml_train.jsonl
    python -m scripts.train_fast --no-dashboard
============================================================================
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Config
# ============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Dataset
# ============================================================================

def format_dataset(ds: Dataset, tokenizer) -> Dataset:
    """Apply chat template to a dataset with 'messages' column."""
    columns = ds.column_names
    if "text" in columns:
        return ds
    if "messages" not in columns:
        raise ValueError(f"Dataset needs 'text' or 'messages' column, got: {columns}")
    log.info("Applying chat template to messages...")
    return ds.map(
        lambda ex: {"text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )},
        remove_columns=[c for c in columns if c != "text"],
        desc="Formatting",
    )


def load_train_data(data_path: str | None, tokenizer) -> Dataset:
    """Load training data from disk or default locations."""
    # Explicit path provided
    if data_path:
        p = Path(data_path)
        if p.suffix == ".jsonl":
            log.info(f"Loading JSONL dataset from {p}...")
            ds = load_dataset("json", data_files=str(p), split="train")
        elif p.is_dir():
            log.info(f"Loading Arrow dataset from {p}...")
            ds = load_from_disk(str(p))
        else:
            raise ValueError(f"Unsupported data format: {p}")
        return format_dataset(ds, tokenizer)

    # Default: check prepared data directory
    local_dir = PROJECT_ROOT / "data"
    if (local_dir / "train").exists():
        log.info(f"Loading prepared dataset from {local_dir / 'train'}...")
        ds = load_from_disk(str(local_dir / "train"))
        return format_dataset(ds, tokenizer)

    # Fallback: check export directory
    export_path = local_dir / "export" / "chatml_train.jsonl"
    if export_path.exists():
        log.info(f"Loading export data from {export_path}...")
        ds = load_dataset("json", data_files=str(export_path), split="train")
        return format_dataset(ds, tokenizer)

    raise FileNotFoundError(
        "No training data found. Run export_training_data.py first, "
        "or specify --data path."
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast iteration trainer using SmolLM2-135M"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data", default=None, help="Override training data path (JSONL or Arrow dir)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard callback")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    fi_cfg = cfg.get("fast_iteration")
    if not fi_cfg:
        log.error("No 'fast_iteration' section in config.yaml")
        sys.exit(1)

    dash_cfg = cfg["dashboard"]
    lora_cfg = fi_cfg["lora"]
    train_cfg = fi_cfg["training"]

    # --- Safety: prevent NVRM deadlock on DGX Spark ---
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = fi_cfg["model"]
    log.info(f"Fast Iteration Trainer — {model_name}")
    log.info(f"  LoRA: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}")
    log.info(f"  Training: {train_cfg['epochs']} epochs, batch={train_cfg['batch_size']}, lr={train_cfg['learning_rate']}")

    # --- Load tokenizer ---
    log.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    attn_impl = fi_cfg.get("attn_implementation", "sdpa")
    log.info(f"Loading {model_name} in BF16 with {attn_impl} attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model = model.to("cuda")
    log.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    # --- Apply LoRA ---
    log.info("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Dataset ---
    train_ds = load_train_data(args.data, tokenizer)
    log.info(f"Training on {len(train_ds)} examples.")

    # --- Callbacks ---
    callbacks = []
    if not args.no_dashboard:
        from dashboard.callback import DashboardCallback
        dashboard_url = f"http://localhost:{dash_cfg['port']}"
        callbacks.append(DashboardCallback(dashboard_url=dashboard_url))
        log.info(f"Dashboard callback → {dashboard_url}")

    # --- Training ---
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        bf16=True,
        logging_steps=train_cfg["logging_steps"],
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        save_strategy="epoch",
        dataset_text_field="text",
        max_length=fi_cfg["max_seq_length"],
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    log.info("Starting fast iteration training...")
    trainer.train()
    log.info("Training complete.")

    # --- Save adapter ---
    adapter_dir = f"{output_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log.info(f"Adapter saved to {adapter_dir}")

    # --- Cleanup ---
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Trainer memory freed. Run eval.py next.")


if __name__ == "__main__":
    main()
