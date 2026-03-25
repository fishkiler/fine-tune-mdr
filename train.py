#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Main Training Script
============================================================================
Fine-tunes Foundation-Sec-8B-Instruct for cybersecurity threat detection
using QLoRA on DGX Spark. Streams live metrics to the dashboard.

Usage:
    python train.py
    python train.py --config config.yaml
============================================================================
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset, load_dataset
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
# Dataset Loading
# ============================================================================

def format_dataset(ds: Dataset, tokenizer) -> Dataset:
    """Apply chat template to a dataset with 'messages' column, producing 'text' column."""
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


def load_and_format_dataset(cfg: dict, tokenizer) -> tuple[Dataset, Dataset | None]:
    """Load dataset from disk (if prepared) or from HuggingFace with on-the-fly formatting."""
    ds_cfg = cfg["dataset"]
    local_dir = Path(ds_cfg["local_dir"])

    # Try loading pre-prepared splits from disk
    if (local_dir / "train").exists():
        log.info(f"Loading prepared dataset from {local_dir}...")
        from datasets import load_from_disk
        train_ds = load_from_disk(str(local_dir / "train"))
        val_ds = None
        if (local_dir / "val").exists():
            val_ds = load_from_disk(str(local_dir / "val"))
        log.info(f"Loaded train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds else ""))
        # Apply chat template if dataset has messages column (not yet formatted)
        train_ds = format_dataset(train_ds, tokenizer)
        if val_ds:
            val_ds = format_dataset(val_ds, tokenizer)
        return train_ds, val_ds

    # Fall back to downloading and formatting on the fly
    log.info(f"No prepared data found — downloading {ds_cfg['name']}...")
    token = os.environ.get("HF_TOKEN")
    hf_config = ds_cfg.get("config")
    raw = load_dataset(ds_cfg["name"], hf_config, split="train", token=token)
    log.info(f"Downloaded {len(raw)} examples.")

    formatted = format_dataset(raw, tokenizer)

    # Split
    split = formatted.train_test_split(
        test_size=ds_cfg["split_ratios"]["val"] + ds_cfg["split_ratios"]["test"],
        seed=ds_cfg["seed"],
    )
    log.info(f"Split: train={len(split['train'])}, val={len(split['test'])}")
    return split["train"], split["test"]


# ============================================================================
# Main
# ============================================================================

def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the most recent checkpoint-* directory in output_dir."""
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=os.path.getmtime)
    if checkpoints:
        log.info(f"Found {len(checkpoints)} checkpoint(s), latest: {checkpoints[-1].name}")
        return str(checkpoints[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Foundation-Sec-8B-Instruct")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    dash_cfg = cfg["dashboard"]

    # --- Load model ---
    quant_mode = "4-bit quantization" if model_cfg["load_in_4bit"] else "BF16 full precision"
    log.info(f"Loading {model_cfg['name']} with {quant_mode}...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    log.info("Model loaded.")

    # --- Apply LoRA ---
    log.info("Attaching LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )
    log.info("LoRA adapters attached.")

    # --- Chat template ---
    log.info(f"Applying {model_cfg['chat_template']} chat template...")
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg["chat_template"])

    # --- Dataset ---
    train_ds, val_ds = load_and_format_dataset(cfg, tokenizer)

    # --- Callbacks ---
    from dashboard.callback import DashboardCallback
    dashboard_url = f"http://localhost:{dash_cfg['port']}"
    dashboard_cb = DashboardCallback(dashboard_url=dashboard_url)
    log.info(f"Dashboard callback pointing to {dashboard_url}")

    # --- Trainer ---
    log.info("Setting up SFTTrainer...")
    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_cfg.get("logging_dir", f"{output_dir}/logs"), exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        optim=train_cfg["optim"],
        max_steps=train_cfg["max_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        logging_dir=train_cfg.get("logging_dir", f"{output_dir}/logs"),
        # Eval if val set is available
        eval_strategy="steps" if val_ds else "no",
        eval_steps=train_cfg["save_steps"] if val_ds else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        args=training_args,
        callbacks=[dashboard_cb],
    )

    # --- Graceful shutdown on SIGTERM/SIGINT ---
    def graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        log.warning(f"Received {sig_name} — saving checkpoint before exit...")
        trainer.save_model(f"{output_dir}/checkpoint-interrupted")
        tokenizer.save_pretrained(f"{output_dir}/checkpoint-interrupted")
        log.info(f"Emergency checkpoint saved to {output_dir}/checkpoint-interrupted")
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    # --- Train ---
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            log.info(f"Resuming training from {resume_checkpoint}")
        else:
            log.warning("--resume passed but no checkpoints found. Starting fresh.")

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    log.info("Training complete.")

    # --- Save ---
    log.info(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved (LoRA adapters + tokenizer).")

    # --- Export ---
    export_cfg = cfg["export"]
    export_dir = export_cfg["output_dir"]
    os.makedirs(export_dir, exist_ok=True)

    if export_cfg.get("save_merged_16bit"):
        log.info("Saving merged 16-bit model...")
        try:
            model.save_pretrained_merged(
                f"{export_dir}/merged_16bit", tokenizer, save_method="merged_16bit"
            )
            log.info(f"Merged model saved to {export_dir}/merged_16bit")
        except Exception as e:
            log.warning(f"Merged save failed: {e}")

    if export_cfg.get("save_gguf"):
        log.info(f"Exporting GGUF ({export_cfg['gguf_quantization']})...")
        try:
            model.save_pretrained_gguf(
                f"{export_dir}/gguf", tokenizer,
                quantization_method=export_cfg["gguf_quantization"],
            )
            log.info(f"GGUF saved to {export_dir}/gguf")
        except Exception as e:
            log.warning(f"GGUF export failed (may need llama.cpp): {e}")
            log.info("Skipping GGUF — safetensors export was already saved.")

    log.info("All done. Run eval.py next.")


if __name__ == "__main__":
    main()
