#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Native Training Script (No Unsloth)
============================================================================
Fine-tunes Foundation-Sec-8B-Instruct using vanilla transformers + peft + trl
with torch.compile() and native SDPA attention, optimized for DGX Spark
Blackwell (sm_121).

Based on NVIDIA's official DGX Spark fine-tuning playbook:
https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/pytorch-fine-tune

Usage:
    python train_native.py
    python train_native.py --config config.yaml
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
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

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

    if (local_dir / "train").exists():
        log.info(f"Loading prepared dataset from {local_dir}...")
        from datasets import load_from_disk
        train_ds = load_from_disk(str(local_dir / "train"))
        val_ds = None
        if (local_dir / "val").exists():
            val_ds = load_from_disk(str(local_dir / "val"))
        log.info(f"Loaded train={len(train_ds)}" + (f", val={len(val_ds)}" if val_ds else ""))
        train_ds = format_dataset(train_ds, tokenizer)
        if val_ds:
            val_ds = format_dataset(val_ds, tokenizer)
        return train_ds, val_ds

    log.info(f"No prepared data found — downloading {ds_cfg['name']}...")
    token = os.environ.get("HF_TOKEN")
    hf_config = ds_cfg.get("config")
    raw = load_dataset(ds_cfg["name"], hf_config, split="train", token=token)
    log.info(f"Downloaded {len(raw)} examples.")

    formatted = format_dataset(raw, tokenizer)

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
    parser = argparse.ArgumentParser(description="Fine-tune Foundation-Sec-8B-Instruct (native)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing checkpoints)")
    parser.add_argument("--data-dir", default=None, help="Override dataset directory (e.g. data/cleaned)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    dash_cfg = cfg["dashboard"]

    # --- CUDA allocator config (optional, DGX GB10-specific) ---
    cuda_conf = train_cfg.get("cuda_alloc_conf")
    if cuda_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cuda_conf)
        log.info(f"CUDA allocator config: {cuda_conf}")

    # --- Cap CUDA memory (optional, DGX GB10-specific) ---
    # On unified memory (GB10), NVRM deadlocks nvidia-modeset when allocation
    # fails at the driver level. Capping at 85% gives a clean Python OOM instead.
    # Not needed on dedicated-VRAM GPUs like H100.
    mem_fraction = train_cfg.get("cuda_memory_fraction")
    if mem_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(float(mem_fraction))
        log.info(f"CUDA memory capped at {float(mem_fraction):.0%} of device memory.")
    else:
        log.info("No CUDA memory cap set (dedicated VRAM mode).")

    # --- Enable TF32 for Tensor Core performance ---
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log.info("TF32 matmul precision enabled for Tensor Core performance.")

    # --- Load tokenizer ---
    log.info(f"Loading tokenizer for {model_cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Tokenizer loaded.")

    # --- Load model with SDPA attention ---
    attn_impl = model_cfg.get("attn_implementation", "sdpa")
    load_in_4bit = model_cfg.get("load_in_4bit", False)

    if load_in_4bit:
        log.info(f"Loading {model_cfg['name']} with 4-bit quantization + {attn_impl} attention...")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
        )
    else:
        log.info(f"Loading {model_cfg['name']} in BF16 with {attn_impl} attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        model = model.to("cuda")
    log.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    # --- Apply LoRA via peft ---
    log.info("Attaching LoRA adapters via peft...")
    use_gc = lora_cfg.get("use_gradient_checkpointing", False)
    if use_gc and use_gc != "unsloth":
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled.")
    elif use_gc == "unsloth":
        log.info("Skipping Unsloth-style gradient checkpointing, using standard if needed.")

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg.get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- torch.compile() for Blackwell optimization ---
    use_compile = train_cfg.get("torch_compile", False)
    if use_compile:
        log.info("Applying torch.compile() — first few steps will be slower (compilation warmup)...")
        model = torch.compile(model)
        log.info("torch.compile() applied.")

    # --- Dataset ---
    if args.data_dir:
        cfg["dataset"]["local_dir"] = args.data_dir
        log.info(f"Using data directory override: {args.data_dir}")
    train_ds, val_ds = load_and_format_dataset(cfg, tokenizer)

    # --- Callbacks ---
    from dashboard.callback import DashboardCallback, TimeLimitCallback
    dashboard_url = f"http://localhost:{dash_cfg['port']}"
    dashboard_cb = DashboardCallback(dashboard_url=dashboard_url)
    log.info(f"Dashboard callback pointing to {dashboard_url}")

    callbacks = [dashboard_cb]
    time_limit = float(train_cfg.get("time_limit_hours", 0))
    if time_limit > 0:
        callbacks.append(TimeLimitCallback(time_limit_hours=time_limit))
        log.info(f"Time limit: {time_limit} hours")

    # --- Training arguments ---
    log.info("Setting up SFTTrainer...")
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_cfg.get("logging_dir", f"{output_dir}/logs"), exist_ok=True)

    sft_config = SFTConfig(
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
        eval_strategy=train_cfg.get("eval_strategy", "steps" if val_ds else "no"),
        eval_steps=train_cfg["save_steps"] if val_ds else None,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        torch_compile=False,
        dataset_text_field="text",
        max_length=model_cfg["max_seq_length"],
        packing=True,  # Pre-tokenizes into fixed-length tensors — faster than dynamic padding even without sequence combining
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    # --- Cooperative shutdown on SIGINT/SIGTERM ---
    # First signal: set trainer control flags so it finishes the current step
    # and saves a proper numbered checkpoint (with optimizer state).
    # Second signal: emergency save + exit.
    _stop_count = [0]

    def graceful_shutdown(signum, frame):
        _stop_count[0] += 1
        sig_name = signal.Signals(signum).name
        if _stop_count[0] == 1:
            log.warning(f"Received {sig_name} — finishing step and saving checkpoint...")
            trainer.control.should_training_stop = True
            trainer.control.should_save = True
        else:
            log.warning(f"Received {sig_name} again — emergency save and exit...")
            trainer.save_model(f"{output_dir}/checkpoint-interrupted")
            tokenizer.save_pretrained(f"{output_dir}/checkpoint-interrupted")
            log.info(f"Emergency checkpoint saved to {output_dir}/checkpoint-interrupted")
            sys.exit(1)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    # --- Train (auto-resume by default, --fresh to start over) ---
    resume_checkpoint = None
    if not args.fresh:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            log.info(f"Auto-resuming from {resume_checkpoint}")
        else:
            log.info("No checkpoints found — starting fresh.")
    else:
        log.info("--fresh flag set — starting fresh (ignoring existing checkpoints).")

    log.info("Starting training...")
    if use_compile:
        log.info("NOTE: First 1-3 steps will be slow due to torch.compile() warmup.")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    log.info("Training complete.")

    # --- Save ---
    log.info(f"Saving LoRA adapters to {output_dir}...")
    # Unwrap compiled model for saving
    save_model = model
    if hasattr(model, "_orig_mod"):
        save_model = model._orig_mod
    save_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved (LoRA adapters + tokenizer).")

    # --- Merge and export ---
    export_cfg = cfg.get("export", {})
    export_dir = export_cfg.get("output_dir", "outputs/exported")
    os.makedirs(export_dir, exist_ok=True)

    if export_cfg.get("save_merged_16bit"):
        # Free trainer + optimizer memory before merge to avoid peak doubling
        import gc
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Freed trainer memory before merge.")

        log.info("Merging LoRA adapters and saving full model in BF16...")
        try:
            merged = save_model.merge_and_unload()
            merged.save_pretrained(f"{export_dir}/merged_16bit")
            tokenizer.save_pretrained(f"{export_dir}/merged_16bit")
            log.info(f"Merged model saved to {export_dir}/merged_16bit")
            del merged
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            log.warning(f"Merged save failed: {e}")

    log.info("All done. Run eval.py next.")


if __name__ == "__main__":
    main()
