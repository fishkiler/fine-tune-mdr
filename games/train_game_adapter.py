#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Game Adapter VLM Training Script
============================================================================
Fine-tunes Qwen3.5-9B with LoRA adapters for game action prediction.
Uses transformers.Trainer with custom multimodal dataset and collator
(SFTTrainer's text-only assumptions don't work with image+text inputs).

Follows train_native.py patterns: config-driven, DashboardCallback,
2-stage SIGINT shutdown, auto-resume from checkpoints, memory cleanup
before merge.

Usage:
    python train_game_adapter.py --game pacman
    python train_game_adapter.py --game pacman --fresh
    python train_game_adapter.py --game pacman --config config.yaml
============================================================================
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================================
# Config
# ============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Dataset
# ============================================================================

class GameFrameDataset(TorchDataset):
    """Multimodal dataset that loads game frames + conversation text.

    Each item returns a dict with input_ids, attention_mask, pixel_values,
    image_grid_thw, and labels — ready for the Qwen3.5 VLM.
    """

    def __init__(self, jsonl_path: str, processor, frames_base_dir: str, max_length: int = 2048):
        self.processor = processor
        self.frames_base_dir = Path(frames_base_dir)
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        log.info(f"Loaded {len(self.examples):,} examples from {jsonl_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example["messages"]

        # Extract image path from user message content
        image_path = None
        text_parts = []

        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "image":
                        image_path = part["image"]
                    elif part.get("type") == "text":
                        text_parts.append(part["text"])

        # Load image
        if image_path:
            full_path = self.frames_base_dir / image_path
            image = Image.open(full_path).convert("RGB")
        else:
            # Fallback: create a black image (shouldn't happen in practice)
            image = Image.new("RGB", (288, 224), (0, 0, 0))

        # Build multimodal messages for the processor's chat template
        # Use list-of-dicts content format so the template inserts
        # <|vision_start|><|image_pad|><|vision_end|> correctly
        template_messages = []
        for msg in messages:
            if msg["role"] == "system":
                template_messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "user":
                if isinstance(msg["content"], list):
                    # Keep multimodal content format: [{"type":"image"}, {"type":"text","text":"..."}]
                    content_parts = []
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            content_parts.append({"type": "image"})
                        elif part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part["text"]})
                    template_messages.append({"role": "user", "content": content_parts})
                else:
                    template_messages.append(msg)
            elif msg["role"] == "assistant":
                template_messages.append({"role": "assistant", "content": msg["content"]})

        # Apply chat template with multimodal content
        text = self.processor.apply_chat_template(
            template_messages, tokenize=False, add_generation_prompt=False
        )

        # Process with the VLM processor (handles image + text tokenization)
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dimension (processor returns [1, ...])
        # Keep image_grid_thw as 2D [1, 3] so collator can cat to [batch, 3]
        result = {}
        for k, v in inputs.items():
            if k == "image_grid_thw":
                result[k] = v  # keep [1, 3]
            else:
                result[k] = v.squeeze(0)

        # Create labels: copy input_ids, mask non-assistant tokens with -100
        labels = result["input_ids"].clone()

        # Find the assistant response section and mask everything before it
        # The assistant response starts after the last assistant header token
        tokenizer = self.processor.tokenizer
        # Mask system and user tokens — only train on assistant output
        assistant_token = tokenizer.encode("assistant", add_special_tokens=False)
        input_ids = result["input_ids"]

        # Find last occurrence of assistant role marker
        mask_end = 0
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i + len(assistant_token)].tolist() == assistant_token:
                mask_end = i + len(assistant_token)

        # Mask everything up to and including the assistant header
        if mask_end > 0:
            labels[:mask_end] = -100

        # Also mask padding tokens
        labels[labels == tokenizer.pad_token_id] = -100

        result["labels"] = labels

        return result


class VLMDataCollator:
    """Collator for variable-size VLM inputs.

    Handles batching of pixel_values (which may have different sizes per image)
    and standard text tensors.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        # Standard text tensors — stack directly
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Image tensors — need special handling for variable sizes
        if "pixel_values" in features[0]:
            # For Qwen2-VL / Qwen3.5, pixel_values can be concatenated
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)

        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        return batch


# ============================================================================
# Main
# ============================================================================

def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the most recent checkpoint-* directory."""
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(out.glob("checkpoint-*"), key=os.path.getmtime)
    if checkpoints:
        log.info(f"Found {len(checkpoints)} checkpoint(s), latest: {checkpoints[-1].name}")
        return str(checkpoints[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Train game adapter (VLM LoRA fine-tuning)")
    parser.add_argument("--game", required=True, help="Game name (e.g. pacman)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoints)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ga_cfg = cfg.get("game_adapters", {})
    game_cfg = ga_cfg.get("games", {}).get(args.game)

    if not game_cfg:
        log.error(f"Game '{args.game}' not found in config.yaml game_adapters.games")
        sys.exit(1)

    base_model = ga_cfg["base_model"]
    lora_cfg = ga_cfg["lora"]
    train_cfg = ga_cfg["training"]
    image_cfg = ga_cfg.get("image", {})
    dash_cfg = cfg.get("dashboard", {})

    # --- Paths ---
    dataset_dir = Path(game_cfg["dataset_dir"])
    adapter_dir = Path(game_cfg["adapter_dir"])
    output_dir = str(adapter_dir / "training_output")
    train_jsonl = dataset_dir / "training" / "train.jsonl"
    test_jsonl = dataset_dir / "training" / "test.jsonl"

    if not train_jsonl.exists():
        log.error(f"Training data not found: {train_jsonl}")
        log.error("Run: python scripts/export_game_training_data.py --game " + args.game)
        sys.exit(1)

    # --- CUDA setup ---
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log.info("TF32 matmul precision enabled.")

    # Cap CUDA memory on unified memory systems (GB10)
    cuda_mem_frac = train_cfg.get("cuda_memory_fraction")
    if cuda_mem_frac:
        torch.cuda.set_per_process_memory_fraction(float(cuda_mem_frac))
        log.info(f"CUDA memory capped at {float(cuda_mem_frac):.0%}")

    # --- Load processor ---
    log.info(f"Loading processor for {base_model}...")
    proc_kwargs = {}
    if image_cfg.get("min_pixels"):
        proc_kwargs["min_pixels"] = image_cfg["min_pixels"]
    if image_cfg.get("max_pixels"):
        proc_kwargs["max_pixels"] = image_cfg["max_pixels"]
    processor = AutoProcessor.from_pretrained(base_model, **proc_kwargs)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if proc_kwargs:
        log.info(f"Processor loaded with {proc_kwargs}")
    else:
        log.info("Processor loaded.")

    # --- Load model ---
    attn_impl = ga_cfg.get("attention", "sdpa")
    precision = ga_cfg.get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    log.info(f"Loading {base_model} in {precision} with {attn_impl} attention...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model = model.to("cuda")
    log.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    # --- Apply LoRA ---
    log.info("Attaching LoRA adapters...")
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled.")

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Datasets ---
    frames_base = str(PROJECT_ROOT / "data" / "games" / args.game)
    max_seq = train_cfg.get("max_seq_length", 2048)

    log.info(f"Loading training data from {train_jsonl}...")
    train_ds = GameFrameDataset(str(train_jsonl), processor, frames_base, max_seq)

    eval_ds = None
    if test_jsonl.exists():
        log.info(f"Loading eval data from {test_jsonl}...")
        eval_ds = GameFrameDataset(str(test_jsonl), processor, frames_base, max_seq)

    # --- Callbacks ---
    from dashboard.callback import DashboardCallback, TimeLimitCallback
    dashboard_url = f"http://localhost:{dash_cfg.get('port', 28000)}"
    callbacks = [DashboardCallback(dashboard_url=dashboard_url)]
    log.info(f"Dashboard callback pointing to {dashboard_url}")

    time_limit = float(train_cfg.get("time_limit_hours", 0))
    if time_limit > 0:
        callbacks.append(TimeLimitCallback(time_limit_hours=time_limit))

    # --- Training arguments ---
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = str(Path(output_dir) / "logs")
    os.makedirs(logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 5),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        optim="adamw_torch",
        logging_steps=1,
        save_steps=500,
        save_total_limit=3,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        seed=42,
        report_to="tensorboard",
        logging_dir=logging_dir,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        remove_unused_columns=False,  # CRITICAL: keep pixel_values through pipeline
        torch_compile=False,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=VLMDataCollator(processor),
        callbacks=callbacks,
    )

    # --- Cooperative shutdown on SIGINT/SIGTERM ---
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
            processor.save_pretrained(f"{output_dir}/checkpoint-interrupted")
            log.info(f"Emergency checkpoint saved to {output_dir}/checkpoint-interrupted")
            sys.exit(1)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    # --- Train ---
    resume_checkpoint = None
    if not args.fresh:
        resume_checkpoint = find_latest_checkpoint(output_dir)
        if resume_checkpoint:
            log.info(f"Auto-resuming from {resume_checkpoint}")
        else:
            log.info("No checkpoints found — starting fresh.")
    else:
        log.info("--fresh flag set — starting fresh.")

    log.info(f"Starting training for {args.game}...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    log.info("Training complete.")

    # --- Save adapter ---
    final_adapter_dir = str(adapter_dir / "v1")
    os.makedirs(final_adapter_dir, exist_ok=True)

    log.info(f"Saving LoRA adapter to {final_adapter_dir}...")
    save_model = model
    if hasattr(model, "_orig_mod"):
        save_model = model._orig_mod
    save_model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    # Save training metadata
    metadata = {
        "game": args.game,
        "base_model": base_model,
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["alpha"],
        "train_examples": len(train_ds),
        "test_examples": len(eval_ds) if eval_ds else 0,
        "epochs": train_cfg.get("epochs", 5),
        "learning_rate": train_cfg.get("learning_rate", 2e-4),
    }
    with open(Path(final_adapter_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Adapter saved.")

    # --- Cleanup ---
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Freed trainer memory.")
    log.info(f"All done. Adapter saved to {final_adapter_dir}")
    log.info(f"Run: python eval_game_adapter.py --game {args.game} --mode offline")


if __name__ == "__main__":
    main()
