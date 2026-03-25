#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Dataset Preparation
============================================================================
Downloads the MITRE STIX CVE dataset, loads custom examples from instruct.md
and data/custom/*.jsonl/*.md, merges everything, applies chat template,
and creates train/val/test splits on disk.

Data sources (in merge order):
  1. HuggingFace dataset (jason-oneal/mitre-stix-cve-exploitdb-dataset)
  2. data/custom/*.md  (markdown conversation format)
  3. data/custom/*.jsonl (ChatML messages format)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config config.yaml
    python scripts/prepare_data.py --custom-only    # skip HuggingFace download
============================================================================
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for scripts.data_utils import
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv

from scripts.data_utils import (
    DEFAULT_SYSTEM_PROMPT,
    load_custom_data,
    parse_jsonl_examples,
    parse_markdown_examples,
)

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
# Format Detection (HuggingFace datasets)
# ============================================================================

def detect_format(dataset: Dataset) -> str:
    """Probe dataset columns to determine the conversation format."""
    columns = dataset.column_names
    log.info(f"Dataset columns: {columns}")

    if "messages" in columns:
        log.info("Detected format: messages (ChatML)")
        return "messages"
    if "conversations" in columns:
        log.info("Detected format: conversations (ShareGPT)")
        return "conversations"
    if "text" in columns:
        log.info("Detected format: text (pre-formatted)")
        return "text"

    raise ValueError(
        f"Cannot detect dataset format. Expected one of 'messages', "
        f"'conversations', or 'text' columns. Got: {columns}"
    )


# ============================================================================
# Formatting Functions
# ============================================================================

def format_messages(example: dict, tokenizer) -> dict:
    """Format a ChatML messages array into a text string."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def format_conversations(example: dict, tokenizer) -> dict:
    """Format ShareGPT-style conversations into a text string."""
    convos = example["conversations"]
    messages = []
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    for turn in convos:
        role = role_map.get(turn.get("from", ""), turn.get("from", ""))
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def format_hf_dataset(raw: Dataset, tokenizer) -> Dataset:
    """Auto-detect and format a HuggingFace dataset."""
    fmt = detect_format(raw)

    if fmt == "text":
        log.info("Text format — no reformatting needed.")
        return raw
    elif fmt == "messages":
        log.info("Applying chat template to messages format...")
        return raw.map(
            lambda ex: format_messages(ex, tokenizer),
            remove_columns=[c for c in raw.column_names if c != "text"],
            desc="Formatting messages",
        )
    elif fmt == "conversations":
        log.info("Applying chat template to conversations format...")
        return raw.map(
            lambda ex: format_conversations(ex, tokenizer),
            remove_columns=[c for c in raw.column_names if c != "text"],
            desc="Formatting conversations",
        )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--custom-only",
        action="store_true",
        help="Skip HuggingFace download, only use custom data",
    )
    parser.add_argument(
        "--no-custom",
        action="store_true",
        help="Skip custom data, only use HuggingFace dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count examples without saving (no model needed)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    # --- Dry run: just count custom examples without loading model ---
    if args.dry_run:
        log.info("Dry run — counting examples without loading model...")
        root = Path(".")
        total = 0

        custom_dir = root / "data" / "custom"
        if custom_dir.exists():
            for f in sorted(custom_dir.glob("*.md")):
                if f.name == "README.md":
                    continue
                convos = parse_markdown_examples(str(f))
                log.info(f"  {f.name}: {len(convos)} examples")
                total += len(convos)
            for f in sorted(custom_dir.glob("*.jsonl")):
                convos = parse_jsonl_examples(str(f))
                log.info(f"  {f.name}: {len(convos)} examples")
                total += len(convos)

        log.info(f"Total custom examples: {total}")
        if not args.custom_only:
            log.info(f"HuggingFace dataset: {ds_cfg['name']} (will be downloaded at runtime)")
        return

    # --- Load tokenizer (native transformers — no Unsloth dependency) ---
    log.info(f"Loading tokenizer for {model_cfg['name']}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Tokenizer loaded.")

    all_texts: list[str] = []

    # --- HuggingFace dataset ---
    if not args.custom_only:
        log.info(f"Downloading dataset: {ds_cfg['name']}...")
        token = os.environ.get("HF_TOKEN")
        raw = load_dataset(ds_cfg["name"], split="train", token=token)
        log.info(f"Loaded {len(raw)} examples from HuggingFace.")

        formatted_hf = format_hf_dataset(raw, tokenizer)
        hf_texts = formatted_hf["text"]
        log.info(f"Formatted {len(hf_texts)} HuggingFace examples.")
        all_texts.extend(hf_texts)

    # --- Custom data ---
    if not args.no_custom:
        custom_texts = load_custom_data(tokenizer, project_root=".")
        all_texts.extend(custom_texts)

    if not all_texts:
        log.error("No training data found. Add examples to instruct.md or data/custom/")
        sys.exit(1)

    log.info(f"Total examples: {len(all_texts)}")

    # --- Build unified dataset ---
    combined = Dataset.from_dict({"text": all_texts})

    # --- Split ---
    ratios = ds_cfg["split_ratios"]
    seed = ds_cfg["seed"]

    log.info(f"Splitting: train={ratios['train']}, val={ratios['val']}, test={ratios['test']}")

    # First split: train vs (val + test)
    split1 = combined.train_test_split(
        test_size=ratios["val"] + ratios["test"],
        seed=seed,
    )
    # Second split: val vs test
    val_test_ratio = ratios["test"] / (ratios["val"] + ratios["test"])
    split2 = split1["test"].train_test_split(
        test_size=val_test_ratio,
        seed=seed,
    )

    splits = DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"],
    })

    log.info(
        f"Split sizes — train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    # --- Save to disk ---
    out_dir = Path(ds_cfg["local_dir"])
    for split_name, split_data in splits.items():
        split_path = out_dir / split_name
        split_data.save_to_disk(str(split_path))
        log.info(f"Saved {split_name} to {split_path}")

    # --- Save manifest ---
    manifest = {
        "total_examples": len(all_texts),
        "hf_examples": len(all_texts) - len(custom_texts) if not args.custom_only else 0,
        "custom_examples": len(custom_texts) if not args.no_custom else 0,
        "splits": {name: len(ds) for name, ds in splits.items()},
        "hf_dataset": ds_cfg["name"] if not args.custom_only else None,
        "seed": seed,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved to {manifest_path}")

    log.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
