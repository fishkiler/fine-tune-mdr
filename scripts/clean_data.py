#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Data Cleaning Pipeline
============================================================================
Runs cleaning operations on the training dataset.

Modes:
  --db mode (default): Operates on the SQLite database, marking records
                       with validation_status='fail' for cleaning issues.
  --arrow mode:        Legacy mode, operates on Arrow files on disk.

DB Cleaning steps:
  1. Strip artifact prefixes from assistant messages in-place
  2. Flag records with too-short or too-long responses
  3. Dedup is handled by content_hash at ingestion (no separate step needed)

Arrow Cleaning steps (legacy):
  1. Strip artifact prefixes (>, **, *) from assistant responses
  2. Filter by response length (50–512 tokens)
  3. Remove exact-duplicate assistant responses

Usage:
    python scripts/clean_data.py                       # DB mode (default)
    python scripts/clean_data.py --db mdr-database/mdr_dataset.db
    python scripts/clean_data.py --arrow --input data --output data/cleaned
============================================================================
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import yaml

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
# Cleaning Functions
# ============================================================================

# Matches the assistant response section in Foundation-Sec chat format
ASSISTANT_MARKERS = [
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|assistant|>\n",
    "### Assistant:\n",
]

# Artifact prefixes commonly found at the start of assistant responses
ARTIFACT_PREFIX_RE = re.compile(r"^(?:[>*]{1,2}\s*)+")


def extract_assistant_response(text: str) -> tuple[str | None, int, int]:
    """Find the assistant response in a formatted training example.

    Returns (response_text, start_idx, end_idx) where indices point to
    the response content within the full text string.
    Returns (None, -1, -1) if no assistant marker is found.
    """
    for marker in ASSISTANT_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            start = idx + len(marker)
            # Find the end of the response (next special token or end of string)
            end = len(text)
            for token in ["<|eot_id|>", "</s>", "<|end_of_text|>"]:
                tok_idx = text.find(token, start)
                if tok_idx != -1:
                    end = min(end, tok_idx)
            return text[start:end].strip(), start, end
    return None, -1, -1


def strip_artifact_prefixes(text: str) -> str:
    """Strip artifact prefixes from a full training example's assistant response."""
    response, start, end = extract_assistant_response(text)
    if response is None:
        return text

    # Strip leading artifact characters from each line of the response
    cleaned_lines = []
    for line in response.split("\n"):
        cleaned_lines.append(ARTIFACT_PREFIX_RE.sub("", line))
    cleaned_response = "\n".join(cleaned_lines)

    if cleaned_response != response:
        # Reconstruct the full text with the cleaned response
        return text[:start] + cleaned_response + text[end:]
    return text


def get_response_token_count(text: str, tokenizer) -> int:
    """Count tokens in the assistant response portion of a training example."""
    response, _, _ = extract_assistant_response(text)
    if response is None:
        return 0
    return len(tokenizer.encode(response, add_special_tokens=False))


# ============================================================================
# Main
# ============================================================================

def clean_database(db_path: str, min_chars: int = 50, max_chars: int = 10000,
                    batch_size: int = 5000) -> dict:
    """Clean records in the SQLite database.

    - Strips artifact prefixes from assistant messages in-place
    - Flags records with too-short or too-long responses as validation failures
    """
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from scripts.db_utils import get_connection, compute_stats

    conn = get_connection(db_path)
    cur = conn.cursor()

    stats = {"total": 0, "artifacts_cleaned": 0, "flagged_short": 0, "flagged_long": 0}

    log.info("Step 1: Stripping artifact prefixes from assistant messages...")
    cur.execute("SELECT id, assistant_message FROM all_records")
    update_batch = []

    for row in cur.fetchall():
        rec_id, assistant_msg = row
        if not assistant_msg:
            continue

        stats["total"] += 1

        # Strip artifact prefixes from each line
        lines = assistant_msg.split("\n")
        cleaned_lines = [ARTIFACT_PREFIX_RE.sub("", line) for line in lines]
        cleaned = "\n".join(cleaned_lines)

        if cleaned != assistant_msg:
            stats["artifacts_cleaned"] += 1
            update_batch.append((cleaned, rec_id))

        if len(update_batch) >= batch_size:
            cur.executemany(
                "UPDATE all_records SET assistant_message = ? WHERE id = ?",
                update_batch,
            )
            conn.commit()
            update_batch = []

    if update_batch:
        cur.executemany(
            "UPDATE all_records SET assistant_message = ? WHERE id = ?",
            update_batch,
        )
        conn.commit()

    log.info(f"  Cleaned {stats['artifacts_cleaned']:,} records with artifacts")

    log.info("Step 2: Flagging records by response length...")
    # Flag too-short responses
    cur.execute(
        "UPDATE all_records SET validation_status = 'fail', "
        "validation_errors = ? "
        "WHERE LENGTH(assistant_message) < ? AND (validation_status IS NULL OR validation_status != 'fail')",
        (json.dumps([{"check": "response_too_short", "severity": "error",
                       "msg": f"Assistant response shorter than {min_chars} chars"}]),
         min_chars),
    )
    stats["flagged_short"] = cur.rowcount

    # Flag too-long responses
    cur.execute(
        "UPDATE all_records SET validation_status = 'fail', "
        "validation_errors = ? "
        "WHERE LENGTH(assistant_message) > ? AND (validation_status IS NULL OR validation_status != 'fail')",
        (json.dumps([{"check": "response_too_long", "severity": "error",
                       "msg": f"Assistant response longer than {max_chars} chars"}]),
         max_chars),
    )
    stats["flagged_long"] = cur.rowcount

    conn.commit()
    log.info(f"  Flagged too short (<{min_chars} chars): {stats['flagged_short']:,}")
    log.info(f"  Flagged too long (>{max_chars} chars): {stats['flagged_long']:,}")

    log.info("Step 3: Dedup is handled by content_hash at ingestion — skipped.")

    compute_stats(conn)
    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean prepared dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--db", default=None, help="Database path (DB mode)")
    parser.add_argument("--arrow", action="store_true", help="Use legacy Arrow mode")
    parser.add_argument("--input", default=None, help="Input data directory (Arrow mode)")
    parser.add_argument("--output", default=None, help="Output directory (Arrow mode)")
    parser.add_argument("--min-tokens", type=int, default=50, help="Min assistant response tokens (Arrow)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max assistant response tokens (Arrow)")
    parser.add_argument("--min-chars", type=int, default=50, help="Min assistant response chars (DB)")
    parser.add_argument("--max-chars", type=int, default=10000, help="Max assistant response chars (DB)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── DB mode (default) ──
    if not args.arrow:
        _project_root = Path(__file__).resolve().parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        from scripts.db_utils import DEFAULT_DB_PATH

        db_path = args.db or str(cfg.get("database", {}).get("path", DEFAULT_DB_PATH))
        if not Path(db_path).exists():
            log.error(f"Database not found: {db_path}")
            log.error("Run 'python mdr-database/build_dataset_db.py' first.")
            sys.exit(1)

        log.info("=" * 60)
        log.info("  MDR Data Cleaning (Database Mode)")
        log.info("=" * 60)

        stats = clean_database(db_path, args.min_chars, args.max_chars)

        log.info("")
        log.info("=" * 60)
        log.info("  CLEANING SUMMARY (Database)")
        log.info("=" * 60)
        log.info(f"  Total records:        {stats['total']:>10,}")
        log.info(f"  Artifacts cleaned:    {stats['artifacts_cleaned']:>10,}")
        log.info(f"  Flagged too short:    {stats['flagged_short']:>10,}")
        log.info(f"  Flagged too long:     {stats['flagged_long']:>10,}")
        return

    # ── Arrow mode (legacy) ──
    from datasets import Dataset, load_from_disk
    from transformers import AutoTokenizer

    input_dir = Path(args.input or cfg["dataset"]["local_dir"])
    output_dir = Path(args.output or "data/cleaned")

    # --- Load tokenizer for token counting ---
    model_name = cfg["model"]["name"]
    log.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log.info("Tokenizer loaded.")

    # --- Process each split ---
    splits = ["train", "val", "test"]
    stats = {}

    for split_name in splits:
        split_path = input_dir / split_name
        if not split_path.exists():
            log.warning(f"Split '{split_name}' not found at {split_path}, skipping.")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Processing split: {split_name}")
        log.info(f"{'='*60}")

        ds = load_from_disk(str(split_path))
        original_count = len(ds)
        log.info(f"Loaded {original_count:,} examples.")

        # --- Step 1: Strip artifact prefixes ---
        log.info("Step 1: Stripping artifact prefixes...")
        artifact_count = 0

        def clean_artifacts(example):
            nonlocal artifact_count
            original = example["text"]
            cleaned = strip_artifact_prefixes(original)
            if cleaned != original:
                artifact_count += 1
            return {"text": cleaned}

        ds = ds.map(clean_artifacts, desc=f"Cleaning artifacts ({split_name})")
        log.info(f"  Cleaned {artifact_count:,} examples with artifact prefixes "
                 f"({artifact_count/original_count:.1%})")

        # --- Step 2: Filter by response length ---
        log.info(f"Step 2: Filtering by response length ({args.min_tokens}–{args.max_tokens} tokens)...")

        def compute_response_length(example):
            return {"_response_tokens": get_response_token_count(example["text"], tokenizer)}

        ds = ds.map(compute_response_length, desc=f"Counting tokens ({split_name})")

        too_short = sum(1 for t in ds["_response_tokens"] if t < args.min_tokens)
        too_long = sum(1 for t in ds["_response_tokens"] if t > args.max_tokens)
        no_response = sum(1 for t in ds["_response_tokens"] if t == 0)

        ds = ds.filter(
            lambda ex: args.min_tokens <= ex["_response_tokens"] <= args.max_tokens,
            desc=f"Length filter ({split_name})",
        )
        ds = ds.remove_columns(["_response_tokens"])

        after_length = len(ds)
        removed_by_length = original_count - after_length
        log.info(f"  Too short (<{args.min_tokens} tokens): {too_short:,}")
        log.info(f"  Too long (>{args.max_tokens} tokens): {too_long:,}")
        log.info(f"  No assistant response: {no_response:,}")
        log.info(f"  Removed: {removed_by_length:,} ({removed_by_length/original_count:.1%})")

        # --- Step 3: Exact deduplication on assistant response ---
        log.info("Step 3: Removing duplicate assistant responses...")

        seen_responses = set()
        dup_indices = []

        for i, text in enumerate(ds["text"]):
            response, _, _ = extract_assistant_response(text)
            if response is None:
                continue
            # Normalize whitespace for dedup comparison
            normalized = " ".join(response.split()).lower()
            if normalized in seen_responses:
                dup_indices.append(i)
            else:
                seen_responses.add(normalized)

        if dup_indices:
            keep_mask = [True] * len(ds)
            for idx in dup_indices:
                keep_mask[idx] = False
            ds = ds.select([i for i, keep in enumerate(keep_mask) if keep])

        dup_count = len(dup_indices)
        final_count = len(ds)
        log.info(f"  Duplicates removed: {dup_count:,} ({dup_count/after_length:.1%})")

        # --- Save cleaned split ---
        out_path = output_dir / split_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out_path))
        log.info(f"Saved {final_count:,} examples to {out_path}")

        stats[split_name] = {
            "original": original_count,
            "artifacts_cleaned": artifact_count,
            "removed_too_short": too_short,
            "removed_too_long": too_long,
            "removed_no_response": no_response,
            "removed_duplicates": dup_count,
            "final": final_count,
            "removed_total": original_count - final_count,
            "removed_pct": (original_count - final_count) / original_count if original_count > 0 else 0,
        }

    # --- Summary ---
    log.info(f"\n{'='*60}")
    log.info("CLEANING SUMMARY")
    log.info(f"{'='*60}")
    for split_name, s in stats.items():
        log.info(f"  {split_name}:")
        log.info(f"    Original:          {s['original']:>8,}")
        log.info(f"    Artifacts cleaned: {s['artifacts_cleaned']:>8,}")
        log.info(f"    Removed (short):   {s['removed_too_short']:>8,}")
        log.info(f"    Removed (long):    {s['removed_too_long']:>8,}")
        log.info(f"    Removed (dedup):   {s['removed_duplicates']:>8,}")
        log.info(f"    Final:             {s['final']:>8,}  ({s['removed_pct']:.1%} removed)")

    total_orig = sum(s["original"] for s in stats.values())
    total_final = sum(s["final"] for s in stats.values())
    log.info(f"\n  TOTAL: {total_orig:,} → {total_final:,} "
             f"({total_orig - total_final:,} removed, "
             f"{(total_orig - total_final) / total_orig:.1%})")

    # --- Save manifest ---
    manifest = {
        "cleaning_config": {
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
        },
        "splits": stats,
        "total_original": total_orig,
        "total_final": total_final,
    }
    manifest_path = output_dir / "cleaning_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
