#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Data Ingestion Pipeline
============================================================================
Incrementally ingests new training data into the SQLite database with
automatic deduplication via content_hash.

Supports:
  - JSONL files: {"messages": [{"role": "...", "content": "..."}, ...]}
  - Arrow datasets: HuggingFace datasets saved with save_to_disk()
  - pentestds output: chatml_train.jsonl / chatml_validate.jsonl

Usage:
    python scripts/ingest_data.py --jsonl data.jsonl
    python scripts/ingest_data.py --jsonl data.jsonl --source "cisa_kev"
    python scripts/ingest_data.py --arrow data/train --split train
    python scripts/ingest_data.py --pentestds .tools/pentest-dataset-builder/data/datasets/dist
============================================================================
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.db_utils import (
    DEFAULT_DB_PATH,
    classify_record,
    compute_stats,
    content_hash,
    create_indexes,
    create_schema,
    extract_all_metadata,
    get_connection,
    migrate_schema,
    parse_messages,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Ingestion from JSONL
# ============================================================================

def ingest_jsonl(db_path: str, jsonl_path: str, source: str = "custom",
                 split: str = "train", batch_size: int = 5000) -> dict:
    """Ingest records from a JSONL file into the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    inserted = 0
    skipped = 0
    errors = 0
    total = 0

    batch = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning(f"Line {line_num}: invalid JSON, skipping")
                errors += 1
                continue

            # Extract messages
            if isinstance(record, dict) and "messages" in record:
                msgs = record["messages"]
            elif isinstance(record, list):
                msgs = record
            else:
                log.warning(f"Line {line_num}: unexpected format, skipping")
                errors += 1
                continue

            user_msg = ""
            assistant_msg = ""
            for m in msgs:
                if m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]

            if not user_msg or not assistant_msg:
                errors += 1
                continue

            meta = extract_all_metadata(user_msg, assistant_msg)

            row = (
                split, meta["domain"], meta["question_type"],
                user_msg, assistant_msg,
                meta["cve_ids"], meta["mitre_techniques"], meta["cwe_ids"],
                meta["severity"], meta["cvss_score"], meta["char_length"],
                meta["content_hash"], source, now,
            )
            batch.append(row)

            if len(batch) >= batch_size:
                i, s = _flush_ingest_batch(cur, batch)
                inserted += i
                skipped += s
                batch = []
                conn.commit()
                log.info(f"  Processed {total:,} lines ({inserted:,} new, {skipped:,} dups)")

    if batch:
        i, s = _flush_ingest_batch(cur, batch)
        inserted += i
        skipped += s
        conn.commit()

    # Update stats
    compute_stats(conn)
    conn.close()

    return {
        "total_lines": total,
        "inserted": inserted,
        "skipped_dups": skipped,
        "errors": errors,
    }


# ============================================================================
# Ingestion from Arrow Datasets
# ============================================================================

def ingest_arrow(db_path: str, arrow_path: str, source: str = "pentestds",
                 split: str = "train", batch_size: int = 5000) -> dict:
    """Ingest records from an Arrow dataset (HuggingFace format)."""
    try:
        from datasets import load_from_disk
    except ImportError:
        log.error("datasets library required. Run: pip install datasets")
        sys.exit(1)

    ds = load_from_disk(arrow_path)
    log.info(f"Loaded {len(ds):,} records from {arrow_path}")

    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    inserted = 0
    skipped = 0
    total = 0
    batch = []

    for record in ds:
        total += 1
        user_msg, assistant_msg = parse_messages(record)

        if not user_msg or not assistant_msg:
            continue

        meta = extract_all_metadata(user_msg, assistant_msg)

        row = (
            split, meta["domain"], meta["question_type"],
            user_msg, assistant_msg,
            meta["cve_ids"], meta["mitre_techniques"], meta["cwe_ids"],
            meta["severity"], meta["cvss_score"], meta["char_length"],
            meta["content_hash"], source, now,
        )
        batch.append(row)

        if len(batch) >= batch_size:
            i, s = _flush_ingest_batch(cur, batch)
            inserted += i
            skipped += s
            batch = []
            conn.commit()

            if total % 50000 == 0:
                log.info(f"  Processed {total:,} / {len(ds):,} ({inserted:,} new, {skipped:,} dups)")

    if batch:
        i, s = _flush_ingest_batch(cur, batch)
        inserted += i
        skipped += s
        conn.commit()

    compute_stats(conn)
    conn.close()

    return {
        "total_records": total,
        "inserted": inserted,
        "skipped_dups": skipped,
    }


# ============================================================================
# Ingestion from pentestds Output
# ============================================================================

def ingest_pentestds(db_path: str, dist_dir: str, batch_size: int = 5000) -> dict:
    """Ingest pentestds ChatML output files."""
    dist_path = Path(dist_dir)
    total_results = {"total_lines": 0, "inserted": 0, "skipped_dups": 0, "errors": 0}

    for jsonl_file in ["chatml_train.jsonl", "chatml_validate.jsonl"]:
        fpath = dist_path / jsonl_file
        if not fpath.exists():
            log.warning(f"File not found: {fpath}")
            continue

        split = "train" if "train" in jsonl_file else "val"
        log.info(f"Ingesting {jsonl_file} as split '{split}'...")
        result = ingest_jsonl(db_path, str(fpath), source="pentestds",
                              split=split, batch_size=batch_size)

        for key in total_results:
            total_results[key] += result.get(key, 0)

    return total_results


# ============================================================================
# Batch Insert Helper
# ============================================================================

def _flush_ingest_batch(cur, batch: list[tuple]) -> tuple[int, int]:
    """Insert a batch of records, silently skipping duplicates.

    Returns (inserted, skipped).
    """
    inserted = 0
    skipped = 0

    for row in batch:
        try:
            cur.execute("""
                INSERT INTO all_records (split, domain, question_type, user_message,
                    assistant_message, cve_ids, mitre_techniques, cwe_ids,
                    severity, cvss_score, char_length, content_hash, source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)

            # Also insert into domain table
            master_id = cur.lastrowid
            domain = row[1]  # domain is at index 1
            domain_row = (master_id, row[0], row[2], row[3], row[4],
                          row[5], row[6], row[7], row[8], row[9], row[10])
            cur.execute(f"""
                INSERT INTO {domain} (master_id, split, question_type,
                    user_message, assistant_message, cve_ids, mitre_techniques,
                    cwe_ids, severity, cvss_score, char_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, domain_row)

            inserted += 1
        except Exception:
            skipped += 1

    return inserted, skipped


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingest data into MDR database")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jsonl", type=str, help="Path to JSONL file")
    group.add_argument("--arrow", type=str, help="Path to Arrow dataset directory")
    group.add_argument("--pentestds", type=str, help="Path to pentestds dist directory")

    parser.add_argument("--source", default=None,
                        help="Source name (default: auto-detect)")
    parser.add_argument("--split", default="train",
                        help="Dataset split name (default: train)")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    db_path = args.db

    # Ensure database exists with current schema
    if not Path(db_path).exists():
        log.info(f"Creating new database at {db_path}")
        conn = get_connection(db_path)
        create_schema(conn)
        create_indexes(conn)
        conn.close()
    else:
        conn = get_connection(db_path)
        migrate_schema(conn)
        conn.close()

    t0 = time.time()

    log.info("=" * 60)
    log.info("  MDR Data Ingestion")
    log.info("=" * 60)

    if args.jsonl:
        source = args.source or "custom"
        log.info(f"  Source: {args.jsonl}")
        log.info(f"  Label: {source}")
        log.info(f"  Split: {args.split}")
        log.info("")
        result = ingest_jsonl(db_path, args.jsonl, source=source,
                              split=args.split, batch_size=args.batch_size)
    elif args.arrow:
        source = args.source or "pentestds"
        log.info(f"  Source: {args.arrow}")
        log.info(f"  Label: {source}")
        log.info(f"  Split: {args.split}")
        log.info("")
        result = ingest_arrow(db_path, args.arrow, source=source,
                              split=args.split, batch_size=args.batch_size)
    elif args.pentestds:
        log.info(f"  Source: {args.pentestds}")
        log.info("")
        result = ingest_pentestds(db_path, args.pentestds,
                                   batch_size=args.batch_size)

    elapsed = time.time() - t0

    log.info("")
    log.info("=" * 60)
    log.info("  INGESTION RESULTS")
    log.info("=" * 60)
    for key, value in result.items():
        log.info(f"  {key:<20} {value:>10,}")
    log.info(f"  {'elapsed':<20} {elapsed:>9.1f}s")


if __name__ == "__main__":
    main()
