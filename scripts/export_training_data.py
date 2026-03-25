#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Quality-Filtered Training Data Export
============================================================================
Exports records from the SQLite database to Arrow format for training,
applying quality filters and domain balancing.

Export criteria:
  - validation_status = 'pass' (or NULL if not yet validated)
  - quality_score >= threshold (configurable, default 3.5)
  - Domain weighting for balanced training

Usage:
    python scripts/export_training_data.py
    python scripts/export_training_data.py --quality-threshold 4.0
    python scripts/export_training_data.py --since 2026-02-21
    python scripts/export_training_data.py --new-only     # only records never exported before
    python scripts/export_training_data.py --no-balance   # export all without rebalancing
    python scripts/export_training_data.py --config config.yaml
============================================================================
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml

from scripts.db_utils import DEFAULT_DB_PATH, DOMAINS, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Default Export Configuration
# ============================================================================

DEFAULT_DOMAIN_WEIGHTS = {
    "cve": 0.08,                # Keep ~50K of 618K (top quality)
    "mitre_attack": 10.0,       # 1,317 x 10 = ~13K
    "secure_code_review": 30.0, # 348 x 30 = ~10K
    "security_general": 8.0,    # 1,236 x 8 = ~10K
    "apt_intel": 1.0,
    "exploitdb": 1.0,
    "stix_general": 1.0,
}

DEFAULT_QUALITY_THRESHOLD = 3.5


# ============================================================================
# Export Functions
# ============================================================================

def fetch_exportable_records(db_path: str, quality_threshold: float,
                              since: str | None = None,
                              domain: str | None = None,
                              require_validation: bool = False,
                              new_only: bool = False) -> dict[str, list[dict]]:
    """Fetch records that pass quality filters, grouped by domain."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    conditions = []
    params = []

    # Quality filter
    if quality_threshold > 0:
        conditions.append("(quality_score >= ? OR quality_score IS NULL)")
        params.append(quality_threshold)

    # Validation filter
    if require_validation:
        conditions.append("validation_status = 'pass'")
    else:
        conditions.append("(validation_status != 'fail' OR validation_status IS NULL)")

    # New-only filter: records never exported before
    if new_only:
        conditions.append("exported_at IS NULL")

    # Date filter for incremental export
    if since:
        conditions.append("ingested_at >= ?")
        params.append(since)

    # Domain filter
    if domain:
        conditions.append("domain = ?")
        params.append(domain)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    cur.execute(
        f"SELECT id, domain, user_message, assistant_message, quality_score "
        f"FROM all_records {where} "
        f"ORDER BY COALESCE(quality_score, 3.0) DESC",
        params,
    )

    by_domain: dict[str, list[dict]] = {}
    for row in cur.fetchall():
        rec = {
            "id": row[0],
            "domain": row[1],
            "user_message": row[2],
            "assistant_message": row[3],
            "quality_score": row[4],
        }
        domain_name = rec["domain"]
        if domain_name not in by_domain:
            by_domain[domain_name] = []
        by_domain[domain_name].append(rec)

    conn.close()
    return by_domain


def apply_domain_weights(records_by_domain: dict[str, list[dict]],
                          weights: dict[str, float],
                          seed: int = 42) -> list[dict]:
    """Apply domain weights to produce a balanced training set.

    weight < 1.0: subsample (keep top N by quality)
    weight = 1.0: keep as-is
    weight > 1.0: upsample by repeating records
    """
    rng = random.Random(seed)
    result = []

    for domain, records in records_by_domain.items():
        weight = weights.get(domain, 1.0)
        n = len(records)

        if weight < 1.0:
            # Subsample: keep top N*weight records (already sorted by quality)
            keep = max(1, int(n * weight))
            selected = records[:keep]
            log.info(f"  {domain}: {n:,} -> {keep:,} (cap {weight:.0%})")
        elif weight == 1.0:
            selected = records
            log.info(f"  {domain}: {n:,} (kept as-is)")
        else:
            # Upsample: repeat records
            repeats = int(weight)
            selected = records * repeats
            # Add partial repeat for fractional part
            frac = weight - repeats
            if frac > 0:
                extra = int(n * frac)
                selected.extend(rng.sample(records, min(extra, n)))
            rng.shuffle(selected)
            log.info(f"  {domain}: {n:,} -> {len(selected):,} (x{weight:.1f})")

        result.extend(selected)

    rng.shuffle(result)
    return result


def export_to_arrow(records: list[dict], output_dir: str, split: str = "train") -> Path:
    """Export records to Arrow dataset format."""
    try:
        from datasets import Dataset
    except ImportError:
        log.error("datasets library required. Run: pip install datasets")
        sys.exit(1)

    # Convert to messages format
    messages_list = []
    for rec in records:
        messages = [
            {"role": "user", "content": rec["user_message"]},
            {"role": "assistant", "content": rec["assistant_message"]},
        ]
        messages_list.append(messages)

    ds = Dataset.from_dict({"messages": messages_list})

    out_path = Path(output_dir) / split
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    return out_path


def export_to_jsonl(records: list[dict], output_path: str) -> Path:
    """Export records to JSONL format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            messages = [
                {"role": "user", "content": rec["user_message"]},
                {"role": "assistant", "content": rec["assistant_message"]},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    return out_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export quality-filtered training data")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--output", default="data/export", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--quality-threshold", type=float, default=None,
                        help="Minimum quality score (default: from config or 3.5)")
    parser.add_argument("--since", type=str,
                        help="Only export records ingested after this date (YYYY-MM-DD)")
    parser.add_argument("--domain", type=str, help="Only export this domain")
    parser.add_argument("--new-only", action="store_true",
                        help="Only export records never exported before (exported_at IS NULL)")
    parser.add_argument("--no-balance", action="store_true",
                        help="Export all records without domain rebalancing")
    parser.add_argument("--require-validation", action="store_true",
                        help="Only export records that passed validation")
    parser.add_argument("--format", choices=["arrow", "jsonl", "both"], default="arrow",
                        help="Output format (default: arrow)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Load config for domain weights
    weights = DEFAULT_DOMAIN_WEIGHTS.copy()
    quality_threshold = args.quality_threshold or DEFAULT_QUALITY_THRESHOLD

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        export_cfg = cfg.get("export", {}).get("training_data", {})
        if export_cfg.get("domain_weights"):
            weights.update(export_cfg["domain_weights"])
        if export_cfg.get("quality_threshold") and args.quality_threshold is None:
            quality_threshold = export_cfg["quality_threshold"]

    t0 = time.time()

    log.info("=" * 60)
    log.info("  MDR Training Data Export")
    log.info("=" * 60)
    log.info(f"  Quality threshold: >= {quality_threshold}")
    log.info(f"  Require validation: {args.require_validation}")
    log.info(f"  New only: {args.new_only}")
    if args.since:
        log.info(f"  Since: {args.since}")
    if args.domain:
        log.info(f"  Domain: {args.domain}")
    log.info(f"  Balance: {'no' if args.no_balance else 'yes'}")

    # Show last export info
    conn = get_connection(args.db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='export_history'")
    if cur.fetchone():
        cur.execute("SELECT exported_at, record_count FROM export_history ORDER BY id DESC LIMIT 1")
        last = cur.fetchone()
        if last:
            log.info(f"  Last export: {last[0]} ({last[1]:,} records)")
        else:
            log.info("  Last export: never")
    cur.execute("SELECT COUNT(*) FROM all_records WHERE exported_at IS NULL")
    never_exported = cur.fetchone()[0]
    log.info(f"  Never exported: {never_exported:,} records")
    conn.close()
    log.info("")

    # Fetch records
    records_by_domain = fetch_exportable_records(
        args.db, quality_threshold, args.since, args.domain,
        args.require_validation, args.new_only,
    )

    total_available = sum(len(recs) for recs in records_by_domain.values())
    log.info(f"Records passing quality filter: {total_available:,}")
    for domain, recs in sorted(records_by_domain.items(), key=lambda x: -len(x[1])):
        log.info(f"  {domain:<25} {len(recs):>10,}")
    log.info("")

    # Apply domain balancing
    if args.no_balance:
        all_records = []
        for recs in records_by_domain.values():
            all_records.extend(recs)
        random.Random(args.seed).shuffle(all_records)
        log.info("Domain balancing: DISABLED")
    else:
        log.info("Applying domain weights:")
        all_records = apply_domain_weights(records_by_domain, weights, args.seed)

    log.info(f"\nFinal training set: {len(all_records):,} records")

    # Export
    if args.format in ("arrow", "both"):
        out_path = export_to_arrow(all_records, args.output, "train")
        log.info(f"Arrow dataset saved to: {out_path}")

    if args.format in ("jsonl", "both"):
        out_path = export_to_jsonl(all_records, str(Path(args.output) / "train.jsonl"))
        log.info(f"JSONL saved to: {out_path}")

    # Save export manifest
    domain_counts = {d: len(r) for d, r in records_by_domain.items()}
    manifest = {
        "total_exported": len(all_records),
        "quality_threshold": quality_threshold,
        "domain_weights": weights if not args.no_balance else "disabled",
        "require_validation": args.require_validation,
        "since": args.since,
        "new_only": args.new_only,
        "seed": args.seed,
        "domain_counts": domain_counts,
    }
    manifest_path = Path(args.output) / "export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Stamp exported records and write export history
    export_ts = datetime.now(timezone.utc).isoformat()
    exported_ids = [rec["id"] for rec in all_records]
    # Deduplicate IDs (upsampling can repeat records)
    unique_ids = list(set(exported_ids))

    conn = get_connection(args.db)
    cur = conn.cursor()

    # Stamp records in batches
    batch_size = 500
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cur.execute(
            f"UPDATE all_records SET exported_at = ? WHERE id IN ({placeholders})",
            [export_ts] + batch,
        )
    log.info(f"Stamped {len(unique_ids):,} records with exported_at = {export_ts}")

    # Ensure export_history table exists (handles pre-v3 databases)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS export_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exported_at TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            quality_threshold REAL,
            domain_weights TEXT,
            since_filter TEXT,
            output_path TEXT,
            format TEXT,
            domain_counts TEXT,
            notes TEXT
        )
    """)

    cur.execute(
        "INSERT INTO export_history "
        "(exported_at, record_count, quality_threshold, domain_weights, "
        "since_filter, output_path, format, domain_counts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            export_ts,
            len(all_records),
            quality_threshold,
            json.dumps(weights if not args.no_balance else "disabled"),
            args.since,
            str(Path(args.output).resolve()),
            args.format,
            json.dumps(domain_counts),
        ),
    )
    conn.commit()
    conn.close()
    log.info(f"Export history recorded (id={cur.lastrowid})")

    elapsed = time.time() - t0
    log.info(f"\nExport complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
