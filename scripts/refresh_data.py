#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Dataset Refresh Pipeline
============================================================================
Pulls fresh CVE data using the pentestds builder, ingests into the SQLite
database (with dedup), then optionally exports quality-filtered training data.

Steps:
  1. Install/update pentestds CLI from GitHub
  2. Run `pentestds build` to pull MITRE, NVD, and ExploitDB data
  3. Ingest pentestds output into SQLite database (dedup via content_hash)
  4. Ingest custom data from data/custom/
  5. Run validation on new records
  6. Optionally export quality-filtered Arrow splits

Legacy mode (--legacy): Uses the old Arrow-based pipeline without the database.

Usage:
    python scripts/refresh_data.py                    # full refresh via DB
    python scripts/refresh_data.py --days-back 90     # last 90 days only
    python scripts/refresh_data.py --skip-build       # re-ingest custom data only
    python scripts/refresh_data.py --dry-run           # preview without executing
    python scripts/refresh_data.py --legacy            # old Arrow-based pipeline
============================================================================
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path for scripts.data_utils import
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PENTESTDS_REPO = "https://github.com/jason-allen-oneal/pentest-dataset-builder.git"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Step 1: Install / Update pentestds
# ============================================================================

def ensure_pentestds(tools_dir: str, dry_run: bool = False) -> Path:
    """Clone or update the pentestds repository and ensure it's pip-installed."""
    tools = Path(tools_dir)
    repo_dir = tools / "pentest-dataset-builder"

    if repo_dir.exists():
        log.info(f"pentestds found at {repo_dir}, pulling latest...")
        if dry_run:
            log.info("[DRY RUN] Would run: git pull")
            return repo_dir
        subprocess.run(
            ["git", "pull"],
            cwd=str(repo_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("pentestds updated.")
    else:
        log.info(f"Cloning pentestds to {repo_dir}...")
        if dry_run:
            log.info(f"[DRY RUN] Would clone {PENTESTDS_REPO}")
            return repo_dir
        tools.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", PENTESTDS_REPO, str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("pentestds cloned.")

    # Ensure pip-installed in editable mode
    if not dry_run:
        log.info("Installing pentestds (pip install -e)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(repo_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        log.info("pentestds installed.")

    return repo_dir


# ============================================================================
# Step 2: Run pentestds build
# ============================================================================

def run_pentestds_build(
    repo_dir: Path,
    nvd_api_key: str | None,
    days_back: int,
    window_days: int,
    sleep_seconds: int,
    dry_run: bool = False,
) -> Path:
    """Run `pentestds build` and return the output directory."""
    dist_dir = repo_dir / "data" / "datasets" / "dist"

    env = os.environ.copy()

    # Scope controls
    env["NVD_MAX_DAYS_BACK"] = str(days_back)
    env["NVD_WINDOW_DAYS"] = str(window_days)

    if nvd_api_key:
        env["NVD_API_KEY"] = nvd_api_key
        env["NVD_SLEEP_SECONDS"] = str(max(sleep_seconds, 1))
        log.info(f"NVD API key set (rate limit: {max(sleep_seconds, 1)}s)")
    else:
        env["NVD_SLEEP_SECONDS"] = str(max(sleep_seconds, 6))
        log.info(f"No NVD API key — using conservative rate limit ({max(sleep_seconds, 6)}s)")

    # Prevent accidental upload to jason-oneal's HuggingFace repo
    env.pop("TOKEN", None)
    env.pop("HF_TOKEN_UPLOAD", None)

    cmd = [sys.executable, "-m", "pentestds", "build"]

    if dry_run:
        log.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        log.info(f"  NVD_MAX_DAYS_BACK={days_back}")
        log.info(f"  NVD_WINDOW_DAYS={window_days}")
        log.info(f"  NVD_SLEEP_SECONDS={env.get('NVD_SLEEP_SECONDS')}")
        log.info(f"  Output dir: {dist_dir}")
        return dist_dir

    log.info("Running pentestds build (this may take a while)...")
    log.info(f"  NVD lookback: {days_back} days, chunk size: {window_days} days")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        log.error(f"pentestds build failed (exit code {result.returncode})")
        log.error(f"stdout: {result.stdout[-2000:] if result.stdout else '(empty)'}")
        log.error(f"stderr: {result.stderr[-2000:] if result.stderr else '(empty)'}")
        sys.exit(1)

    log.info("pentestds build completed.")

    # Validate expected output files
    expected = ["chatml_train.jsonl", "chatml_validate.jsonl"]
    for fname in expected:
        fpath = dist_dir / fname
        if not fpath.exists():
            log.error(f"Expected output file missing: {fpath}")
            sys.exit(1)
        size_mb = fpath.stat().st_size / (1024 * 1024)
        log.info(f"  {fname}: {size_mb:.1f} MB")

    return dist_dir


# ============================================================================
# Step 3: Convert pentestds output → our data/ format
# ============================================================================

def convert_and_merge(
    dist_dir: Path,
    data_dir: Path,
    split_ratios: dict,
    seed: int,
    project_root: str = ".",
    dry_run: bool = False,
) -> dict:
    """
    Read pentestds ChatML output, merge with custom examples,
    re-split into train/val/test, and save to data/.
    """
    from datasets import Dataset, DatasetDict

    from scripts.data_utils import load_custom_conversations

    chatml_train = dist_dir / "chatml_train.jsonl"
    chatml_val = dist_dir / "chatml_validate.jsonl"

    # Count lines for dry run
    if dry_run:
        train_count = sum(1 for _ in open(chatml_train)) if chatml_train.exists() else 0
        val_count = sum(1 for _ in open(chatml_val)) if chatml_val.exists() else 0
        custom_convos = load_custom_conversations(project_root)
        log.info(f"[DRY RUN] pentestds train examples: {train_count}")
        log.info(f"[DRY RUN] pentestds validation examples: {val_count}")
        log.info(f"[DRY RUN] Custom examples: {len(custom_convos)}")
        log.info(f"[DRY RUN] Total before split: {train_count + val_count + len(custom_convos)}")
        log.info(f"[DRY RUN] Split ratios: {split_ratios}")
        return {
            "pentestds_examples": train_count + val_count,
            "custom_examples": len(custom_convos),
            "total_examples": train_count + val_count + len(custom_convos),
        }

    # Load pentestds output — each line is {"messages": [...]}
    log.info("Loading pentestds ChatML output...")
    all_messages: list[list[dict]] = []

    for fpath in [chatml_train, chatml_val]:
        if not fpath.exists():
            log.warning(f"File not found, skipping: {fpath}")
            continue
        with open(fpath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"{fpath.name}:{line_num} — invalid JSON, skipping")
                    continue

                # pentestds format: {"messages": [{"role": ..., "content": ...}]}
                if isinstance(record, dict) and "messages" in record:
                    all_messages.append(record["messages"])
                elif isinstance(record, list):
                    all_messages.append(record)

    pentestds_count = len(all_messages)
    log.info(f"Loaded {pentestds_count} examples from pentestds output.")

    # Load custom examples (raw conversations, no tokenizer needed)
    custom_convos = load_custom_conversations(project_root)
    custom_count = len(custom_convos)
    log.info(f"Loaded {custom_count} custom examples.")

    all_messages.extend(custom_convos)
    total = len(all_messages)
    log.info(f"Total examples after merge: {total}")

    # Build dataset with messages column (ChatML format for train.py)
    dataset = Dataset.from_dict({"messages": all_messages})

    # Split: train / val / test
    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]
    test_ratio = split_ratios["test"]

    log.info(f"Splitting: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    split1 = dataset.train_test_split(
        test_size=val_ratio + test_ratio,
        seed=seed,
    )
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
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

    # Save to disk
    for split_name, split_data in splits.items():
        split_path = data_dir / split_name
        split_data.save_to_disk(str(split_path))
        log.info(f"Saved {split_name} to {split_path}")

    # Write manifest
    manifest = {
        "total_examples": total,
        "pentestds_examples": pentestds_count,
        "custom_examples": custom_count,
        "splits": {name: len(ds) for name, ds in splits.items()},
        "source": "pentestds + custom",
        "refresh_timestamp": datetime.now().isoformat(),
        "seed": seed,
    }
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved to {manifest_path}")

    return manifest


# ============================================================================
# Step 4: Backup previous data
# ============================================================================

def backup_data(data_dir: Path, keep_backups: int, dry_run: bool = False) -> None:
    """Move existing data splits to a timestamped backup directory."""
    split_dirs = [data_dir / name for name in ("train", "val", "test")]
    existing = [d for d in split_dirs if d.exists()]

    if not existing:
        log.info("No existing data to back up.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / f"backup_{timestamp}"

    if dry_run:
        log.info(f"[DRY RUN] Would backup {len(existing)} splits to {backup_dir}")
        return

    backup_dir.mkdir(parents=True, exist_ok=True)
    for split_dir in existing:
        dest = backup_dir / split_dir.name
        shutil.move(str(split_dir), str(dest))
        log.info(f"Backed up {split_dir.name} → {backup_dir.name}/")

    # Also backup manifest if it exists
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        shutil.copy2(str(manifest), str(backup_dir / "manifest.json"))

    # Prune old backups
    backups = sorted(data_dir.glob("backup_*"), key=lambda p: p.name, reverse=True)
    for old_backup in backups[keep_backups:]:
        log.info(f"Removing old backup: {old_backup.name}")
        shutil.rmtree(str(old_backup))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Refresh CVE dataset using pentestds builder"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=None,
        help="Override NVD lookback window (days). Default: from config",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip pentestds build, just re-ingest custom data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use old Arrow-based pipeline (no database)",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation after ingestion",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export quality-filtered training data after refresh",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    refresh_cfg = cfg.get("refresh", {})

    tools_dir = refresh_cfg.get("tools_dir", ".tools")
    days_back = args.days_back or refresh_cfg.get("nvd_max_days_back", 5475)
    window_days = refresh_cfg.get("nvd_window_days", 90)
    sleep_seconds = refresh_cfg.get("nvd_sleep_seconds", 6)
    keep_backups = refresh_cfg.get("keep_backups", 2)
    data_dir = Path(ds_cfg.get("local_dir", "data"))

    nvd_api_key = os.environ.get("NVD_API_KEY")

    log.info("=" * 60)
    log.info("Fine-Tune MDR — Dataset Refresh")
    log.info("=" * 60)

    if args.dry_run:
        log.info("DRY RUN MODE — no changes will be made")
        log.info("")

    # Step 1: Install/update pentestds
    repo_dir = ensure_pentestds(tools_dir, dry_run=args.dry_run)

    # Step 2: Run pentestds build (unless --skip-build)
    dist_dir = repo_dir / "data" / "datasets" / "dist"
    if not args.skip_build:
        dist_dir = run_pentestds_build(
            repo_dir=repo_dir,
            nvd_api_key=nvd_api_key,
            days_back=days_back,
            window_days=window_days,
            sleep_seconds=sleep_seconds,
            dry_run=args.dry_run,
        )
    else:
        log.info("Skipping pentestds build (--skip-build)")
        if not dist_dir.exists() and not args.dry_run:
            log.error(f"No pentestds output found at {dist_dir}")
            log.error("Run without --skip-build first to generate data.")
            sys.exit(1)

    # ── Legacy mode: old Arrow-based pipeline ──
    if args.legacy:
        log.info("Using LEGACY Arrow-based pipeline")
        backup_data(data_dir, keep_backups, dry_run=args.dry_run)
        manifest = convert_and_merge(
            dist_dir=dist_dir,
            data_dir=data_dir,
            split_ratios=ds_cfg["split_ratios"],
            seed=ds_cfg["seed"],
            project_root=".",
            dry_run=args.dry_run,
        )
        log.info("")
        log.info("=" * 60)
        if args.dry_run:
            log.info("DRY RUN complete. No changes were made.")
        else:
            log.info("Dataset refresh complete! (legacy mode)")
            log.info(f"  Total examples: {manifest['total_examples']}")
        log.info("=" * 60)
        return

    # ── Database pipeline (default) ──
    if args.dry_run:
        log.info("[DRY RUN] Would ingest pentestds output into database")
        log.info("[DRY RUN] Would ingest custom data from data/custom/")
        log.info("[DRY RUN] Would run validation on new records")
        if args.export:
            log.info("[DRY RUN] Would export quality-filtered training data")
        log.info("")
        log.info("DRY RUN complete. No changes were made.")
        return

    from scripts.ingest_data import ingest_pentestds, ingest_jsonl
    from scripts.validate_data import run_validation
    from scripts.db_utils import DEFAULT_DB_PATH, get_connection, create_schema, create_indexes, migrate_schema
    from scripts.data_utils import load_custom_conversations

    db_path = str(cfg.get("database", {}).get("path", DEFAULT_DB_PATH))

    # Ensure database exists
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

    # Step 3: Ingest pentestds output into database
    log.info("")
    log.info("Step 3: Ingesting pentestds output into database...")
    pentestds_result = ingest_pentestds(db_path, str(dist_dir))
    log.info(f"  Inserted: {pentestds_result['inserted']:,}")
    log.info(f"  Skipped (dups): {pentestds_result['skipped_dups']:,}")

    # Step 4: Ingest custom data
    custom_dir = Path(".") / "data" / "custom"
    custom_total = 0
    if custom_dir.exists():
        log.info("")
        log.info("Step 4: Ingesting custom data...")
        import tempfile
        custom_convos = load_custom_conversations(".")
        if custom_convos:
            # Write to temp JSONL and ingest
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                             delete=False, encoding="utf-8") as f:
                for convo in custom_convos:
                    json.dump({"messages": convo}, f, ensure_ascii=False)
                    f.write("\n")
                tmp_path = f.name

            custom_result = ingest_jsonl(db_path, tmp_path, source="custom", split="train")
            custom_total = custom_result["inserted"]
            log.info(f"  Inserted: {custom_result['inserted']:,}")
            log.info(f"  Skipped (dups): {custom_result['skipped_dups']:,}")
            Path(tmp_path).unlink(missing_ok=True)
    else:
        log.info("Step 4: No custom data directory found, skipping.")

    # Step 5: Validate new records
    if not args.skip_validate:
        log.info("")
        log.info("Step 5: Running validation on all records...")
        val_stats = run_validation(db_path)
        log.info(f"  Validated: {val_stats['total']:,}")
        log.info(f"  Pass: {val_stats['pass']:,} | Warn: {val_stats['warn']:,} | Fail: {val_stats['fail']:,}")

    # Step 6: Export (optional)
    if args.export:
        log.info("")
        log.info("Step 6: Exporting quality-filtered training data...")
        from scripts.export_training_data import fetch_exportable_records, apply_domain_weights, export_to_arrow
        export_cfg = cfg.get("export", {}).get("training_data", {})
        quality_threshold = export_cfg.get("quality_threshold", 3.5)
        weights = export_cfg.get("domain_weights", {})
        output_dir = export_cfg.get("output_dir", "data/export")

        records_by_domain = fetch_exportable_records(db_path, quality_threshold)
        total_available = sum(len(r) for r in records_by_domain.values())
        log.info(f"  Records passing quality filter: {total_available:,}")

        if weights:
            all_records = apply_domain_weights(records_by_domain, weights)
        else:
            all_records = []
            for recs in records_by_domain.values():
                all_records.extend(recs)

        out_path = export_to_arrow(all_records, output_dir)
        log.info(f"  Exported {len(all_records):,} records to {out_path}")

    log.info("")
    log.info("=" * 60)
    log.info("Dataset refresh complete!")
    log.info(f"  pentestds new records: {pentestds_result['inserted']:,}")
    log.info(f"  Custom new records:    {custom_total:,}")
    log.info("")
    log.info("Next steps:")
    log.info("  python scripts/review_data.py --sample 1000  # LLM quality review")
    log.info("  python scripts/export_training_data.py       # export for training")
    log.info("  python train_native.py                       # train the model")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
