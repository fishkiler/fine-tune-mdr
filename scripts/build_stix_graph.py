#!/usr/bin/env python3
"""
============================================================================
STIX Relationship Graph — Master Orchestrator
============================================================================
Runs the complete STIX pipeline in order:
  1. Migrate schema to v5 (STIX tables)
  2. Fetch & ingest ATT&CK + MISP objects/relationships
  3. Link existing training records to STIX objects
  4. Generate new training pairs from graph traversal
  5. Print summary stats + verification

Usage:
    python scripts/build_stix_graph.py
    python scripts/build_stix_graph.py --skip-fetch
    python scripts/build_stix_graph.py --skip-generate
    python scripts/build_stix_graph.py --db mdr-database/mdr_dataset.db
============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as `python scripts/build_stix_graph.py` (adds project root to path)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.db_utils import DEFAULT_DB_PATH, get_connection, migrate_schema, compute_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build STIX relationship graph — full pipeline"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip STIX object download/ingestion")
    parser.add_argument("--skip-link", action="store_true", help="Skip training record linking")
    parser.add_argument("--skip-generate", action="store_true", help="Skip training pair generation")
    args = parser.parse_args()

    t0 = time.time()

    # ── Step 1: Schema Migration ──
    log.info("=" * 70)
    log.info("STEP 1: Schema Migration")
    log.info("=" * 70)
    conn = get_connection(args.db)
    migrate_schema(conn)
    conn.close()
    log.info("Schema migration complete.\n")

    # ── Step 2: Fetch & Ingest STIX Objects ──
    if not args.skip_fetch:
        log.info("=" * 70)
        log.info("STEP 2: STIX Object Ingestion (ATT&CK + MISP + CVEs)")
        log.info("=" * 70)
        from scripts.sources.fetch_stix_objects import run_ingestion
        run_ingestion(args.db)
        log.info("")
    else:
        log.info("Skipping STIX object ingestion (--skip-fetch)\n")

    # ── Step 3: Link Training Records ──
    if not args.skip_link:
        log.info("=" * 70)
        log.info("STEP 3: Link Training Records → STIX Objects")
        log.info("=" * 70)
        from scripts.link_stix_training import run_linking
        run_linking(args.db)
        log.info("")
    else:
        log.info("Skipping training record linking (--skip-link)\n")

    # ── Step 4: Generate Training Pairs ──
    if not args.skip_generate:
        log.info("=" * 70)
        log.info("STEP 4: Generate Training Pairs from Graph")
        log.info("=" * 70)
        from scripts.generate_stix_training_pairs import run_generation
        run_generation(args.db)
        log.info("")
    else:
        log.info("Skipping training pair generation (--skip-generate)\n")

    # ── Step 5: Summary & Verification ──
    log.info("=" * 70)
    log.info("STEP 5: Summary & Verification")
    log.info("=" * 70)

    conn = get_connection(args.db)
    cur = conn.cursor()

    # STIX Objects
    cur.execute("SELECT type, COUNT(*) FROM stix_objects GROUP BY type ORDER BY COUNT(*) DESC")
    rows = cur.fetchall()
    log.info("\nSTIX Objects:")
    total_obj = 0
    for obj_type, count in rows:
        log.info(f"  {obj_type:25s} {count:>8,}")
        total_obj += count
    log.info(f"  {'TOTAL':25s} {total_obj:>8,}")

    # STIX Relationships
    cur.execute(
        "SELECT relationship_type, COUNT(*) FROM stix_relationships "
        "GROUP BY relationship_type ORDER BY COUNT(*) DESC"
    )
    rows = cur.fetchall()
    log.info("\nSTIX Relationships:")
    total_rel = 0
    for rel_type, count in rows:
        log.info(f"  {rel_type:25s} {count:>8,}")
        total_rel += count
    log.info(f"  {'TOTAL':25s} {total_rel:>8,}")

    # Training Links
    cur.execute(
        "SELECT link_type, COUNT(*) FROM stix_training_links "
        "GROUP BY link_type ORDER BY COUNT(*) DESC"
    )
    rows = cur.fetchall()
    log.info("\nTraining Links:")
    total_links = 0
    for link_type, count in rows:
        log.info(f"  {link_type:25s} {count:>8,}")
        total_links += count
    log.info(f"  {'TOTAL':25s} {total_links:>8,}")

    # Generated pairs
    cur.execute("SELECT COUNT(*) FROM all_records WHERE source = 'stix_graph'")
    gen_count = cur.fetchone()[0]
    log.info(f"\nGenerated training pairs: {gen_count:,}")

    # Total dataset size
    cur.execute("SELECT COUNT(*) FROM all_records")
    total = cur.fetchone()[0]
    log.info(f"Total dataset size: {total:,}")

    # Schema version
    cur.execute("SELECT value FROM schema_info WHERE key = 'version'")
    version = cur.fetchone()[0]
    log.info(f"Schema version: v{version}")

    conn.close()

    elapsed = time.time() - t0
    log.info(f"\nPipeline completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
