#!/usr/bin/env python3
"""
============================================================================
STIX Relationship Layer — Schema Migration (v4 → v5)
============================================================================
Thin CLI wrapper around db_utils.migrate_schema() for the v5 STIX tables.
Creates: stix_objects, stix_relationships, stix_training_links, stix_stats.

Usage:
    python -m scripts.migrate_v5_stix
    python -m scripts.migrate_v5_stix --db mdr-database/mdr_dataset.db
    python -m scripts.migrate_v5_stix --verify-only
============================================================================
"""

import argparse
import logging
import sys

from scripts.db_utils import (
    DEFAULT_DB_PATH,
    SCHEMA_VERSION,
    get_connection,
    migrate_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STIX_TABLES = ["stix_objects", "stix_relationships", "stix_training_links", "stix_stats"]


def verify_tables(conn) -> bool:
    """Verify all STIX tables exist and report their column counts."""
    cur = conn.cursor()
    all_ok = True
    for table in STIX_TABLES:
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if cur.fetchone() is None:
            log.error(f"  MISSING: {table}")
            all_ok = False
        else:
            cur.execute(f"PRAGMA table_info({table})")
            cols = cur.fetchall()
            log.info(f"  OK: {table} ({len(cols)} columns)")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Migrate MDR database schema to v5 (STIX relationship layer)"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify tables exist, don't migrate",
    )
    args = parser.parse_args()

    conn = get_connection(args.db)

    if not args.verify_only:
        migrate_schema(conn)

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT value FROM schema_info WHERE key = 'version'")
    row = cur.fetchone()
    version = int(row[0]) if row else 0

    log.info(f"Schema version: v{version} (target: v{SCHEMA_VERSION})")
    log.info("Verifying STIX tables:")
    ok = verify_tables(conn)

    conn.close()

    if not ok:
        log.error("Migration verification FAILED — missing tables.")
        sys.exit(1)
    else:
        log.info("Migration verified successfully.")


if __name__ == "__main__":
    main()
