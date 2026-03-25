#!/usr/bin/env python3
"""
============================================================================
STIX Training Record Linker
============================================================================
Populates stix_training_links by scanning all_records and linking them
to STIX objects via 5 strategies:

  1. CVE-based links:     cve_ids → vulnerability objects     (about, 1.0)
  2. MITRE technique:     mitre_techniques → attack-pattern   (about, 1.0)
  3. APT intel:           actor name/alias matching           (about, 0.8)
  4. Detection rules:     siem/sigma + mitre_techniques       (detects, 1.0)
  5. Log analysis:        log_analysis + mitre_techniques     (analyzes, 1.0)

Usage:
    python -m scripts.link_stix_training
    python -m scripts.link_stix_training --db mdr-database/mdr_dataset.db
    python -m scripts.link_stix_training --relink
============================================================================
"""

import argparse
import json
import logging
import re
import uuid

from scripts.db_utils import DEFAULT_DB_PATH, get_connection, migrate_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 1000


# ============================================================================
# Lookup Index Builders
# ============================================================================

def build_cve_lookup(conn) -> dict[str, str]:
    """Build CVE ID → stix_id lookup from vulnerability objects."""
    cur = conn.cursor()
    cur.execute(
        "SELECT stix_id, name FROM stix_objects WHERE type = 'vulnerability'"
    )
    lookup = {}
    for stix_id, name in cur.fetchall():
        lookup[name] = stix_id
    log.info(f"Built CVE lookup: {len(lookup):,} entries")
    return lookup


def build_technique_lookup(conn) -> dict[str, str]:
    """Build technique external_id (T1566, T1059.001) → stix_id lookup."""
    cur = conn.cursor()
    cur.execute(
        "SELECT stix_id, external_ids FROM stix_objects WHERE type = 'attack-pattern'"
    )
    lookup = {}
    for stix_id, ext_json in cur.fetchall():
        if ext_json:
            ext = json.loads(ext_json)
            attack_id = ext.get("mitre_attack_id")
            if attack_id:
                lookup[attack_id] = stix_id
    log.info(f"Built technique lookup: {len(lookup)} entries")
    return lookup


def build_actor_lookup(conn) -> tuple[dict[str, str], list[re.Pattern]]:
    """Build actor name/alias → stix_id lookup and compiled regex patterns.

    Returns (name_lookup, patterns) where patterns is a list of
    (compiled_regex, stix_id) tuples for text scanning.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT stix_id, name, aliases FROM stix_objects WHERE type = 'intrusion-set'"
    )
    name_lookup = {}
    patterns = []

    for stix_id, name, aliases_json in cur.fetchall():
        # Add primary name
        name_lookup[name.lower()] = stix_id
        # Only create regex for names >= 4 chars to avoid false positives
        if len(name) >= 4:
            try:
                pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
                patterns.append((pat, stix_id))
            except re.error:
                pass

        # Add aliases
        if aliases_json:
            for alias in json.loads(aliases_json):
                name_lookup[alias.lower()] = stix_id
                if len(alias) >= 4:
                    try:
                        pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                        patterns.append((pat, stix_id))
                    except re.error:
                        pass

    log.info(f"Built actor lookup: {len(name_lookup):,} names, {len(patterns)} patterns")
    return name_lookup, patterns


# ============================================================================
# Linking Strategies
# ============================================================================

def link_cve_records(conn, cve_lookup: dict[str, str]) -> int:
    """Strategy 1: Link records with CVE IDs to vulnerability objects."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, cve_ids FROM all_records "
        "WHERE cve_ids IS NOT NULL AND cve_ids != ''"
    )

    links = []
    total = 0

    for record_id, cve_ids_str in cur.fetchall():
        for cve_id in cve_ids_str.split(","):
            cve_id = cve_id.strip()
            stix_id = cve_lookup.get(cve_id)
            if stix_id:
                links.append((stix_id, record_id, "about", 1.0))

        if len(links) >= BATCH_SIZE:
            total += _flush_links(conn, links)
            links = []

    if links:
        total += _flush_links(conn, links)

    log.info(f"  CVE links: {total:,}")
    return total


def link_technique_records(conn, tech_lookup: dict[str, str]) -> int:
    """Strategy 2: Link records with technique IDs to attack-pattern objects."""
    cur = conn.cursor()
    # Only link records in domains that aren't detection-specific
    # (those get 'detects' or 'analyzes' link types instead)
    cur.execute(
        "SELECT id, mitre_techniques, domain FROM all_records "
        "WHERE mitre_techniques IS NOT NULL AND mitre_techniques != '' "
        "AND domain NOT IN ('siem_queries', 'sigma_rules', 'log_analysis')"
    )

    links = []
    total = 0

    for record_id, techniques_str, domain in cur.fetchall():
        for tech_id in techniques_str.split(","):
            tech_id = tech_id.strip()
            stix_id = tech_lookup.get(tech_id)
            if stix_id:
                links.append((stix_id, record_id, "about", 1.0))

        if len(links) >= BATCH_SIZE:
            total += _flush_links(conn, links)
            links = []

    if links:
        total += _flush_links(conn, links)

    log.info(f"  Technique links: {total:,}")
    return total


def link_apt_records(conn, actor_patterns: list[tuple]) -> int:
    """Strategy 3: Link apt_intel records to intrusion-set objects via text matching."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user_message, assistant_message FROM all_records "
        "WHERE domain = 'apt_intel'"
    )

    links = []
    total = 0

    for record_id, user_msg, asst_msg in cur.fetchall():
        text = f"{user_msg or ''} {asst_msg or ''}"
        matched_ids = set()

        for pat, stix_id in actor_patterns:
            if stix_id not in matched_ids and pat.search(text):
                links.append((stix_id, record_id, "about", 0.8))
                matched_ids.add(stix_id)

        if len(links) >= BATCH_SIZE:
            total += _flush_links(conn, links)
            links = []

    if links:
        total += _flush_links(conn, links)

    log.info(f"  APT intel links: {total:,}")
    return total


def link_detection_records(conn, tech_lookup: dict[str, str]) -> int:
    """Strategy 4: Link SIEM/Sigma records to technique objects as 'detects'."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, mitre_techniques FROM all_records "
        "WHERE mitre_techniques IS NOT NULL AND mitre_techniques != '' "
        "AND domain IN ('siem_queries', 'sigma_rules')"
    )

    links = []
    total = 0

    for record_id, techniques_str in cur.fetchall():
        for tech_id in techniques_str.split(","):
            tech_id = tech_id.strip()
            stix_id = tech_lookup.get(tech_id)
            if stix_id:
                links.append((stix_id, record_id, "detects", 1.0))

        if len(links) >= BATCH_SIZE:
            total += _flush_links(conn, links)
            links = []

    if links:
        total += _flush_links(conn, links)

    log.info(f"  Detection links: {total:,}")
    return total


def link_log_records(conn, tech_lookup: dict[str, str]) -> int:
    """Strategy 5: Link log_analysis records to technique objects as 'analyzes'."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, mitre_techniques FROM all_records "
        "WHERE mitre_techniques IS NOT NULL AND mitre_techniques != '' "
        "AND domain = 'log_analysis'"
    )

    links = []
    total = 0

    for record_id, techniques_str in cur.fetchall():
        for tech_id in techniques_str.split(","):
            tech_id = tech_id.strip()
            stix_id = tech_lookup.get(tech_id)
            if stix_id:
                links.append((stix_id, record_id, "analyzes", 1.0))

        if len(links) >= BATCH_SIZE:
            total += _flush_links(conn, links)
            links = []

    if links:
        total += _flush_links(conn, links)

    log.info(f"  Log analysis links: {total:,}")
    return total


# ============================================================================
# Helpers
# ============================================================================

def _flush_links(conn, links: list[tuple]) -> int:
    """INSERT OR IGNORE a batch of links and return count inserted."""
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO stix_training_links "
        "(stix_id, record_id, link_type, confidence) VALUES (?, ?, ?, ?)",
        links,
    )
    inserted = cur.rowcount
    conn.commit()
    return inserted


# ============================================================================
# Main
# ============================================================================

def run_linking(db_path: str, relink: bool = False):
    """Run all 5 linking strategies."""
    conn = get_connection(db_path)
    migrate_schema(conn)

    if relink:
        log.info("--relink: clearing existing stix_training_links...")
        conn.execute("DELETE FROM stix_training_links")
        conn.commit()

    # Build lookup indexes
    log.info("Building lookup indexes...")
    cve_lookup = build_cve_lookup(conn)
    tech_lookup = build_technique_lookup(conn)
    _name_lookup, actor_patterns = build_actor_lookup(conn)

    # Run all strategies
    log.info("Linking records to STIX objects...")
    total = 0
    total += link_cve_records(conn, cve_lookup)
    total += link_technique_records(conn, tech_lookup)
    total += link_apt_records(conn, actor_patterns)
    total += link_detection_records(conn, tech_lookup)
    total += link_log_records(conn, tech_lookup)

    # Update stats
    from scripts.sources.fetch_stix_objects import update_stix_stats
    update_stix_stats(conn)

    # Summary
    log.info("=" * 60)
    log.info("Linking Summary:")
    cur = conn.cursor()
    cur.execute(
        "SELECT link_type, COUNT(*) FROM stix_training_links "
        "GROUP BY link_type ORDER BY COUNT(*) DESC"
    )
    for link_type, count in cur.fetchall():
        log.info(f"  {link_type}: {count:,}")
    cur.execute("SELECT COUNT(*) FROM stix_training_links")
    log.info(f"  TOTAL links: {cur.fetchone()[0]:,}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Link existing training records to STIX objects"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument(
        "--relink", action="store_true",
        help="Clear existing links before relinking",
    )
    args = parser.parse_args()

    run_linking(args.db, relink=args.relink)


if __name__ == "__main__":
    main()
