#!/usr/bin/env python3
"""
============================================================================
STIX Object & Relationship Ingestion Pipeline
============================================================================
Downloads ATT&CK Enterprise STIX bundle + MISP Galaxy threat actors,
extracts ALL object types and relationships, and populates the STIX
graph tables (stix_objects, stix_relationships, stix_stats).

Steps:
  2a. Download ATT&CK STIX bundle (cached 7 days)
  2b. Extract & store STIX objects (all types)
  2c. Extract & store STIX relationships
  2d. Ingest MISP Galaxy actors (deduplicated against ATT&CK)
  2e. Ingest CVE vulnerability objects from all_records

Usage:
    python -m scripts.sources.fetch_stix_objects
    python -m scripts.sources.fetch_stix_objects --skip-misp --skip-cve
    python -m scripts.sources.fetch_stix_objects --db mdr-database/mdr_dataset.db
============================================================================
"""

import argparse
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

from scripts.db_utils import DEFAULT_DB_PATH, get_connection, migrate_schema
from scripts.sources.fetch_mitre_groups import (
    fetch_stix_bundle,
    _is_active,
    _get_attack_id,
    _get_aliases,
    ENTERPRISE_URL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CACHE_DIR = Path("data/sources")
ATTACK_CACHE = CACHE_DIR / "enterprise-attack.json"
MISP_CACHE = CACHE_DIR / "misp-threat-actor.json"
MISP_URL = (
    "https://raw.githubusercontent.com/MISP/misp-galaxy/"
    "main/clusters/threat-actor.json"
)

CACHE_MAX_AGE_DAYS = 7
BATCH_SIZE = 5000

# STIX types we extract from the ATT&CK bundle
ATTACK_TYPES = {
    "intrusion-set", "attack-pattern", "malware", "tool",
    "campaign", "course-of-action", "x-mitre-tactic",
    "x-mitre-data-source", "x-mitre-data-component",
    "x-mitre-detection-strategy",
}

MISP_CACHE_GALAXY = {
    "ransomware": CACHE_DIR / "misp-ransomware.json",
}
MISP_RANSOMWARE_URL = (
    "https://raw.githubusercontent.com/MISP/misp-galaxy/"
    "main/clusters/ransomware.json"
)


# ============================================================================
# Caching
# ============================================================================

def _cache_valid(path: Path) -> bool:
    """Check if a cached file exists and is fresh enough."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_MAX_AGE_DAYS * 86400


def _download_json(url: str, cache_path: Path, label: str) -> dict:
    """Download JSON from URL with local file caching."""
    if _cache_valid(cache_path):
        log.info(f"Using cached {label}: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    log.info(f"Downloading {label} from {url}...")
    req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
    with urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    log.info(f"Cached {label} to {cache_path}")
    return data


# ============================================================================
# Step 2b: Extract STIX Objects from ATT&CK Bundle
# ============================================================================

def extract_stix_objects(bundle: dict) -> tuple[list[dict], list[dict]]:
    """Extract all STIX objects and relationships from ATT&CK bundle.

    Returns (objects_list, relationships_list) ready for DB insertion.
    """
    raw_objects = bundle.get("objects", [])
    objects = []
    relationships = []
    stix_ids_seen = set()

    for obj in raw_objects:
        otype = obj.get("type", "")
        stix_id = obj.get("id", "")

        if not _is_active(obj):
            continue

        if otype in ATTACK_TYPES:
            attack_id = _get_attack_id(obj)
            aliases = _get_aliases(obj)

            # Build external_ids
            external_ids = {}
            if attack_id:
                external_ids["mitre_attack_id"] = attack_id
            for ref in obj.get("external_references", []):
                src = ref.get("source_name", "")
                eid = ref.get("external_id", "")
                if src and eid and src != "mitre-attack":
                    external_ids[src] = eid

            # Kill chain phases
            phases = []
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    phases.append(phase["phase_name"])

            platforms = obj.get("x_mitre_platforms", [])

            record = {
                "stix_id": stix_id,
                "type": otype,
                "name": obj.get("name", ""),
                "aliases": json.dumps(aliases) if aliases else None,
                "description": obj.get("description", ""),
                "external_ids": json.dumps(external_ids) if external_ids else None,
                "source": "mitre_attack",
                "platforms": json.dumps(platforms) if platforms else None,
                "kill_chain_phases": json.dumps(phases) if phases else None,
                "severity": None,
                "cvss_score": None,
                "raw_stix_json": json.dumps(obj, ensure_ascii=False),
            }
            objects.append(record)
            stix_ids_seen.add(stix_id)

        elif otype == "relationship":
            relationships.append({
                "relationship_id": stix_id,
                "source_ref": obj.get("source_ref", ""),
                "target_ref": obj.get("target_ref", ""),
                "relationship_type": obj.get("relationship_type", ""),
                "description": obj.get("description", ""),
                "source": "mitre_attack",
                "confidence": None,
            })

    log.info(
        f"Extracted {len(objects)} STIX objects, "
        f"{len(relationships)} relationships from ATT&CK bundle"
    )

    # Count by type
    type_counts = {}
    for o in objects:
        type_counts[o["type"]] = type_counts.get(o["type"], 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {t}: {c}")

    return objects, relationships


# ============================================================================
# Step 2d: Extract MISP Galaxy Actors
# ============================================================================

def extract_misp_actors(
    galaxy: dict,
    existing_names: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Extract MISP actors, deduplicating against ATT&CK intrusion-sets.

    existing_names: lowercase name/alias → stix_id mapping from ATT&CK.
    Returns (new_objects, merge_updates) where merge_updates are dicts
    with {stix_id, aliases_to_add} for existing objects.
    """
    actors = galaxy.get("values", [])
    new_objects = []
    merge_updates = []

    for actor in actors:
        name = actor.get("value", "")
        if not name:
            continue

        misp_uuid = actor.get("uuid", "")
        description = actor.get("description", "")
        meta = actor.get("meta", {})
        synonyms = meta.get("synonyms", [])
        country = meta.get("country")
        sponsor = meta.get("cfr-suspected-state-sponsor")

        # Check if this actor already exists in ATT&CK data
        matched_stix_id = existing_names.get(name.lower())
        if not matched_stix_id:
            for syn in synonyms:
                matched_stix_id = existing_names.get(syn.lower())
                if matched_stix_id:
                    break

        # Collect rich metadata from MISP
        victims = meta.get("cfr-suspected-victims", [])
        target_cats = meta.get("cfr-target-category", [])
        motive = meta.get("motive")
        incident_type = meta.get("cfr-type-of-incident")

        if matched_stix_id:
            # Merge: add MISP aliases + metadata to existing object
            merge_updates.append({
                "stix_id": matched_stix_id,
                "aliases_to_add": synonyms,
                "misp_uuid": misp_uuid,
                "country": country,
                "sponsor": sponsor,
                "victims": victims,
                "target_categories": target_cats,
                "motive": motive,
                "incident_type": incident_type,
            })
        else:
            # New actor from MISP
            stix_id = f"intrusion-set--{misp_uuid}" if misp_uuid else None
            if not stix_id:
                continue

            external_ids = {"misp_uuid": misp_uuid}
            if country:
                external_ids["country"] = country
            if sponsor:
                external_ids["cfr-suspected-state-sponsor"] = sponsor
            if victims:
                external_ids["cfr-suspected-victims"] = victims
            if target_cats:
                external_ids["cfr-target-category"] = target_cats
            if motive:
                external_ids["motive"] = motive
            if incident_type:
                external_ids["cfr-type-of-incident"] = incident_type

            new_objects.append({
                "stix_id": stix_id,
                "type": "intrusion-set",
                "name": name,
                "aliases": json.dumps(synonyms) if synonyms else None,
                "description": description[:5000] if description else None,
                "external_ids": json.dumps(external_ids),
                "source": "misp_galaxy",
                "platforms": None,
                "kill_chain_phases": None,
                "severity": None,
                "cvss_score": None,
                "raw_stix_json": json.dumps(actor, ensure_ascii=False),
            })

    log.info(
        f"MISP Galaxy: {len(new_objects)} new actors, "
        f"{len(merge_updates)} merged with ATT&CK"
    )
    return new_objects, merge_updates


# ============================================================================
# Step 2e: Extract CVE Vulnerability Objects
# ============================================================================

def extract_cve_objects(conn) -> list[dict]:
    """Pull unique CVE IDs from all_records and create vulnerability objects."""
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT cve_ids FROM all_records "
        "WHERE cve_ids IS NOT NULL AND cve_ids != ''"
    )
    rows = cur.fetchall()

    # Collect unique CVE IDs
    cve_set = set()
    for (cve_ids_str,) in rows:
        for cve_id in cve_ids_str.split(","):
            cve_id = cve_id.strip()
            if re.match(r"CVE-\d{4}-\d+", cve_id):
                cve_set.add(cve_id)

    log.info(f"Found {len(cve_set)} unique CVE IDs in all_records")

    # Pull best description/severity/cvss for each CVE
    cve_meta = {}
    cur.execute(
        "SELECT cve_ids, severity, cvss_score, assistant_message, source "
        "FROM all_records WHERE cve_ids IS NOT NULL AND cve_ids != '' "
        "ORDER BY quality_score DESC NULLS LAST"
    )
    for cve_ids_str, severity, cvss, assistant, source in cur.fetchall():
        for cve_id in cve_ids_str.split(","):
            cve_id = cve_id.strip()
            if cve_id not in cve_meta and re.match(r"CVE-\d{4}-\d+", cve_id):
                cve_meta[cve_id] = {
                    "severity": severity,
                    "cvss_score": cvss,
                    "source": source or "cveorg",
                    # Extract first 500 chars of assistant as description
                    "description": (assistant or "")[:500],
                }

    objects = []
    for cve_id in sorted(cve_set):
        stix_id = f"vulnerability--{uuid.uuid5(uuid.NAMESPACE_URL, cve_id)}"
        meta = cve_meta.get(cve_id, {})
        external_ids = {"cve_id": cve_id}

        objects.append({
            "stix_id": stix_id,
            "type": "vulnerability",
            "name": cve_id,
            "aliases": None,
            "description": meta.get("description"),
            "external_ids": json.dumps(external_ids),
            "source": meta.get("source", "cveorg"),
            "platforms": None,
            "kill_chain_phases": None,
            "severity": meta.get("severity"),
            "cvss_score": meta.get("cvss_score"),
            "raw_stix_json": None,
        })

    log.info(f"Generated {len(objects)} CVE vulnerability STIX objects")
    return objects


# ============================================================================
# Database Ingestion
# ============================================================================

def ingest_objects(conn, objects: list[dict]) -> dict:
    """Batch INSERT OR REPLACE STIX objects into the database."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for i in range(0, len(objects), BATCH_SIZE):
        batch = objects[i:i + BATCH_SIZE]
        for obj in batch:
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO stix_objects
                    (stix_id, type, name, aliases, description, external_ids,
                     source, platforms, kill_chain_phases, severity, cvss_score,
                     raw_stix_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    obj["stix_id"], obj["type"], obj["name"],
                    obj["aliases"], obj["description"], obj["external_ids"],
                    obj["source"], obj["platforms"], obj["kill_chain_phases"],
                    obj["severity"], obj["cvss_score"], obj["raw_stix_json"],
                ))
                inserted += 1
            except Exception as e:
                log.warning(f"Skip object {obj.get('stix_id', '?')}: {e}")
                skipped += 1

        conn.commit()
        log.info(f"  Objects: {inserted:,} inserted, {skipped} skipped (batch {i // BATCH_SIZE + 1})")

    return {"inserted": inserted, "skipped": skipped}


def ingest_relationships(conn, rels: list[dict], valid_ids: set[str]) -> dict:
    """Batch INSERT OR IGNORE STIX relationships, skipping orphans."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    orphaned = 0

    for i in range(0, len(rels), BATCH_SIZE):
        batch = rels[i:i + BATCH_SIZE]
        for rel in batch:
            src = rel["source_ref"]
            tgt = rel["target_ref"]

            if src not in valid_ids or tgt not in valid_ids:
                orphaned += 1
                continue

            try:
                cur.execute("""
                    INSERT OR IGNORE INTO stix_relationships
                    (relationship_id, source_ref, target_ref, relationship_type,
                     description, source, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel["relationship_id"], src, tgt,
                    rel["relationship_type"], rel["description"],
                    rel["source"], rel["confidence"],
                ))
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                log.warning(f"Skip relationship {rel.get('relationship_id', '?')}: {e}")
                skipped += 1

        conn.commit()

    log.info(
        f"  Relationships: {inserted:,} inserted, "
        f"{skipped} skipped, {orphaned} orphaned"
    )
    return {"inserted": inserted, "skipped": skipped, "orphaned": orphaned}


def apply_misp_merges(conn, merges: list[dict]) -> int:
    """Merge MISP aliases into existing ATT&CK objects."""
    cur = conn.cursor()
    merged = 0

    for merge in merges:
        stix_id = merge["stix_id"]
        cur.execute("SELECT aliases, external_ids FROM stix_objects WHERE stix_id = ?", (stix_id,))
        row = cur.fetchone()
        if not row:
            continue

        # Merge aliases
        existing_aliases = json.loads(row[0]) if row[0] else []
        existing_lower = {a.lower() for a in existing_aliases}
        new_aliases = [a for a in merge["aliases_to_add"] if a.lower() not in existing_lower]

        if not new_aliases:
            continue

        combined = existing_aliases + new_aliases
        # Merge external_ids
        existing_ext = json.loads(row[1]) if row[1] else {}
        if merge.get("misp_uuid"):
            existing_ext["misp_uuid"] = merge["misp_uuid"]
        if merge.get("country"):
            existing_ext["country"] = merge["country"]
        if merge.get("sponsor"):
            existing_ext["cfr-suspected-state-sponsor"] = merge["sponsor"]
        if merge.get("victims"):
            existing_ext["cfr-suspected-victims"] = merge["victims"]
        if merge.get("target_categories"):
            existing_ext["cfr-target-category"] = merge["target_categories"]
        if merge.get("motive"):
            existing_ext["motive"] = merge["motive"]
        if merge.get("incident_type"):
            existing_ext["cfr-type-of-incident"] = merge["incident_type"]

        cur.execute("""
            UPDATE stix_objects
            SET aliases = ?, external_ids = ?, updated_at = datetime('now')
            WHERE stix_id = ?
        """, (json.dumps(combined), json.dumps(existing_ext), stix_id))
        merged += 1

    conn.commit()
    log.info(f"  Merged MISP data into {merged} existing ATT&CK objects")
    return merged


# ============================================================================
# Sigma Rule Objects & Relationships
# ============================================================================

def ingest_sigma_rules(conn) -> dict:
    """Create STIX objects for ALL Sigma rules and link to attack-pattern objects.

    Uses build_full_sigma_index() to ingest all 3,100+ SigmaHQ rules (not just
    1 per technique). Each rule becomes a sigma-rule STIX object with a detects
    relationship for every technique it maps to.
    """
    from scripts.sources.sigma_hq import build_full_sigma_index

    all_rules = build_full_sigma_index()
    if not all_rules:
        log.warning("No Sigma rules found. Check network/cache.")
        return {"objects": 0, "relationships": 0}

    log.info(f"Processing {len(all_rules)} Sigma rules into STIX objects...")

    # Build technique external_id → stix_id lookup
    cur = conn.cursor()
    cur.execute("SELECT stix_id, external_ids FROM stix_objects WHERE type = 'attack-pattern'")
    tech_lookup = {}
    for stix_id, ext_json in cur.fetchall():
        if ext_json:
            ext = json.loads(ext_json)
            aid = ext.get("mitre_attack_id")
            if aid:
                tech_lookup[aid] = stix_id

    objects = []
    relationships = []
    unmatched_techniques = set()

    for rule in all_rules:
        rule_path = rule["path"]
        sigma_stix_id = f"sigma-rule--{uuid.uuid5(uuid.NAMESPACE_URL, rule_path)}"
        tech_ids = rule["technique_ids"]

        objects.append({
            "stix_id": sigma_stix_id,
            "type": "sigma-rule",
            "name": rule["title"],
            "aliases": None,
            "description": (
                f"Sigma detection rule: {rule['title']}. "
                f"Status: {rule['status']}, Level: {rule['level']}. "
                f"Techniques: {', '.join(tech_ids)}."
            ),
            "external_ids": json.dumps({
                "sigma_rule_path": rule_path,
                "mitre_attack_ids": tech_ids,
                "status": rule["status"],
                "level": rule["level"],
            }),
            "source": "sigma_hq",
            "platforms": None,
            "kill_chain_phases": None,
            "severity": None,
            "cvss_score": None,
            "raw_stix_json": rule["yaml_content"],
        })

        # Create a detects relationship for each technique
        for tech_id in tech_ids:
            tech_stix_id = tech_lookup.get(tech_id)
            if tech_stix_id:
                rel_id = f"relationship--sigma-{uuid.uuid5(uuid.NAMESPACE_URL, f'{sigma_stix_id}-{tech_stix_id}')}"
                relationships.append({
                    "relationship_id": rel_id,
                    "source_ref": sigma_stix_id,
                    "target_ref": tech_stix_id,
                    "relationship_type": "detects",
                    "description": f"Sigma rule '{rule['title']}' detects {tech_id}",
                    "source": "sigma_hq",
                    "confidence": None,
                })
            else:
                unmatched_techniques.add(tech_id)

    if unmatched_techniques:
        log.info(f"  {len(unmatched_techniques)} technique IDs not found in attack-patterns "
                 f"(subtechniques or deprecated): {sorted(unmatched_techniques)[:10]}...")

    # Ingest
    if objects:
        ingest_objects(conn, objects)
    if relationships:
        # Need valid IDs including newly added sigma objects
        cur.execute("SELECT stix_id FROM stix_objects")
        valid_ids = {row[0] for row in cur.fetchall()}
        ingest_relationships(conn, relationships, valid_ids)

    log.info(f"  Sigma rules: {len(objects)} objects, {len(relationships)} detects relationships")
    return {"objects": len(objects), "relationships": len(relationships)}


# ============================================================================
# CISA KEV → Technique Bridging
# ============================================================================

def ingest_kev_relationships(conn) -> dict:
    """Create relationships between KEV CVEs and known malware/ransomware.

    Scans CISA KEV records for ransomware flags and product names,
    then links CVE vulnerability objects to malware/tool STIX objects
    via 'exploited-by' synthetic relationships.
    """
    cur = conn.cursor()

    # Find KEV records with ransomware flag
    cur.execute(
        "SELECT user_message, assistant_message FROM all_records "
        "WHERE source = 'cisa_kev'"
    )
    kev_rows = cur.fetchall()
    if not kev_rows:
        log.info("  No CISA KEV records found in database")
        return {"relationships": 0}

    # Build malware/tool name → stix_id lookup (for product matching)
    cur.execute(
        "SELECT stix_id, name, aliases FROM stix_objects "
        "WHERE type IN ('malware', 'tool')"
    )
    sw_lookup = {}
    for stix_id, name, aliases_json in cur.fetchall():
        if len(name) >= 4:
            sw_lookup[name.lower()] = stix_id
        if aliases_json:
            for alias in json.loads(aliases_json):
                if len(alias) >= 4:
                    sw_lookup[alias.lower()] = stix_id

    # Build CVE name → stix_id lookup
    cur.execute(
        "SELECT stix_id, name FROM stix_objects WHERE type = 'vulnerability'"
    )
    cve_stix_lookup = {name: stix_id for stix_id, name in cur.fetchall()}

    relationships = []
    ransomware_cves = set()

    for user_msg, asst_msg in kev_rows:
        # Extract CVE ID from the message
        cve_match = re.search(r"(CVE-\d{4}-\d+)", user_msg or "")
        if not cve_match:
            continue
        cve_id = cve_match.group(1)
        cve_stix_id = cve_stix_lookup.get(cve_id)
        if not cve_stix_id:
            continue

        full_text = f"{user_msg} {asst_msg}"

        # Check for ransomware flag
        if "ransomware" in full_text.lower():
            ransomware_cves.add(cve_id)

        # Try to match software names in the KEV text
        for sw_name, sw_stix_id in sw_lookup.items():
            if re.search(r"\b" + re.escape(sw_name) + r"\b", full_text, re.IGNORECASE):
                rel_id = f"relationship--kev-{uuid.uuid5(uuid.NAMESPACE_URL, f'{cve_stix_id}-{sw_stix_id}')}"
                relationships.append({
                    "relationship_id": rel_id,
                    "source_ref": sw_stix_id,
                    "target_ref": cve_stix_id,
                    "relationship_type": "exploits",
                    "description": f"Software linked to exploitation of {cve_id} via CISA KEV",
                    "source": "cisa_kev",
                    "confidence": 60,
                })

    # Deduplicate relationships by ID
    seen = set()
    unique_rels = []
    for rel in relationships:
        if rel["relationship_id"] not in seen:
            seen.add(rel["relationship_id"])
            unique_rels.append(rel)

    if unique_rels:
        cur.execute("SELECT stix_id FROM stix_objects")
        valid_ids = {row[0] for row in cur.fetchall()}
        ingest_relationships(conn, unique_rels, valid_ids)

    log.info(f"  KEV bridging: {len(unique_rels)} exploits relationships, {len(ransomware_cves)} ransomware CVEs")
    return {"relationships": len(unique_rels), "ransomware_cves": len(ransomware_cves)}


# ============================================================================
# Stats
# ============================================================================

def update_stix_stats(conn) -> None:
    """Populate the stix_stats table with current counts."""
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # Object counts by type
    cur.execute("SELECT type, COUNT(*) FROM stix_objects GROUP BY type")
    type_counts = dict(cur.fetchall())

    # Relationship counts by type
    cur.execute(
        "SELECT relationship_type, COUNT(*) FROM stix_relationships GROUP BY relationship_type"
    )
    rel_counts = dict(cur.fetchall())

    # Training link counts by STIX object type
    cur.execute("""
        SELECT so.type, COUNT(stl.record_id)
        FROM stix_training_links stl
        JOIN stix_objects so ON stl.stix_id = so.stix_id
        GROUP BY so.type
    """)
    link_counts = dict(cur.fetchall())

    # Merge all type keys
    all_types = set(type_counts) | set(rel_counts) | set(link_counts)
    for t in all_types:
        cur.execute("""
            INSERT OR REPLACE INTO stix_stats
            (type, object_count, relationship_count, training_link_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (
            t,
            type_counts.get(t, 0),
            rel_counts.get(t, 0),
            link_counts.get(t, 0),
            now,
        ))

    conn.commit()
    log.info("Updated stix_stats table")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_ingestion(db_path: str, skip_misp: bool = False, skip_cve: bool = False):
    """Run the full STIX ingestion pipeline."""
    conn = get_connection(db_path)
    migrate_schema(conn)

    # Step 2a-c: ATT&CK bundle
    log.info("=" * 60)
    log.info("Step 2a: Downloading ATT&CK STIX bundle...")
    bundle = _download_json(ENTERPRISE_URL, ATTACK_CACHE, "ATT&CK STIX bundle")

    log.info("Step 2b: Extracting STIX objects...")
    objects, relationships = extract_stix_objects(bundle)

    log.info("Step 2b: Ingesting STIX objects...")
    obj_result = ingest_objects(conn, objects)

    # Build valid ID set for relationship validation
    cur = conn.cursor()
    cur.execute("SELECT stix_id FROM stix_objects")
    valid_ids = {row[0] for row in cur.fetchall()}

    log.info("Step 2c: Ingesting STIX relationships...")
    rel_result = ingest_relationships(conn, relationships, valid_ids)

    # Step 2d: MISP Galaxy
    if not skip_misp:
        log.info("=" * 60)
        log.info("Step 2d: Downloading MISP Galaxy threat actors...")
        galaxy = _download_json(MISP_URL, MISP_CACHE, "MISP Galaxy threat actors")

        # Build name/alias lookup from existing ATT&CK intrusion-sets
        cur.execute(
            "SELECT stix_id, name, aliases FROM stix_objects WHERE type = 'intrusion-set'"
        )
        existing_names = {}
        for stix_id, name, aliases_json in cur.fetchall():
            existing_names[name.lower()] = stix_id
            if aliases_json:
                for alias in json.loads(aliases_json):
                    existing_names[alias.lower()] = stix_id

        new_actors, merge_updates = extract_misp_actors(galaxy, existing_names)
        if new_actors:
            ingest_objects(conn, new_actors)
        if merge_updates:
            apply_misp_merges(conn, merge_updates)
    else:
        log.info("Skipping MISP Galaxy ingestion (--skip-misp)")

    # Step 2e: CVE vulnerability objects
    if not skip_cve:
        log.info("=" * 60)
        log.info("Step 2e: Generating CVE vulnerability objects...")
        cve_objects = extract_cve_objects(conn)
        if cve_objects:
            ingest_objects(conn, cve_objects)
    else:
        log.info("Skipping CVE ingestion (--skip-cve)")

    # Step 2f: Sigma rules as STIX objects
    log.info("=" * 60)
    log.info("Step 2f: Ingesting Sigma rules as STIX objects...")
    ingest_sigma_rules(conn)

    # Step 2g: CISA KEV → technique bridging
    log.info("=" * 60)
    log.info("Step 2g: CISA KEV → malware/tool bridging...")
    ingest_kev_relationships(conn)

    # Update stats
    update_stix_stats(conn)

    # Final summary
    log.info("=" * 60)
    log.info("STIX Ingestion Summary:")
    cur.execute("SELECT type, COUNT(*) FROM stix_objects GROUP BY type ORDER BY COUNT(*) DESC")
    for obj_type, count in cur.fetchall():
        log.info(f"  {obj_type}: {count:,}")
    cur.execute("SELECT COUNT(*) FROM stix_objects")
    log.info(f"  TOTAL objects: {cur.fetchone()[0]:,}")

    cur.execute(
        "SELECT relationship_type, COUNT(*) FROM stix_relationships "
        "GROUP BY relationship_type ORDER BY COUNT(*) DESC"
    )
    for rel_type, count in cur.fetchall():
        log.info(f"  {rel_type}: {count:,}")
    cur.execute("SELECT COUNT(*) FROM stix_relationships")
    log.info(f"  TOTAL relationships: {cur.fetchone()[0]:,}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest STIX objects and relationships from ATT&CK + MISP"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--skip-misp", action="store_true", help="Skip MISP Galaxy ingestion")
    parser.add_argument("--skip-cve", action="store_true", help="Skip CVE object generation")
    args = parser.parse_args()

    run_ingestion(args.db, skip_misp=args.skip_misp, skip_cve=args.skip_cve)


if __name__ == "__main__":
    main()
