#!/usr/bin/env python3
"""
============================================================================
CVE.org Training Data Generator
============================================================================
Parses raw CVE JSON files from cve.org and generates high-quality training
examples with specific CVSS scores, CWE IDs, affected versions, and
remediation data — replacing generic boilerplate from pentestds.

Source: docs/cves/{year}/ directories containing CVE JSON v5.x files
Output: JSONL for ingestion via ingest_data.py

Usage:
    python -m scripts.sources.ingest_cveorg
    python -m scripts.sources.ingest_cveorg --delete-existing
    python -m scripts.sources.ingest_cveorg --year-start 2023 --year-end 2025
============================================================================
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.db_utils import DEFAULT_DB_PATH, compute_stats, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Question Templates ──────────────────────────────────────────────────────

QUESTION_TEMPLATES = {
    "summary": [
        "Summarize {cve_id} and include severity context.",
        "What is {cve_id} and how critical is it?",
        "Provide a security brief on {cve_id}.",
        "Describe the vulnerability {cve_id} including its severity rating.",
    ],
    "impact": [
        "What is the impact of {cve_id} and how serious is it?",
        "Analyze the security impact of {cve_id}.",
        "How does {cve_id} affect confidentiality, integrity, and availability?",
        "What are the potential consequences of exploiting {cve_id}?",
    ],
    "mitigation": [
        "How should {cve_id} be mitigated in a production environment?",
        "What steps should be taken to remediate {cve_id}?",
        "What patches or workarounds are available for {cve_id}?",
        "How can organizations protect against {cve_id}?",
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(obj, *keys, default=None):
    """Safely traverse nested dicts/lists without KeyError/IndexError."""
    for key in keys:
        if obj is None:
            return default
        if isinstance(key, int):
            if isinstance(obj, list) and 0 <= key < len(obj):
                obj = obj[key]
            else:
                return default
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return default
    return obj if obj is not None else default


def strip_html(text: str) -> str:
    """Remove HTML tags, converting block elements to whitespace."""
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<li\s*>", "- ", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _select_template(templates: list[str], cve_id: str) -> str:
    """Deterministically select a template variant based on CVE ID hash."""
    h = int(hashlib.md5(cve_id.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


def _severity_from_score(score: float) -> str:
    """Map CVSS base score to severity label."""
    if score >= 9.0:
        return "CRITICAL"
    elif score >= 7.0:
        return "HIGH"
    elif score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"


# ── CVSS Extraction ─────────────────────────────────────────────────────────

def extract_cvss(cve_data: dict) -> dict | None:
    """Extract CVSS data. Priority: CNA v3.1 → ADP v3.1 → CNA v4.0 → CNA v3.0."""
    cna = _get(cve_data, "containers", "cna", default={})
    adp_list = _get(cve_data, "containers", "adp", default=[])

    def _parse_cvss(cvss: dict, version: str) -> dict:
        return {
            "version": version,
            "score": cvss.get("baseScore"),
            "severity": (cvss.get("baseSeverity") or "").upper(),
            "vector": cvss.get("vectorString"),
            "attackVector": cvss.get("attackVector"),
            "attackComplexity": cvss.get("attackComplexity"),
            "privilegesRequired": cvss.get("privilegesRequired"),
            "userInteraction": cvss.get("userInteraction"),
            "scope": cvss.get("scope"),
            "confidentialityImpact": cvss.get("confidentialityImpact"),
            "integrityImpact": cvss.get("integrityImpact"),
            "availabilityImpact": cvss.get("availabilityImpact"),
        }

    # CNA v3.1
    for metric in _get(cna, "metrics", default=[]):
        if metric.get("cvssV3_1"):
            return _parse_cvss(metric["cvssV3_1"], "3.1")

    # ADP v3.1 (CISA Vulnrichment)
    for adp in adp_list:
        for metric in _get(adp, "metrics", default=[]):
            if metric.get("cvssV3_1"):
                return _parse_cvss(metric["cvssV3_1"], "3.1")

    # CNA v4.0
    for metric in _get(cna, "metrics", default=[]):
        if metric.get("cvssV4_0"):
            return _parse_cvss(metric["cvssV4_0"], "4.0")

    # CNA v3.0 fallback
    for metric in _get(cna, "metrics", default=[]):
        if metric.get("cvssV3_0"):
            return _parse_cvss(metric["cvssV3_0"], "3.0")

    return None


# ── CWE Extraction ──────────────────────────────────────────────────────────

def extract_cwes(cve_data: dict) -> list[dict]:
    """Extract CWE IDs and descriptions from CNA and ADP containers."""
    cwes = []
    seen = set()

    # CNA problemTypes
    cna = _get(cve_data, "containers", "cna", default={})
    for pt in _get(cna, "problemTypes", default=[]):
        for desc in _get(pt, "descriptions", default=[]):
            cwe_id = desc.get("cweId")
            if cwe_id and cwe_id not in seen:
                seen.add(cwe_id)
                raw_desc = desc.get("description", "")
                # Strip the CWE ID prefix from description if present
                clean = re.sub(r"^CWE-\d+\s*", "", raw_desc).strip()
                cwes.append({"id": cwe_id, "description": clean})

    # ADP problemTypes (CISA enrichment)
    for adp in _get(cve_data, "containers", "adp", default=[]):
        for pt in _get(adp, "problemTypes", default=[]):
            for desc in _get(pt, "descriptions", default=[]):
                cwe_id = desc.get("cweId")
                if cwe_id and cwe_id not in seen:
                    seen.add(cwe_id)
                    raw_desc = desc.get("description", "")
                    clean = re.sub(r"^CWE-\d+\s*", "", raw_desc).strip()
                    cwes.append({"id": cwe_id, "description": clean})

    return cwes


# ── Affected Products Extraction ────────────────────────────────────────────

def extract_affected(cve_data: dict) -> list[dict]:
    """Extract affected products with version ranges and fix versions."""
    products = []
    cna = _get(cve_data, "containers", "cna", default={})

    for entry in _get(cna, "affected", default=[]):
        vendor = entry.get("vendor", "")
        product = entry.get("product", "")

        if not vendor or not product:
            continue
        # Skip placeholder n/a entries
        if vendor.lower() == "n/a" and product.lower() == "n/a":
            continue

        affected_versions = []
        fix_versions = []

        for ver in entry.get("versions", []):
            status = ver.get("status", "")
            version = ver.get("version", "")
            version_type = ver.get("versionType", "")

            # Skip git commit hashes and kernel commit markers
            if version_type in ("git", "original_commit_for_fix"):
                continue

            less_than = ver.get("lessThan")
            less_than_or_equal = ver.get("lessThanOrEqual")

            if status == "affected":
                if less_than and version:
                    affected_versions.append(f"{version} to < {less_than}")
                elif less_than_or_equal and version:
                    affected_versions.append(f"{version} to {less_than_or_equal}")
                elif version and version.lower() != "n/a":
                    affected_versions.append(version)
            elif status == "unaffected":
                # This is a fix version
                if version and version != "0" and version.lower() != "n/a":
                    # Skip wildcard-only entries like "6.1.*"
                    if not re.match(r"^\d+\.\d+\.\*$", version):
                        fix_versions.append(version)

        products.append({
            "vendor": vendor,
            "product": product,
            "affected_versions": affected_versions,
            "fix_versions": fix_versions,
        })

    return products


# ── Solutions / References Extraction ────────────────────────────────────────

def extract_solutions(cve_data: dict) -> str:
    """Extract solutions/workarounds text from CNA container."""
    cna = _get(cve_data, "containers", "cna", default={})

    for sol in _get(cna, "solutions", default=[]):
        text = sol.get("value", "")
        if text:
            return strip_html(text)

    for wa in _get(cna, "workarounds", default=[]):
        text = wa.get("value", "")
        if text:
            return strip_html(text)

    return ""


def extract_patch_refs(cve_data: dict) -> list[str]:
    """Extract vendor-advisory and patch reference URLs."""
    cna = _get(cve_data, "containers", "cna", default={})
    refs = []
    for ref in _get(cna, "references", default=[]):
        tags = ref.get("tags", [])
        url = ref.get("url", "")
        if not url:
            continue
        if any(t in tags for t in ("vendor-advisory", "patch", "x_refsource_CONFIRM")):
            refs.append(url)
    return refs


# ── Answer Composition ──────────────────────────────────────────────────────

def _format_products_brief(affected: list[dict]) -> str:
    """Format affected products as a brief inline string."""
    parts = []
    for p in affected[:3]:
        vendor = p["vendor"]
        product = p["product"]
        if vendor.lower() in product.lower():
            label = product
        else:
            label = f"{vendor} {product}"
        versions = p.get("affected_versions", [])
        if versions:
            label += f" {versions[0]}"
        parts.append(label)

    result = ", ".join(parts)
    if len(affected) > 3:
        result += f", and {len(affected) - 3} more products"
    return result


def _format_fix_versions(affected: list[dict]) -> str:
    """Format fix versions across all products as bullet list."""
    lines = []
    for p in affected:
        fixes = p.get("fix_versions", [])
        if not fixes:
            continue
        vendor = p["vendor"]
        product = p["product"]
        if vendor.lower() in product.lower():
            label = product
        else:
            label = f"{vendor} {product}"
        lines.append(f"- {label}: upgrade to {' or '.join(fixes)}")
    return "\n".join(lines)


def compose_summary(cve_id: str, description: str, cvss: dict | None,
                    cwes: list[dict], affected: list[dict],
                    date_published: str) -> str:
    """Compose a summary answer from structured CVE data."""
    parts = []

    # Opening: CVE ID + severity tag + description
    severity_tag = ""
    if cvss and cvss.get("score"):
        sev = cvss.get("severity") or _severity_from_score(cvss["score"])
        severity_tag = f" ({sev.lower()}, CVSS {cvss['score']})"
    parts.append(f"{cve_id}{severity_tag} — {description}")

    # CWE line
    if cwes:
        cwe_strs = []
        for c in cwes:
            s = c["id"]
            if c["description"] and c["description"].lower() not in ("n/a", ""):
                s += f" ({c['description']})"
            cwe_strs.append(s)
        parts.append(f"\nWeakness: {', '.join(cwe_strs)}")

    # Affected products
    if affected:
        parts.append(f"\nAffected: {_format_products_brief(affected)}")

    # Fix versions
    fix_str = _format_fix_versions(affected)
    if fix_str:
        parts.append(f"\nFix versions:\n{fix_str}")

    # Published date
    if date_published:
        parts.append(f"\nPublished: {date_published[:10]}")

    return "\n".join(parts)


def compose_impact(cve_id: str, description: str, cvss: dict | None,
                   cwes: list[dict], affected: list[dict]) -> str | None:
    """Compose an impact answer. Returns None if insufficient data."""
    if not cvss and not cwes:
        return None

    parts = []

    if cvss and cvss.get("score"):
        sev = cvss.get("severity") or _severity_from_score(cvss["score"])
        parts.append(
            f"{cve_id} has a CVSS {cvss['version']} base score of "
            f"{cvss['score']} ({sev})."
        )

        # Attack characteristics
        attack_info = []
        if cvss.get("attackVector"):
            attack_info.append(f"Attack Vector: {cvss['attackVector']}")
        if cvss.get("attackComplexity"):
            attack_info.append(f"Attack Complexity: {cvss['attackComplexity']}")
        if cvss.get("privilegesRequired"):
            attack_info.append(f"Privileges Required: {cvss['privilegesRequired']}")
        if cvss.get("userInteraction"):
            attack_info.append(f"User Interaction: {cvss['userInteraction']}")
        if cvss.get("scope"):
            attack_info.append(f"Scope: {cvss['scope']}")
        if attack_info:
            parts.append("\n" + " | ".join(attack_info))

        # CIA impact
        cia = []
        if cvss.get("confidentialityImpact"):
            cia.append(f"Confidentiality: {cvss['confidentialityImpact']}")
        if cvss.get("integrityImpact"):
            cia.append(f"Integrity: {cvss['integrityImpact']}")
        if cvss.get("availabilityImpact"):
            cia.append(f"Availability: {cvss['availabilityImpact']}")
        if cia:
            parts.append("Impact — " + " | ".join(cia))
    elif cwes:
        # No CVSS — pivot to CWE-based assessment
        cwe_strs = []
        for c in cwes:
            s = c["id"]
            if c["description"] and c["description"].lower() not in ("n/a", ""):
                s += f" ({c['description']})"
            cwe_strs.append(s)
        parts.append(f"{cve_id} is categorized as {', '.join(cwe_strs)}.")

    parts.append(f"\n{description}")

    if affected:
        parts.append(f"\nAffected products: {_format_products_brief(affected)}")

    return "\n".join(parts)


def compose_mitigation(cve_id: str, affected: list[dict],
                       solutions_text: str, patch_refs: list[str]) -> str | None:
    """Compose a mitigation answer. Returns None if no actionable data.

    Requires fix versions or solutions text — bare URL refs alone produce
    low-quality answers and are excluded.
    """
    fix_str = _format_fix_versions(affected)
    if not fix_str and not solutions_text:
        return None

    parts = [f"To remediate {cve_id}:"]

    if fix_str:
        parts.append(f"\n{fix_str}")

    if solutions_text:
        parts.append(f"\n{solutions_text}")

    if patch_refs:
        parts.append("\nReferences:")
        for ref in patch_refs[:5]:
            parts.append(f"- {ref}")

    return "\n".join(parts)


# ── CVE File Processing ─────────────────────────────────────────────────────

def process_cve_file(filepath: Path, min_desc_len: int = 50) -> list[dict]:
    """Parse a single CVE JSON and generate training examples."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            cve_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    # Only PUBLISHED records
    state = _get(cve_data, "cveMetadata", "state", default="")
    if state != "PUBLISHED":
        return []

    cve_id = _get(cve_data, "cveMetadata", "cveId", default="")
    if not cve_id:
        return []

    # Extract description
    description = _get(
        cve_data, "containers", "cna", "descriptions", 0, "value", default=""
    )
    description = strip_html(description).strip()
    if len(description) < min_desc_len:
        return []

    # Truncate excessively long descriptions (kernel stack traces etc.)
    if len(description) > 2000:
        description = description[:2000].rsplit(".", 1)[0] + "."

    # Extract structured data
    cvss = extract_cvss(cve_data)
    cwes = extract_cwes(cve_data)
    affected = extract_affected(cve_data)
    solutions_text = extract_solutions(cve_data)
    patch_refs = extract_patch_refs(cve_data)
    date_published = _get(cve_data, "cveMetadata", "datePublished", default="")

    examples = []

    # 1. Summary — always generated
    q = _select_template(QUESTION_TEMPLATES["summary"], cve_id).format(cve_id=cve_id)
    a = compose_summary(cve_id, description, cvss, cwes, affected, date_published)
    examples.append({"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]})

    # 2. Impact — requires CVSS or CWE
    impact_answer = compose_impact(cve_id, description, cvss, cwes, affected)
    if impact_answer:
        q = _select_template(QUESTION_TEMPLATES["impact"], cve_id).format(cve_id=cve_id)
        examples.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": impact_answer},
        ]})

    # 3. Mitigation — requires fix versions, solutions, or patch refs
    mitigation_answer = compose_mitigation(cve_id, affected, solutions_text, patch_refs)
    if mitigation_answer:
        q = _select_template(QUESTION_TEMPLATES["mitigation"], cve_id).format(cve_id=cve_id)
        examples.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": mitigation_answer},
        ]})

    return examples


# ── Database Deletion ────────────────────────────────────────────────────────

def delete_existing_pentestds(db_path: str) -> int:
    """Delete pentestds CVE records from database, preserving CISA KEV."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # Count before deletion
    cur.execute(
        "SELECT COUNT(*) FROM all_records WHERE domain = 'cve' AND source = 'pentestds'"
    )
    count = cur.fetchone()[0]

    if count == 0:
        log.info("No pentestds CVE records to delete.")
        conn.close()
        return 0

    log.info(f"Deleting {count:,} pentestds CVE records...")

    # Delete from domain table first (FK constraint)
    cur.execute("""
        DELETE FROM cve WHERE master_id IN (
            SELECT id FROM all_records WHERE domain = 'cve' AND source = 'pentestds'
        )
    """)
    domain_deleted = cur.rowcount
    log.info(f"  Deleted {domain_deleted:,} from cve table")

    # Delete from master table
    cur.execute(
        "DELETE FROM all_records WHERE domain = 'cve' AND source = 'pentestds'"
    )
    master_deleted = cur.rowcount
    log.info(f"  Deleted {master_deleted:,} from all_records table")

    # Verify CISA KEV preserved
    cur.execute("SELECT COUNT(*) FROM all_records WHERE source = 'cisa_kev'")
    kev_count = cur.fetchone()[0]
    log.info(f"  CISA KEV records preserved: {kev_count:,}")

    compute_stats(conn)
    conn.commit()
    conn.close()
    return master_deleted


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from cve.org JSON files"
    )
    parser.add_argument(
        "--cve-dir", default="docs/cves",
        help="Root directory with year subdirs (default: docs/cves)",
    )
    parser.add_argument(
        "--output", default="data/sources/cveorg_2020_2025.jsonl",
        help="Output JSONL path (default: data/sources/cveorg_2020_2025.jsonl)",
    )
    parser.add_argument("--year-start", type=int, default=2020)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument(
        "--delete-existing", action="store_true",
        help="Delete pentestds CVE records from DB before generating",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument(
        "--min-description-length", type=int, default=50,
        help="Skip CVEs with descriptions shorter than this (default: 50)",
    )
    args = parser.parse_args()

    t0 = time.time()

    log.info("=" * 60)
    log.info("  CVE.org Training Data Generator")
    log.info("=" * 60)
    log.info(f"  CVE dir:    {args.cve_dir}")
    log.info(f"  Output:     {args.output}")
    log.info(f"  Years:      {args.year_start}–{args.year_end}")
    log.info(f"  Min desc:   {args.min_description_length} chars")
    log.info("")

    # Step 1: Delete old records if requested
    if args.delete_existing:
        delete_existing_pentestds(args.db)
        log.info("")

    # Step 2: Walk CVE JSON files and generate training examples
    cve_root = Path(args.cve_dir)
    if not cve_root.exists():
        log.error(f"CVE directory not found: {cve_root}")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_generated = 0
    total_skipped = 0
    total_errors = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for year in range(args.year_start, args.year_end + 1):
            year_dir = cve_root / str(year)
            if not year_dir.exists():
                log.warning(f"Year directory not found: {year_dir}")
                continue

            # Collect all JSON files under this year (subdirs like 0xxx, 1xxx, ...)
            json_files = sorted(year_dir.rglob("CVE-*.json"))
            year_files = 0
            year_generated = 0
            year_skipped = 0
            year_errors = 0

            log.info(f"[{year}] Found {len(json_files):,} CVE files")

            for filepath in json_files:
                year_files += 1
                total_files += 1

                try:
                    examples = process_cve_file(
                        filepath, min_desc_len=args.min_description_length
                    )
                except Exception as e:
                    year_errors += 1
                    total_errors += 1
                    if year_errors <= 3:
                        log.warning(f"  Error processing {filepath.name}: {e}")
                    continue

                if not examples:
                    year_skipped += 1
                    total_skipped += 1
                    continue

                for ex in examples:
                    out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    year_generated += 1
                    total_generated += 1

                # Progress every 5000 files
                if year_files % 5000 == 0:
                    log.info(
                        f"  [{year}] {year_files:,} files → "
                        f"{year_generated:,} generated, "
                        f"{year_skipped:,} skipped, "
                        f"{year_errors:,} errors"
                    )

            log.info(
                f"  [{year}] Done: {year_files:,} files → "
                f"{year_generated:,} generated, "
                f"{year_skipped:,} skipped, "
                f"{year_errors:,} errors"
            )

    elapsed = time.time() - t0

    log.info("")
    log.info("=" * 60)
    log.info("  GENERATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Files processed:     {total_files:>10,}")
    log.info(f"  Examples generated:  {total_generated:>10,}")
    log.info(f"  Files skipped:       {total_skipped:>10,}")
    log.info(f"  Errors:              {total_errors:>10,}")
    log.info(f"  Elapsed:             {elapsed:>9.1f}s")
    log.info(f"  Output:              {out_path}")
    log.info("")
    log.info(
        f"Next step: python scripts/ingest_data.py "
        f"--jsonl {out_path} --source cveorg"
    )


if __name__ == "__main__":
    main()
