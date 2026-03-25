#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Data Validation (Rule-Based)
============================================================================
Runs automated format/content checks against every record in the database.
Fast, free, and catches obvious data quality issues before LLM review.

Validation checks by domain:
  - CVE: Valid CVE IDs, CVSS range, severity-score match, CWE validity
  - MITRE ATT&CK: Valid technique IDs, tactic names, technique-tactic mapping
  - Secure Code Review: Code blocks present, both secure/insecure examples
  - Security General: Substantive responses, actionable guidance

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --domain cve --limit 1000
    python scripts/validate_data.py --db mdr-database/mdr_dataset.db
============================================================================
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.db_utils import DEFAULT_DB_PATH, DOMAINS, get_connection, compute_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Valid MITRE ATT&CK Tactics
# ============================================================================

VALID_TACTICS = {
    "reconnaissance", "resource development", "initial access", "execution",
    "persistence", "privilege escalation", "defense evasion",
    "credential access", "discovery", "lateral movement", "collection",
    "command and control", "exfiltration", "impact",
}

# Severity-to-CVSS ranges
SEVERITY_RANGES = {
    "CRITICAL": (9.0, 10.0),
    "HIGH": (7.0, 8.9),
    "MEDIUM": (4.0, 6.9),
    "LOW": (0.1, 3.9),
}


# ============================================================================
# Validation Functions
# ============================================================================

def validate_cve_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a CVE domain record."""
    errors = []
    full_text = f"{user_msg} {assistant_msg}"

    # Check CVE ID format
    cve_ids = metadata.get("cve_ids")
    if cve_ids:
        for cve_id in cve_ids.split(","):
            if not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id.strip()):
                errors.append({"check": "cve_id_format", "severity": "error",
                               "msg": f"Invalid CVE ID format: {cve_id}"})

    # Check CVSS score range
    cvss = metadata.get("cvss_score")
    if cvss is not None:
        if not (0.0 <= cvss <= 10.0):
            errors.append({"check": "cvss_range", "severity": "error",
                           "msg": f"CVSS score {cvss} out of range [0.0-10.0]"})

    # Check severity-CVSS consistency
    severity = metadata.get("severity")
    if severity and cvss is not None:
        expected_range = SEVERITY_RANGES.get(severity)
        if expected_range:
            lo, hi = expected_range
            if not (lo <= cvss <= hi):
                errors.append({"check": "severity_cvss_mismatch", "severity": "warn",
                               "msg": f"Severity {severity} expects CVSS {lo}-{hi}, got {cvss}"})

    # Check CWE ID format
    cwe_ids = metadata.get("cwe_ids")
    if cwe_ids:
        for cwe_id in cwe_ids.split(","):
            if not re.match(r"^CWE-\d+$", cwe_id.strip()):
                errors.append({"check": "cwe_id_format", "severity": "warn",
                               "msg": f"Invalid CWE ID format: {cwe_id}"})

    # Check assistant response is substantive (not just echoing the CVE ID)
    if len(assistant_msg.strip()) < 50:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Assistant response too short ({len(assistant_msg)} chars)"})

    # Check question-response alignment
    question_type = metadata.get("question_type")
    if question_type == "impact" and "impact" not in assistant_msg.lower() and "affect" not in assistant_msg.lower():
        errors.append({"check": "question_response_align", "severity": "warn",
                       "msg": "Impact question but response doesn't discuss impact"})

    if question_type == "mitigation" and not any(
        kw in assistant_msg.lower() for kw in ["mitigat", "patch", "update", "fix", "remediat", "workaround"]
    ):
        errors.append({"check": "question_response_align", "severity": "warn",
                       "msg": "Mitigation question but response lacks mitigation guidance"})

    return errors


def validate_mitre_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a MITRE ATT&CK domain record."""
    errors = []
    full_text = f"{user_msg} {assistant_msg}"

    # Check technique ID format
    techniques = metadata.get("mitre_techniques")
    if techniques:
        for tech_id in techniques.split(","):
            tech_id = tech_id.strip()
            if not re.match(r"^T\d{4}(?:\.\d{3})?$", tech_id):
                errors.append({"check": "technique_id_format", "severity": "error",
                               "msg": f"Invalid technique ID format: {tech_id}"})

    # Check for valid tactic names
    text_lower = full_text.lower()
    found_tactics = [t for t in VALID_TACTICS if t in text_lower]
    if not found_tactics and techniques:
        errors.append({"check": "missing_tactic", "severity": "warn",
                       "msg": "Technique referenced but no valid tactic mentioned"})

    # Check response substance
    if len(assistant_msg.strip()) < 30:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Assistant response too short ({len(assistant_msg)} chars)"})

    return errors


def validate_code_review_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a secure code review domain record."""
    errors = []

    # Check for code blocks in response
    has_code = bool(re.search(r"```[\s\S]*?```", assistant_msg) or
                    re.search(r"    \S", assistant_msg))
    if not has_code:
        errors.append({"check": "missing_code_blocks", "severity": "warn",
                       "msg": "Code review response missing code examples"})

    # Check for security explanation
    security_keywords = ["vulnerab", "secur", "risk", "attack", "exploit",
                         "injection", "xss", "csrf", "sanitiz", "validat"]
    has_security = any(kw in assistant_msg.lower() for kw in security_keywords)
    if not has_security:
        errors.append({"check": "missing_security_context", "severity": "warn",
                       "msg": "Code review response lacks security-specific language"})

    # Check response substance
    if len(assistant_msg.strip()) < 50:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Assistant response too short ({len(assistant_msg)} chars)"})

    return errors


def validate_apt_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate an APT intelligence domain record."""
    errors = []
    full_text = f"{user_msg} {assistant_msg}"

    # Check response substance
    if len(assistant_msg.strip()) < 100:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"APT response too short ({len(assistant_msg)} chars, need 100+)"})

    # Check for group name or ID presence in response
    has_group_id = bool(re.search(r"\bG\d{4}\b", assistant_msg))
    has_apt_name = bool(re.search(r"APT\d+", assistant_msg))
    has_actor_name = bool(re.search(
        r"(?:Also known as|Attribution|tracked|associated with)\b",
        assistant_msg, re.IGNORECASE,
    ))
    if not (has_group_id or has_apt_name or has_actor_name):
        errors.append({"check": "missing_group_identity", "severity": "warn",
                       "msg": "APT response lacks group ID, APT name, or identity markers"})

    # Check for specificity signals (ATT&CK IDs, country names, technique keywords)
    specificity_patterns = [
        r"T\d{4}",           # technique IDs
        r"S\d{4}",           # software IDs
        r"C\d{4}",           # campaign IDs
        r"\bG\d{4}\b",       # group IDs
    ]
    specificity_keywords = [
        "initial access", "lateral movement", "persistence", "exfiltration",
        "credential access", "execution", "defense evasion", "discovery",
        "command and control", "collection", "privilege escalation",
        "reconnaissance", "resource development", "impact",
        "malware", "tool", "campaign", "technique",
    ]
    has_specificity = (
        any(re.search(p, assistant_msg) for p in specificity_patterns) or
        any(kw in assistant_msg.lower() for kw in specificity_keywords)
    )
    if not has_specificity:
        errors.append({"check": "low_specificity", "severity": "warn",
                       "msg": "APT response lacks ATT&CK IDs, tactics, or technical specifics"})

    return errors


def validate_log_analysis_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a log analysis domain record."""
    errors = []

    # Check for embedded JSON log in question
    if "```" not in user_msg and "{" not in user_msg:
        errors.append({"check": "missing_embedded_log", "severity": "error",
                       "msg": "Log analysis question missing embedded log data"})

    # Check answer contains technique identification
    if not re.search(r"T\d{4}(?:\.\d{3})?", assistant_msg):
        errors.append({"check": "missing_technique_id", "severity": "warn",
                       "msg": "Log analysis answer missing MITRE technique ID"})

    # Check for severity assessment in answer
    severity_terms = ["critical", "high", "medium", "low", "severity"]
    if not any(term in assistant_msg.lower() for term in severity_terms):
        errors.append({"check": "missing_severity", "severity": "warn",
                       "msg": "Log analysis answer missing severity assessment"})

    # Check response substance
    if len(assistant_msg.strip()) < 100:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Log analysis response too short ({len(assistant_msg)} chars, need 100+)"})

    return errors


def validate_siem_queries_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a SIEM queries domain record."""
    errors = []
    full_text = f"{user_msg} {assistant_msg}"

    # Check for code block, raw SPL content, or query breakdown in answer
    has_code_block = "```" in assistant_msg
    has_raw_spl = bool(re.search(
        r'\|\s*(stats|eval|table|where|search|rename|tstats|streamstats|timechart|chart)',
        assistant_msg))
    has_query_breakdown = "Query Breakdown" in assistant_msg or "Base search" in assistant_msg
    if not has_code_block and not has_raw_spl and not has_query_breakdown:
        errors.append({"check": "missing_code_block", "severity": "error",
                       "msg": "SIEM query answer missing code block or SPL content"})

    # Check for SPL or KQL syntax keywords (broader set for production queries)
    spl_keywords = ["index=", "sourcetype=", "| stats", "| where", "| table",
                    "| search", "| eval", "| tstats", "| streamstats",
                    "| timechart", "| chart", "| rename", "| rex",
                    "| transaction", "| eventstats", "| inputlookup",
                    "| outputlookup", "| dedup", "| join", "| append",
                    "datamodel=", "summariesonly"]
    kql_keywords = ["SecurityEvent", "DeviceProcessEvents", "| where", "| project", "| summarize"]
    has_spl = any(kw in full_text for kw in spl_keywords)
    has_kql = any(kw in full_text for kw in kql_keywords)
    if not has_spl and not has_kql:
        errors.append({"check": "missing_query_syntax", "severity": "warn",
                       "msg": "SIEM query answer lacks SPL or KQL syntax markers"})

    # Check response substance
    if len(assistant_msg.strip()) < 80:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"SIEM query response too short ({len(assistant_msg)} chars, need 80+)"})

    return errors


def validate_sigma_rules_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a Sigma rules domain record."""
    errors = []

    # Check for YAML structure markers in answer
    yaml_markers = ["title:", "logsource:", "detection:", "condition:"]
    has_yaml = sum(1 for m in yaml_markers if m in assistant_msg)
    if has_yaml < 2:
        # May be a sigma explanation rather than a rule — check for explanation content
        if not any(kw in assistant_msg.lower() for kw in ["this rule", "this sigma", "detects", "triggers"]):
            errors.append({"check": "missing_yaml_structure", "severity": "warn",
                           "msg": f"Sigma answer missing YAML markers (found {has_yaml}/4)"})

    # Check response substance
    if len(assistant_msg.strip()) < 80:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Sigma response too short ({len(assistant_msg)} chars, need 80+)"})

    return errors


def validate_security_general_record(user_msg: str, assistant_msg: str, metadata: dict) -> list[dict]:
    """Validate a security general domain record."""
    errors = []

    # Check response substance (no generic filler)
    if len(assistant_msg.strip()) < 30:
        errors.append({"check": "response_too_short", "severity": "error",
                       "msg": f"Assistant response too short ({len(assistant_msg)} chars)"})

    # Check for actionable content
    filler_patterns = [
        r"^(yes|no|ok|sure|correct)[.!]?$",
        r"^I('m| am) not sure",
        r"^I don't know",
    ]
    stripped = assistant_msg.strip()
    for pattern in filler_patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            errors.append({"check": "generic_filler", "severity": "error",
                           "msg": "Response is generic filler, not actionable guidance"})
            break

    return errors


def validate_record(record_id: int, domain: str, user_msg: str,
                    assistant_msg: str, metadata: dict) -> tuple[str, list[dict]]:
    """Run all applicable validation checks on a record.

    Returns (status, errors) where status is 'pass', 'fail', or 'warn'.
    """
    errors = []

    # Universal checks
    if not user_msg or not user_msg.strip():
        errors.append({"check": "empty_user_message", "severity": "error",
                       "msg": "User message is empty"})
    if not assistant_msg or not assistant_msg.strip():
        errors.append({"check": "empty_assistant_message", "severity": "error",
                       "msg": "Assistant message is empty"})

    if errors:
        return "fail", errors

    # Domain-specific checks
    if domain == "cve":
        errors.extend(validate_cve_record(user_msg, assistant_msg, metadata))
    elif domain == "mitre_attack":
        errors.extend(validate_mitre_record(user_msg, assistant_msg, metadata))
    elif domain == "secure_code_review":
        errors.extend(validate_code_review_record(user_msg, assistant_msg, metadata))
    elif domain == "apt_intel":
        errors.extend(validate_apt_record(user_msg, assistant_msg, metadata))
    elif domain == "security_general":
        errors.extend(validate_security_general_record(user_msg, assistant_msg, metadata))
    elif domain == "log_analysis":
        errors.extend(validate_log_analysis_record(user_msg, assistant_msg, metadata))
    elif domain == "siem_queries":
        errors.extend(validate_siem_queries_record(user_msg, assistant_msg, metadata))
    elif domain == "sigma_rules":
        errors.extend(validate_sigma_rules_record(user_msg, assistant_msg, metadata))
    # exploitdb, stix_general: basic checks only for now

    # Determine overall status
    has_errors = any(e["severity"] == "error" for e in errors)
    has_warnings = any(e["severity"] == "warn" for e in errors)

    if has_errors:
        return "fail", errors
    elif has_warnings:
        return "warn", errors
    else:
        return "pass", []


# ============================================================================
# Main Runner
# ============================================================================

def run_validation(db_path: str, domain_filter: str | None = None,
                   limit: int | None = None, batch_size: int = 5000) -> dict:
    """Run validation on all records in the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # Build query
    conditions = []
    params = []
    if domain_filter:
        conditions.append("domain = ?")
        params.append(domain_filter)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    cur.execute(
        f"SELECT id, domain, user_message, assistant_message, "
        f"cve_ids, mitre_techniques, cwe_ids, severity, cvss_score, "
        f"question_type FROM all_records {where} {limit_clause}",
        params,
    )

    stats = {
        "total": 0,
        "pass": 0,
        "fail": 0,
        "warn": 0,
        "errors_by_check": {},
        "errors_by_domain": {},
    }

    update_batch = []

    for row in cur.fetchall():
        (rec_id, domain, user_msg, assistant_msg,
         cve_ids, techniques, cwe_ids, severity, cvss_score, qtype) = row

        metadata = {
            "cve_ids": cve_ids,
            "mitre_techniques": techniques,
            "cwe_ids": cwe_ids,
            "severity": severity,
            "cvss_score": cvss_score,
            "question_type": qtype,
        }

        status, errors = validate_record(rec_id, domain, user_msg, assistant_msg, metadata)

        stats["total"] += 1
        stats[status] += 1

        for err in errors:
            check = err["check"]
            stats["errors_by_check"][check] = stats["errors_by_check"].get(check, 0) + 1
            if domain not in stats["errors_by_domain"]:
                stats["errors_by_domain"][domain] = {}
            stats["errors_by_domain"][domain][check] = \
                stats["errors_by_domain"][domain].get(check, 0) + 1

        errors_json = json.dumps(errors) if errors else None
        update_batch.append((status, errors_json, rec_id))

        if len(update_batch) >= batch_size:
            _flush_updates(conn, update_batch)
            update_batch = []
            log.info(f"  Validated {stats['total']:,} records...")

    if update_batch:
        _flush_updates(conn, update_batch)

    # Update domain_stats
    compute_stats(conn)
    conn.execute("PRAGMA optimize")
    conn.close()

    return stats


def _flush_updates(conn, batch):
    """Update validation status for a batch of records."""
    cur = conn.cursor()
    cur.executemany(
        "UPDATE all_records SET validation_status = ?, validation_errors = ? WHERE id = ?",
        batch,
    )
    conn.commit()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate MDR dataset records")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--domain", choices=DOMAINS, help="Only validate this domain")
    parser.add_argument("--limit", type=int, help="Max records to validate")
    parser.add_argument("--batch-size", type=int, default=5000, help="Update batch size")
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error(f"Database not found: {args.db}")
        log.error("Run 'python mdr-database/build_dataset_db.py' first.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  MDR Data Validation")
    log.info("=" * 60)
    if args.domain:
        log.info(f"  Domain filter: {args.domain}")
    if args.limit:
        log.info(f"  Record limit: {args.limit:,}")
    log.info("")

    stats = run_validation(args.db, args.domain, args.limit, args.batch_size)

    # Print results
    log.info("")
    log.info("=" * 60)
    log.info("  VALIDATION RESULTS")
    log.info("=" * 60)
    log.info(f"  Total validated: {stats['total']:,}")
    log.info(f"  Pass:            {stats['pass']:,} ({stats['pass']/max(stats['total'],1):.1%})")
    log.info(f"  Warn:            {stats['warn']:,} ({stats['warn']/max(stats['total'],1):.1%})")
    log.info(f"  Fail:            {stats['fail']:,} ({stats['fail']/max(stats['total'],1):.1%})")

    if stats["errors_by_check"]:
        log.info("")
        log.info("  Errors by check:")
        for check, count in sorted(stats["errors_by_check"].items(), key=lambda x: -x[1]):
            log.info(f"    {check:<35} {count:>8,}")

    if stats["errors_by_domain"]:
        log.info("")
        log.info("  Errors by domain:")
        for domain, checks in sorted(stats["errors_by_domain"].items()):
            total_domain_errors = sum(checks.values())
            log.info(f"    {domain}: {total_domain_errors:,} issues")
            for check, count in sorted(checks.items(), key=lambda x: -x[1])[:5]:
                log.info(f"      {check:<33} {count:>8,}")


if __name__ == "__main__":
    main()
