#!/usr/bin/env python3
"""
============================================================================
Splunk SPL Database Ingestion
============================================================================
Converts production Splunk searches from the splunk-spl-generator database
into training examples for the siem_queries domain.

Sources:
  - ts_mdr_app (108): MDR-specific detection rules (highest quality)
  - splunk_searches (2826): Production scheduled searches
  - analysis_results: AI-generated suggested + tstats searches (26 log sources)

Usage:
    python -m scripts.sources.ingest_spl_db
    python -m scripts.sources.ingest_spl_db --db data/sources/spl_generator.db
    python -m scripts.sources.ingest_spl_db --output data/sources/spl_production.jsonl
============================================================================
"""

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "sources" / "spl_generator.db"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sources" / "spl_production.jsonl"

# Minimum query length for inclusion
MIN_QUERY_LEN = 50

# ============================================================================
# Template Selection
# ============================================================================

def _select_template(templates: list[str], key: str) -> str:
    """Deterministically select a template variant based on key hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


# ============================================================================
# Question Templates
# ============================================================================

DETECTION_RULE_TEMPLATES = [
    "Write a Splunk SPL detection rule for: {title}. The search should be "
    "production-ready and suitable for scheduled alerting.",

    "As a detection engineer, create a Splunk SPL search to detect: {title}. "
    "Include appropriate field filters and explain the detection logic.",

    "Create a Splunk SPL correlation search for the following use case: {title}. "
    "The query should be optimized for scheduled execution.",
]

MDR_RULE_TEMPLATES = [
    "Write a Splunk SPL detection rule for an MDR platform that detects: {title}. "
    "The search should run on a {schedule} schedule and be suitable for SOC alerting.",

    "As an MDR analyst, create a Splunk SPL search for: {title}. "
    "Design it for automated alerting with a {schedule} execution interval.",

    "Create a production Splunk detection rule for MDR monitoring: {title}. "
    "It should be optimized for {schedule} scheduled execution.",
]

GENERAL_SPL_TEMPLATES = [
    "Write a Splunk SPL query for: {title}.",

    "Create a Splunk search that implements: {title}. "
    "Include relevant field extractions and filtering.",

    "As a Splunk administrator, write an SPL query for: {title}.",
]

TSTATS_TEMPLATES = [
    "Write a Splunk tstats query using accelerated data models for: {title}. "
    "Use the {data_model} data model.",

    "Create an optimized Splunk tstats search for: {title}. "
    "Target the {data_model} data model for performance.",

    "Write a high-performance Splunk tstats query for: {title}. "
    "Leverage the {data_model} accelerated data model.",
]

SUGGESTED_SEARCH_TEMPLATES = [
    "Write a Splunk SPL query for analyzing {log_source} logs: {title}. {description}",

    "Create a Splunk detection search for {log_source} data: {title}. {description}",

    "As a Splunk analyst, write an SPL query targeting {log_source}: {title}. {description}",
]

EXPLAIN_TEMPLATES = [
    "Explain what this Splunk SPL query does and what security threats it detects:"
    "\n\n```spl\n{query}\n```",

    "As a SOC analyst, analyze this Splunk detection rule. Describe its purpose, "
    "detection logic, and what events would trigger it:"
    "\n\n```spl\n{query}\n```",

    "What does this Splunk SPL search detect? Break down the query logic and "
    "explain the key field filters:"
    "\n\n```spl\n{query}\n```",
]


# ============================================================================
# Answer Composers
# ============================================================================

def compose_spl_answer(title: str, query: str, schedule: str | None = None,
                       is_mdr: bool = False, description: str | None = None) -> str:
    """Compose a training answer for a SPL query."""
    parts = []

    if is_mdr:
        parts.append(f"**MDR Detection Rule: {title}**")
    else:
        parts.append(f"**{title}**")

    parts.extend(["", "```spl", query.strip(), "```", ""])

    # Generate detection logic explanation
    parts.append("**Detection Logic:**")
    parts.append(_explain_query_logic(title, query))

    # Add schedule info if available
    if schedule and schedule != "None":
        parts.extend(["", f"**Recommended Schedule:** `{schedule}`"])
        interval = _human_schedule(schedule)
        if interval:
            parts.append(f"This search runs {interval} for near-real-time detection.")

    # Add MITRE mapping if detectable from title/query
    mitre_mapping = _extract_mitre_context(title, query)
    if mitre_mapping:
        parts.extend(["", f"**Security Context:** {mitre_mapping}"])

    if description:
        parts.extend(["", f"**Use Case:** {description}"])

    return "\n".join(parts)


def compose_explain_answer(title: str, query: str, schedule: str | None = None) -> str:
    """Compose an explanation of an existing SPL query."""
    parts = [
        f"**Purpose:** {title}",
        "",
        "**Query Breakdown:**",
        _detailed_query_breakdown(query),
    ]

    # Key fields and commands
    commands = _extract_spl_commands(query)
    if commands:
        parts.extend(["", "**Key SPL Commands Used:**"])
        for cmd in commands[:8]:
            parts.append(f"- `{cmd}`")

    # Sourcetypes
    sourcetypes = re.findall(r'sourcetype\s*=\s*["\']?([^\s"\'|]+)', query)
    if sourcetypes:
        parts.extend(["", f"**Data Sources:** {', '.join(set(sourcetypes))}"])

    # Schedule
    if schedule and schedule != "None":
        interval = _human_schedule(schedule)
        if interval:
            parts.extend(["", f"**Schedule:** Runs {interval}"])

    mitre = _extract_mitre_context(title, query)
    if mitre:
        parts.extend(["", f"**Security Context:** {mitre}"])

    return "\n".join(parts)


def compose_tstats_answer(title: str, query: str, data_model: str,
                          use_case: str | None = None) -> str:
    """Compose a training answer for a tstats query."""
    parts = [
        f"**{title}**",
        "",
        "```spl",
        query.strip(),
        "```",
        "",
        "**Detection Logic:**",
        _explain_query_logic(title, query),
        "",
        f"**Data Model:** `{data_model}`",
        "",
        "**Performance Notes:**",
        "- Uses `tstats` for accelerated data model searches (10-100x faster than raw search)",
        f"- Requires the `{data_model}` data model to be accelerated",
        "- `summariesonly=true` ensures only accelerated data is searched",
    ]

    if use_case:
        parts.extend(["", f"**Use Case:** {use_case}"])

    return "\n".join(parts)


# ============================================================================
# Query Analysis Helpers
# ============================================================================

def _explain_query_logic(title: str, query: str) -> str:
    """Generate a brief explanation of query logic from its structure."""
    parts = []
    title_lower = title.lower()
    query_lower = query.lower()

    # Detect patterns from title
    if any(kw in title_lower for kw in ["brute force", "failed login", "multiple failed"]):
        parts.append("This search detects brute force authentication attacks by "
                     "identifying multiple failed login attempts, optionally followed "
                     "by a successful login indicating credential compromise.")
    elif any(kw in title_lower for kw in ["password spray", "spraying"]):
        parts.append("This search detects password spraying attacks where an attacker "
                     "tries common passwords across many accounts to avoid lockout thresholds.")
    elif any(kw in title_lower for kw in ["anomal", "unusual", "unfamiliar"]):
        parts.append("This search identifies anomalous behavior by comparing current "
                     "activity against established baselines to detect deviations.")
    elif any(kw in title_lower for kw in ["threat", "malware", "ransomware"]):
        parts.append("This search detects threat activity by correlating security events "
                     "and matching against known threat indicators.")
    elif "inactive" in title_lower or "dormant" in title_lower:
        parts.append("This search monitors for usage of inactive or dormant accounts, "
                     "which may indicate compromised credentials being leveraged by attackers.")
    else:
        parts.append(f"This search implements detection logic for: {title}.")

    # Explain key SPL patterns
    if "streamstats" in query_lower:
        parts.append("Uses `streamstats` for running calculations to track "
                     "sequential event patterns (e.g., failure-then-success sequences).")
    if "tstats" in query_lower:
        parts.append("Uses `tstats` with accelerated data models for high-performance searching.")
    if "transaction" in query_lower:
        parts.append("Uses `transaction` to group related events into sessions for correlation.")
    if "eventstats" in query_lower:
        parts.append("Uses `eventstats` to compute aggregations while preserving individual events.")
    if "iplocation" in query_lower:
        parts.append("Uses `iplocation` to enrich events with geographic data for "
                     "geo-anomaly detection.")
    if re.search(r'\|\s*where\s+\w+\s*>\s*\d+', query):
        parts.append("Applies threshold-based alerting to filter high-confidence detections.")

    return " ".join(parts)


def _detailed_query_breakdown(query: str) -> str:
    """Break down a query pipe-by-pipe."""
    # Split on pipes (but not pipes inside quotes)
    pipes = re.split(r'\|\s*(?=\w)', query)
    if len(pipes) <= 1:
        return f"Single-stage query that searches for specific events matching the filter criteria."

    parts = []
    for i, pipe in enumerate(pipes[:6]):
        pipe = pipe.strip()
        if i == 0:
            parts.append(f"1. **Base search:** Filters events using `{pipe[:100]}...`"
                        if len(pipe) > 100 else f"1. **Base search:** `{pipe}`")
        else:
            cmd = pipe.split()[0] if pipe.split() else "unknown"
            parts.append(f"{i+1}. **`{cmd}`:** {_describe_spl_command(cmd, pipe)}")

    if len(pipes) > 6:
        parts.append(f"...plus {len(pipes) - 6} additional pipeline stages.")

    return "\n".join(parts)


def _describe_spl_command(cmd: str, pipe: str) -> str:
    """Brief description of what an SPL command does in context."""
    descriptions = {
        "stats": "Aggregates events using statistical functions",
        "eval": "Creates or transforms fields with calculated values",
        "where": "Filters results based on a condition",
        "table": "Formats output with selected fields",
        "sort": "Orders results",
        "search": "Further filters events",
        "rename": "Renames fields for clarity",
        "streamstats": "Computes running statistics across ordered events",
        "eventstats": "Adds aggregated values to each event",
        "bin": "Groups time values into buckets",
        "timechart": "Creates time-series aggregations",
        "tstats": "Searches accelerated data models for performance",
        "lookup": "Enriches events with lookup table data",
        "inputlookup": "Loads data from a lookup table",
        "transaction": "Groups events into transactions",
        "dedup": "Removes duplicate events",
        "fillnull": "Replaces null values",
        "iplocation": "Enriches with geographic location data",
        "head": "Limits to first N results",
        "rex": "Extracts fields using regex",
        "mvexpand": "Expands multivalue fields into separate events",
        "append": "Appends results from a subsearch",
        "join": "Joins results from two searches",
        "chart": "Creates a chart with aggregated data",
        "outputlookup": "Writes results to a lookup table",
    }
    return descriptions.get(cmd, f"Processes events with `{cmd}`")


def _extract_spl_commands(query: str) -> list[str]:
    """Extract unique SPL commands from a query."""
    pipes = re.split(r'\|\s*', query)
    commands = []
    seen = set()
    for pipe in pipes:
        parts = pipe.strip().split()
        if parts:
            cmd = parts[0].lower().strip('`')
            if cmd not in seen and cmd not in ("search", ""):
                seen.add(cmd)
                commands.append(cmd)
    return commands


def _extract_mitre_context(title: str, query: str) -> str | None:
    """Infer MITRE ATT&CK context from title/query patterns."""
    combined = f"{title} {query}".lower()

    mappings = [
        (["brute force", "password guess"], "T1110 — Brute Force (Credential Access)"),
        (["password spray"], "T1110.003 — Password Spraying (Credential Access)"),
        (["credential stuff"], "T1110.004 — Credential Stuffing (Credential Access)"),
        (["phish", "spearphish"], "T1566 — Phishing (Initial Access)"),
        (["lateral movement", "pass the hash", "pth"], "T1550 — Use Alternate Authentication Material (Lateral Movement)"),
        (["privilege escalat"], "T1068 — Exploitation for Privilege Escalation"),
        (["data exfil"], "T1041 — Exfiltration Over C2 Channel"),
        (["ransomware", "encrypt"], "T1486 — Data Encrypted for Impact"),
        (["malware", "malicious"], "Malware detection and response"),
        (["failed login", "failed auth", "login fail"], "T1110 — Brute Force (Credential Access)"),
        (["account lockout"], "T1110 — Brute Force (Credential Access)"),
        (["new account", "account creat"], "T1136 — Create Account (Persistence)"),
        (["account delet", "account remov"], "T1531 — Account Access Removal (Impact)"),
        (["service creat", "new service"], "T1543.003 — Windows Service (Persistence)"),
        (["scheduled task"], "T1053.005 — Scheduled Task (Persistence)"),
        (["registry"], "T1112 — Modify Registry (Defense Evasion)"),
        (["dns tunnel", "dns exfil"], "T1071.004 — DNS (Command and Control)"),
        (["vpn", "remote access"], "T1133 — External Remote Services (Initial Access)"),
        (["firewall", "ids", "ips"], "Network security monitoring and threat detection"),
        (["o365", "office 365", "office365"], "Microsoft 365 security monitoring"),
        (["azure", "aad", "entra"], "Azure/Entra ID security monitoring"),
        (["aws", "cloudtrail"], "AWS cloud security monitoring"),
    ]

    for keywords, mapping in mappings:
        if any(kw in combined for kw in keywords):
            return mapping

    return None


def _human_schedule(cron: str) -> str | None:
    """Convert cron schedule to human-readable interval."""
    cron = cron.strip()
    if not cron or cron == "None":
        return None

    if cron.startswith("*/"):
        try:
            mins = int(cron.split()[0].replace("*/", ""))
            return f"every {mins} minutes"
        except (ValueError, IndexError):
            pass

    parts = cron.split()
    if len(parts) >= 2:
        if parts[0].isdigit() and parts[1] == "*":
            return f"every hour (at minute {parts[0]})"
        if parts[0] == "0" and parts[1].isdigit():
            return f"daily at {parts[1]}:00"

    return f"on cron schedule `{cron}`"


def _log_source_from_filename(filename: str) -> str:
    """Extract human-readable log source from analysis filename."""
    name = Path(filename).stem.replace("_", " ").replace("-", " ")
    # Clean up common prefixes
    for prefix in ["test data/", "test_data/", "lsuam "]:
        name = name.replace(prefix, "")
    return name.strip().title()


# ============================================================================
# Record Generators
# ============================================================================

def generate_from_ts_mdr(conn: sqlite3.Connection) -> list[dict]:
    """Generate training examples from ts_mdr_app (MDR detection rules)."""
    cur = conn.cursor()
    cur.execute("SELECT title, qualified_search, cron_schedule FROM ts_mdr_app")
    examples = []

    for title, query, schedule in cur.fetchall():
        if not query or len(query.strip()) < MIN_QUERY_LEN:
            continue

        schedule_str = schedule or "*/15 * * * *"
        human_sched = _human_schedule(schedule_str) or "every 15 minutes"

        # 1. Write query from title
        q = _select_template(MDR_RULE_TEMPLATES, f"mdr_{title}").format(
            title=title, schedule=human_sched,
        )
        a = compose_spl_answer(title, query, schedule_str, is_mdr=True)
        examples.append(_make_record(q, a))

        # 2. Explain existing query (for queries >300 chars — substantial enough)
        if len(query) > 300:
            q = _select_template(EXPLAIN_TEMPLATES, f"mdr_explain_{title}").format(
                query=query.strip(),
            )
            a = compose_explain_answer(title, query, schedule_str)
            examples.append(_make_record(q, a))

    log.info(f"  ts_mdr_app: {len(examples)} examples from 108 rules")
    return examples


def generate_from_splunk_searches(conn: sqlite3.Connection) -> list[dict]:
    """Generate training examples from splunk_searches (production queries)."""
    cur = conn.cursor()
    cur.execute("SELECT title, qualified_search, cron_schedule, is_scheduled FROM splunk_searches")
    examples = []

    for title, query, schedule, is_scheduled in cur.fetchall():
        if not query or len(query.strip()) < MIN_QUERY_LEN:
            continue

        # Determine if security-focused for template selection
        combined = f"{title} {query}".lower()
        security_kw = ["threat", "attack", "malware", "brute", "fail", "alert",
                       "authentication", "credential", "anomal", "suspicious",
                       "security", "vulnerability", "unauthorized", "blocked",
                       "denied", "detect", "incident", "intrusion"]
        is_security = any(kw in combined for kw in security_kw)

        if is_security:
            templates = DETECTION_RULE_TEMPLATES
        else:
            templates = GENERAL_SPL_TEMPLATES

        # 1. Write query from title
        q = _select_template(templates, f"ss_{title}").format(title=title)
        a = compose_spl_answer(title, query, schedule if is_scheduled else None)
        examples.append(_make_record(q, a))

        # 2. Explain query (for longer, security-focused queries)
        if is_security and len(query) > 200:
            q = _select_template(EXPLAIN_TEMPLATES, f"ss_explain_{title}").format(
                query=query.strip(),
            )
            a = compose_explain_answer(title, query, schedule if is_scheduled else None)
            examples.append(_make_record(q, a))

    log.info(f"  splunk_searches: {len(examples)} examples from 2,826 rows")
    return examples


def generate_from_analysis_results(conn: sqlite3.Connection) -> list[dict]:
    """Generate training examples from analysis_results (suggested + tstats searches)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, log_format, suggested_searches, tstats_searches
        FROM analysis_results
    """)
    examples = []

    for filename, log_format, ss_raw, ts_raw in cur.fetchall():
        log_source = _log_source_from_filename(filename)

        # Process suggested_searches
        try:
            searches = json.loads(ss_raw) if isinstance(ss_raw, str) else ss_raw
            if isinstance(searches, list):
                for s in searches:
                    if not isinstance(s, dict):
                        continue
                    stitle = s.get("title", "")
                    squery = s.get("query", "")
                    sdesc = s.get("description", stitle)
                    if not squery or len(squery.strip()) < MIN_QUERY_LEN:
                        continue

                    q = _select_template(SUGGESTED_SEARCH_TEMPLATES,
                                        f"ar_{filename}_{stitle}").format(
                        log_source=log_source, title=stitle, description=sdesc,
                    )
                    a = compose_spl_answer(stitle, squery, description=sdesc)
                    examples.append(_make_record(q, a))
        except (json.JSONDecodeError, TypeError):
            pass

        # Process tstats_searches
        try:
            tstats = json.loads(ts_raw) if isinstance(ts_raw, str) else ts_raw
            search_list = []
            if isinstance(tstats, dict) and "searches" in tstats:
                search_list = tstats["searches"]
            elif isinstance(tstats, list):
                search_list = tstats

            for s in search_list:
                if not isinstance(s, dict):
                    continue
                stitle = s.get("title", "")
                squery = s.get("query", s.get("tstats_query", ""))
                use_case = s.get("use_case", "")
                data_model = s.get("data_model", "Unknown")
                if not squery or len(squery.strip()) < MIN_QUERY_LEN:
                    continue

                q = _select_template(TSTATS_TEMPLATES,
                                    f"ts_{filename}_{stitle}").format(
                    title=stitle, data_model=data_model,
                )
                a = compose_tstats_answer(stitle, squery, data_model, use_case)
                examples.append(_make_record(q, a))
        except (json.JSONDecodeError, TypeError):
            pass

    log.info(f"  analysis_results: {len(examples)} examples from 26 analyses")
    return examples


def _make_record(user_msg: str, assistant_msg: str) -> dict:
    """Create a standard training record."""
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert SPL generator database to training examples"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to spl_generator.db")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    args = parser.parse_args()

    if not args.db.exists():
        log.error(f"Database not found: {args.db}")
        log.error("Copy it from 192.168.1.67 first:")
        log.error("  scp 192.168.1.67:/home/jayoung/Documents/code_bank/"
                  "splunk-spl-generator/data/analysis_history.db "
                  "data/sources/spl_generator.db")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  SPL Production Query Ingestion")
    log.info("=" * 60)
    log.info(f"  Database: {args.db}")
    log.info(f"  Output:   {args.output}")
    log.info("")

    conn = sqlite3.connect(str(args.db))

    all_examples = []

    # Generate from each source
    all_examples.extend(generate_from_ts_mdr(conn))
    all_examples.extend(generate_from_splunk_searches(conn))
    all_examples.extend(generate_from_analysis_results(conn))

    conn.close()

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"\nTotal: {len(all_examples)} training examples written to {args.output}")


if __name__ == "__main__":
    main()
