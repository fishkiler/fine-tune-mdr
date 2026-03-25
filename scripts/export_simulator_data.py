#!/usr/bin/env python3
"""
============================================================================
MDR Log Simulator — Bulk Event Exporter
============================================================================
Pulls events from the MDR Log Simulator API and converts them to ChatML
JSONL format for training. This is the "fast bulk" companion to
fetch_attack_logs.py (which generates rich multi-template pairs).

Usage:
    python -m scripts.export_simulator_data
    python -m scripts.export_simulator_data --limit 1000
    python -m scripts.export_simulator_data --output data/export/simulator_events.jsonl
============================================================================
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sources.fetch_attack_logs import SimulatorClient, sanitize_log

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "export" / "simulator_events.jsonl"
RATE_LIMIT_SEC = 0.15  # 150ms between requests


# ============================================================================
# Event Fetching
# ============================================================================

def fetch_events(client: SimulatorClient, limit: int = 0) -> list[dict]:
    """Fetch preview events for all mapped techniques.

    Args:
        client: SimulatorClient instance.
        limit: Max techniques to fetch (0 = all).

    Returns:
        List of event dicts with technique metadata attached.
    """
    log.info("Fetching technique list...")
    techniques = client.get_techniques()
    log.info(f"Found {len(techniques)} mapped techniques.")

    if limit > 0:
        techniques = techniques[:limit]
        log.info(f"Limited to {limit} techniques.")

    events = []
    errors = 0

    for i, tech in enumerate(techniques, 1):
        tech_id = tech.get("technique_id", tech.get("id", "unknown"))
        tech_name = tech.get("technique_name", tech.get("name", "Unknown"))
        tactic = tech.get("tactic", "unknown")

        try:
            preview = client.get_preview(tech_id)
            preview_events = preview.get("events", preview.get("logs", []))

            for event in preview_events:
                events.append({
                    "raw": event,
                    "technique_id": tech_id,
                    "technique_name": tech_name,
                    "tactic": tactic,
                })
        except Exception as e:
            error_str = str(e)
            if "400" in error_str or "404" in error_str:
                log.debug(f"  Skipped {tech_id} ({error_str})")
            else:
                log.warning(f"  Failed {tech_id}: {error_str}")
            errors += 1

        if i % 25 == 0:
            log.info(f"  Progress: {i}/{len(techniques)} techniques, {len(events)} events")

        time.sleep(RATE_LIMIT_SEC)

    log.info(f"Fetched {len(events)} events from {len(techniques)} techniques ({errors} errors).")
    return events


# ============================================================================
# ChatML Conversion
# ============================================================================

def events_to_chatml(events: list[dict]) -> list[dict]:
    """Convert raw events to ChatML training pairs.

    Each pair asks the model to analyze a security log and identify the
    MITRE ATT&CK technique, providing structured analysis.
    """
    records = []

    for entry in events:
        raw = entry["raw"]
        tech_id = entry["technique_id"]
        tech_name = entry["technique_name"]
        tactic = entry["tactic"]

        # Strip ground-truth fields from the question side
        sanitized = sanitize_log(raw) if isinstance(raw, dict) else raw
        log_text = json.dumps(sanitized, indent=2) if isinstance(sanitized, dict) else str(sanitized)

        user_msg = (
            "Analyze the following security log event and identify any malicious activity. "
            "If an attack is detected, specify the MITRE ATT&CK technique, describe the "
            "evidence, assess the severity, and recommend response actions.\n\n"
            f"```json\n{log_text}\n```"
        )

        assistant_msg = (
            f"**Detection: {tech_id} — {tech_name}**\n\n"
            f"**Tactic:** {tactic}\n\n"
            f"**Analysis:** This log event exhibits behavior consistent with "
            f"{tech_id} ({tech_name}). The activity falls under the {tactic} tactic "
            f"in the MITRE ATT&CK framework.\n\n"
            f"**Recommended Actions:**\n"
            f"1. Isolate the affected host and preserve forensic evidence\n"
            f"2. Search for related {tech_id} indicators across the environment\n"
            f"3. Review surrounding logs for lateral movement or persistence\n"
            f"4. Escalate to incident response if confirmed malicious"
        )

        records.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    return records


# ============================================================================
# Output
# ============================================================================

def write_jsonl(records: list[dict], output_path: Path) -> None:
    """Write records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    log.info(f"Wrote {len(records)} records to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export MDR Log Simulator events as ChatML training data"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--simulator-url", type=str, default=None,
        help="Simulator URL (default: from config.yaml)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max techniques to fetch (0 = all)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    # Resolve simulator URL
    simulator_url = args.simulator_url
    if not simulator_url:
        config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            cfg = yaml.safe_load(open(config_path))
            simulator_url = cfg.get("sources", {}).get("attack_logs", {}).get("simulator_url")
    if not simulator_url:
        log.error("No simulator URL provided. Use --simulator-url or set in config.yaml")
        sys.exit(1)

    log.info(f"MDR Log Simulator Bulk Exporter")
    log.info(f"  Simulator: {simulator_url}")
    log.info(f"  Output:    {args.output}")
    log.info(f"  Limit:     {'all' if args.limit == 0 else args.limit}")

    client = SimulatorClient(simulator_url)
    events = fetch_events(client, limit=args.limit)

    if not events:
        log.warning("No events fetched. Check simulator connectivity.")
        sys.exit(1)

    records = events_to_chatml(events)
    write_jsonl(records, args.output)

    log.info(f"Export complete: {len(records)} training pairs from {len(events)} events.")


if __name__ == "__main__":
    main()
