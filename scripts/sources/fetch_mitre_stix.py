#!/usr/bin/env python3
"""
============================================================================
MITRE ATT&CK STIX Data Fetcher
============================================================================
Downloads MITRE ATT&CK STIX data from GitHub and generates training examples
covering technique descriptions, tactic mappings, and detection guidance.

Source: https://github.com/mitre-attack/attack-stix-data
Format: STIX 2.1 JSON bundles

Usage:
    python -m scripts.sources.fetch_mitre_stix
    python -m scripts.sources.fetch_mitre_stix --output data/sources/mitre_stix.jsonl
============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.request import urlopen, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Direct URLs to STIX bundles on GitHub
ENTERPRISE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


def fetch_stix_bundle(url: str) -> dict:
    """Download a STIX 2.1 bundle."""
    log.info(f"Fetching STIX bundle...")
    req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    objects = data.get("objects", [])
    log.info(f"Fetched {len(objects)} STIX objects.")
    return data


def extract_techniques(bundle: dict) -> list[dict]:
    """Extract attack-pattern objects (techniques) from a STIX bundle."""
    techniques = []
    objects = bundle.get("objects", [])

    # Build ID-to-object lookup
    id_map = {obj.get("id"): obj for obj in objects}

    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        # Extract technique ID from external_references
        tech_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id")
                break

        if not tech_id:
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")
        platforms = obj.get("x_mitre_platforms", [])
        detection = obj.get("x_mitre_detection", "")
        data_sources = obj.get("x_mitre_data_sources", [])

        # Get tactics from kill_chain_phases
        tactics = []
        for phase in obj.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                tactic = phase.get("phase_name", "").replace("-", " ").title()
                tactics.append(tactic)

        techniques.append({
            "id": tech_id,
            "name": name,
            "description": description,
            "tactics": tactics,
            "platforms": platforms,
            "detection": detection,
            "data_sources": data_sources,
        })

    return techniques


def generate_training_examples(techniques: list[dict]) -> list[dict]:
    """Generate training Q&A pairs from techniques."""
    examples = []

    for tech in techniques:
        tech_id = tech["id"]
        name = tech["name"]
        description = tech["description"][:2000]  # Truncate long descriptions
        tactics = tech["tactics"]
        platforms = tech["platforms"]
        detection = tech["detection"]
        data_sources = tech["data_sources"]

        if not description:
            continue

        # 1. Technique description question
        q1 = f"Describe the MITRE ATT&CK technique {tech_id} ({name})."
        a1 = f"{tech_id}: {name}\n\n{description}"
        if tactics:
            a1 += f"\n\nTactics: {', '.join(tactics)}"
        if platforms:
            a1 += f"\nPlatforms: {', '.join(platforms)}"
        examples.append({"messages": [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
        ]})

        # 2. Tactic lookup question
        if tactics:
            q2 = f"What tactic does the technique {tech_id} ({name}) belong to in MITRE ATT&CK?"
            tactic_str = ", ".join(tactics)
            a2 = (
                f"The technique {tech_id} ({name}) belongs to the "
                f"{'tactic' if len(tactics) == 1 else 'tactics'}: {tactic_str}.\n\n"
                f"{description[:500]}"
            )
            examples.append({"messages": [
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2},
            ]})

        # 3. Detection question
        if detection:
            q3 = f"How can you detect the use of {tech_id} ({name}) in your environment?"
            a3 = f"Detection guidance for {tech_id} ({name}):\n\n{detection}"
            if data_sources:
                a3 += f"\n\nRelevant data sources: {', '.join(data_sources[:10])}"
            examples.append({"messages": [
                {"role": "user", "content": q3},
                {"role": "assistant", "content": a3},
            ]})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Fetch MITRE ATT&CK STIX data for training")
    parser.add_argument("--output", default="data/sources/mitre_stix.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--url", default=ENTERPRISE_URL,
                        help="STIX bundle URL")
    args = parser.parse_args()

    bundle = fetch_stix_bundle(args.url)
    techniques = extract_techniques(bundle)
    log.info(f"Extracted {len(techniques)} techniques.")

    examples = generate_training_examples(techniques)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"Generated {len(examples)} training examples")
    log.info(f"Saved to {out_path}")
    log.info(f"\nNext step: python scripts/ingest_data.py --jsonl {out_path} --source mitre_stix")


if __name__ == "__main__":
    main()
