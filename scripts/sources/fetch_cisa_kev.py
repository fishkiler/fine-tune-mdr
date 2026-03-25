#!/usr/bin/env python3
"""
============================================================================
CISA Known Exploited Vulnerabilities (KEV) Fetcher
============================================================================
Downloads the CISA KEV catalog and generates training examples covering:
  - CVE summary and impact assessment
  - Required remediation actions and deadlines
  - Attack vector and exploitation context

Source: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
API: Free, no key required.

Usage:
    python -m scripts.sources.fetch_cisa_kev
    python -m scripts.sources.fetch_cisa_kev --output data/sources/cisa_kev.jsonl
============================================================================
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

QUESTION_TEMPLATES = [
    {
        "type": "summary",
        "template": "Summarize the vulnerability {cve_id} and explain why it is listed in CISA's Known Exploited Vulnerabilities catalog.",
    },
    {
        "type": "impact",
        "template": "How serious is the impact of {cve_id} ({product})? What are the risks if left unpatched?",
    },
    {
        "type": "mitigation",
        "template": "What are the recommended mitigations for {cve_id} affecting {product}?",
    },
]


def fetch_kev_catalog() -> dict:
    """Download the CISA KEV catalog."""
    log.info(f"Fetching CISA KEV catalog from {KEV_URL}...")
    req = Request(KEV_URL, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    log.info(f"Fetched {data.get('count', '?')} vulnerabilities.")
    return data


def generate_training_examples(catalog: dict) -> list[dict]:
    """Convert KEV entries into training examples."""
    examples = []
    vulns = catalog.get("vulnerabilities", [])

    for vuln in vulns:
        cve_id = vuln.get("cveID", "")
        product = vuln.get("product", "Unknown Product")
        vendor = vuln.get("vendorProject", "Unknown Vendor")
        name = vuln.get("vulnerabilityName", "")
        description = vuln.get("shortDescription", "")
        action = vuln.get("requiredAction", "Apply updates per vendor instructions.")
        due_date = vuln.get("dueDate", "")
        date_added = vuln.get("dateAdded", "")
        known_ransomware = vuln.get("knownRansomwareCampaignUse", "Unknown")
        notes = vuln.get("notes", "")

        if not cve_id or not description:
            continue

        # Generate summary example
        answer_parts = [
            f"{cve_id} ({name})",
            f"",
            f"Vendor/Product: {vendor} {product}",
            f"Description: {description}",
            f"",
            f"This vulnerability is listed in CISA's Known Exploited Vulnerabilities (KEV) catalog, "
            f"meaning it has been observed being actively exploited in the wild.",
        ]

        if known_ransomware and known_ransomware.lower() == "known":
            answer_parts.append(
                "This vulnerability has been used in known ransomware campaigns."
            )

        if date_added:
            answer_parts.append(f"Date added to KEV: {date_added}")
        if due_date:
            answer_parts.append(f"Required remediation deadline: {due_date}")

        answer_parts.extend([
            f"",
            f"Required Action: {action}",
        ])

        if notes:
            answer_parts.append(f"Notes: {notes}")

        for tmpl in QUESTION_TEMPLATES:
            question = tmpl["template"].format(cve_id=cve_id, product=f"{vendor} {product}")

            if tmpl["type"] == "summary":
                answer = "\n".join(answer_parts)
            elif tmpl["type"] == "impact":
                answer = (
                    f"{cve_id} affects {vendor} {product}.\n\n"
                    f"{description}\n\n"
                    f"This vulnerability is in CISA's KEV catalog, confirming active exploitation "
                    f"in the wild. "
                )
                if known_ransomware and known_ransomware.lower() == "known":
                    answer += "It has been used in ransomware campaigns. "
                answer += (
                    f"Organizations using {vendor} {product} should treat this as "
                    f"a high-priority remediation item."
                )
            elif tmpl["type"] == "mitigation":
                answer = (
                    f"For {cve_id} ({vendor} {product}):\n\n"
                    f"Required Action: {action}\n"
                )
                if due_date:
                    answer += f"CISA Remediation Deadline: {due_date}\n"
                answer += (
                    f"\nThis is a CISA KEV-listed vulnerability with confirmed active exploitation. "
                    f"Patching should be prioritized immediately."
                )

            examples.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Fetch CISA KEV data for training")
    parser.add_argument("--output", default="data/sources/cisa_kev.jsonl",
                        help="Output JSONL file path")
    args = parser.parse_args()

    catalog = fetch_kev_catalog()
    examples = generate_training_examples(catalog)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"Generated {len(examples)} training examples")
    log.info(f"Saved to {out_path}")
    log.info(f"\nNext step: python scripts/ingest_data.py --jsonl {out_path} --source cisa_kev")


if __name__ == "__main__":
    main()
