#!/usr/bin/env python3
"""
============================================================================
MITRE ATT&CK Groups / APT Training Data Generator
============================================================================
Downloads the same Enterprise ATT&CK STIX bundle used by fetch_mitre_stix.py
but extracts groups (intrusion-sets), software, campaigns, and relationships
to generate APT-focused training pairs.

Q&A types generated:
  - Group profile briefings
  - TTP mapping (techniques by tactic)
  - Attribution from software/malware indicators
  - Detection guidance for group activity
  - Software/tooling analysis
  - Campaign analysis

Source: https://github.com/mitre-attack/attack-stix-data
Format: STIX 2.1 JSON bundles

Usage:
    python -m scripts.sources.fetch_mitre_groups
    python -m scripts.sources.fetch_mitre_groups --output data/sources/mitre_groups.jsonl
============================================================================
"""

import argparse
import hashlib
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

ENTERPRISE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


# ── STIX Bundle Download ─────────────────────────────────────────────────────

def fetch_stix_bundle(url: str) -> dict:
    """Download a STIX 2.1 bundle."""
    log.info("Fetching STIX bundle...")
    req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    objects = data.get("objects", [])
    log.info(f"Fetched {len(objects)} STIX objects.")
    return data


# ── Template Selection ────────────────────────────────────────────────────────

def _select_template(templates: list[str], key: str) -> str:
    """Deterministically select a template variant based on key hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


# ── STIX Data Extraction ─────────────────────────────────────────────────────

def _is_active(obj: dict) -> bool:
    """Check if a STIX object is not revoked or deprecated."""
    return not obj.get("revoked", False) and not obj.get("x_mitre_deprecated", False)


def _get_attack_id(obj: dict) -> str | None:
    """Extract ATT&CK external ID (G0007, S0154, C0024, T1566, etc.)."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _get_aliases(obj: dict) -> list[str]:
    """Get all aliases for a group or software object."""
    aliases = list(obj.get("aliases", []))
    # intrusion-set also stores aliases in x_mitre_aliases
    aliases.extend(obj.get("x_mitre_aliases", []))
    # Deduplicate preserving order, excluding the primary name
    name = obj.get("name", "")
    seen = {name.lower()}
    result = []
    for a in aliases:
        if a.lower() not in seen:
            seen.add(a.lower())
            result.append(a)
    return result


def extract_all(bundle: dict) -> dict:
    """Extract groups, software, campaigns, techniques, and relationships."""
    objects = bundle.get("objects", [])

    groups = {}       # stix_id -> group dict
    software = {}     # stix_id -> software dict
    campaigns = {}    # stix_id -> campaign dict
    techniques = {}   # stix_id -> technique dict
    relationships = []  # list of relationship dicts

    for obj in objects:
        otype = obj.get("type")
        if not _is_active(obj):
            continue

        stix_id = obj.get("id", "")
        attack_id = _get_attack_id(obj)

        if otype == "intrusion-set" and attack_id:
            groups[stix_id] = {
                "attack_id": attack_id,
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "aliases": _get_aliases(obj),
                "stix_id": stix_id,
                "techniques": [],      # filled from relationships
                "software": [],        # filled from relationships
                "campaigns": [],       # filled from relationships
            }

        elif otype in ("malware", "tool") and attack_id:
            software[stix_id] = {
                "attack_id": attack_id,
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "type": otype,
                "stix_id": stix_id,
                "aliases": _get_aliases(obj),
                "techniques": [],      # filled from relationships
            }

        elif otype == "campaign" and attack_id:
            campaigns[stix_id] = {
                "attack_id": attack_id,
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "stix_id": stix_id,
                "first_seen": obj.get("first_seen", ""),
                "last_seen": obj.get("last_seen", ""),
                "group": None,         # filled from relationships
                "techniques": [],      # filled from relationships
                "software": [],        # filled from relationships
            }

        elif otype == "attack-pattern" and attack_id:
            tactics = []
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    tactics.append(phase["phase_name"].replace("-", " ").title())
            techniques[stix_id] = {
                "attack_id": attack_id,
                "name": obj.get("name", ""),
                "tactics": tactics,
                "stix_id": stix_id,
            }

        elif otype == "relationship":
            relationships.append({
                "source_ref": obj.get("source_ref", ""),
                "target_ref": obj.get("target_ref", ""),
                "relationship_type": obj.get("relationship_type", ""),
                "description": obj.get("description", ""),
            })

    log.info(
        f"Extracted: {len(groups)} groups, {len(software)} software, "
        f"{len(campaigns)} campaigns, {len(techniques)} techniques, "
        f"{len(relationships)} relationships"
    )

    # ── Wire up relationships ──
    for rel in relationships:
        src = rel["source_ref"]
        tgt = rel["target_ref"]
        rtype = rel["relationship_type"]
        desc = rel["description"]

        if rtype == "uses":
            # group uses technique
            if src in groups and tgt in techniques:
                groups[src]["techniques"].append({
                    **techniques[tgt],
                    "context": desc,
                })
            # group uses software
            elif src in groups and tgt in software:
                groups[src]["software"].append(software[tgt])
            # software uses technique
            elif src in software and tgt in techniques:
                software[src]["techniques"].append(techniques[tgt])
            # campaign uses technique
            elif src in campaigns and tgt in techniques:
                campaigns[src]["techniques"].append(techniques[tgt])
            # campaign uses software
            elif src in campaigns and tgt in software:
                campaigns[src]["software"].append(software[tgt])

        elif rtype == "attributed-to":
            # campaign attributed-to group
            if src in campaigns and tgt in groups:
                campaigns[src]["group"] = groups[tgt]
                groups[tgt]["campaigns"].append(campaigns[src])

    return {
        "groups": groups,
        "software": software,
        "campaigns": campaigns,
        "techniques": techniques,
    }


# ── Training Example Generation ──────────────────────────────────────────────

PROFILE_TEMPLATES = [
    "Who is {name}? Provide a threat intelligence briefing.",
    "Brief me on the threat actor {name}.",
    "What is known about the APT group {name} ({attack_id})?",
    "Provide an overview of {name} and their operations.",
]

TTP_TEMPLATES = [
    "What techniques does {name} use for {tactic}?",
    "Describe the {tactic} techniques associated with {name}.",
    "How does {name} perform {tactic} operations?",
]

ATTRIBUTION_TEMPLATES = [
    "An attacker is using {software}. Which threat group might be responsible?",
    "We detected {software} in our environment. What group is known to use it?",
    "What threat actor is associated with {software}?",
]

DETECTION_TEMPLATES = [
    "How would you detect {name} activity in an enterprise network?",
    "What indicators and techniques should analysts monitor to identify {name}?",
    "Describe detection strategies for {name} ({attack_id}) operations.",
]

SOFTWARE_TEMPLATES = [
    "What malware and tools does {name} use?",
    "Describe the software arsenal of {name}.",
    "What tools are associated with {name} ({attack_id})?",
]

CAMPAIGN_TEMPLATES = [
    "Describe the {campaign_name} campaign.",
    "What is known about the {campaign_name} campaign ({campaign_id})?",
    "Provide analysis of the {campaign_name} campaign attributed to {name}.",
]


def _group_techniques_by_tactic(techniques: list[dict]) -> dict[str, list[dict]]:
    """Group a list of technique dicts by their tactic(s)."""
    by_tactic = {}
    for tech in techniques:
        for tactic in tech.get("tactics", ["Other"]):
            by_tactic.setdefault(tactic, []).append(tech)
    return by_tactic


def generate_group_examples(data: dict) -> list[dict]:
    """Generate training Q&A pairs from enriched group data."""
    examples = []
    groups = data["groups"]

    for stix_id, group in groups.items():
        name = group["name"]
        attack_id = group["attack_id"]
        description = group["description"]
        aliases = group["aliases"]
        techniques = group["techniques"]
        sw = group["software"]
        camps = group["campaigns"]
        key = attack_id  # for template selection

        # ── 1. Group Profile ──
        if len(description) >= 50:
            q = _select_template(PROFILE_TEMPLATES, key).format(
                name=name, attack_id=attack_id
            )
            parts = [f"{name} ({attack_id})"]
            if aliases:
                parts.append(f"Also known as: {', '.join(aliases)}")
            parts.append(f"\n{description[:2000]}")

            if techniques:
                parts.append(f"\n{name} has been observed using {len(techniques)} "
                             f"ATT&CK techniques across their operations.")
            if sw:
                sw_names = sorted(set(s["name"] for s in sw))[:10]
                parts.append(f"Associated tools and malware: {', '.join(sw_names)}")
            if camps:
                camp_names = [c["name"] for c in camps]
                parts.append(f"Known campaigns: {', '.join(camp_names)}")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 2. TTP Mapping (per tactic with 2+ techniques) ──
        by_tactic = _group_techniques_by_tactic(techniques)
        for tactic, tactic_techs in by_tactic.items():
            if len(tactic_techs) < 2:
                continue

            q = _select_template(TTP_TEMPLATES, f"{key}_{tactic}").format(
                name=name, tactic=tactic.lower()
            )
            parts = [f"{name} ({attack_id}) uses the following techniques "
                     f"for {tactic.lower()}:\n"]

            for tech in sorted(tactic_techs, key=lambda t: t["attack_id"]):
                line = f"- {tech['attack_id']} ({tech['name']})"
                ctx = tech.get("context", "")
                if ctx:
                    # Trim context to first sentence for conciseness
                    first_sentence = ctx.split(". ")[0].rstrip(".")
                    if len(first_sentence) < 300:
                        line += f": {first_sentence}."
                parts.append(line)

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 3. Attribution (from distinctive software) ──
        if len(sw) >= 1:
            # Pick the first software alphabetically for determinism
            target_sw = sorted(sw, key=lambda s: s["name"])[0]
            sw_label = f"{target_sw['name']} ({target_sw['attack_id']})"

            q = _select_template(ATTRIBUTION_TEMPLATES, f"{key}_attr").format(
                software=target_sw["name"]
            )
            parts = [
                f"{target_sw['name']} ({target_sw['attack_id']}) is a "
                f"{target_sw['type']} associated with {name} ({attack_id}).",
            ]
            if aliases:
                parts.append(f"{name} is also tracked as: {', '.join(aliases[:5])}")
            if target_sw.get("description"):
                parts.append(f"\n{target_sw['description'][:500]}")

            # Other groups that use the same software
            other_users = []
            for g_id, g in groups.items():
                if g_id == stix_id:
                    continue
                if any(s["stix_id"] == target_sw["stix_id"] for s in g["software"]):
                    other_users.append(f"{g['name']} ({g['attack_id']})")
            if other_users:
                parts.append(f"\nNote: {target_sw['name']} is also used by "
                             f"{', '.join(other_users[:5])}.")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 4. Detection Guidance (3+ mapped techniques) ──
        if len(techniques) >= 3:
            q = _select_template(DETECTION_TEMPLATES, f"{key}_det").format(
                name=name, attack_id=attack_id
            )
            by_tactic_det = _group_techniques_by_tactic(techniques)
            parts = [
                f"To detect {name} ({attack_id}) activity, monitor for "
                f"the following techniques across {len(by_tactic_det)} "
                f"ATT&CK tactics:\n"
            ]
            for tactic in sorted(by_tactic_det.keys()):
                tactic_techs = by_tactic_det[tactic]
                tech_strs = [f"{t['attack_id']} {t['name']}" for t in
                             sorted(tactic_techs, key=lambda t: t["attack_id"])[:5]]
                parts.append(f"**{tactic}**: {', '.join(tech_strs)}")

            if sw:
                sw_names = sorted(set(s["name"] for s in sw))[:5]
                parts.append(f"\nAlso watch for indicators of: {', '.join(sw_names)}")

            parts.append(f"\nTotal mapped techniques: {len(techniques)}")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 5. Software/Tooling Analysis (2+ software) ──
        if len(sw) >= 2:
            q = _select_template(SOFTWARE_TEMPLATES, f"{key}_sw").format(
                name=name, attack_id=attack_id
            )
            parts = [f"{name} ({attack_id}) is associated with the following "
                     f"tools and malware:\n"]

            for s in sorted(sw, key=lambda s: s["name"]):
                line = f"- {s['name']} ({s['attack_id']}) [{s['type']}]"
                if s.get("description"):
                    first_sentence = s["description"].split(". ")[0].rstrip(".")
                    if len(first_sentence) < 200:
                        line += f": {first_sentence}."
                parts.append(line)

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 6. Campaign Analysis ──
        for camp in camps:
            if not camp.get("description") or len(camp["description"]) < 50:
                continue

            q = _select_template(CAMPAIGN_TEMPLATES, f"{key}_{camp['attack_id']}").format(
                campaign_name=camp["name"],
                campaign_id=camp["attack_id"],
                name=name,
            )
            parts = [f"{camp['name']} ({camp['attack_id']})"]
            if camp.get("first_seen") or camp.get("last_seen"):
                timeline = []
                if camp["first_seen"]:
                    timeline.append(f"First seen: {camp['first_seen'][:10]}")
                if camp["last_seen"]:
                    timeline.append(f"Last seen: {camp['last_seen'][:10]}")
                parts.append(" | ".join(timeline))

            parts.append(f"\nAttributed to: {name} ({attack_id})")
            parts.append(f"\n{camp['description'][:1500]}")

            camp_techs = camp.get("techniques", [])
            if camp_techs:
                tech_strs = [f"{t['attack_id']} ({t['name']})"
                             for t in sorted(camp_techs, key=lambda t: t["attack_id"])[:10]]
                parts.append(f"\nTechniques used: {', '.join(tech_strs)}")

            camp_sw = camp.get("software", [])
            if camp_sw:
                sw_strs = [f"{s['name']} ({s['attack_id']})"
                           for s in sorted(camp_sw, key=lambda s: s["name"])]
                parts.append(f"Software used: {', '.join(sw_strs)}")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate APT training data from MITRE ATT&CK STIX groups"
    )
    parser.add_argument(
        "--output", default="data/sources/mitre_groups.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--url", default=ENTERPRISE_URL, help="STIX bundle URL")
    args = parser.parse_args()

    bundle = fetch_stix_bundle(args.url)
    data = extract_all(bundle)

    examples = generate_group_examples(data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary stats
    groups_with_examples = sum(
        1 for g in data["groups"].values()
        if len(g["description"]) >= 50 or len(g["techniques"]) >= 2
    )
    log.info(f"Generated {len(examples)} training examples "
             f"from {groups_with_examples} groups")
    log.info(f"Saved to {out_path}")
    log.info(f"\nNext step: python scripts/ingest_data.py "
             f"--jsonl {out_path} --source mitre_stix_groups")


if __name__ == "__main__":
    main()
