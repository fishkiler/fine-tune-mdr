#!/usr/bin/env python3
"""
============================================================================
MISP Galaxy Threat Actor Training Data Generator
============================================================================
Downloads the MISP Galaxy threat-actor.json from GitHub and generates
attribution, targeting, and actor identification training pairs.

Covers data MITRE ATT&CK doesn't include well:
  - Country attribution and state sponsorship
  - Target sectors and victim countries
  - Motivation and operational goals
  - Extensive alias/synonym mappings

Source: https://github.com/MISP/misp-galaxy
Format: MISP Galaxy JSON (threat-actor cluster)

Usage:
    python -m scripts.sources.fetch_misp_galaxy
    python -m scripts.sources.fetch_misp_galaxy --output data/sources/misp_galaxy.jsonl
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

GALAXY_URL = (
    "https://raw.githubusercontent.com/MISP/misp-galaxy/"
    "main/clusters/threat-actor.json"
)


# ── Download ──────────────────────────────────────────────────────────────────

def fetch_galaxy(url: str) -> dict:
    """Download the MISP Galaxy threat-actor cluster."""
    log.info("Fetching MISP Galaxy threat-actor.json...")
    req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    actors = data.get("values", [])
    log.info(f"Fetched {len(actors)} threat actor entries.")
    return data


# ── Template Selection ────────────────────────────────────────────────────────

def _select_template(templates: list[str], key: str) -> str:
    """Deterministically select a template variant based on key hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


# ── MITRE Cross-Reference ────────────────────────────────────────────────────

def build_mitre_lookup(mitre_groups: dict | None) -> dict[str, str]:
    """Build a name/alias → ATT&CK ID lookup for cross-referencing.

    If mitre_groups is None, returns empty dict (graceful degradation).
    """
    lookup = {}
    if not mitre_groups:
        return lookup
    for g in mitre_groups.values():
        attack_id = g.get("attack_id", "")
        if not attack_id:
            continue
        lookup[g["name"].lower()] = attack_id
        for alias in g.get("aliases", []):
            lookup[alias.lower()] = attack_id
    return lookup


# ── Training Example Generation ──────────────────────────────────────────────

PROFILE_TEMPLATES = [
    "What is known about the threat actor {name}?",
    "Provide a threat intelligence profile for {name}.",
    "Who is the threat actor known as {name}?",
    "Brief me on {name} and their operations.",
]

ATTRIBUTION_TEMPLATES = [
    "What country is {name} believed to operate from?",
    "What is the state sponsorship attribution for {name}?",
    "Which nation-state is associated with {name}?",
]

TARGETING_TEMPLATES = [
    "What sectors does {name} target?",
    "What are the known victims of {name}?",
    "What industries and countries has {name} targeted?",
]

ALIAS_TEMPLATES = [
    "Is {alias1} the same group as {alias2}?",
    "What are the known aliases for {name}?",
    "How is {name} tracked across different threat intelligence vendors?",
]


def _get_meta(actor: dict, key: str, default=None):
    """Safely get a field from the actor's meta dict."""
    return actor.get("meta", {}).get(key, default)


def generate_training_examples(
    galaxy: dict,
    mitre_lookup: dict[str, str] | None = None,
) -> list[dict]:
    """Generate training Q&A pairs from MISP Galaxy threat actors."""
    if mitre_lookup is None:
        mitre_lookup = {}

    examples = []
    actors = galaxy.get("values", [])

    for actor in actors:
        name = actor.get("value", "")
        if not name:
            continue

        description = actor.get("description", "")
        synonyms = _get_meta(actor, "synonyms", [])
        country = _get_meta(actor, "country")
        sponsor = _get_meta(actor, "cfr-suspected-state-sponsor")
        victims = _get_meta(actor, "cfr-suspected-victims", [])
        target_cats = _get_meta(actor, "cfr-target-category", [])
        motive = _get_meta(actor, "motive")
        refs = _get_meta(actor, "refs", [])

        key = name  # for template selection

        # Check for MITRE cross-reference
        mitre_id = mitre_lookup.get(name.lower())
        if not mitre_id:
            for syn in synonyms:
                mitre_id = mitre_lookup.get(syn.lower())
                if mitre_id:
                    break

        # ── 1. Actor Profile ──
        if len(description) >= 50:
            q = _select_template(PROFILE_TEMPLATES, key).format(name=name)
            parts = [name]
            if synonyms:
                parts.append(f"Also known as: {', '.join(synonyms[:10])}")
            if mitre_id:
                parts.append(f"Tracked in MITRE ATT&CK as {mitre_id}.")

            parts.append(f"\n{description[:2000]}")

            if country or sponsor:
                attr_parts = []
                if country:
                    attr_parts.append(f"Country: {country}")
                if sponsor:
                    attr_parts.append(f"Suspected state sponsor: {sponsor}")
                parts.append(f"\nAttribution: {' | '.join(attr_parts)}")

            if motive:
                parts.append(f"Motivation: {motive}")
            if target_cats:
                parts.append(f"Target sectors: {', '.join(target_cats)}")
            if victims:
                parts.append(f"Known victims: {', '.join(victims[:10])}")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 2. Attribution / Country ──
        if country or sponsor:
            q = _select_template(ATTRIBUTION_TEMPLATES, f"{key}_attr").format(
                name=name
            )
            parts = []
            if sponsor:
                parts.append(f"{name} is believed to be sponsored by {sponsor}.")
            if country:
                parts.append(f"The group is attributed to {country}.")
            if synonyms:
                parts.append(f"\n{name} is also known as: {', '.join(synonyms[:8])}")
            if mitre_id:
                parts.append(f"MITRE ATT&CK tracking ID: {mitre_id}")

            # Add evidence context from description
            if description:
                # Include first two sentences for attribution context
                sentences = description.split(". ")
                context = ". ".join(sentences[:2]).rstrip(".")
                if len(context) >= 30:
                    parts.append(f"\n{context}.")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 3. Targeting / Victimology ──
        if victims or target_cats:
            q = _select_template(TARGETING_TEMPLATES, f"{key}_tgt").format(
                name=name
            )
            parts = [f"{name} targeting profile:"]
            if target_cats:
                parts.append(f"\nTarget sectors: {', '.join(target_cats)}")
            if victims:
                parts.append(f"Known victim countries: {', '.join(victims)}")
            if motive:
                parts.append(f"Motivation: {motive}")
            if country or sponsor:
                attr = sponsor or country
                parts.append(f"Attribution: {attr}")
            if mitre_id:
                parts.append(f"MITRE ATT&CK ID: {mitre_id}")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

        # ── 4. Alias Lookup (3+ synonyms) ──
        if len(synonyms) >= 3:
            # Pick two aliases deterministically for the question
            sorted_syns = sorted(synonyms)
            alias1 = sorted_syns[0]
            alias2 = name

            q = _select_template(ALIAS_TEMPLATES, f"{key}_alias").format(
                name=name, alias1=alias1, alias2=alias2
            )
            parts = [
                f"Yes, {name} is tracked under multiple names across "
                f"different threat intelligence providers:",
                f"\nKnown aliases: {', '.join(synonyms[:15])}",
            ]
            if mitre_id:
                parts.append(f"MITRE ATT&CK ID: {mitre_id}")
            if country or sponsor:
                attr = sponsor or country
                parts.append(f"Attribution: {attr}")
            if description:
                sentences = description.split(". ")
                context = ". ".join(sentences[:2]).rstrip(".")
                if len(context) >= 30:
                    parts.append(f"\n{context}.")

            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": "\n".join(parts)},
            ]})

    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate APT training data from MISP Galaxy threat actors"
    )
    parser.add_argument(
        "--output", default="data/sources/misp_galaxy.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--url", default=GALAXY_URL, help="Galaxy JSON URL")
    parser.add_argument(
        "--mitre-stix-url",
        default=(
            "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
            "master/enterprise-attack/enterprise-attack.json"
        ),
        help="MITRE ATT&CK STIX bundle URL for cross-referencing",
    )
    parser.add_argument(
        "--skip-mitre-xref", action="store_true",
        help="Skip MITRE ATT&CK cross-referencing (faster, no second download)",
    )
    args = parser.parse_args()

    galaxy = fetch_galaxy(args.url)

    # Build MITRE cross-reference lookup
    mitre_lookup = {}
    if not args.skip_mitre_xref:
        try:
            log.info("Building MITRE ATT&CK cross-reference lookup...")
            from scripts.sources.fetch_mitre_groups import fetch_stix_bundle, extract_all
            bundle = fetch_stix_bundle(args.mitre_stix_url)
            data = extract_all(bundle)
            mitre_lookup = build_mitre_lookup(data["groups"])
            log.info(f"Built lookup with {len(mitre_lookup)} name→ID mappings.")
        except Exception as e:
            log.warning(f"Could not build MITRE cross-reference: {e}")
            log.warning("Continuing without cross-references.")

    examples = generate_training_examples(galaxy, mitre_lookup)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"Generated {len(examples)} training examples")
    log.info(f"Saved to {out_path}")
    log.info(f"\nNext step: python scripts/ingest_data.py "
             f"--jsonl {out_path} --source misp_galaxy")


if __name__ == "__main__":
    main()
