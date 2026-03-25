#!/usr/bin/env python3
"""
============================================================================
STIX Graph Training Pair Generator
============================================================================
Traverses the STIX relationship graph to generate ~15K instruction-tuning
pairs that teach the model to reason about cross-domain relationships.

14 categories:
   1. Actor → Technique Mapping        (~6K pairs)
   2. Actor → Malware/Tool Arsenal     (~2K pairs, incl. software profiles)
   3. Technique/SW → Actor Attribution (~1K pairs, incl. campaign attribution)
   4. Vulnerability → Kill Chain       (~150 pairs)
   5. Full Kill Chain Narratives       (~200 pairs, incl. campaign breakdowns)
   6. Detection → Attribution          (~600 pairs)
   7. Mitigation Mapping               (~750 pairs, incl. actor-level defenses)
   8. Actor → Target Sectors/Victims   (~200 pairs, MISP metadata)
   9. Sigma Detection → Attribution    (~400 pairs)
  10. Attack Log → Actor Inference     (~200 pairs)
  11. Data Source → Detection Guidance  (~700 pairs)
  12. Technique Overview Profiles      (~700 pairs)
  13. Malware/Tool Overview Profiles   (~800 pairs)
  14. Actor Overview Profiles          (~900 pairs)

Usage:
    python -m scripts.generate_stix_training_pairs
    python -m scripts.generate_stix_training_pairs --dry-run
    python -m scripts.generate_stix_training_pairs --categories 1,2,3
    python -m scripts.generate_stix_training_pairs --max-per-category 500
============================================================================
"""

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.db_utils import (
    DEFAULT_DB_PATH,
    content_hash,
    extract_all_metadata,
    get_connection,
    migrate_schema,
)
from scripts.stix_graph import STIXGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/sources/stix_graph_pairs.jsonl")
MAX_ASSISTANT_LEN = 2000
BATCH_SIZE = 1000


# ============================================================================
# Template Selection (reuses pattern from fetch_mitre_groups.py)
# ============================================================================

def _select_template(templates: list[str], key: str) -> str:
    """Deterministically select a template variant based on key hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


def _make_example(user_msg: str, assistant_msg: str) -> dict:
    """Create a training example dict."""
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ============================================================================
# Category 1: Actor → Technique Mapping
# ============================================================================

CAT1_BROAD = [
    "What MITRE ATT&CK techniques does {name} use?",
    "What are the TTPs associated with {name}?",
    "List the MITRE ATT&CK techniques used by {name} ({attack_id}).",
    "Describe the attack techniques employed by {name}.",
    "What techniques has {name} been observed using in their operations?",
]

CAT1_TACTIC = [
    "Which {tactic} techniques has {name} been observed using?",
    "Describe {name}'s {tactic} methods.",
    "What {tactic} techniques does {name} ({attack_id}) employ?",
    "How does {name} perform {tactic}?",
    "What are {name}'s known {tactic} capabilities?",
]

CAT1_NARROW = [
    "How does {name} use {tech_name} ({tech_id})?",
    "Describe how {name} ({attack_id}) employs {tech_name}.",
    "What is known about {name}'s use of {tech_id} ({tech_name})?",
    "Explain {name}'s implementation of {tech_name} ({tech_id}).",
    "How has {name} been observed using {tech_name}?",
]

CAT1_SUBTECH = [
    "What subtechniques of {parent_name} ({parent_id}) does {name} use?",
    "Break down {name}'s use of {parent_name} into specific subtechniques.",
    "Which specific variants of {parent_name} ({parent_id}) has {name} been observed using?",
    "How does {name} implement {parent_name}? What subtechniques are involved?",
    "Detail {name}'s use of {parent_id} ({parent_name}) subtechniques.",
]


def generate_cat1(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 1: Actor → Technique Mapping — broad, per-tactic, and narrow."""
    examples = []
    actors = graph.get_actors()

    for actor in actors:
        if len(examples) >= max_pairs:
            break

        techs = graph.actor_techniques(actor["stix_id"])
        if not techs:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]
        aliases = actor.get("aliases", [])

        # ── Broad: all techniques ──
        if len(techs) >= 2:
            q = _select_template(CAT1_BROAD, name).format(name=name, attack_id=attack_id)

            by_tactic = defaultdict(list)
            for t in techs:
                for phase in t.get("kill_chain_phases", ["other"]):
                    by_tactic[phase].append(t)

            parts = [f"{name} ({attack_id})"]
            if aliases:
                parts.append(f"Also known as: {', '.join(aliases[:5])}")
            parts.append(f"\n{name} has been observed using {len(techs)} ATT&CK techniques:\n")

            for tactic in sorted(by_tactic.keys()):
                tactic_label = tactic.replace("-", " ").title()
                tactic_techs = by_tactic[tactic]
                parts.append(f"**{tactic_label}:**")
                for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:6]:
                    tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                    parts.append(f"  - {tid} {t['name']}")

            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

        # ── Per-tactic questions (2+ techniques in tactic) ──
        by_tactic = defaultdict(list)
        for t in techs:
            for phase in t.get("kill_chain_phases", []):
                by_tactic[phase].append(t)

        for tactic, tactic_techs in by_tactic.items():
            if len(tactic_techs) < 2 or len(examples) >= max_pairs:
                continue

            tactic_label = tactic.replace("-", " ").title()
            q = _select_template(CAT1_TACTIC, f"{name}_{tactic}").format(
                name=name, attack_id=attack_id, tactic=tactic_label.lower()
            )
            parts = [f"{name} ({attack_id}) uses the following {tactic_label.lower()} techniques:\n"]
            for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", "")):
                tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                desc = t.get("rel_description", "")
                line = f"- {tid} {t['name']}"
                if desc:
                    first = desc.split(". ")[0].rstrip(".")
                    if len(first) < 200:
                        line += f": {first}."
                parts.append(line)
            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

        # ── Narrow: individual technique usage (where context exists) ──
        for t in techs:
            if len(examples) >= max_pairs:
                break
            rel_desc = t.get("rel_description", "")
            tech_desc = t.get("description", "")
            desc = rel_desc if rel_desc and len(rel_desc) >= 30 else tech_desc
            if not desc or len(desc) < 30:
                continue

            tid = t.get("external_ids", {}).get("mitre_attack_id", "")
            q = _select_template(CAT1_NARROW, f"{name}_{tid}").format(
                name=name, attack_id=attack_id,
                tech_name=t["name"], tech_id=tid
            )
            parts = [
                f"{name} ({attack_id}) has been observed using {t['name']} ({tid}).",
                f"\n{desc[:1500]}",
            ]
            phases = t.get("kill_chain_phases", [])
            if phases:
                labels = [p.replace("-", " ").title() for p in phases]
                parts.append(f"\nThis technique is used in the {', '.join(labels)} phase(s).")
            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

    # ── Subtechnique grouping ──
    # Group actor's subtechniques under parent techniques
    for actor in actors:
        if len(examples) >= max_pairs:
            break

        techs = graph.actor_techniques(actor["stix_id"])
        if not techs:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]

        # Find subtechniques and group by parent
        parent_subs = defaultdict(list)  # parent_stix_id → [subtechniques]
        for t in techs:
            # Check if this technique is a subtechnique
            parents = graph.get_connected_objects(
                t["stix_id"], rel_type="subtechnique-of", target_type="attack-pattern"
            )
            for parent in parents:
                parent_subs[parent["stix_id"]].append(t)

        for parent_id, subs in parent_subs.items():
            if len(subs) < 2 or len(examples) >= max_pairs:
                continue

            parent = graph.nodes.get(parent_id)
            if not parent:
                continue

            parent_ext_id = parent.get("external_ids", {}).get("mitre_attack_id", "")
            q = _select_template(CAT1_SUBTECH, f"{name}_{parent_ext_id}").format(
                name=name, attack_id=attack_id,
                parent_name=parent["name"], parent_id=parent_ext_id
            )

            parts = [f"{name} ({attack_id}) uses the following subtechniques of {parent['name']} ({parent_ext_id}):\n"]
            for sub in sorted(subs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", "")):
                sub_id = sub.get("external_ids", {}).get("mitre_attack_id", "")
                desc = sub.get("rel_description", "")
                line = f"- **{sub_id} {sub['name']}**"
                if desc:
                    first = desc.split(". ")[0].rstrip(".")
                    if len(first) < 200:
                        line += f": {first}."
                parts.append(line)

            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

    log.info(f"  Category 1 (Actor→Technique): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 2: Actor → Malware/Tool Arsenal
# ============================================================================

CAT2_BROAD = [
    "What malware and tools does {name} use?",
    "Describe the software arsenal of {name} ({attack_id}).",
    "What tools are associated with {name}?",
    "List the malware families linked to {name}.",
    "What software does {name} deploy in their operations?",
]

CAT2_NARROW = [
    "How does {name} use {sw_name}?",
    "What is {name}'s use of {sw_name} ({sw_id})?",
    "Describe {name}'s deployment of {sw_name}.",
    "How has {name} ({attack_id}) been observed using {sw_name}?",
    "What role does {sw_name} play in {name}'s operations?",
]

CAT2_SW_PROFILE = [
    "What ATT&CK techniques does {sw_name} implement?",
    "Describe the capabilities of {sw_name} ({sw_id}).",
    "What MITRE techniques are associated with {sw_name}?",
    "Provide a technical profile of {sw_name} ({sw_id}) including its ATT&CK mappings.",
    "What attack techniques can be performed using {sw_name}?",
]


def generate_cat2(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 2: Actor → Malware/Tool Arsenal — broad and narrow."""
    examples = []
    actors = graph.get_actors()

    for actor in actors:
        if len(examples) >= max_pairs:
            break

        software = graph.actor_software(actor["stix_id"])
        if not software:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]

        # ── Broad: all software ──
        if len(software) >= 2:
            q = _select_template(CAT2_BROAD, name).format(name=name, attack_id=attack_id)

            malware = [s for s in software if s["type"] == "malware"]
            tools = [s for s in software if s["type"] == "tool"]

            parts = [f"{name} ({attack_id}) has been associated with the following software:\n"]

            if malware:
                parts.append("**Malware:**")
                for m in sorted(malware, key=lambda x: x["name"]):
                    mid = m.get("external_ids", {}).get("mitre_attack_id", "")
                    desc = m.get("description", "")
                    line = f"- {m['name']} ({mid})"
                    if desc:
                        first = desc.split(". ")[0].rstrip(".")
                        if len(first) < 200:
                            line += f": {first}."
                    parts.append(line)

            if tools:
                parts.append("\n**Tools:**")
                for t in sorted(tools, key=lambda x: x["name"]):
                    tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                    desc = t.get("description", "")
                    line = f"- {t['name']} ({tid})"
                    if desc:
                        first = desc.split(". ")[0].rstrip(".")
                        if len(first) < 200:
                            line += f": {first}."
                    parts.append(line)

            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

        # ── Narrow: per-software usage (where context exists) ──
        for sw in software:
            if len(examples) >= max_pairs:
                break
            desc = sw.get("rel_description", "")
            sw_desc = sw.get("description", "")
            if not desc and not sw_desc:
                continue

            sw_id = sw.get("external_ids", {}).get("mitre_attack_id", "")
            q = _select_template(CAT2_NARROW, f"{name}_{sw['name']}").format(
                name=name, attack_id=attack_id,
                sw_name=sw["name"], sw_id=sw_id
            )
            parts = [
                f"{sw['name']} ({sw_id}) is a {sw['type']} used by {name} ({attack_id}).",
            ]
            if desc:
                parts.append(f"\n{desc[:800]}")
            elif sw_desc:
                parts.append(f"\n{sw_desc[:800]}")

            # What techniques does this software implement?
            sw_techs = graph.get_connected_objects(
                sw["stix_id"], rel_type="uses", target_type="attack-pattern"
            )
            if sw_techs:
                tech_strs = [
                    f"{t.get('external_ids', {}).get('mitre_attack_id', '')} {t['name']}"
                    for t in sorted(sw_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:8]
                ]
                parts.append(f"\nTechniques implemented: {', '.join(tech_strs)}")

            answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
            examples.append(_make_example(q, answer))

    # ── Software profiles (malware/tool → techniques) ──
    seen_sw = set()
    for node in graph.nodes.values():
        if len(examples) >= max_pairs:
            break
        if node["type"] not in ("malware", "tool"):
            continue
        if node["stix_id"] in seen_sw:
            continue
        seen_sw.add(node["stix_id"])

        sw_techs = graph.get_connected_objects(
            node["stix_id"], rel_type="uses", target_type="attack-pattern"
        )
        if len(sw_techs) < 2:
            continue

        sw_id = node["external_ids"].get("mitre_attack_id", "")
        q = _select_template(CAT2_SW_PROFILE, node["name"]).format(
            sw_name=node["name"], sw_id=sw_id
        )

        sw_type = node["type"]
        parts = [f"**{node['name']}** ({sw_id}) is a {sw_type} with the following ATT&CK technique mappings:\n"]

        if node.get("description"):
            first_sentence = node["description"].split(". ")[0].rstrip(".")
            if len(first_sentence) < 300:
                parts.append(f"{first_sentence}.\n")

        by_tactic = defaultdict(list)
        for t in sw_techs:
            for phase in t.get("kill_chain_phases", ["other"]):
                by_tactic[phase].append(t)

        for tactic in sorted(by_tactic.keys()):
            tactic_label = tactic.replace("-", " ").title()
            tactic_techs = by_tactic[tactic]
            parts.append(f"**{tactic_label}:**")
            for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:6]:
                tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                parts.append(f"  - {tid} {t['name']}")

        # Which actors use this software?
        actors = graph.get_incoming_objects(
            node["stix_id"], rel_type="uses", source_type="intrusion-set"
        )
        if actors:
            actor_names = sorted(set(a["name"] for a in actors))[:5]
            parts.append(f"\nUsed by: {', '.join(actor_names)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 2 (Actor→Arsenal): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 3: Technique → Actor Attribution
# ============================================================================

CAT3_TEMPLATES = [
    "Which threat actors have been observed using {tech_id} ({tech_name})?",
    "What threat groups use {tech_name} ({tech_id})?",
    "If an attacker is using {tech_name}, which APT groups should I investigate?",
    "What groups are known to employ {tech_id} ({tech_name})?",
    "Which nation-state actors use {tech_name}?",
]

CAT3_SW_TEMPLATES = [
    "Which threat actors are associated with {sw_name}?",
    "What groups use {sw_name} ({sw_id})?",
    "We detected {sw_name} in our environment. What group is likely responsible?",
    "What threat actor is known to deploy {sw_name}?",
    "Which APT groups are linked to {sw_name} ({sw_id})?",
]

CAT3_CAMPAIGN_TEMPLATES = [
    "What is the {campaign_name} campaign and who is behind it?",
    "Who conducted the {campaign_name} campaign?",
    "Provide details about the {campaign_name} campaign, including attribution.",
    "What threat actor is responsible for the {campaign_name} campaign?",
    "Describe the {campaign_name} campaign and its attributed threat group.",
]


def generate_cat3(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 3: Technique/Software → Actor Attribution."""
    examples = []

    # ── Technique → Actor ──
    techniques = graph.get_techniques()
    for tech in techniques:
        if len(examples) >= max_pairs:
            break

        actors = graph.technique_actors(tech["stix_id"])
        if not actors:
            continue

        tech_id = tech["external_ids"].get("mitre_attack_id", "")
        tech_name = tech["name"]
        tactics = tech.get("kill_chain_phases", [])

        q = _select_template(CAT3_TEMPLATES, tech_id).format(
            tech_id=tech_id, tech_name=tech_name
        )

        parts = [f"The following threat actors have been observed using {tech_name} ({tech_id}):\n"]

        for a in sorted(actors, key=lambda x: x["name"]):
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            country = a.get("external_ids", {}).get("country", "")
            desc = a.get("rel_description", "")
            line = f"- {a['name']} ({aid})"
            if country:
                line += f" [{country}]"
            if desc:
                first = desc.split(". ")[0].rstrip(".")
                if len(first) < 200:
                    line += f": {first}."
            parts.append(line)

        if tactics:
            tactic_labels = [t.replace("-", " ").title() for t in tactics]
            parts.append(f"\nThis technique is in the {', '.join(tactic_labels)} phase(s).")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    # ── Software → Actor (reverse attribution) ──
    for node in graph.nodes.values():
        if len(examples) >= max_pairs:
            break
        if node["type"] not in ("malware", "tool"):
            continue

        # Find actors that use this software
        actors = graph.get_incoming_objects(
            node["stix_id"], rel_type="uses", source_type="intrusion-set"
        )
        if not actors:
            continue

        sw_id = node["external_ids"].get("mitre_attack_id", "")
        q = _select_template(CAT3_SW_TEMPLATES, node["name"]).format(
            sw_name=node["name"], sw_id=sw_id
        )

        parts = [f"{node['name']} ({sw_id}) is a {node['type']} associated with:\n"]
        for a in sorted(actors, key=lambda x: x["name"]):
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            desc = a.get("rel_description", "")
            line = f"- {a['name']} ({aid})"
            if desc:
                first = desc.split(". ")[0].rstrip(".")
                if len(first) < 150:
                    line += f": {first}."
            parts.append(line)

        if node.get("description"):
            parts.append(f"\n{node['description'][:500]}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    # ── Campaign → Actor Attribution ──
    campaigns = [n for n in graph.nodes.values() if n["type"] == "campaign"]
    for camp in campaigns:
        if len(examples) >= max_pairs:
            break

        # campaign --attributed-to--> intrusion-set
        actors = graph.get_connected_objects(
            camp["stix_id"], rel_type="attributed-to", target_type="intrusion-set"
        )
        if not actors:
            continue

        camp_desc = camp.get("description", "")
        q = _select_template(CAT3_CAMPAIGN_TEMPLATES, camp["name"]).format(
            campaign_name=camp["name"]
        )

        parts = [f"The **{camp['name']}** campaign is attributed to:\n"]
        for a in actors:
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            country = a.get("external_ids", {}).get("country", "")
            line = f"- **{a['name']}** ({aid})"
            if country:
                line += f" [{country}]"
            parts.append(line)

        if camp_desc:
            parts.append(f"\n{camp_desc[:800]}")

        # What techniques were used in this campaign?
        camp_techs = graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="attack-pattern"
        )
        if camp_techs:
            tech_strs = [
                f"{t.get('external_ids', {}).get('mitre_attack_id', '')} {t['name']}"
                for t in sorted(camp_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:10]
            ]
            parts.append(f"\nTechniques used in this campaign: {', '.join(tech_strs)}")

        # What software was used?
        camp_sw = graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="malware"
        ) + graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="tool"
        )
        if camp_sw:
            sw_names = sorted(set(s["name"] for s in camp_sw))[:8]
            parts.append(f"Software used: {', '.join(sw_names)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 3 (Technique/SW/Campaign→Attribution): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 4: Vulnerability → Kill Chain
# ============================================================================

CAT4_TEMPLATES = [
    "Analyze {cve_id} from a threat intelligence perspective. Which actors exploit it and through what techniques?",
    "What threat actors and attack techniques are associated with {cve_id}?",
    "Provide a kill chain analysis for {cve_id}.",
    "How is {cve_id} exploited in real-world attacks? Which groups use it?",
    "What is the threat landscape around {cve_id}?",
]


def generate_cat4(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 4: Vulnerability → Kill Chain (multi-hop).

    Strategy: Scan technique descriptions for CVE mentions to build
    implicit vulnerability→technique→actor kill chains.
    """
    examples = []
    import re as _re

    conn = get_connection(graph.db_path)
    cur = conn.cursor()

    # Build CVE→vulnerability stix_id lookup
    cur.execute(
        "SELECT stix_id, name, description, severity, cvss_score "
        "FROM stix_objects WHERE type = 'vulnerability'"
    )
    cve_lookup = {}
    for stix_id, name, desc, sev, cvss in cur.fetchall():
        cve_lookup[name] = {
            "stix_id": stix_id, "name": name,
            "description": desc, "severity": sev, "cvss_score": cvss,
        }

    # Scan technique descriptions + relationship descriptions for CVE mentions
    cve_tech_map = defaultdict(list)  # cve_id → [(tech_stix_id, tech_name, tech_ext_id)]

    for tech in graph.get_techniques():
        tech_id = tech["external_ids"].get("mitre_attack_id", "")

        # Check technique description
        desc = tech.get("description", "")
        cves_in_desc = _re.findall(r"CVE-\d{4}-\d+", desc)

        # Check relationship descriptions (how actors use this technique)
        for rt, src_id, rel_desc in graph.incoming.get(tech["stix_id"], []):
            if rel_desc:
                cves_in_desc.extend(_re.findall(r"CVE-\d{4}-\d+", rel_desc))

        for cve_id in set(cves_in_desc):
            if cve_id in cve_lookup:
                cve_tech_map[cve_id].append((tech["stix_id"], tech["name"], tech_id))

    log.info(f"  Cat4: found {len(cve_tech_map)} CVEs with technique cross-refs")

    for cve_id, tech_list in sorted(cve_tech_map.items()):
        if len(examples) >= max_pairs:
            break

        vuln = cve_lookup[cve_id]

        # Find actors via techniques
        actors_for_cve = []
        seen_actors = set()
        for tech_sid, tech_name, tech_ext_id in tech_list:
            for a in graph.technique_actors(tech_sid):
                if a["stix_id"] not in seen_actors:
                    seen_actors.add(a["stix_id"])
                    actors_for_cve.append(a)

        q = _select_template(CAT4_TEMPLATES, cve_id).format(cve_id=cve_id)

        parts = [f"**{cve_id}**"]
        if vuln["severity"]:
            sev_str = f"Severity: {vuln['severity']}"
            if vuln["cvss_score"]:
                sev_str += f" (CVSS: {vuln['cvss_score']})"
            parts.append(sev_str)
        if vuln["description"]:
            parts.append(f"\n{vuln['description'][:300]}")

        parts.append("\n**Associated Techniques:**")
        for tech_sid, tech_name, tech_ext_id in tech_list[:8]:
            parts.append(f"- {tech_ext_id} {tech_name}")

        if actors_for_cve:
            parts.append("\n**Known Threat Actors:**")
            for a in actors_for_cve[:8]:
                aid = a.get("external_ids", {}).get("mitre_attack_id", "")
                parts.append(f"- {a['name']} ({aid})")

        # Mitigations
        mitigations = set()
        for tech_sid, _, _ in tech_list[:5]:
            for m in graph.technique_mitigations(tech_sid):
                mitigations.add((m["name"], m.get("external_ids", {}).get("mitre_attack_id", "")))
        if mitigations:
            parts.append("\n**Recommended Mitigations:**")
            for mname, mid in sorted(mitigations)[:5]:
                parts.append(f"- {mname} ({mid})")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    conn.close()
    log.info(f"  Category 4 (Vulnerability→Kill Chain): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 5: Full Kill Chain Narratives
# ============================================================================

CAT5_TEMPLATES = [
    "Describe the full attack lifecycle of {name}, including initial access, execution, persistence, and impact.",
    "Walk me through a typical {name} attack from initial access to data exfiltration.",
    "What does a {name} intrusion look like end-to-end?",
    "Provide a full kill chain analysis for {name} ({attack_id}).",
    "How does {name} operate across the MITRE ATT&CK kill chain?",
]

TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact",
]


def generate_cat5(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 5: Full Kill Chain Narratives."""
    examples = []
    actors = graph.get_actors()

    for actor in actors:
        if len(examples) >= max_pairs:
            break

        techs = graph.actor_techniques(actor["stix_id"])
        software = graph.actor_software(actor["stix_id"])
        campaigns = graph.actor_campaigns(actor["stix_id"])

        by_tactic = defaultdict(list)
        for t in techs:
            for phase in t.get("kill_chain_phases", []):
                by_tactic[phase].append(t)

        if len(by_tactic) < 2:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]

        q = _select_template(CAT5_TEMPLATES, name).format(name=name, attack_id=attack_id)

        parts = [f"## {name} ({attack_id}) Attack Lifecycle\n"]

        for tactic in TACTIC_ORDER:
            if tactic not in by_tactic:
                continue
            tactic_label = tactic.replace("-", " ").title()
            tactic_techs = by_tactic[tactic]
            tech_strs = []
            for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:4]:
                tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                desc = t.get("rel_description", "")
                s = f"{tid} {t['name']}"
                if desc:
                    first = desc.split(". ")[0].rstrip(".")
                    if len(first) < 120:
                        s += f" — {first}"
                tech_strs.append(s)
            parts.append(f"**{tactic_label}:** {'; '.join(tech_strs)}")

        if software:
            sw_names = sorted(set(s["name"] for s in software))[:8]
            parts.append(f"\n**Tools & Malware:** {', '.join(sw_names)}")

        if campaigns:
            camp_strs = [c["name"] for c in campaigns[:5]]
            parts.append(f"**Known Campaigns:** {', '.join(camp_strs)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    # ── Campaign technique breakdowns ──
    CAT5_CAMPAIGN_TEMPLATES = [
        "What techniques were used in the {campaign_name} campaign?",
        "Describe the TTPs observed in the {campaign_name} campaign.",
        "Provide a technical breakdown of the {campaign_name} campaign.",
        "What attack techniques were employed during the {campaign_name} campaign?",
        "Analyze the tactics and techniques used in {campaign_name}.",
    ]

    campaigns = [n for n in graph.nodes.values() if n["type"] == "campaign"]
    for camp in campaigns:
        if len(examples) >= max_pairs:
            break

        camp_techs = graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="attack-pattern"
        )
        if len(camp_techs) < 2:
            continue

        camp_sw = graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="malware"
        ) + graph.get_connected_objects(
            camp["stix_id"], rel_type="uses", target_type="tool"
        )

        actors = graph.get_connected_objects(
            camp["stix_id"], rel_type="attributed-to", target_type="intrusion-set"
        )

        q = _select_template(CAT5_CAMPAIGN_TEMPLATES, camp["name"]).format(
            campaign_name=camp["name"]
        )

        parts = [f"## {camp['name']} Campaign\n"]

        if actors:
            actor_names = [a["name"] for a in actors]
            parts.append(f"**Attributed to:** {', '.join(actor_names)}\n")

        if camp.get("description"):
            first_para = camp["description"].split("\n")[0][:400]
            parts.append(f"{first_para}\n")

        # Organize by tactic
        by_tactic = defaultdict(list)
        for t in camp_techs:
            for phase in t.get("kill_chain_phases", ["other"]):
                by_tactic[phase].append(t)

        for tactic in TACTIC_ORDER:
            if tactic not in by_tactic:
                continue
            tactic_label = tactic.replace("-", " ").title()
            tactic_techs = by_tactic[tactic]
            parts.append(f"**{tactic_label}:**")
            for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:6]:
                tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                parts.append(f"  - {tid} {t['name']}")

        if camp_sw:
            sw_names = sorted(set(s["name"] for s in camp_sw))[:8]
            parts.append(f"\n**Software Used:** {', '.join(sw_names)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 5 (Kill Chain Narratives): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 6: Detection → Attribution
# ============================================================================

CAT6_TEMPLATES = [
    "A detection rule triggered for {tech_id} ({tech_name}). What threat actors should I investigate?",
    "Our SIEM alerted on {tech_name} ({tech_id}). Which APT groups use this technique?",
    "We detected activity matching {tech_id}. Who could be behind this?",
    "A Sigma rule flagged {tech_name}. What groups commonly use this technique?",
    "Our monitoring detected {tech_id} ({tech_name}). Provide threat attribution guidance.",
]

CAT6_MULTI = [
    "We detected {tech_id1} ({tech_name1}) and {tech_id2} ({tech_name2}) in our environment. What threat actor may be responsible?",
    "Our SIEM flagged both {tech_id1} and {tech_id2}. Which groups use this combination?",
    "Correlated alerts for {tech_name1} ({tech_id1}) and {tech_name2} ({tech_id2}). What threat actors should we investigate?",
    "We observed {tech_id1} followed by {tech_id2}. Which APT groups use both techniques?",
    "Two detections fired: {tech_id1} ({tech_name1}) and {tech_id2} ({tech_name2}). What's the likely attribution?",
]


def generate_cat6(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 6: Detection → Attribution — single technique and multi-technique."""
    examples = []
    techniques = graph.get_techniques()

    # ── Single technique detection ──
    for tech in techniques:
        if len(examples) >= max_pairs:
            break

        actors = graph.technique_actors(tech["stix_id"])
        if not actors:
            continue

        tech_id = tech["external_ids"].get("mitre_attack_id", "")
        tech_name = tech["name"]
        tactics = tech.get("kill_chain_phases", [])

        q = _select_template(CAT6_TEMPLATES, f"det_{tech_id}").format(
            tech_id=tech_id, tech_name=tech_name
        )

        parts = [
            f"Detection of {tech_name} ({tech_id}) should prompt investigation "
            f"of the following threat actors:\n"
        ]

        for a in sorted(actors, key=lambda x: x["name"])[:10]:
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            country = a.get("external_ids", {}).get("country", "")
            line = f"- **{a['name']}** ({aid})"
            if country:
                line += f" [{country}]"
            parts.append(line)

        if tactics:
            tactic_labels = [t.replace("-", " ").title() for t in tactics]
            parts.append(f"\nThis technique falls under: {', '.join(tactic_labels)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    # ── Multi-technique correlation ──
    # Find pairs of techniques that share actors
    tech_list = [t for t in techniques if graph.technique_actors(t["stix_id"])]
    for i in range(0, len(tech_list) - 1, 2):
        if len(examples) >= max_pairs:
            break

        t1 = tech_list[i]
        t2 = tech_list[i + 1]

        actors1 = {a["stix_id"] for a in graph.technique_actors(t1["stix_id"])}
        actors2 = {a["stix_id"] for a in graph.technique_actors(t2["stix_id"])}
        shared = actors1 & actors2

        if not shared:
            continue

        t1_id = t1["external_ids"].get("mitre_attack_id", "")
        t2_id = t2["external_ids"].get("mitre_attack_id", "")

        q = _select_template(CAT6_MULTI, f"multi_{t1_id}_{t2_id}").format(
            tech_id1=t1_id, tech_name1=t1["name"],
            tech_id2=t2_id, tech_name2=t2["name"],
        )

        parts = [
            f"The combination of {t1['name']} ({t1_id}) and {t2['name']} ({t2_id}) "
            f"is associated with the following threat actors:\n"
        ]

        for sid in sorted(shared):
            node = graph.nodes.get(sid)
            if node:
                aid = node.get("external_ids", {}).get("mitre_attack_id", "")
                parts.append(f"- **{node['name']}** ({aid})")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 6 (Detection→Attribution): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 7: Mitigation Mapping
# ============================================================================

CAT7_TEMPLATES = [
    "What mitigations are effective against {tech_id} ({tech_name})?",
    "How can we defend against {tech_name} ({tech_id})?",
    "What MITRE ATT&CK mitigations address {tech_id}?",
    "Recommend mitigations for {tech_name} ({tech_id}).",
    "What security controls counter {tech_name}?",
]

CAT7_ACTOR_TEMPLATES = [
    "How can we defend against {name}?",
    "What mitigations should we deploy to protect against {name} ({attack_id})?",
    "What security controls are effective against {name}'s techniques?",
    "Recommend defenses against {name} ({attack_id}).",
    "What is the defensive strategy for protecting against {name}?",
]


def generate_cat7(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 7: Mitigation Mapping — per technique and per actor."""
    examples = []
    techniques = graph.get_techniques()

    # ── Per-technique mitigations ──
    for tech in techniques:
        if len(examples) >= max_pairs:
            break

        mitigations = graph.technique_mitigations(tech["stix_id"])
        if not mitigations:
            continue

        actors = graph.technique_actors(tech["stix_id"])
        tech_id = tech["external_ids"].get("mitre_attack_id", "")
        tech_name = tech["name"]

        q = _select_template(CAT7_TEMPLATES, tech_id).format(
            tech_id=tech_id, tech_name=tech_name
        )

        parts = [f"The following mitigations address {tech_name} ({tech_id}):\n"]

        for m in sorted(mitigations, key=lambda x: x["name"]):
            mid = m.get("external_ids", {}).get("mitre_attack_id", "")
            desc = m.get("rel_description", "") or m.get("description", "")
            line = f"- **{m['name']}** ({mid})"
            if desc:
                first = desc.split(". ")[0].rstrip(".")
                if len(first) < 200:
                    line += f": {first}."
            parts.append(line)

        if actors:
            actor_names = sorted(set(a["name"] for a in actors))[:8]
            parts.append(f"\nThis technique is used by: {', '.join(actor_names)}")
            if len(actors) >= 5:
                parts.append("Priority: HIGH — widely used across multiple threat groups.")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    # ── Actor-level mitigations (aggregate across techniques) ──
    actors = graph.get_actors()
    for actor in actors:
        if len(examples) >= max_pairs:
            break

        techs = graph.actor_techniques(actor["stix_id"])
        if len(techs) < 3:
            continue

        # Collect all mitigations across this actor's techniques
        mitigation_techs = defaultdict(set)  # mitigation_id → set of technique names
        mitigation_nodes = {}
        for t in techs:
            mits = graph.technique_mitigations(t["stix_id"])
            tid = t.get("external_ids", {}).get("mitre_attack_id", "")
            for m in mits:
                mitigation_techs[m["stix_id"]].add(f"{tid} {t['name']}")
                mitigation_nodes[m["stix_id"]] = m

        if len(mitigation_nodes) < 2:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]

        q = _select_template(CAT7_ACTOR_TEMPLATES, name).format(
            name=name, attack_id=attack_id
        )

        parts = [f"To defend against {name} ({attack_id}), prioritize these mitigations:\n"]

        # Sort by number of techniques covered (most impactful first)
        sorted_mits = sorted(
            mitigation_nodes.items(),
            key=lambda x: len(mitigation_techs[x[0]]),
            reverse=True,
        )
        for mid, m in sorted_mits[:10]:
            mit_id = m.get("external_ids", {}).get("mitre_attack_id", "")
            covered = mitigation_techs[mid]
            line = f"- **{m['name']}** ({mit_id}) — covers {len(covered)} technique(s)"
            parts.append(line)

        parts.append(f"\n{name} uses {len(techs)} known ATT&CK techniques across their operations.")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 7 (Mitigation Mapping): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 8: Actor → Target Sectors/Victims (MISP metadata)
# ============================================================================

CAT8_SECTOR_TEMPLATES = [
    "What sectors does {name} target?",
    "Which industries has {name} been observed targeting?",
    "What is {name}'s targeting profile?",
    "Describe the sectors and industries targeted by {name} ({attack_id}).",
    "What verticals does {name} focus on?",
]

CAT8_VICTIM_TEMPLATES = [
    "Which countries has {name} targeted?",
    "What are the known victim countries of {name}?",
    "What is {name}'s geographic targeting profile?",
    "Which regions does {name} ({attack_id}) operate against?",
    "What countries have been victimized by {name}?",
]

CAT8_SECTOR_REVERSE = [
    "Which APT groups target the {sector} sector?",
    "What threat actors are known to target {sector}?",
    "What nation-state groups attack {sector} organizations?",
    "Which groups should {sector} organizations be most concerned about?",
    "Identify threat actors that target {sector}.",
]

CAT8_COUNTRY_REVERSE = [
    "Which APT groups target {country}?",
    "What threat actors have been observed targeting {country}?",
    "What nation-state groups attack organizations in {country}?",
    "Which threat actors should organizations in {country} watch for?",
    "Identify APT groups that target {country}.",
]


def generate_cat8(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 8: Actor targeting — sectors, victims, and reverse lookups."""
    examples = []
    actors = graph.get_actors()

    # Collect sector→actors and country→actors mappings for reverse queries
    sector_actors = defaultdict(list)
    country_actors = defaultdict(list)

    for actor in actors:
        ext = actor.get("external_ids", {})
        name = actor["name"]
        attack_id = ext.get("mitre_attack_id", "")
        victims = ext.get("cfr-suspected-victims", [])
        target_cats = ext.get("cfr-target-category", [])
        country = ext.get("country", "")
        sponsor = ext.get("cfr-suspected-state-sponsor", "")
        motive = ext.get("motive", "")

        # ── Sector targeting ──
        if target_cats and len(examples) < max_pairs:
            q = _select_template(CAT8_SECTOR_TEMPLATES, f"{name}_sec").format(
                name=name, attack_id=attack_id
            )
            parts = [f"{name} ({attack_id}) targeting profile:\n"]
            parts.append(f"**Target sectors:** {', '.join(target_cats)}")
            if victims:
                parts.append(f"**Known victim countries:** {', '.join(victims[:10])}")
            if motive:
                parts.append(f"**Motivation:** {motive}")
            if sponsor:
                parts.append(f"**State sponsor:** {sponsor}")

            techs = graph.actor_techniques(actor["stix_id"])
            if techs:
                parts.append(f"\n{name} employs {len(techs)} ATT&CK techniques in their operations.")

            examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

            for cat in target_cats:
                sector_actors[cat].append(actor)

        # ── Victim country targeting ──
        if victims and len(examples) < max_pairs:
            q = _select_template(CAT8_VICTIM_TEMPLATES, f"{name}_vic").format(
                name=name, attack_id=attack_id
            )
            parts = [f"{name} ({attack_id}) geographic targeting:\n"]
            parts.append(f"**Known victim countries:** {', '.join(victims)}")
            if target_cats:
                parts.append(f"**Target sectors:** {', '.join(target_cats)}")
            if country or sponsor:
                attr = sponsor or country
                parts.append(f"**Attribution:** {attr}")
            if motive:
                parts.append(f"**Motivation:** {motive}")

            examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

            for v in victims:
                country_actors[v].append(actor)

    # ── Reverse: sector → actors ──
    for sector, sector_actor_list in sorted(sector_actors.items()):
        if len(examples) >= max_pairs or len(sector_actor_list) < 2:
            continue
        q = _select_template(CAT8_SECTOR_REVERSE, sector).format(sector=sector)
        parts = [f"The following threat actors are known to target the {sector} sector:\n"]
        for a in sorted(sector_actor_list, key=lambda x: x["name"]):
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            sp = a.get("external_ids", {}).get("cfr-suspected-state-sponsor", "")
            line = f"- **{a['name']}** ({aid})"
            if sp:
                line += f" [{sp}]"
            parts.append(line)
        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    # ── Reverse: country → actors ──
    for ctry, ctry_actor_list in sorted(country_actors.items()):
        if len(examples) >= max_pairs or len(ctry_actor_list) < 2:
            continue
        q = _select_template(CAT8_COUNTRY_REVERSE, ctry).format(country=ctry)
        parts = [f"The following threat actors have been observed targeting {ctry}:\n"]
        for a in sorted(ctry_actor_list, key=lambda x: x["name"]):
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            sp = a.get("external_ids", {}).get("cfr-suspected-state-sponsor", "")
            line = f"- **{a['name']}** ({aid})"
            if sp:
                line += f" [{sp}]"
            parts.append(line)
        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    log.info(f"  Category 8 (Targeting/Victims): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 9: Sigma Detection → Attribution
# ============================================================================

CAT9_TEMPLATES = [
    "A Sigma rule '{rule_name}' triggered detecting {tech_id} ({tech_name}). What threat actors should I investigate?",
    "Our SIEM fired Sigma rule '{rule_name}' for {tech_name} ({tech_id}). Who could be behind this?",
    "Sigma detection for {tech_name} ({tech_id}) via rule '{rule_name}'. Provide threat attribution.",
    "Rule '{rule_name}' detected {tech_id}. Which APT groups use this technique?",
    "We got a Sigma alert for {tech_name}. Provide an attribution analysis.",
]

CAT9_COVERAGE = [
    "What Sigma detection coverage exists for {name}'s techniques?",
    "Which of {name}'s techniques have Sigma rules?",
    "Assess the Sigma detection posture against {name} ({attack_id}).",
    "What detection gaps exist for {name}'s TTPs in our Sigma ruleset?",
    "How well do Sigma rules cover {name}'s attack techniques?",
]


def generate_cat9(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 9: Sigma detection rule → actor attribution and coverage."""
    examples = []

    # Find sigma-rule objects and their technique links
    sigma_rules = [n for n in graph.nodes.values() if n["type"] == "sigma-rule"]
    if not sigma_rules:
        log.info("  Category 9 (Sigma Detection): 0 pairs (no Sigma rules in graph)")
        return examples

    # ── Per-rule detection→attribution ──
    for rule in sigma_rules:
        if len(examples) >= max_pairs:
            break

        # Find technique this rule detects
        techs = graph.get_connected_objects(
            rule["stix_id"], rel_type="detects", target_type="attack-pattern"
        )
        if not techs:
            continue

        tech = techs[0]
        tech_id = tech.get("external_ids", {}).get("mitre_attack_id", "")
        tech_name = tech["name"]

        actors = graph.technique_actors(tech["stix_id"])
        if not actors:
            continue

        q = _select_template(CAT9_TEMPLATES, rule["name"]).format(
            rule_name=rule["name"], tech_id=tech_id, tech_name=tech_name
        )

        parts = [
            f"The Sigma rule '{rule['name']}' detects {tech_name} ({tech_id}).",
            f"\nThe following threat actors use this technique:\n",
        ]

        for a in sorted(actors, key=lambda x: x["name"])[:10]:
            aid = a.get("external_ids", {}).get("mitre_attack_id", "")
            country = a.get("external_ids", {}).get("country", "")
            line = f"- **{a['name']}** ({aid})"
            if country:
                line += f" [{country}]"
            parts.append(line)

        tactics = tech.get("kill_chain_phases", [])
        if tactics:
            labels = [t.replace("-", " ").title() for t in tactics]
            parts.append(f"\nATT&CK tactic: {', '.join(labels)}")

        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    # ── Per-actor detection coverage assessment ──
    # Build technique→has_sigma_rule lookup
    sigma_tech_ids = set()
    for rule in sigma_rules:
        ext = rule.get("external_ids", {})
        tid = ext.get("mitre_attack_id")
        if tid:
            sigma_tech_ids.add(tid)

    actors = graph.get_actors()
    for actor in actors:
        if len(examples) >= max_pairs:
            break

        techs = graph.actor_techniques(actor["stix_id"])
        if len(techs) < 3:
            continue

        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        name = actor["name"]

        covered = []
        uncovered = []
        for t in techs:
            tid = t.get("external_ids", {}).get("mitre_attack_id", "")
            if tid in sigma_tech_ids:
                covered.append(f"{tid} {t['name']}")
            else:
                uncovered.append(f"{tid} {t['name']}")

        if not covered and not uncovered:
            continue

        q = _select_template(CAT9_COVERAGE, name).format(name=name, attack_id=attack_id)
        total = len(covered) + len(uncovered)
        pct = len(covered) * 100 // total if total else 0

        parts = [f"Sigma detection coverage for {name} ({attack_id}): {pct}% ({len(covered)}/{total} techniques)\n"]

        if covered:
            parts.append("**Covered by Sigma rules:**")
            for t in sorted(covered)[:10]:
                parts.append(f"- {t}")

        if uncovered:
            parts.append(f"\n**Detection gaps ({len(uncovered)} techniques without Sigma rules):**")
            for t in sorted(uncovered)[:10]:
                parts.append(f"- {t}")

        if pct < 50:
            parts.append(f"\n**Assessment:** LOW detection coverage. {name} can operate with minimal Sigma-based detection.")
        elif pct < 80:
            parts.append(f"\n**Assessment:** MODERATE detection coverage. Some techniques may evade Sigma-based monitoring.")
        else:
            parts.append(f"\n**Assessment:** HIGH detection coverage. Most of {name}'s techniques are detectable via Sigma rules.")

        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    log.info(f"  Category 9 (Sigma Detection): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 10: Attack Log → Actor Inference
# ============================================================================

CAT10_TEMPLATES = [
    "We observed the following security log events involving {tech_id} ({tech_name}). What threat actors could be responsible?",
    "Our SOC detected activity matching {tech_name} ({tech_id}) in security logs. Provide attribution guidance.",
    "Log analysis shows indicators of {tech_name} ({tech_id}). Which APT groups use this technique?",
    "Security events flagged {tech_id} ({tech_name}) in our environment. Who should we investigate?",
    "Alert triage: {tech_name} ({tech_id}) detected in log events. Provide threat actor context.",
]

CAT10_MULTI = [
    "Our log analysis found multiple techniques: {tech_list}. What threat actors match this pattern?",
    "We observed {tech_list} across correlated log events. Which APT groups use this combination?",
    "Multiple ATT&CK techniques detected in security logs: {tech_list}. Provide attribution analysis.",
    "Log correlation identified {tech_list}. What threat actors match these TTPs?",
    "Triage result shows {tech_list} across security events. Who is the likely adversary?",
]


def generate_cat10(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 10: Attack log → actor inference from technique IDs."""
    examples = []

    conn = get_connection(graph.db_path)
    cur = conn.cursor()

    # Get log_analysis records with technique IDs
    cur.execute(
        "SELECT id, mitre_techniques, user_message FROM all_records "
        "WHERE domain = 'log_analysis' AND mitre_techniques IS NOT NULL "
        "AND mitre_techniques != ''"
    )
    log_records = cur.fetchall()

    if not log_records:
        log.info("  Category 10 (Log→Actor): 0 pairs (no log records with techniques)")
        conn.close()
        return examples

    # Build technique ext_id → stix_id lookup
    tech_lookup = {}
    for tech in graph.get_techniques():
        tid = tech["external_ids"].get("mitre_attack_id", "")
        if tid:
            tech_lookup[tid] = tech

    # ── Single technique log→actor pairs ──
    seen_techs = set()
    for record_id, tech_str, user_msg in log_records:
        if len(examples) >= max_pairs:
            break

        tech_ids = [t.strip() for t in tech_str.split(",") if t.strip()]

        for tid in tech_ids:
            if tid in seen_techs or tid not in tech_lookup:
                continue
            seen_techs.add(tid)

            tech = tech_lookup[tid]
            actors = graph.technique_actors(tech["stix_id"])
            if not actors:
                continue

            q = _select_template(CAT10_TEMPLATES, f"log_{tid}").format(
                tech_id=tid, tech_name=tech["name"]
            )

            parts = [
                f"Log events indicating {tech['name']} ({tid}) suggest the following threat actors:\n"
            ]

            for a in sorted(actors, key=lambda x: x["name"])[:10]:
                aid = a.get("external_ids", {}).get("mitre_attack_id", "")
                country = a.get("external_ids", {}).get("country", "")
                line = f"- **{a['name']}** ({aid})"
                if country:
                    line += f" [{country}]"
                parts.append(line)

            tactics = tech.get("kill_chain_phases", [])
            if tactics:
                labels = [t.replace("-", " ").title() for t in tactics]
                parts.append(f"\nThis technique is in the {', '.join(labels)} phase(s).")

            parts.append("\n**Recommended next steps:**")
            parts.append("- Correlate with other technique detections for more precise attribution")
            parts.append("- Check for related IOCs associated with these groups")

            examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    # ── Multi-technique log correlation ──
    for record_id, tech_str, user_msg in log_records:
        if len(examples) >= max_pairs:
            break

        tech_ids = [t.strip() for t in tech_str.split(",") if t.strip() and t.strip() in tech_lookup]
        if len(tech_ids) < 2:
            continue

        # Find actors that use ALL techniques (intersection)
        actor_sets = []
        for tid in tech_ids[:4]:  # Cap at 4 techniques
            tech = tech_lookup[tid]
            actor_ids = {a["stix_id"] for a in graph.technique_actors(tech["stix_id"])}
            actor_sets.append(actor_ids)

        shared = set.intersection(*actor_sets) if actor_sets else set()
        if not shared:
            continue

        tech_labels = [f"{tid} ({tech_lookup[tid]['name']})" for tid in tech_ids[:4]]
        tech_list = ", ".join(tech_labels)

        q = _select_template(CAT10_MULTI, f"multi_{'_'.join(tech_ids[:4])}").format(
            tech_list=tech_list
        )

        parts = [f"The combination of techniques {tech_list} is associated with:\n"]
        for sid in sorted(shared):
            node = graph.nodes.get(sid)
            if node:
                aid = node.get("external_ids", {}).get("mitre_attack_id", "")
                parts.append(f"- **{node['name']}** ({aid})")

        parts.append(f"\nThese {len(shared)} group(s) are known to use all {len(tech_ids[:4])} detected techniques.")

        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    conn.close()
    log.info(f"  Category 10 (Log→Actor): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 11: Data Source → Detection Guidance
# ============================================================================

CAT11_TEMPLATES = [
    "What data sources are needed to detect {tech_id} ({tech_name})?",
    "Which telemetry sources should I collect to identify {tech_name} ({tech_id})?",
    "What monitoring is required to detect {tech_id}?",
    "Describe the data collection requirements for detecting {tech_name}.",
    "What log sources and sensors detect {tech_name} ({tech_id})?",
]


def generate_cat11(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 11: Data source/component → technique detection guidance."""
    examples = []

    # Find detection-strategy objects (these have 'detects' relationships to techniques)
    det_strategies = [n for n in graph.nodes.values() if n["type"] == "x-mitre-detection-strategy"]
    # Also find data-component objects for supplemental descriptions
    data_components = [n for n in graph.nodes.values() if n["type"] == "x-mitre-data-component"]

    if not det_strategies and not data_components:
        log.info("  Category 11 (Data Source): 0 pairs (no detection strategy or data component objects)")
        return examples

    # Build technique → detection strategies via 'detects' relationships
    tech_datasources = defaultdict(list)
    # Detection strategies have explicit 'detects' relationships
    for ds in det_strategies:
        techs = graph.get_connected_objects(ds["stix_id"], rel_type="detects", target_type="attack-pattern")
        for tech in techs:
            tech_datasources[tech["stix_id"]].append(ds)
    # Also check incoming detects from data components (if any exist)
    for dc in data_components:
        techs = graph.get_connected_objects(dc["stix_id"], rel_type="detects", target_type="attack-pattern")
        for tech in techs:
            if dc not in tech_datasources[tech["stix_id"]]:
                tech_datasources[tech["stix_id"]].append(dc)

    techniques = graph.get_techniques()
    for tech in techniques:
        if len(examples) >= max_pairs:
            break

        ds_list = tech_datasources.get(tech["stix_id"], [])
        if not ds_list:
            continue

        tech_id = tech["external_ids"].get("mitre_attack_id", "")
        tech_name = tech["name"]
        actors = graph.technique_actors(tech["stix_id"])

        q = _select_template(CAT11_TEMPLATES, tech_id).format(
            tech_id=tech_id, tech_name=tech_name
        )

        parts = [f"To detect {tech_name} ({tech_id}), monitor the following data sources:\n"]

        for dc in sorted(ds_list, key=lambda x: x["name"]):
            desc = dc.get("description", "")
            line = f"- **{dc['name']}**"
            if desc:
                first = desc.split(". ")[0].rstrip(".")
                if len(first) < 200:
                    line += f": {first}."
            parts.append(line)

        if actors:
            actor_names = sorted(set(a["name"] for a in actors))[:5]
            parts.append(f"\nThis technique is used by: {', '.join(actor_names)}")

        tactics = tech.get("kill_chain_phases", [])
        if tactics:
            labels = [t.replace("-", " ").title() for t in tactics]
            parts.append(f"ATT&CK tactic: {', '.join(labels)}")

        examples.append(_make_example(q, "\n".join(parts)[:MAX_ASSISTANT_LEN]))

    log.info(f"  Category 11 (Data Source): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 12: Technique Overview (comprehensive technique profiles)
# ============================================================================

CAT12_TEMPLATES = [
    "What is {tech_id} ({tech_name}) in the MITRE ATT&CK framework?",
    "Explain {tech_name} ({tech_id}) in detail.",
    "Describe the ATT&CK technique {tech_id} ({tech_name}).",
    "Provide an overview of {tech_name} ({tech_id}) including usage, actors, and mitigations.",
    "What do I need to know about {tech_id} ({tech_name})?",
]


def generate_cat12(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 12: Comprehensive technique profiles."""
    examples = []
    techniques = graph.get_techniques()

    for tech in techniques:
        if len(examples) >= max_pairs:
            break

        tech_id = tech["external_ids"].get("mitre_attack_id", "")
        tech_name = tech["name"]
        desc = tech.get("description", "")
        if not desc or len(desc) < 50:
            continue

        actors = graph.technique_actors(tech["stix_id"])
        mitigations = graph.technique_mitigations(tech["stix_id"])
        tactics = tech.get("kill_chain_phases", [])
        platforms = tech.get("platforms", [])

        # Get subtechniques
        subtechs = graph.get_incoming_objects(
            tech["stix_id"], rel_type="subtechnique-of", source_type="attack-pattern"
        )

        # Get software that implements this technique
        software = graph.get_incoming_objects(
            tech["stix_id"], rel_type="uses", source_type="malware"
        ) + graph.get_incoming_objects(
            tech["stix_id"], rel_type="uses", source_type="tool"
        )

        q = _select_template(CAT12_TEMPLATES, tech_id).format(
            tech_id=tech_id, tech_name=tech_name
        )

        parts = [f"## {tech_name} ({tech_id})\n"]

        if tactics:
            tactic_labels = [t.replace("-", " ").title() for t in tactics]
            parts.append(f"**Tactic:** {', '.join(tactic_labels)}")

        if platforms:
            parts.append(f"**Platforms:** {', '.join(platforms)}")

        parts.append(f"\n{desc[:800]}")

        if subtechs:
            sub_strs = [
                f"{s.get('external_ids', {}).get('mitre_attack_id', '')} {s['name']}"
                for s in sorted(subtechs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:10]
            ]
            parts.append(f"\n**Subtechniques:** {', '.join(sub_strs)}")

        if actors:
            actor_names = sorted(set(a["name"] for a in actors))[:8]
            parts.append(f"\n**Known Users:** {', '.join(actor_names)}")
            if len(actors) > 8:
                parts.append(f"(and {len(actors) - 8} more groups)")

        if software:
            sw_names = sorted(set(s["name"] for s in software))[:8]
            parts.append(f"\n**Associated Software:** {', '.join(sw_names)}")

        if mitigations:
            mit_strs = [
                f"{m.get('external_ids', {}).get('mitre_attack_id', '')} {m['name']}"
                for m in sorted(mitigations, key=lambda x: x["name"])[:5]
            ]
            parts.append(f"\n**Mitigations:** {', '.join(mit_strs)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 12 (Technique Overview): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 13: Malware/Tool Overview (comprehensive profiles)
# ============================================================================

CAT13_MALWARE_TEMPLATES = [
    "What is {sw_name} malware and how is it used in attacks?",
    "Provide a threat intelligence profile for {sw_name} ({sw_id}).",
    "Describe {sw_name} ({sw_id}) including capabilities, techniques, and associated actors.",
    "What do we know about {sw_name} from a security perspective?",
    "Analyze {sw_name} ({sw_id}) — what are its capabilities and who uses it?",
]

CAT13_TOOL_TEMPLATES = [
    "What is {sw_name} and how is it used by threat actors?",
    "Describe the dual-use tool {sw_name} ({sw_id}) and its abuse by adversaries.",
    "What techniques does {sw_name} enable and which groups use it?",
    "Provide a security analysis of {sw_name} ({sw_id}).",
    "What should defenders know about {sw_name} ({sw_id})?",
]


def generate_cat13(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 13: Comprehensive malware/tool profiles."""
    examples = []

    for node in graph.nodes.values():
        if len(examples) >= max_pairs:
            break
        if node["type"] not in ("malware", "tool"):
            continue

        desc = node.get("description", "")
        if not desc or len(desc) < 50:
            continue

        sw_id = node.get("external_ids", {}).get("mitre_attack_id", "")

        templates = CAT13_MALWARE_TEMPLATES if node["type"] == "malware" else CAT13_TOOL_TEMPLATES
        q = _select_template(templates, node["name"]).format(
            sw_name=node["name"], sw_id=sw_id
        )

        parts = [f"## {node['name']} ({sw_id})\n"]
        parts.append(f"**Type:** {node['type'].title()}")

        if node.get("platforms"):
            parts.append(f"**Platforms:** {', '.join(node['platforms'])}")

        parts.append(f"\n{desc[:800]}")

        # Techniques
        techs = graph.get_connected_objects(
            node["stix_id"], rel_type="uses", target_type="attack-pattern"
        )
        if techs:
            by_tactic = defaultdict(list)
            for t in techs:
                for phase in t.get("kill_chain_phases", ["other"]):
                    by_tactic[phase].append(t)

            parts.append(f"\n**ATT&CK Techniques ({len(techs)}):**")
            for tactic in sorted(by_tactic.keys()):
                tactic_label = tactic.replace("-", " ").title()
                for t in sorted(by_tactic[tactic], key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", ""))[:4]:
                    tid = t.get("external_ids", {}).get("mitre_attack_id", "")
                    parts.append(f"  - {tid} {t['name']} ({tactic_label})")

        # Associated actors
        actors = graph.get_incoming_objects(
            node["stix_id"], rel_type="uses", source_type="intrusion-set"
        )
        if actors:
            actor_names = sorted(set(a["name"] for a in actors))[:8]
            parts.append(f"\n**Used By:** {', '.join(actor_names)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 13 (Malware/Tool Overview): {len(examples)} pairs")
    return examples


# ============================================================================
# Category 14: Actor Overview (comprehensive threat actor profiles)
# ============================================================================

CAT14_TEMPLATES = [
    "Who is {name}? Provide a comprehensive threat actor profile.",
    "What do we know about the threat actor {name} ({attack_id})?",
    "Give me an intelligence briefing on {name}.",
    "Describe the APT group {name} including origin, targets, and capabilities.",
    "Provide a threat intelligence summary for {name} ({attack_id}).",
]


def generate_cat14(graph: STIXGraph, max_pairs: int) -> list[dict]:
    """Category 14: Comprehensive threat actor profiles."""
    examples = []
    actors = graph.get_actors()

    for actor in actors:
        if len(examples) >= max_pairs:
            break

        name = actor["name"]
        attack_id = actor["external_ids"].get("mitre_attack_id", "")
        desc = actor.get("description", "")
        aliases = actor.get("aliases", [])
        ext = actor.get("external_ids", {})

        # Need at least some content to generate a useful profile
        techs = graph.actor_techniques(actor["stix_id"])
        software = graph.actor_software(actor["stix_id"])
        campaigns = graph.actor_campaigns(actor["stix_id"])

        country = ext.get("country", "")
        sponsor = ext.get("suspected_state_sponsor", "")
        victims = ext.get("cfr-suspected-victims", [])
        target_cats = ext.get("cfr-target-category", [])
        motive = ext.get("motive", "")

        # Skip actors with almost no info
        has_content = bool(desc) or techs or software or country or victims
        if not has_content:
            continue

        q = _select_template(CAT14_TEMPLATES, name).format(
            name=name, attack_id=attack_id
        )

        parts = [f"## {name}"]
        if attack_id:
            parts[0] += f" ({attack_id})"
        parts.append("")

        if aliases:
            parts.append(f"**Also known as:** {', '.join(aliases[:8])}")

        if country:
            line = f"**Origin:** {country}"
            if sponsor:
                line += f" (suspected state sponsor: {sponsor})"
            parts.append(line)

        if motive:
            parts.append(f"**Motivation:** {motive}")

        if desc:
            parts.append(f"\n{desc[:600]}")

        if target_cats:
            cats = target_cats if isinstance(target_cats, list) else [target_cats]
            parts.append(f"\n**Target Sectors:** {', '.join(cats[:8])}")

        if victims:
            vics = victims if isinstance(victims, list) else [victims]
            parts.append(f"**Target Countries:** {', '.join(vics[:8])}")

        if techs:
            parts.append(f"\n**ATT&CK Techniques:** {len(techs)} known techniques")
            # Show top tactics
            by_tactic = defaultdict(int)
            for t in techs:
                for phase in t.get("kill_chain_phases", []):
                    by_tactic[phase] += 1
            top_tactics = sorted(by_tactic.items(), key=lambda x: -x[1])[:5]
            tactic_strs = [f"{t.replace('-', ' ').title()} ({c})" for t, c in top_tactics]
            parts.append(f"Primary tactics: {', '.join(tactic_strs)}")

        if software:
            sw_names = sorted(set(s["name"] for s in software))[:8]
            parts.append(f"\n**Software Arsenal:** {', '.join(sw_names)}")

        if campaigns:
            camp_names = [c["name"] for c in campaigns[:5]]
            parts.append(f"**Campaigns:** {', '.join(camp_names)}")

        answer = "\n".join(parts)[:MAX_ASSISTANT_LEN]
        examples.append(_make_example(q, answer))

    log.info(f"  Category 14 (Actor Overview): {len(examples)} pairs")
    return examples


# ============================================================================
# Deduplication & Ingestion
# ============================================================================

def deduplicate_examples(examples: list[dict], conn) -> list[dict]:
    """Remove duplicates within the batch and against existing DB records."""
    cur = conn.cursor()
    cur.execute("SELECT content_hash FROM all_records")
    existing_hashes = {row[0] for row in cur.fetchall()}

    seen = set()
    unique = []
    for ex in examples:
        msgs = ex["messages"]
        h = content_hash(msgs[0]["content"], msgs[1]["content"])
        if h not in seen and h not in existing_hashes:
            seen.add(h)
            unique.append(ex)

    removed = len(examples) - len(unique)
    if removed:
        log.info(f"  Dedup: removed {removed} duplicates")
    return unique


def ingest_to_db(conn, examples: list[dict]) -> int:
    """Ingest generated training pairs into all_records + domain tables."""
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    for ex in examples:
        msgs = ex["messages"]
        user_msg = msgs[0]["content"]
        asst_msg = msgs[1]["content"]

        meta = extract_all_metadata(user_msg, asst_msg)

        try:
            cur.execute("""
                INSERT INTO all_records
                (split, domain, question_type, user_message, assistant_message,
                 cve_ids, mitre_techniques, cwe_ids, severity, cvss_score,
                 char_length, content_hash, source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "train", meta["domain"], meta["question_type"],
                user_msg, asst_msg,
                meta["cve_ids"], meta["mitre_techniques"], meta["cwe_ids"],
                meta["severity"], meta["cvss_score"],
                meta["char_length"], meta["content_hash"],
                "stix_graph", now,
            ))

            master_id = cur.lastrowid
            domain = meta["domain"]

            cur.execute(f"""
                INSERT INTO {domain}
                (master_id, split, question_type, user_message, assistant_message,
                 cve_ids, mitre_techniques, cwe_ids, severity, cvss_score, char_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                master_id, "train", meta["question_type"],
                user_msg, asst_msg,
                meta["cve_ids"], meta["mitre_techniques"], meta["cwe_ids"],
                meta["severity"], meta["cvss_score"], meta["char_length"],
            ))
            inserted += 1

        except Exception as e:
            if "UNIQUE constraint" not in str(e):
                log.warning(f"Insert error: {e}")

    conn.commit()
    return inserted


# ============================================================================
# Main
# ============================================================================

def run_generation(
    db_path: str,
    dry_run: bool = False,
    categories: list[int] | None = None,
    max_per_category: int | None = None,
):
    """Run the full training pair generation pipeline."""
    conn = get_connection(db_path)
    migrate_schema(conn)

    log.info("Loading STIX graph...")
    graph = STIXGraph(db_path)

    all_categories = categories or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    defaults = {
        1: 8000, 2: 5000, 3: 5000, 4: 3000, 5: 2000, 6: 3000, 7: 2500,
        8: 2000, 9: 2000, 10: 1500, 11: 1500, 12: 1500, 13: 1500, 14: 2000,
    }

    all_examples = []
    category_counts = {}

    log.info("Generating training pairs...")

    generators = {
        1: generate_cat1,
        2: generate_cat2,
        3: generate_cat3,
        4: generate_cat4,
        5: generate_cat5,
        6: generate_cat6,
        7: generate_cat7,
        8: generate_cat8,
        9: generate_cat9,
        10: generate_cat10,
        11: generate_cat11,
        12: generate_cat12,
        13: generate_cat13,
        14: generate_cat14,
    }

    for cat_num in sorted(all_categories):
        if cat_num not in generators:
            log.warning(f"Unknown category: {cat_num}")
            continue

        limit = max_per_category or defaults[cat_num]
        examples = generators[cat_num](graph, limit)
        category_counts[cat_num] = len(examples)
        all_examples.extend(examples)

    log.info(f"\nTotal generated: {len(all_examples)} pairs")

    # Dedup
    all_examples = deduplicate_examples(all_examples, conn)
    log.info(f"After dedup: {len(all_examples)} unique pairs")

    if dry_run:
        log.info("DRY RUN — no output written.")
        conn.close()
        return

    # Write JSONL output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"Saved to {OUTPUT_PATH} ({len(all_examples)} lines)")

    # Ingest into DB
    log.info("Ingesting into database...")
    inserted = ingest_to_db(conn, all_examples)
    log.info(f"Ingested {inserted:,} records into all_records")

    # Update stats
    from scripts.sources.fetch_stix_objects import update_stix_stats
    from scripts.db_utils import compute_stats
    update_stix_stats(conn)
    compute_stats(conn)

    # Summary
    log.info("=" * 60)
    log.info("Generation Summary:")
    for cat_num, count in sorted(category_counts.items()):
        log.info(f"  Category {cat_num}: {count:,} pairs")
    log.info(f"  TOTAL unique: {len(all_examples):,}")
    log.info(f"  Ingested: {inserted:,}")
    log.info(f"  JSONL: {OUTPUT_PATH}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training pairs from STIX relationship graph"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't write")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated category numbers (e.g. 1,2,3)"
    )
    parser.add_argument(
        "--max-per-category", type=int, default=None,
        help="Max pairs per category (overrides defaults)"
    )
    args = parser.parse_args()

    cats = None
    if args.categories:
        cats = [int(c.strip()) for c in args.categories.split(",")]

    run_generation(
        args.db,
        dry_run=args.dry_run,
        categories=cats,
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
