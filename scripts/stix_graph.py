#!/usr/bin/env python3
"""
============================================================================
STIX Graph Query Library & CLI
============================================================================
In-memory graph built from stix_objects + stix_relationships tables.
Provides traversal functions for the training pair generator and CLI
for ad-hoc queries.

Library usage:
    from scripts.stix_graph import STIXGraph
    g = STIXGraph("mdr-database/mdr_dataset.db")
    techs = g.get_connected_objects(apt28_id, rel_type="uses", target_type="attack-pattern")

CLI usage:
    python scripts/stix_graph.py --stats
    python scripts/stix_graph.py --query actor-techniques --name "APT28"
    python scripts/stix_graph.py --query cve-actors --cve "CVE-2024-3400"
    python scripts/stix_graph.py --query subgraph --name "APT28" --depth 2
============================================================================
"""

import argparse
import json
import logging
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

from scripts.db_utils import DEFAULT_DB_PATH, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class STIXGraph:
    """In-memory STIX relationship graph with traversal operations."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self.nodes: dict[str, dict] = {}          # stix_id → object dict
        self.outgoing: dict[str, list] = defaultdict(list)  # stix_id → [(rel_type, target_id, desc)]
        self.incoming: dict[str, list] = defaultdict(list)  # stix_id → [(rel_type, source_id, desc)]
        self.name_index: dict[str, str] = {}      # lowercase name/alias → stix_id
        self._load()

    def _load(self):
        """Load graph from database into memory."""
        conn = get_connection(self.db_path)
        cur = conn.cursor()

        # Load nodes
        cur.execute(
            "SELECT stix_id, type, name, aliases, description, external_ids, "
            "source, platforms, kill_chain_phases, severity, cvss_score "
            "FROM stix_objects"
        )
        for row in cur.fetchall():
            stix_id = row[0]
            node = {
                "stix_id": stix_id,
                "type": row[1],
                "name": row[2],
                "aliases": json.loads(row[3]) if row[3] else [],
                "description": row[4] or "",
                "external_ids": json.loads(row[5]) if row[5] else {},
                "source": row[6],
                "platforms": json.loads(row[7]) if row[7] else [],
                "kill_chain_phases": json.loads(row[8]) if row[8] else [],
                "severity": row[9],
                "cvss_score": row[10],
            }
            self.nodes[stix_id] = node

            # Name index (skip vulnerability objects — too many and names are CVE IDs)
            if node["type"] != "vulnerability":
                self.name_index[node["name"].lower()] = stix_id
                for alias in node["aliases"]:
                    self.name_index[alias.lower()] = stix_id
                # Index external IDs (T1059, G0007, S0154, etc.)
                # Only for primary object types — sigma-rules and detection-strategies
                # reference technique IDs but shouldn't own them in the index.
                if node["type"] in (
                    "attack-pattern", "intrusion-set", "malware", "tool",
                    "campaign", "course-of-action", "x-mitre-tactic",
                    "x-mitre-data-component",
                ):
                    for ext_val in node["external_ids"].values():
                        if isinstance(ext_val, str) and ext_val:
                            self.name_index[ext_val.lower()] = stix_id

        # Load edges
        cur.execute(
            "SELECT source_ref, target_ref, relationship_type, description "
            "FROM stix_relationships"
        )
        for src, tgt, rel_type, desc in cur.fetchall():
            self.outgoing[src].append((rel_type, tgt, desc))
            self.incoming[tgt].append((rel_type, src, desc))

        conn.close()
        log.info(
            f"Loaded graph: {len(self.nodes):,} nodes, "
            f"{sum(len(v) for v in self.outgoing.values()):,} edges"
        )

    # ── Core Query Functions ──────────────────────────────────────────────

    def get_object_by_name(self, name: str) -> dict | None:
        """Find a STIX object by name or alias (case-insensitive)."""
        stix_id = self.name_index.get(name.lower())
        if stix_id:
            return self.nodes.get(stix_id)
        # Also check CVE IDs directly
        for node in self.nodes.values():
            if node["name"].lower() == name.lower():
                return node
        return None

    def get_relationships(
        self,
        stix_id: str,
        direction: str = "both",
        rel_type: str | None = None,
    ) -> list[dict]:
        """Get all relationships for a STIX object.

        direction: 'outgoing', 'incoming', or 'both'
        rel_type: optional filter by relationship type
        """
        results = []

        if direction in ("outgoing", "both"):
            for rt, tgt, desc in self.outgoing.get(stix_id, []):
                if rel_type and rt != rel_type:
                    continue
                target_node = self.nodes.get(tgt, {})
                results.append({
                    "direction": "outgoing",
                    "relationship_type": rt,
                    "related_id": tgt,
                    "related_name": target_node.get("name", ""),
                    "related_type": target_node.get("type", ""),
                    "description": desc,
                })

        if direction in ("incoming", "both"):
            for rt, src, desc in self.incoming.get(stix_id, []):
                if rel_type and rt != rel_type:
                    continue
                source_node = self.nodes.get(src, {})
                results.append({
                    "direction": "incoming",
                    "relationship_type": rt,
                    "related_id": src,
                    "related_name": source_node.get("name", ""),
                    "related_type": source_node.get("type", ""),
                    "description": desc,
                })

        return results

    def get_connected_objects(
        self,
        stix_id: str,
        rel_type: str | None = None,
        target_type: str | None = None,
    ) -> list[dict]:
        """Get all objects connected to this one via outgoing relationships.

        Optionally filtered by relationship type and target object type.
        """
        results = []
        for rt, tgt, desc in self.outgoing.get(stix_id, []):
            if rel_type and rt != rel_type:
                continue
            node = self.nodes.get(tgt)
            if not node:
                continue
            if target_type and node["type"] != target_type:
                continue
            results.append({**node, "rel_description": desc, "rel_type": rt})
        return results

    def get_incoming_objects(
        self,
        stix_id: str,
        rel_type: str | None = None,
        source_type: str | None = None,
    ) -> list[dict]:
        """Get all objects that point TO this one via incoming relationships."""
        results = []
        for rt, src, desc in self.incoming.get(stix_id, []):
            if rel_type and rt != rel_type:
                continue
            node = self.nodes.get(src)
            if not node:
                continue
            if source_type and node["type"] != source_type:
                continue
            results.append({**node, "rel_description": desc, "rel_type": rt})
        return results

    def traverse_path(
        self,
        start_id: str,
        path_spec: list[tuple[str, str]],
    ) -> list[dict]:
        """Multi-hop graph traversal.

        path_spec: list of (relationship_type, target_object_type) tuples.
        Returns objects reached at the end of the path.
        """
        current_ids = {start_id}

        for rel_type, target_type in path_spec:
            next_ids = set()
            for sid in current_ids:
                for connected in self.get_connected_objects(sid, rel_type, target_type):
                    next_ids.add(connected["stix_id"])
            current_ids = next_ids

        return [self.nodes[sid] for sid in current_ids if sid in self.nodes]

    def get_subgraph(self, stix_id: str, depth: int = 2) -> dict:
        """BFS subgraph extraction up to N hops.

        Returns {"nodes": [...], "edges": [...]} suitable for visualization.
        """
        visited = set()
        queue = deque([(stix_id, 0)])
        nodes = []
        edges = []

        while queue:
            current_id, d = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self.nodes.get(current_id)
            if node:
                entry = {
                    "stix_id": node["stix_id"],
                    "type": node["type"],
                    "name": node["name"],
                    "depth": d,
                }
                if node.get("description"):
                    entry["description"] = node["description"][:300]
                if node.get("aliases"):
                    entry["aliases"] = node["aliases"][:5]
                if node.get("kill_chain_phases"):
                    entry["kill_chain_phases"] = node["kill_chain_phases"]
                ext = node.get("external_ids", {})
                if ext:
                    entry["external_ids"] = ext
                nodes.append(entry)

            if d >= depth:
                continue

            # Outgoing
            for rt, tgt, desc in self.outgoing.get(current_id, []):
                # Skip vulnerability nodes in subgraph (too many)
                tgt_node = self.nodes.get(tgt)
                if tgt_node and tgt_node["type"] == "vulnerability":
                    continue
                edges.append({
                    "source": current_id,
                    "target": tgt,
                    "type": rt,
                })
                if tgt not in visited:
                    queue.append((tgt, d + 1))

            # Incoming
            for rt, src, desc in self.incoming.get(current_id, []):
                src_node = self.nodes.get(src)
                if src_node and src_node["type"] == "vulnerability":
                    continue
                edges.append({
                    "source": src,
                    "target": current_id,
                    "type": rt,
                })
                if src not in visited:
                    queue.append((src, d + 1))

        return {"nodes": nodes, "edges": edges}

    def get_training_records_for_object(self, stix_id: str) -> list[dict]:
        """Get all training records linked to a STIX object."""
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT ar.id, ar.domain, ar.question_type,
                   ar.user_message, ar.assistant_message,
                   stl.link_type, stl.confidence
            FROM stix_training_links stl
            JOIN all_records ar ON stl.record_id = ar.id
            WHERE stl.stix_id = ?
        """, (stix_id,))
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0], "domain": row[1], "question_type": row[2],
                "user_message": row[3], "assistant_message": row[4],
                "link_type": row[5], "confidence": row[6],
            })
        conn.close()
        return results

    def get_stix_objects_for_record(self, record_id: int) -> list[dict]:
        """Get all STIX objects linked to a training record."""
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT so.stix_id, so.type, so.name, stl.link_type, stl.confidence
            FROM stix_training_links stl
            JOIN stix_objects so ON stl.stix_id = so.stix_id
            WHERE stl.record_id = ?
        """, (record_id,))
        results = []
        for row in cur.fetchall():
            results.append({
                "stix_id": row[0], "type": row[1], "name": row[2],
                "link_type": row[3], "confidence": row[4],
            })
        conn.close()
        return results

    def get_stats(self) -> dict:
        """Return counts: objects by type, relationships by type."""
        obj_counts = defaultdict(int)
        for node in self.nodes.values():
            obj_counts[node["type"]] += 1

        rel_counts = defaultdict(int)
        for edges in self.outgoing.values():
            for rt, _, _ in edges:
                rel_counts[rt] += 1

        # Training links from DB
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT link_type, COUNT(*) FROM stix_training_links GROUP BY link_type"
        )
        link_counts = dict(cur.fetchall())
        conn.close()

        return {
            "objects": dict(obj_counts),
            "relationships": dict(rel_counts),
            "training_links": link_counts,
        }

    # ── Convenience methods for training pair generation ──────────────────

    def get_actors(self) -> list[dict]:
        """Get all intrusion-set objects."""
        return [n for n in self.nodes.values() if n["type"] == "intrusion-set"]

    def get_techniques(self) -> list[dict]:
        """Get all attack-pattern objects."""
        return [n for n in self.nodes.values() if n["type"] == "attack-pattern"]

    def get_mitigations(self) -> list[dict]:
        """Get all course-of-action objects."""
        return [n for n in self.nodes.values() if n["type"] == "course-of-action"]

    def actor_techniques(self, actor_id: str) -> list[dict]:
        """Get techniques used by an actor, with tactic info."""
        return self.get_connected_objects(actor_id, rel_type="uses", target_type="attack-pattern")

    def actor_software(self, actor_id: str) -> list[dict]:
        """Get malware/tools used by an actor."""
        malware = self.get_connected_objects(actor_id, rel_type="uses", target_type="malware")
        tools = self.get_connected_objects(actor_id, rel_type="uses", target_type="tool")
        return malware + tools

    def actor_campaigns(self, actor_id: str) -> list[dict]:
        """Get campaigns attributed to an actor."""
        return self.get_incoming_objects(actor_id, rel_type="attributed-to", source_type="campaign")

    # ── Campaign export methods ──────────────────────────────────────────

    KILL_CHAIN_ORDER = [
        "reconnaissance", "resource-development", "initial-access", "execution",
        "persistence", "privilege-escalation", "defense-evasion", "credential-access",
        "discovery", "lateral-movement", "collection", "command-and-control",
        "exfiltration", "impact",
    ]

    def get_campaigns_summary(self) -> list[dict]:
        """Return summary list of all campaign objects with technique/malware/tool counts."""
        campaigns = [n for n in self.nodes.values() if n["type"] == "campaign"]
        results = []
        for c in campaigns:
            sid = c["stix_id"]
            techniques = self.get_connected_objects(sid, rel_type="uses", target_type="attack-pattern")
            malware = self.get_connected_objects(sid, rel_type="uses", target_type="malware")
            tools = self.get_connected_objects(sid, rel_type="uses", target_type="tool")
            actors = self.get_connected_objects(sid, rel_type="attributed-to", target_type="intrusion-set")

            # Collect unique kill-chain phases
            phases = set()
            for t in techniques:
                for p in t.get("kill_chain_phases", []):
                    phases.add(p)
            sorted_phases = [p for p in self.KILL_CHAIN_ORDER if p in phases]

            actor_name = actors[0]["name"] if actors else None

            results.append({
                "stix_id": sid,
                "name": c["name"],
                "mitre_id": c["external_ids"].get("mitre_attack_id", ""),
                "actor": actor_name,
                "description": c["description"][:300] if c["description"] else "",
                "technique_count": len(techniques),
                "malware_count": len(malware),
                "tool_count": len(tools),
                "phases": sorted_phases,
                "malware": [m["name"] for m in malware],
                "tools": [t["name"] for t in tools],
            })

        results.sort(key=lambda x: x["name"])
        return results

    def get_campaign_detail(self, stix_id: str) -> dict | None:
        """Extract full campaign data for simulator import.

        Returns techniques (kill-chain ordered), actor attribution,
        malware, tools, and relationship context descriptions.
        """
        node = self.nodes.get(stix_id)
        if not node or node["type"] != "campaign":
            return None

        # Techniques with kill-chain phase and relationship context
        raw_techs = self.get_connected_objects(stix_id, rel_type="uses", target_type="attack-pattern")
        techniques = []
        for t in raw_techs:
            phases = t.get("kill_chain_phases", [])
            primary_phase = None
            if phases:
                for p in self.KILL_CHAIN_ORDER:
                    if p in phases:
                        primary_phase = p
                        break
                if not primary_phase:
                    primary_phase = phases[0]
            techniques.append({
                "stix_id": t["stix_id"],
                "id": t.get("external_ids", {}).get("mitre_attack_id", ""),
                "name": t["name"],
                "phase": primary_phase or "unknown",
                "phases": [p for p in self.KILL_CHAIN_ORDER if p in phases] if phases else [],
                "description": t.get("description", "")[:500],
                "context": t.get("rel_description", "") or "",
            })

        # Sort by kill-chain phase order
        phase_order = {p: i for i, p in enumerate(self.KILL_CHAIN_ORDER)}
        techniques.sort(key=lambda t: (phase_order.get(t["phase"], 99), t["id"]))

        # Actor attribution
        actors = self.get_connected_objects(stix_id, rel_type="attributed-to", target_type="intrusion-set")
        actor_info = None
        if actors:
            a = actors[0]
            actor_info = {
                "stix_id": a["stix_id"],
                "name": a["name"],
                "aliases": a.get("aliases", []),
                "description": a.get("description", "")[:500],
                "external_id": a.get("external_ids", {}).get("mitre_attack_id", ""),
            }

        # Malware
        raw_malware = self.get_connected_objects(stix_id, rel_type="uses", target_type="malware")
        malware = [{
            "stix_id": m["stix_id"],
            "name": m["name"],
            "description": m.get("description", "")[:500],
            "aliases": m.get("aliases", [])[:5],
        } for m in raw_malware]

        # Tools
        raw_tools = self.get_connected_objects(stix_id, rel_type="uses", target_type="tool")
        tools = [{
            "stix_id": t["stix_id"],
            "name": t["name"],
            "description": t.get("description", "")[:500],
        } for t in raw_tools]

        # Unique phases in order
        all_phases = set()
        for t in techniques:
            all_phases.update(t["phases"])
        sorted_phases = [p for p in self.KILL_CHAIN_ORDER if p in all_phases]

        return {
            "stix_id": stix_id,
            "name": node["name"],
            "mitre_id": node["external_ids"].get("mitre_attack_id", ""),
            "description": node["description"] or "",
            "actor": actor_info,
            "techniques": techniques,
            "malware": malware,
            "tools": tools,
            "phases": sorted_phases,
            "technique_count": len(techniques),
        }

    def technique_actors(self, tech_id: str) -> list[dict]:
        """Get actors that use a technique."""
        return self.get_incoming_objects(tech_id, rel_type="uses", source_type="intrusion-set")

    def technique_mitigations(self, tech_id: str) -> list[dict]:
        """Get mitigations for a technique."""
        return self.get_incoming_objects(tech_id, rel_type="mitigates", source_type="course-of-action")

    def technique_software(self, tech_id: str) -> list[dict]:
        """Get software that implements a technique."""
        malware = self.get_incoming_objects(tech_id, rel_type="uses", source_type="malware")
        tools = self.get_incoming_objects(tech_id, rel_type="uses", source_type="tool")
        return malware + tools


# ============================================================================
# CLI
# ============================================================================

def cli_stats(graph: STIXGraph):
    stats = graph.get_stats()
    print("\n=== STIX Graph Statistics ===\n")
    print("Objects by type:")
    for t, c in sorted(stats["objects"].items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {c:>8,}")
    total_obj = sum(stats["objects"].values())
    print(f"  {'TOTAL':25s} {total_obj:>8,}")

    print("\nRelationships by type:")
    for t, c in sorted(stats["relationships"].items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {c:>8,}")
    total_rel = sum(stats["relationships"].values())
    print(f"  {'TOTAL':25s} {total_rel:>8,}")

    print("\nTraining links by type:")
    for t, c in sorted(stats["training_links"].items(), key=lambda x: -x[1]):
        print(f"  {t:25s} {c:>8,}")
    total_links = sum(stats["training_links"].values())
    print(f"  {'TOTAL':25s} {total_links:>8,}")


def cli_actor_techniques(graph: STIXGraph, name: str):
    obj = graph.get_object_by_name(name)
    if not obj:
        print(f"Actor not found: {name}")
        return

    techs = graph.actor_techniques(obj["stix_id"])
    attack_id = obj["external_ids"].get("mitre_attack_id", "")
    print(f"\n=== Techniques used by {obj['name']} ({attack_id}) ===\n")

    # Group by tactic
    by_tactic = defaultdict(list)
    for t in techs:
        phases = t.get("kill_chain_phases", [])
        if phases:
            for phase in phases:
                by_tactic[phase].append(t)
        else:
            by_tactic["unknown"].append(t)

    for tactic in sorted(by_tactic.keys()):
        tactic_techs = by_tactic[tactic]
        print(f"  {tactic.replace('-', ' ').title()}:")
        for t in sorted(tactic_techs, key=lambda x: x.get("external_ids", {}).get("mitre_attack_id", "")):
            tid = t.get("external_ids", {}).get("mitre_attack_id", "???")
            print(f"    - {tid} {t['name']}")
        print()

    print(f"Total: {len(techs)} techniques")


def cli_cve_actors(graph: STIXGraph, cve_id: str):
    obj = graph.get_object_by_name(cve_id)
    if not obj:
        print(f"CVE not found in STIX objects: {cve_id}")
        return

    print(f"\n=== Actors linked to {cve_id} ===\n")

    # Multi-hop: vulnerability ← technique ← actor
    # First find techniques that reference this CVE
    # (via exploit relationships or training links)
    actors_found = set()
    records = graph.get_training_records_for_object(obj["stix_id"])
    for rec in records:
        # Check if there are technique IDs in the record
        stix_objs = graph.get_stix_objects_for_record(rec["id"])
        for so in stix_objs:
            if so["type"] == "attack-pattern":
                # Find actors that use this technique
                actors = graph.technique_actors(so["stix_id"])
                for actor in actors:
                    if actor["stix_id"] not in actors_found:
                        actors_found.add(actor["stix_id"])
                        aid = actor.get("external_ids", {}).get("mitre_attack_id", "")
                        print(f"  - {actor['name']} ({aid})")

    if not actors_found:
        print("  No actors found via technique cross-references.")

    print(f"\nTotal: {len(actors_found)} actors")


def cli_subgraph(graph: STIXGraph, name: str, depth: int):
    obj = graph.get_object_by_name(name)
    if not obj:
        print(f"Object not found: {name}")
        return

    subgraph = graph.get_subgraph(obj["stix_id"], depth=depth)
    print(f"\n=== Subgraph around {obj['name']} (depth={depth}) ===\n")
    print(f"Nodes: {len(subgraph['nodes'])}")
    print(f"Edges: {len(subgraph['edges'])}")

    print("\nNodes by type:")
    type_counts = defaultdict(int)
    for n in subgraph["nodes"]:
        type_counts[n["type"]] += 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    print("\nEdges by type:")
    edge_counts = defaultdict(int)
    for e in subgraph["edges"]:
        edge_counts[e["type"]] += 1
    for t, c in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


def main():
    parser = argparse.ArgumentParser(
        description="STIX Graph query CLI"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument("--query", choices=["actor-techniques", "cve-actors", "subgraph"])
    parser.add_argument("--name", help="Actor or object name")
    parser.add_argument("--cve", help="CVE ID for cve-actors query")
    parser.add_argument("--depth", type=int, default=2, help="Subgraph depth")
    args = parser.parse_args()

    graph = STIXGraph(args.db)

    if args.stats:
        cli_stats(graph)
    elif args.query == "actor-techniques":
        if not args.name:
            parser.error("--name required for actor-techniques query")
        cli_actor_techniques(graph, args.name)
    elif args.query == "cve-actors":
        if not args.cve:
            parser.error("--cve required for cve-actors query")
        cli_cve_actors(graph, args.cve)
    elif args.query == "subgraph":
        if not args.name:
            parser.error("--name required for subgraph query")
        cli_subgraph(graph, args.name, args.depth)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
