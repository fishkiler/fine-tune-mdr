#!/usr/bin/env python3
"""
============================================================================
STIX 2.1 Interactive Graph Visualization — API Server
============================================================================
FastAPI server exposing the STIXGraph for browser-based visualization.
Serves the vis.js frontend and provides search, subgraph, and detail APIs.

Usage:
    python -m scripts.stix_graph_server
    python -m scripts.stix_graph_server --port 11970
============================================================================
"""

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from scripts.stix_graph import STIXGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "static"
MAX_SUBGRAPH_NODES = 200  # cap for vis.js rendering performance

graph: STIXGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    log.info("Loading STIX graph into memory...")
    graph = STIXGraph()
    stats = graph.get_stats()
    total_nodes = sum(stats["objects"].values())
    total_edges = sum(stats["relationships"].values())
    log.info(f"Graph ready: {total_nodes:,} nodes, {total_edges:,} edges")
    yield


app = FastAPI(title="STIX Graph Visualization", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.1.67:6970",
        "http://192.168.1.67:6971",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Static files & root redirect ──────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/static/stix_graph.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon") \
        if (STATIC_DIR / "favicon.ico").exists() else RedirectResponse(url="/static/stix_graph.html")


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/stix/stats")
def api_stats():
    """Graph statistics: object counts, relationship counts, training links."""
    return graph.get_stats()


@app.get("/api/stix/search")
def api_search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """Search objects by name/alias substring match. Returns top matches."""
    q_lower = q.lower()
    matches = []

    # Exact match first
    exact_id = graph.name_index.get(q_lower)
    if exact_id and exact_id in graph.nodes:
        node = graph.nodes[exact_id]
        matches.append({
            "stix_id": node["stix_id"],
            "type": node["type"],
            "name": node["name"],
            "aliases": node["aliases"][:5],
            "score": 100,
        })

    # Substring matches
    seen = {exact_id} if exact_id else set()
    for name_key, stix_id in graph.name_index.items():
        if stix_id in seen:
            continue
        if q_lower in name_key:
            node = graph.nodes.get(stix_id)
            if not node:
                continue
            seen.add(stix_id)
            # Score: starts-with > contains, shorter = better
            score = 80 if name_key.startswith(q_lower) else 50
            score += max(0, 30 - len(name_key))
            matches.append({
                "stix_id": node["stix_id"],
                "type": node["type"],
                "name": node["name"],
                "aliases": node["aliases"][:5],
                "score": score,
            })
            if len(matches) >= limit:
                break

    # Also search CVE IDs directly if query looks like a CVE
    if q_lower.startswith("cve-") and len(matches) < limit:
        for stix_id, node in graph.nodes.items():
            if node["type"] == "vulnerability" and q_lower in node["name"].lower():
                if stix_id not in seen:
                    seen.add(stix_id)
                    matches.append({
                        "stix_id": node["stix_id"],
                        "type": node["type"],
                        "name": node["name"],
                        "aliases": [],
                        "score": 60,
                    })
                    if len(matches) >= limit:
                        break

    matches.sort(key=lambda x: -x["score"])
    return {"results": matches[:limit], "total": len(matches)}


@app.get("/api/stix/subgraph/{stix_id:path}")
def api_subgraph(
    stix_id: str,
    depth: int = Query(2, ge=1, le=3),
    max_nodes: int = Query(MAX_SUBGRAPH_NODES, ge=10, le=500),
):
    """BFS subgraph extraction. Returns nodes + edges for visualization.

    If the raw subgraph exceeds max_nodes, it is pruned to keep the
    closest / highest-degree nodes so vis.js stays responsive.
    """
    if stix_id not in graph.nodes:
        return {"error": f"Object not found: {stix_id}", "nodes": [], "edges": []}

    subgraph = graph.get_subgraph(stix_id, depth=depth)
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]
    truncated = False

    # Prune if too large: keep center node, then sort remaining by
    # (lower depth first, higher degree second) and take top max_nodes.
    if len(nodes) > max_nodes:
        truncated = True
        # Compute degree within this subgraph
        deg = {}
        for e in edges:
            deg[e["source"]] = deg.get(e["source"], 0) + 1
            deg[e["target"]] = deg.get(e["target"], 0) + 1

        # Center always stays
        others = [n for n in nodes if n["stix_id"] != stix_id]
        others.sort(key=lambda n: (n["depth"], -deg.get(n["stix_id"], 0)))
        kept = {stix_id}
        for n in others[:max_nodes - 1]:
            kept.add(n["stix_id"])
        nodes = [n for n in nodes if n["stix_id"] in kept]
        edges = [e for e in edges if e["source"] in kept and e["target"] in kept]

    # Add degree info for node sizing
    degree_map = {}
    for edge in edges:
        degree_map[edge["source"]] = degree_map.get(edge["source"], 0) + 1
        degree_map[edge["target"]] = degree_map.get(edge["target"], 0) + 1
    for node in nodes:
        node["degree"] = degree_map.get(node["stix_id"], 0)

    return {"nodes": nodes, "edges": edges, "truncated": truncated}


@app.get("/api/stix/object/{stix_id:path}")
def api_object(stix_id: str):
    """Full object detail + all relationships."""
    node = graph.nodes.get(stix_id)
    if not node:
        return {"error": f"Object not found: {stix_id}"}
    rels = graph.get_relationships(stix_id)
    return {"object": node, "relationships": rels}


@app.get("/api/stix/objects")
def api_objects(type: str = Query(...), limit: int = Query(200, ge=1, le=1000)):
    """List all objects of a given type."""
    results = []
    for node in graph.nodes.values():
        if node["type"] == type:
            results.append({
                "stix_id": node["stix_id"],
                "type": node["type"],
                "name": node["name"],
                "aliases": node["aliases"][:5],
            })
            if len(results) >= limit:
                break
    results.sort(key=lambda x: x["name"])
    return {"results": results, "total": len(results)}


# ── Campaign Export API (for MDR Log Simulator integration) ────────────────

@app.get("/api/stix/campaigns")
def api_campaigns():
    """List all MITRE ATT&CK campaigns with technique/malware/tool metadata."""
    return graph.get_campaigns_summary()


@app.get("/api/stix/campaigns/{stix_id:path}")
def api_campaign_detail(stix_id: str):
    """Full campaign detail: kill-chain-ordered techniques, actor, malware, tools."""
    detail = graph.get_campaign_detail(stix_id)
    if not detail:
        return {"error": f"Campaign not found: {stix_id}"}
    return detail


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STIX Graph Visualization Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=11970, help="Port (default: 11970)")
    args = parser.parse_args()

    log.info(f"Starting STIX Graph server on http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
