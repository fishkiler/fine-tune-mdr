#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — LLM Detection Strategy Description Generator
============================================================================
Enriches the 691 x-mitre-detection-strategy objects in the STIX graph with
LLM-generated detection descriptions. Each detection-strategy currently has
an empty description field — MITRE's STIX bundle only includes name/ID.

For each detection-strategy, the script:
  1. Follows its `detects` relationship to find the attack-pattern
  2. Gathers context: technique description, tactics, platforms, mitigations,
     sigma rules, and threat actors that use the technique
  3. Prompts an LLM to write a structured detection strategy
  4. Writes the description back to the stix_objects table

Uses OpenRouter API (same pattern as review_data.py).

Usage:
    python -m scripts.generate_detection_descriptions --dry-run --limit 5
    python -m scripts.generate_detection_descriptions --model x-ai/grok-4.1-fast
    python -m scripts.generate_detection_descriptions --model x-ai/grok-4.1-fast --force
============================================================================
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env file from project root
_env_file = _project_root / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if value and key not in os.environ:
                    os.environ[key] = value

from scripts.db_utils import DEFAULT_DB_PATH, get_connection
from scripts.stix_graph import STIXGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Prompt Template
# ============================================================================

SYSTEM_PROMPT = """You are a senior detection engineer writing MITRE ATT&CK detection strategies for a security operations center (SOC). Write concise, actionable detection guidance that a security analyst can implement immediately.

You MUST respond with ONLY the detection strategy text — no preamble, no markdown headers, no "Here is..." introduction. Start directly with the content."""

TECHNIQUE_PROMPT_TEMPLATE = """Write a detection strategy for the following MITRE ATT&CK technique.

Technique: {name} ({attack_id})
Tactics: {tactics}
Platforms: {platforms}

Technique Description:
{technique_description}

{context_sections}

Write a concise detection strategy (200-400 words) covering:
1. Data Sources — what logs/telemetry to collect
2. Detection Logic — behavioral patterns and anomalies to monitor
3. Key Indicators — specific artifacts, event IDs, command patterns
4. False Positives — legitimate activity that may look malicious
5. Implementation Notes — SIEM/EDR recommendations

Be specific and actionable. Reference real log sources (Sysmon, Windows Security, auditd, cloud audit logs) and event IDs where applicable. Do not use markdown headers — write in flowing paragraphs with numbered sections."""

BATCH_PROMPT_TEMPLATE = """Write detection strategies for each of the following {count} MITRE ATT&CK techniques. For each technique, write a separate detection strategy (200-400 words) covering: (1) Data Sources, (2) Detection Logic, (3) Key Indicators, (4) False Positives, (5) Implementation Notes.

Be specific and actionable. Reference real log sources and event IDs. Do not use markdown headers — write in flowing paragraphs with numbered sections.

Respond with a JSON array of objects, one per technique, in order. Each object must have: {{"id": "<DET_ID>", "description": "<the detection strategy text>"}}

{techniques_block}"""


# ============================================================================
# Context Gathering
# ============================================================================

def gather_technique_context(graph: STIXGraph, det_stix_id: str) -> dict | None:
    """Gather all context for a detection-strategy's linked attack-pattern.

    Returns a dict with technique info and related objects, or None if
    the detection-strategy has no linked attack-pattern.
    """
    det_node = graph.nodes.get(det_stix_id)
    if not det_node:
        return None

    # Follow detects relationship to find the attack-pattern
    attack_patterns = graph.get_connected_objects(
        det_stix_id, rel_type="detects", target_type="attack-pattern"
    )
    if not attack_patterns:
        return None

    tech = attack_patterns[0]
    tech_id = tech["stix_id"]
    attack_id = tech.get("external_ids", {}).get("mitre_attack_id", "")

    # Mitigations
    mitigations = graph.technique_mitigations(tech_id)
    mitigation_names = [m["name"] for m in mitigations[:5]]

    # Sigma rules that detect this technique
    sigma_rules = graph.get_incoming_objects(
        tech_id, rel_type="detects", source_type="sigma-rule"
    )
    sigma_info = []
    for sr in sigma_rules[:5]:
        sigma_info.append(sr["name"])

    # Software (malware/tools) that uses this technique
    software = graph.technique_software(tech_id)
    software_names = [s["name"] for s in software[:5]]

    # Actors that use this technique
    actors = graph.technique_actors(tech_id)
    actor_names = [a["name"] for a in actors[:5]]

    # Platforms and tactics
    platforms = tech.get("platforms", [])
    tactics = tech.get("kill_chain_phases", [])

    return {
        "det_stix_id": det_stix_id,
        "det_name": det_node["name"],
        "det_external_id": det_node.get("external_ids", {}).get("mitre_attack_id", ""),
        "attack_id": attack_id,
        "name": tech["name"],
        "description": tech.get("description", "")[:1500],
        "platforms": platforms,
        "tactics": tactics,
        "mitigations": mitigation_names,
        "sigma_rules": sigma_info,
        "software": software_names,
        "actors": actor_names,
    }


def build_single_prompt(ctx: dict) -> str:
    """Build a prompt for a single technique."""
    sections = []

    if ctx["mitigations"]:
        sections.append(f"Mitigations: {', '.join(ctx['mitigations'])}")
    if ctx["sigma_rules"]:
        sections.append(f"Existing Sigma Rules: {', '.join(ctx['sigma_rules'])}")
    if ctx["software"]:
        sections.append(f"Known Software Using This Technique: {', '.join(ctx['software'])}")
    if ctx["actors"]:
        sections.append(f"Threat Actors Using This Technique: {', '.join(ctx['actors'])}")

    return TECHNIQUE_PROMPT_TEMPLATE.format(
        name=ctx["name"],
        attack_id=ctx["attack_id"],
        tactics=", ".join(ctx["tactics"]) if ctx["tactics"] else "N/A",
        platforms=", ".join(ctx["platforms"]) if ctx["platforms"] else "N/A",
        technique_description=ctx["description"] or "No description available.",
        context_sections="\n".join(sections) if sections else "",
    )


def build_batch_prompt(contexts: list[dict]) -> str:
    """Build a batched prompt for multiple techniques."""
    blocks = []
    for i, ctx in enumerate(contexts):
        sections = []
        if ctx["mitigations"]:
            sections.append(f"  Mitigations: {', '.join(ctx['mitigations'])}")
        if ctx["sigma_rules"]:
            sections.append(f"  Existing Sigma Rules: {', '.join(ctx['sigma_rules'])}")
        if ctx["software"]:
            sections.append(f"  Known Software: {', '.join(ctx['software'])}")
        if ctx["actors"]:
            sections.append(f"  Threat Actors: {', '.join(ctx['actors'])}")

        block = (
            f"--- Technique {i+1}: {ctx['name']} ({ctx['attack_id']}) [ID: {ctx['det_external_id']}] ---\n"
            f"  Tactics: {', '.join(ctx['tactics']) if ctx['tactics'] else 'N/A'}\n"
            f"  Platforms: {', '.join(ctx['platforms']) if ctx['platforms'] else 'N/A'}\n"
            f"  Description: {ctx['description'][:800] or 'No description available.'}\n"
        )
        if sections:
            block += "\n".join(sections) + "\n"
        blocks.append(block)

    return BATCH_PROMPT_TEMPLATE.format(
        count=len(contexts),
        techniques_block="\n".join(blocks),
    )


# ============================================================================
# LLM Client
# ============================================================================

def parse_batch_response(text: str, expected_ids: list[str]) -> dict[str, str]:
    """Parse a batched LLM response into {det_external_id: description}.

    Tries JSON array first, falls back to splitting by detection IDs.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try JSON array parse
    try:
        items = json.loads(text)
        if isinstance(items, list):
            result = {}
            for item in items:
                det_id = item.get("id", "")
                desc = item.get("description", "").strip()
                if det_id and desc:
                    result[det_id] = desc
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Fallback: try to split by DET IDs
    result = {}
    for i, det_id in enumerate(expected_ids):
        # Look for the ID in the text as a delimiter
        pattern = re.escape(det_id)
        matches = list(re.finditer(pattern, text))
        if matches:
            start = matches[0].end()
            # Find next DET ID or end of text
            end = len(text)
            for next_id in expected_ids[i+1:]:
                next_match = re.search(re.escape(next_id), text[start:])
                if next_match:
                    end = start + next_match.start()
                    break
            desc = text[start:end].strip().strip(":").strip("-").strip()
            # Clean up leading/trailing cruft
            desc = re.sub(r'^[\s\-:]+', '', desc)
            if len(desc) > 50:  # Only accept if it's substantial
                result[det_id] = desc

    return result


def parse_single_response(text: str) -> str:
    """Parse a single-technique LLM response."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    # Strip common LLM preambles
    for prefix in ["Here is", "Here's", "Below is"]:
        if text.startswith(prefix):
            # Skip to end of first sentence
            idx = text.find("\n")
            if idx > 0 and idx < 200:
                text = text[idx:].strip()
            break
    return text


def call_openrouter(system: str, user_prompt: str, model: str,
                    max_tokens: int = 4096) -> str:
    """Call OpenRouter API and return response text."""
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=120.0,
    )

    extra = {}
    if "grok" in model or "x-ai" in model:
        extra["extra_body"] = {"reasoning": {"enabled": False}}

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        **extra,
    )
    return response.choices[0].message.content.strip()


# ============================================================================
# Database Operations
# ============================================================================

def get_empty_detection_strategies(db_path: str, force: bool = False) -> list[dict]:
    """Get detection-strategy objects that need descriptions."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    if force:
        cur.execute(
            "SELECT stix_id, name, external_ids FROM stix_objects "
            "WHERE type = 'x-mitre-detection-strategy'"
        )
    else:
        cur.execute(
            "SELECT stix_id, name, external_ids FROM stix_objects "
            "WHERE type = 'x-mitre-detection-strategy' "
            "AND (description IS NULL OR description = '')"
        )

    results = []
    for row in cur.fetchall():
        results.append({
            "stix_id": row[0],
            "name": row[1],
            "external_ids": json.loads(row[2]) if row[2] else {},
        })
    conn.close()
    return results


def save_description(db_path: str, stix_id: str, description: str) -> None:
    """Write a detection description back to the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "UPDATE stix_objects SET description = ?, updated_at = ? WHERE stix_id = ?",
        (description, now, stix_id),
    )
    conn.commit()
    conn.close()


# ============================================================================
# Main Processing
# ============================================================================

def process_batch(graph: STIXGraph, contexts: list[dict], model: str,
                  db_path: str) -> int:
    """Process a batch of techniques via a single LLM call. Returns count saved."""
    if len(contexts) == 1:
        # Single technique — simpler prompt/parse
        prompt = build_single_prompt(contexts[0])
        try:
            text = call_openrouter(SYSTEM_PROMPT, prompt, model, max_tokens=2048)
            description = parse_single_response(text)
            if len(description) > 50:
                save_description(db_path, contexts[0]["det_stix_id"], description)
                return 1
            else:
                log.warning(f"  Response too short for {contexts[0]['det_external_id']}, skipping")
                return 0
        except Exception as e:
            log.warning(f"  API call failed: {e}")
            return 0

    # Batch prompt
    prompt = build_batch_prompt(contexts)
    expected_ids = [ctx["det_external_id"] for ctx in contexts]
    id_to_stix = {ctx["det_external_id"]: ctx["det_stix_id"] for ctx in contexts}

    # Scale max_tokens with batch size (~500 tokens per detection strategy)
    max_tok = max(4096, len(contexts) * 600)

    try:
        text = call_openrouter(SYSTEM_PROMPT, prompt, model, max_tokens=max_tok)
        descriptions = parse_batch_response(text, expected_ids)

        saved = 0
        for det_id, desc in descriptions.items():
            stix_id = id_to_stix.get(det_id)
            if stix_id and len(desc) > 50:
                save_description(db_path, stix_id, desc)
                saved += 1

        # If batch parse failed, fall back to individual calls
        if saved < len(contexts) // 2:
            log.warning(f"  Batch parse got {saved}/{len(contexts)}, "
                        f"retrying missed items individually...")
            for ctx in contexts:
                if ctx["det_external_id"] not in descriptions:
                    single_prompt = build_single_prompt(ctx)
                    try:
                        single_text = call_openrouter(
                            SYSTEM_PROMPT, single_prompt, model, max_tokens=2048
                        )
                        desc = parse_single_response(single_text)
                        if len(desc) > 50:
                            save_description(db_path, ctx["det_stix_id"], desc)
                            saved += 1
                        time.sleep(0.3)
                    except Exception as e:
                        log.warning(f"  Individual retry failed for "
                                    f"{ctx['det_external_id']}: {e}")

        return saved

    except Exception as e:
        log.warning(f"  Batch API call failed: {e}")
        return 0


def run(db_path: str, model: str, batch_size: int = 5,
        limit: int | None = None, force: bool = False,
        dry_run: bool = False) -> dict:
    """Main entry point for detection description generation."""

    # Get detection-strategies needing descriptions
    det_strategies = get_empty_detection_strategies(db_path, force=force)
    if limit:
        det_strategies = det_strategies[:limit]

    if not det_strategies:
        log.info("No detection-strategy objects need descriptions.")
        return {"total": 0, "generated": 0, "failed": 0}

    log.info(f"Found {len(det_strategies)} detection-strategy objects to process.")

    # Load the STIX graph for context gathering
    log.info("Loading STIX graph...")
    graph = STIXGraph(db_path)

    # Gather context for each detection-strategy
    contexts = []
    skipped = 0
    for ds in det_strategies:
        ctx = gather_technique_context(graph, ds["stix_id"])
        if ctx:
            contexts.append(ctx)
        else:
            skipped += 1

    if skipped:
        log.info(f"Skipped {skipped} detection-strategies with no linked attack-pattern.")

    if not contexts:
        log.info("No techniques found to generate descriptions for.")
        return {"total": len(det_strategies), "generated": 0, "failed": 0}

    log.info(f"Gathered context for {len(contexts)} techniques.")

    if dry_run:
        log.info("\n[DRY RUN] Preview of prompts that would be sent:\n")
        for ctx in contexts[:3]:
            prompt = build_single_prompt(ctx)
            log.info(f"{'='*60}")
            log.info(f"Detection Strategy: {ctx['det_name']} ({ctx['det_external_id']})")
            log.info(f"Technique: {ctx['name']} ({ctx['attack_id']})")
            log.info(f"Tactics: {', '.join(ctx['tactics'])}")
            log.info(f"Platforms: {', '.join(ctx['platforms'])}")
            log.info(f"Mitigations: {ctx['mitigations']}")
            log.info(f"Sigma Rules: {ctx['sigma_rules']}")
            log.info(f"Software: {ctx['software']}")
            log.info(f"Actors: {ctx['actors']}")
            log.info(f"\nPrompt length: {len(prompt)} chars")
            log.info(f"Technique description: {ctx['description'][:200]}...")
            log.info("")

        if len(contexts) > 3:
            log.info(f"... and {len(contexts) - 3} more techniques")

        return {"total": len(contexts), "generated": 0, "failed": 0}

    # Process in batches
    generated = 0
    failed = 0

    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(contexts) + batch_size - 1) // batch_size

        names = [f"{c['attack_id']}" for c in batch]
        log.info(f"Batch {batch_num}/{total_batches} "
                 f"({generated}/{len(contexts)} done) — {', '.join(names)}")

        saved = process_batch(graph, batch, model, db_path)
        generated += saved
        failed += len(batch) - saved

        if saved < len(batch):
            log.warning(f"  {len(batch) - saved} techniques in batch failed")

        # Rate limiting between batches
        time.sleep(0.5)

    return {
        "total": len(contexts),
        "generated": generated,
        "failed": failed,
    }


# ============================================================================
# CLI
# ============================================================================

OPENROUTER_MODELS = {
    "grok-fast": "x-ai/grok-4.1-fast",
    "grok": "x-ai/grok-4.1",
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    "deepseek": "deepseek/deepseek-chat-v3-0324",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
}


def main():
    model_help = (
        "OpenRouter model to use. Full model ID or shortcut: "
        + ", ".join(f"{k}" for k in OPENROUTER_MODELS.keys())
    )

    parser = argparse.ArgumentParser(
        description="Generate LLM detection strategy descriptions for STIX graph"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--model", default="x-ai/grok-4.1-fast", help=model_help)
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Techniques per API call (default: 5)")
    parser.add_argument("--limit", type=int, help="Max techniques to process")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if description already populated")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview prompts without calling API")
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Resolve model shortcuts
    model = args.model
    if model in OPENROUTER_MODELS:
        model = OPENROUTER_MODELS[model]

    log.info("=" * 60)
    log.info("  STIX Detection Strategy Description Generator")
    log.info("=" * 60)
    log.info(f"  Model:      {model}")
    log.info(f"  Batch size: {args.batch_size}")
    if args.limit:
        log.info(f"  Limit:      {args.limit}")
    if args.force:
        log.info(f"  Force:      regenerating all")
    if args.dry_run:
        log.info(f"  Mode:       DRY RUN")
    log.info("")

    results = run(
        db_path=args.db,
        model=model,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
    )

    log.info("")
    log.info("=" * 60)
    log.info("  RESULTS")
    log.info("=" * 60)
    log.info(f"  Total techniques:  {results['total']}")
    log.info(f"  Generated:         {results['generated']}")
    log.info(f"  Failed:            {results['failed']}")


if __name__ == "__main__":
    main()
