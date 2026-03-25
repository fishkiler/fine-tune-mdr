#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — LLM Quality Scoring
============================================================================
Uses an LLM (Claude API) to score database records for training data quality.

Scoring dimensions (1-5 each):
  - Accuracy:     Is the information factually correct?
  - Completeness: Does the response fully address the question?
  - Clarity:      Is the response well-written and understandable?
  - Relevance:    Does the response stay on topic?
  - Usefulness:   Would this help a security analyst in practice?

quality_score = average of all dimensions (1.0-5.0)

Usage:
    python scripts/review_data.py --sample 100           # review 100 random records
    python scripts/review_data.py --domain cve --sample 500
    python scripts/review_data.py --all                   # review all unreviewed records
    python scripts/review_data.py --ids 1,2,3             # review specific records
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

from scripts.db_utils import DEFAULT_DB_PATH, DOMAINS, get_connection, compute_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Review Prompt
# ============================================================================

REVIEW_SYSTEM_PROMPT = """You are a data quality reviewer for a cybersecurity AI training dataset.
You will be given a user question and an assistant response from a training dataset.
Rate the assistant response on 5 dimensions using a 1-5 scale.

Scoring guide:
  1 = Very poor (wrong, incomplete, or harmful)
  2 = Poor (significant issues)
  3 = Adequate (acceptable but has room for improvement)
  4 = Good (solid response with minor issues)
  5 = Excellent (accurate, complete, clear, and useful)

You MUST respond with ONLY a JSON object in this exact format:
{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "relevance": <1-5>,
  "usefulness": <1-5>,
  "notes": "<brief explanation of any issues>"
}"""


def build_review_prompt(records: list[dict]) -> str:
    """Build a review prompt for a batch of records."""
    parts = []
    for i, rec in enumerate(records):
        parts.append(f"--- Record {rec['id']} (domain: {rec['domain']}) ---")
        parts.append(f"USER: {rec['user_message'][:2000]}")
        parts.append(f"ASSISTANT: {rec['assistant_message'][:3000]}")
        parts.append("")

    if len(records) == 1:
        return "\n".join(parts) + "\nRate this response. Respond with ONLY the JSON object."
    else:
        return (
            "\n".join(parts)
            + f"\nRate each of the {len(records)} responses above. "
            + "Respond with a JSON array of objects, one per record, in order. "
            + "Each object must have the fields: accuracy, completeness, clarity, relevance, usefulness, notes."
        )


# ============================================================================
# LLM Client
# ============================================================================

def _parse_scores_response(text: str) -> list[dict]:
    """Parse LLM response text into a list of score dicts."""
    text = text.strip()

    # Strip markdown code fences if present (e.g. ```json ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Handle both single object and array formats
    if text.startswith("["):
        return json.loads(text)
    else:
        return [json.loads(text)]


def review_with_claude(records: list[dict], model: str = "claude-sonnet-4-6") -> list[dict]:
    """Send records to Claude API for quality review."""
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_review_prompt(records)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return _parse_scores_response(text)

    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse LLM response as JSON: {e}")
        log.warning(f"Response text: {text[:500]}")
        return []
    except Exception as e:
        log.warning(f"API call failed: {e}")
        return []


def review_with_openrouter(records: list[dict], model: str) -> list[dict]:
    """Send records to OpenRouter API (OpenAI-compatible) for quality review."""
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
        timeout=60.0,
    )
    prompt = build_review_prompt(records)

    try:
        # Build extra params — disable reasoning for models that support it
        extra = {}
        if "grok" in model or "x-ai" in model:
            extra["extra_body"] = {"reasoning": {"enabled": False}}

        # Scale max_tokens with batch size (~200 tokens per record's JSON score)
        max_tok = max(2048, len(records) * 200)

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tok,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **extra,
        )
        text = response.choices[0].message.content.strip()
        return _parse_scores_response(text)

    except json.JSONDecodeError as e:
        log.warning(f"Failed to parse LLM response as JSON: {e}")
        log.warning(f"Response text: {text[:500]}")
        return []
    except Exception as e:
        log.warning(f"API call failed: {e}")
        return []


def review_records(records: list[dict], model: str, provider: str) -> list[dict]:
    """Route review to the appropriate provider."""
    if provider == "openrouter":
        return review_with_openrouter(records, model=model)
    else:
        return review_with_claude(records, model=model)


# ============================================================================
# Database Operations
# ============================================================================

def fetch_records_for_review(db_path: str, domain: str | None = None,
                              source: str | None = None,
                              sample_size: int | None = None,
                              record_ids: list[int] | None = None,
                              only_unreviewed: bool = True) -> list[dict]:
    """Fetch records from the database for review."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    conditions = []
    params = []

    if only_unreviewed:
        conditions.append("quality_score IS NULL")

    if domain:
        conditions.append("domain = ?")
        params.append(domain)

    if source:
        conditions.append("source = ?")
        params.append(source)

    if record_ids:
        placeholders = ",".join("?" * len(record_ids))
        conditions.append(f"id IN ({placeholders})")
        params.extend(record_ids)

    # Also prefer records that passed validation
    if only_unreviewed:
        conditions.append("(validation_status = 'pass' OR validation_status IS NULL)")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    if sample_size and not record_ids:
        query = f"SELECT id, domain, user_message, assistant_message FROM all_records {where} ORDER BY RANDOM() LIMIT ?"
        params.append(sample_size)
    else:
        query = f"SELECT id, domain, user_message, assistant_message FROM all_records {where}"

    cur.execute(query, params)
    records = [
        {"id": row[0], "domain": row[1], "user_message": row[2], "assistant_message": row[3]}
        for row in cur.fetchall()
    ]
    conn.close()
    return records


def save_review_scores(db_path: str, record_id: int, scores: dict,
                        reviewer: str) -> None:
    """Save quality scores for a record."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    dims = ["accuracy", "completeness", "clarity", "relevance", "usefulness"]
    dim_scores = {d: scores.get(d, 3) for d in dims}
    quality_score = sum(dim_scores.values()) / len(dims)
    now = datetime.now(timezone.utc).isoformat()

    cur.execute(
        "UPDATE all_records SET quality_score = ?, quality_scores = ?, "
        "quality_reviewed_at = ?, quality_reviewer = ? WHERE id = ?",
        (quality_score, json.dumps(dim_scores), now, reviewer, record_id),
    )
    conn.commit()
    conn.close()


# ============================================================================
# Main Runner
# ============================================================================

def run_review(db_path: str, domain: str | None = None,
               source: str | None = None,
               sample_size: int | None = None,
               record_ids: list[int] | None = None,
               model: str = "claude-sonnet-4-6",
               provider: str = "anthropic",
               batch_size: int = 5,
               dry_run: bool = False) -> dict:
    """Run LLM quality review on database records."""
    records = fetch_records_for_review(db_path, domain, source=source,
                                       sample_size=sample_size, record_ids=record_ids)

    if not records:
        log.info("No records to review.")
        return {"total": 0, "reviewed": 0, "avg_score": 0}

    log.info(f"Found {len(records)} records to review.")

    if dry_run:
        log.info("[DRY RUN] Would review these records with LLM.")
        for rec in records[:5]:
            log.info(f"  Record #{rec['id']} ({rec['domain']}): "
                     f"{rec['user_message'][:80]}...")
        return {"total": len(records), "reviewed": 0, "avg_score": 0}

    reviewer_tag = f"{provider}/{model}" if provider == "openrouter" else model
    reviewed = 0
    total_score = 0.0
    failed = 0

    # Process in batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        log.info(f"Reviewing batch {i // batch_size + 1} "
                 f"({len(batch)} records, {reviewed}/{len(records)} done)...")

        scores_list = review_records(batch, model=model, provider=provider)

        if len(scores_list) != len(batch):
            log.warning(f"Expected {len(batch)} scores, got {len(scores_list)}. "
                        f"Falling back to individual review.")
            # Fall back to reviewing one at a time
            for rec in batch:
                individual_scores = review_records([rec], model=model, provider=provider)
                if individual_scores:
                    save_review_scores(db_path, rec["id"], individual_scores[0], reviewer_tag)
                    dims = ["accuracy", "completeness", "clarity", "relevance", "usefulness"]
                    score = sum(individual_scores[0].get(d, 3) for d in dims) / len(dims)
                    total_score += score
                    reviewed += 1
                else:
                    failed += 1
                time.sleep(0.3)  # Rate limiting
        else:
            for rec, scores in zip(batch, scores_list):
                save_review_scores(db_path, rec["id"], scores, reviewer_tag)
                dims = ["accuracy", "completeness", "clarity", "relevance", "usefulness"]
                score = sum(scores.get(d, 3) for d in dims) / len(dims)
                total_score += score
                reviewed += 1

        time.sleep(0.5)  # Rate limiting between batches

    # Update stats
    conn = get_connection(db_path)
    compute_stats(conn)
    conn.close()

    avg_score = total_score / reviewed if reviewed > 0 else 0
    return {
        "total": len(records),
        "reviewed": reviewed,
        "failed": failed,
        "avg_score": round(avg_score, 2),
    }


# ============================================================================
# Main
# ============================================================================

OPENROUTER_MODELS = {
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    "llama-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-70b": "meta-llama/llama-3.1-70b-instruct",
    "deepseek": "deepseek/deepseek-chat-v3-0324",
    "qwen-32b": "qwen/qwen-2.5-32b-instruct",
    "mistral-small": "mistralai/mistral-small-3.1-24b-instruct",
}


def main():
    model_help = (
        "Model to use. For Anthropic: claude-sonnet-4-6, claude-haiku-4-5-20251001. "
        "For OpenRouter: use full model ID (e.g. google/gemini-2.0-flash-001) or a shortcut: "
        + ", ".join(f"{k}" for k in OPENROUTER_MODELS.keys())
    )

    parser = argparse.ArgumentParser(description="LLM Quality Review for MDR dataset")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--domain", choices=DOMAINS, help="Only review this domain")
    parser.add_argument("--source", type=str, help="Only review records from this source (e.g. stix_graph)")
    parser.add_argument("--sample", type=int, help="Random sample size to review")
    parser.add_argument("--all", action="store_true", help="Review all unreviewed records")
    parser.add_argument("--ids", type=str, help="Comma-separated record IDs to review")
    parser.add_argument("--model", default="claude-sonnet-4-6", help=model_help)
    parser.add_argument("--provider", choices=["anthropic", "openrouter"],
                        help="API provider (auto-detected from model if not set)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Records per API call (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be reviewed")
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Resolve model shortcuts
    model = args.model
    if model in OPENROUTER_MODELS:
        model = OPENROUTER_MODELS[model]

    # Auto-detect provider from model name (slash = OpenRouter)
    provider = args.provider
    if not provider:
        provider = "openrouter" if "/" in model else "anthropic"

    record_ids = None
    if args.ids:
        record_ids = [int(x.strip()) for x in args.ids.split(",")]

    sample_size = args.sample
    if args.all:
        sample_size = None
    elif not sample_size and not record_ids:
        sample_size = 100  # Default sample

    log.info("=" * 60)
    log.info("  MDR LLM Quality Review")
    log.info("=" * 60)
    log.info(f"  Provider: {provider}")
    log.info(f"  Model:    {model}")
    if args.domain:
        log.info(f"  Domain:   {args.domain}")
    if args.source:
        log.info(f"  Source:   {args.source}")
    if sample_size:
        log.info(f"  Sample:   {sample_size}")
    if record_ids:
        log.info(f"  IDs:      {record_ids}")
    log.info("")

    results = run_review(
        db_path=args.db,
        domain=args.domain,
        source=args.source,
        sample_size=sample_size,
        record_ids=record_ids,
        model=model,
        provider=provider,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    log.info("")
    log.info("=" * 60)
    log.info("  REVIEW RESULTS")
    log.info("=" * 60)
    log.info(f"  Total records:  {results['total']}")
    log.info(f"  Reviewed:       {results['reviewed']}")
    if results.get("failed"):
        log.info(f"  Failed:         {results['failed']}")
    log.info(f"  Avg score:      {results['avg_score']:.2f} / 5.0")


if __name__ == "__main__":
    main()
