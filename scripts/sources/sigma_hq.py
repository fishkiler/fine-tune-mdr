#!/usr/bin/env python3
"""
============================================================================
SigmaHQ Rule Downloader & Indexer
============================================================================
Downloads Sigma detection rules from the SigmaHQ GitHub repository and
indexes them by MITRE ATT&CK technique ID for use in training data.

Rules are cached locally in data/sources/sigma_cache/ to avoid repeated
GitHub API calls.

Usage:
    python -m scripts.sources.sigma_hq
    python -m scripts.sources.sigma_hq --cache-dir data/sources/sigma_cache
============================================================================
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "sources" / "sigma_cache"

# GitHub API endpoint for SigmaHQ rule tree
SIGMAHQ_TREE_URL = "https://api.github.com/repos/SigmaHQ/sigma/git/trees/master?recursive=1"
SIGMAHQ_RAW_BASE = "https://raw.githubusercontent.com/SigmaHQ/sigma/master/"

# Status and level preferences for rule selection
STATUS_PRIORITY = {"stable": 0, "test": 1, "experimental": 2, "deprecated": 3}
LEVEL_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}


def fetch_rule_tree() -> list[str]:
    """Fetch the full file tree from SigmaHQ and return rule YAML paths."""
    log.info("Fetching SigmaHQ repository tree...")
    req = Request(SIGMAHQ_TREE_URL, headers={
        "User-Agent": "MDR-Training-Pipeline/1.0",
        "Accept": "application/vnd.github.v3+json",
    })
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    paths = []
    for item in data.get("tree", []):
        p = item.get("path", "")
        if p.startswith("rules/") and p.endswith(".yml"):
            paths.append(p)

    log.info(f"Found {len(paths)} rule files in SigmaHQ tree.")
    return paths


def download_rule(path: str, cache_dir: Path) -> str | None:
    """Download a single Sigma rule YAML, using cache if available."""
    cache_file = cache_dir / path.replace("/", "_")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    url = SIGMAHQ_RAW_BASE + path
    try:
        req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
        with urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        cache_file.write_text(content, encoding="utf-8")
        return content
    except (URLError, TimeoutError) as e:
        log.debug(f"Failed to download {path}: {e}")
        return None


def extract_technique_ids(rule_text: str) -> list[str]:
    """Extract ATT&CK technique IDs from a Sigma rule's tags field."""
    technique_ids = []
    in_tags = False
    for line in rule_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("tags:"):
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("- "):
                tag = stripped[2:].strip()
                # Match attack.tXXXX or attack.tXXXX.XXX
                m = re.match(r"attack\.(t\d{4}(?:\.\d{3})?)$", tag, re.IGNORECASE)
                if m:
                    technique_ids.append(m.group(1).upper())
            elif stripped and not stripped.startswith("#"):
                in_tags = False
    return technique_ids


def extract_rule_metadata(rule_text: str) -> dict:
    """Extract status and level from a Sigma rule for quality ranking."""
    status = "experimental"
    level = "medium"
    for line in rule_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("status:"):
            status = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("level:"):
            level = stripped.split(":", 1)[1].strip()
    return {"status": status, "level": level}


def rule_quality_score(metadata: dict) -> int:
    """Score a rule for selection (lower is better)."""
    s = STATUS_PRIORITY.get(metadata["status"], 5)
    l = LEVEL_PRIORITY.get(metadata["level"], 5)
    return s * 10 + l


def build_full_sigma_index(cache_dir: Path | None = None,
                           max_rules: int = 0) -> list[dict]:
    """Returns ALL sigma rules with metadata and technique mappings.

    Unlike build_sigma_index() which picks 1 best rule per technique,
    this returns every rule that has at least one technique tag.

    Returns list of dicts:
      {"path": str, "title": str, "status": str, "level": str,
       "technique_ids": [str], "yaml_content": str}
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    rule_paths = fetch_rule_tree()
    if max_rules > 0:
        rule_paths = rule_paths[:max_rules]

    results = []
    downloaded = 0
    skipped = 0
    no_technique = 0

    for i, path in enumerate(rule_paths):
        content = download_rule(path, cache_dir)
        if content is None:
            skipped += 1
            continue
        downloaded += 1

        tech_ids = extract_technique_ids(content)
        if not tech_ids:
            no_technique += 1
            continue

        meta = extract_rule_metadata(content)

        # Extract title
        title = path.rsplit("/", 1)[-1].replace(".yml", "")
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("title:"):
                title = stripped.split(":", 1)[1].strip()
                break

        results.append({
            "path": path,
            "title": title,
            "status": meta["status"],
            "level": meta["level"],
            "technique_ids": tech_ids,
            "yaml_content": content,
        })

        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i+1}/{len(rule_paths)} rules, "
                     f"{len(results)} with techniques...")

    log.info(f"Full Sigma index: {downloaded} downloaded, {skipped} skipped, "
             f"{no_technique} without technique tags, {len(results)} rules indexed.")

    # Cache metadata index (without YAML content) for quick reload
    full_index_file = cache_dir / "_full_rule_index.json"
    index_meta = [
        {"path": r["path"], "title": r["title"], "status": r["status"],
         "level": r["level"], "technique_ids": r["technique_ids"]}
        for r in results
    ]
    full_index_file.write_text(json.dumps(index_meta, indent=2), encoding="utf-8")

    return results


def build_sigma_index(cache_dir: Path | None = None,
                      max_rules: int = 0) -> dict[str, str]:
    """Download SigmaHQ rules and build a {technique_id: best_rule_yaml} index.

    Args:
        cache_dir: Directory to cache downloaded rules.
        max_rules: Max rules to download (0 = all). For testing.

    Returns:
        Dict mapping technique IDs (e.g. "T1053.005") to the best matching
        Sigma rule YAML string.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached index
    index_file = cache_dir / "_technique_index.json"
    rules_dir_marker = cache_dir / "_tree_fetched"

    rule_paths = fetch_rule_tree()
    if max_rules > 0:
        rule_paths = rule_paths[:max_rules]

    # Collect all candidates per technique
    candidates: dict[str, list[tuple[int, str, str]]] = {}  # tech_id -> [(score, path, content)]
    downloaded = 0
    skipped = 0

    for i, path in enumerate(rule_paths):
        content = download_rule(path, cache_dir)
        if content is None:
            skipped += 1
            continue
        downloaded += 1

        tech_ids = extract_technique_ids(content)
        if not tech_ids:
            continue

        meta = extract_rule_metadata(content)
        score = rule_quality_score(meta)

        for tid in tech_ids:
            if tid not in candidates:
                candidates[tid] = []
            candidates[tid].append((score, path, content))

        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i+1}/{len(rule_paths)} rules, "
                     f"{len(candidates)} techniques matched...")

    # Select best rule per technique
    index: dict[str, str] = {}
    for tid, entries in candidates.items():
        entries.sort(key=lambda x: x[0])  # lowest score = best
        index[tid] = entries[0][2]  # the rule content

    log.info(f"SigmaHQ index: {downloaded} rules downloaded, {skipped} skipped, "
             f"{len(index)} techniques covered.")

    # Cache the technique-to-path mapping (not the full content)
    path_index = {}
    for tid, entries in candidates.items():
        entries.sort(key=lambda x: x[0])
        path_index[tid] = entries[0][1]
    index_file.write_text(json.dumps(path_index, indent=2), encoding="utf-8")

    return index


def main():
    parser = argparse.ArgumentParser(description="Download and index SigmaHQ rules")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Cache directory for downloaded rules")
    parser.add_argument("--max-rules", type=int, default=0,
                        help="Max rules to download (0 = all)")
    parser.add_argument("--full", action="store_true",
                        help="Build full index (all rules, not just best-per-technique)")
    args = parser.parse_args()

    if args.full:
        results = build_full_sigma_index(cache_dir=args.cache_dir, max_rules=args.max_rules)

        # Compute stats
        techniques = set()
        status_counts: dict[str, int] = {}
        level_counts: dict[str, int] = {}
        for r in results:
            techniques.update(r["technique_ids"])
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1

        print(f"\nSigmaHQ Full Index Summary:")
        print(f"  Total rules with technique tags: {len(results)}")
        print(f"  Unique techniques covered: {len(techniques)}")
        print(f"  By status: {dict(sorted(status_counts.items(), key=lambda x: -x[1]))}")
        print(f"  By level: {dict(sorted(level_counts.items(), key=lambda x: -x[1]))}")
    else:
        index = build_sigma_index(cache_dir=args.cache_dir, max_rules=args.max_rules)

        print(f"\nSigmaHQ Index Summary:")
        print(f"  Techniques covered: {len(index)}")
        sample_ids = sorted(index.keys())[:10]
        print(f"  Sample technique IDs: {', '.join(sample_ids)}")


if __name__ == "__main__":
    main()
