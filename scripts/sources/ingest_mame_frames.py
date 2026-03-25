#!/usr/bin/env python3
"""
============================================================================
MAME Frame Ingestion Script
============================================================================
Imports captured MAME gameplay frames and action labels into the database.
Follows the same patterns as fetch_attack_logs.py — deduplication by
perceptual hash, incremental ingestion, batch inserts.

Usage:
    python -m scripts.sources.ingest_mame_frames --game pacman
    python -m scripts.sources.ingest_mame_frames --game pacman --frames-dir data/games/pacman
    python -m scripts.sources.ingest_mame_frames --game pacman --dry-run
============================================================================
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.db_utils import DEFAULT_DB_PATH, get_connection, migrate_schema

# Pac-Man action space (matches config.yaml and capture script)
PACMAN_ACTIONS = ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]


def convert_ppm_to_png(ppm_path: Path) -> Path:
    """Convert a PPM frame to PNG (10-12x smaller) and return the PNG path."""
    from PIL import Image
    png_path = ppm_path.with_suffix(".png")
    if png_path.exists():
        return png_path
    img = Image.open(ppm_path)
    img.save(png_path, "PNG")
    ppm_path.unlink()  # Remove the PPM to save disk space
    return png_path


def compute_frame_hash(frame_path: Path) -> str:
    """Compute perceptual hash of a frame image for deduplication."""
    from PIL import Image
    try:
        import imagehash
        img = Image.open(frame_path)
        return str(imagehash.phash(img))
    except ImportError:
        # Fallback: use file content SHA-256 if imagehash not installed
        import hashlib
        data = frame_path.read_bytes()
        return hashlib.sha256(data).hexdigest()


def parse_actions_csv(csv_path: Path) -> dict[tuple[int, int], dict]:
    """Parse actions.csv into a lookup keyed by (episode, frame_index)."""
    lookup = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["episode"]), int(row["frame_index"]))
            lookup[key] = {
                "frame_path": row["frame_path"],
                "action": row["action"],
                "timestamp": float(row.get("timestamp", 0)),
            }
    return lookup


def get_action_id(action: str, actions: list[str]) -> int:
    """Map action name to numeric ID."""
    try:
        return actions.index(action)
    except ValueError:
        log.warning(f"Unknown action '{action}', mapping to 0 (NONE)")
        return 0


def ingest_frames(
    game_name: str,
    frames_dir: Path,
    db_path: str,
    actions: list[str],
    dry_run: bool = False,
    dedup_threshold: int = 4,
    batch_size: int = 500,
) -> dict:
    """Ingest frames from disk into the database.

    Returns stats dict with counts.
    """
    csv_path = frames_dir / "actions.csv"
    if not csv_path.exists():
        log.error(f"actions.csv not found at {csv_path}")
        sys.exit(1)

    log.info(f"Parsing {csv_path}...")
    action_lookup = parse_actions_csv(csv_path)
    log.info(f"Found {len(action_lookup):,} action entries")

    # Connect to database and ensure schema is current
    conn = get_connection(db_path)
    migrate_schema(conn)
    cur = conn.cursor()

    # Get existing frame hashes for dedup
    cur.execute("SELECT frame_hash FROM game_frames WHERE game_name = ?", (game_name,))
    existing_hashes = {row[0] for row in cur.fetchall()}
    log.info(f"Existing frames in DB: {len(existing_hashes):,}")

    # Ensure game_adapters row exists
    cur.execute(
        "INSERT OR IGNORE INTO game_adapters (game_name, base_model, action_space, status) "
        "VALUES (?, ?, ?, ?)",
        (game_name, "Qwen/Qwen3.5-9B", str(actions), "collecting"),
    )

    # Scan frames
    stats = {
        "total_csv_entries": len(action_lookup),
        "frames_found": 0,
        "frames_ingested": 0,
        "duplicates_skipped": 0,
        "missing_files": 0,
        "action_distribution": {a: 0 for a in actions},
    }

    batch = []

    for (episode, frame_idx), entry in sorted(action_lookup.items()):
        frame_path = frames_dir / entry["frame_path"]

        if not frame_path.exists():
            # Try PPM variant (capture script saves as PPM)
            ppm_path = frame_path.with_suffix(".ppm")
            if not ppm_path.exists() and frame_path.suffix == ".ppm":
                ppm_path = frame_path
            if ppm_path.exists():
                frame_path = ppm_path
            else:
                stats["missing_files"] += 1
                continue

        stats["frames_found"] += 1

        # Convert PPM to PNG if needed (10-12x smaller)
        if frame_path.suffix == ".ppm":
            frame_path = convert_ppm_to_png(frame_path)

        # Compute perceptual hash
        frame_hash = compute_frame_hash(frame_path)

        # Check for duplicates (exact hash match)
        if frame_hash in existing_hashes:
            stats["duplicates_skipped"] += 1
            continue

        action = entry["action"]
        action_id = get_action_id(action, actions)
        # Use current frame_path (may have been converted from PPM to PNG)
        relative_path = str(frame_path.relative_to(frames_dir))

        batch.append((
            game_name,
            relative_path,
            frame_hash,
            action,
            action_id,
            episode,
            frame_idx,
            None,  # cumulative_score (not tracked in v1 capture)
        ))

        existing_hashes.add(frame_hash)
        stats["frames_ingested"] += 1
        stats["action_distribution"][action] = stats["action_distribution"].get(action, 0) + 1

        # Batch insert
        if len(batch) >= batch_size:
            if not dry_run:
                cur.executemany(
                    "INSERT INTO game_frames "
                    "(game_name, frame_path, frame_hash, action_label, action_id, "
                    "episode_id, frame_index, cumulative_score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
            log.info(f"  Ingested {stats['frames_ingested']:,} frames...")
            batch = []

    # Final batch
    if batch and not dry_run:
        cur.executemany(
            "INSERT INTO game_frames "
            "(game_name, frame_path, frame_hash, action_label, action_id, "
            "episode_id, frame_index, cumulative_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    # Update game_adapters with total frame count
    if not dry_run:
        cur.execute(
            "SELECT COUNT(*) FROM game_frames WHERE game_name = ?", (game_name,)
        )
        total_frames = cur.fetchone()[0]
        cur.execute(
            "UPDATE game_adapters SET total_frames = ? WHERE game_name = ?",
            (total_frames, game_name),
        )
        conn.commit()

    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Ingest MAME gameplay frames into database")
    parser.add_argument("--game", required=True, help="Game name (e.g. pacman)")
    parser.add_argument("--frames-dir", default=None,
                        help="Directory containing frames/ and actions.csv (default: data/games/<game>)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--dry-run", action="store_true", help="Parse and hash without writing to DB")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    args = parser.parse_args()

    game = args.game
    frames_dir = Path(args.frames_dir) if args.frames_dir else _project_root / "data" / "games" / game

    if not frames_dir.exists():
        log.error(f"Frames directory not found: {frames_dir}")
        sys.exit(1)

    # Determine action space from config or use default
    actions = PACMAN_ACTIONS  # Default for pacman
    try:
        import yaml
        config_path = _project_root / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            game_cfg = cfg.get("game_adapters", {}).get("games", {}).get(game, {})
            if game_cfg.get("actions"):
                actions = game_cfg["actions"]
    except Exception:
        pass

    t0 = time.time()
    log.info("=" * 60)
    log.info("  MAME Frame Ingestion")
    log.info("=" * 60)
    log.info(f"  Game: {game}")
    log.info(f"  Frames dir: {frames_dir}")
    log.info(f"  Actions: {actions}")
    log.info(f"  Dry run: {args.dry_run}")
    log.info("")

    stats = ingest_frames(
        game_name=game,
        frames_dir=frames_dir,
        db_path=args.db,
        actions=actions,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - t0
    log.info("")
    log.info("Results:")
    log.info(f"  CSV entries:       {stats['total_csv_entries']:,}")
    log.info(f"  Frames found:      {stats['frames_found']:,}")
    log.info(f"  Frames ingested:   {stats['frames_ingested']:,}")
    log.info(f"  Duplicates skipped:{stats['duplicates_skipped']:,}")
    log.info(f"  Missing files:     {stats['missing_files']:,}")
    log.info("")
    log.info("Action distribution:")
    for action, count in sorted(stats["action_distribution"].items(), key=lambda x: -x[1]):
        pct = count / max(stats["frames_ingested"], 1) * 100
        log.info(f"  {action:<10} {count:>8,}  ({pct:.1f}%)")
    log.info(f"\nIngestion complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
