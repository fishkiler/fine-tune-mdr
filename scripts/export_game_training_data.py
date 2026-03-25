#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Game Adapter Training Data Export
============================================================================
Exports game frames from the database into Qwen3.5-9B VLM conversation
format JSONL, applying action balancing and template-based reasoning.

Follows export_training_data.py patterns:
  - Quality filtering
  - Action balancing (downsample NONE, oversample decision points)
  - Episode-boundary train/test split (prevents data leakage)
  - JSONL output with export manifest

Usage:
    python scripts/export_game_training_data.py --game pacman
    python scripts/export_game_training_data.py --game pacman --test-ratio 0.15
    python scripts/export_game_training_data.py --game pacman --no-balance
============================================================================
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml

from scripts.db_utils import DEFAULT_DB_PATH, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Reasoning Templates (Strategy A — template-based)
# ============================================================================
# Keyed by action. Randomly selected during export. The model learns the
# visual→action mapping; reasoning text is scaffolding for chat format.

REASONING_TEMPLATES = {
    "NONE": [
        "Pac-Man is moving through a clear corridor. No direction change needed.\n\nAction: NONE",
        "The current path is safe with pellets ahead. Continue in the current direction.\n\nAction: NONE",
        "No ghosts nearby and the path is open. Maintain current heading.\n\nAction: NONE",
        "Pac-Man is on a straight stretch collecting pellets. Keep going.\n\nAction: NONE",
        "The corridor ahead is clear. No need to change direction.\n\nAction: NONE",
    ],
    "UP": [
        "There's an opening upward with pellets to collect. Turn up.\n\nAction: UP",
        "A ghost is approaching from below. Move up to avoid it.\n\nAction: UP",
        "The upper corridor has uncollected pellets. Head up.\n\nAction: UP",
        "Turning up at this intersection to reach more pellets.\n\nAction: UP",
        "Moving up to navigate toward the upper section of the maze.\n\nAction: UP",
    ],
    "DOWN": [
        "There's a path downward with pellets available. Turn down.\n\nAction: DOWN",
        "Moving down to avoid a ghost approaching from above.\n\nAction: DOWN",
        "The lower corridor has pellets to collect. Head down.\n\nAction: DOWN",
        "Turning down at this junction to reach the lower maze area.\n\nAction: DOWN",
        "A ghost is blocking the upper path. Move down to safety.\n\nAction: DOWN",
    ],
    "LEFT": [
        "Turning left at the intersection to collect pellets.\n\nAction: LEFT",
        "A ghost is visible to the right. Move left to avoid danger.\n\nAction: LEFT",
        "The left corridor has uncollected pellets. Turn left.\n\nAction: LEFT",
        "Moving left to navigate around a ghost ahead.\n\nAction: LEFT",
        "There's an opening to the left with a clear path. Go left.\n\nAction: LEFT",
    ],
    "RIGHT": [
        "Turning right at the intersection to collect pellets.\n\nAction: RIGHT",
        "The right corridor is clear with pellets ahead. Move right.\n\nAction: RIGHT",
        "A ghost is approaching from the left. Turn right to escape.\n\nAction: RIGHT",
        "Moving right to reach uncollected pellets in the east corridor.\n\nAction: RIGHT",
        "The path to the right is open and safe. Head right.\n\nAction: RIGHT",
    ],
}


def get_reasoning(action: str, rng: random.Random) -> str:
    """Select a random reasoning template for the given action."""
    templates = REASONING_TEMPLATES.get(action, REASONING_TEMPLATES["NONE"])
    return rng.choice(templates)


# ============================================================================
# Data Loading
# ============================================================================

def fetch_game_frames(db_path: str, game_name: str) -> list[dict]:
    """Fetch all non-excluded frames for a game from the database."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT id, game_name, frame_path, action_label, action_id, "
        "episode_id, frame_index, cumulative_score "
        "FROM game_frames "
        "WHERE game_name = ? AND excluded = 0 "
        "ORDER BY episode_id, frame_index",
        (game_name,),
    )

    frames = []
    for row in cur.fetchall():
        frames.append({
            "id": row[0],
            "game_name": row[1],
            "frame_path": row[2],
            "action_label": row[3],
            "action_id": row[4],
            "episode_id": row[5],
            "frame_index": row[6],
            "cumulative_score": row[7],
        })

    conn.close()
    return frames


# ============================================================================
# Action Balancing
# ============================================================================

def apply_action_balancing(
    frames: list[dict],
    max_noop_ratio: float = 0.25,
    decision_oversample: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Balance action distribution by downsampling NONE and oversampling decision points.

    Decision points = frames where action changes from previous frame's action.
    """
    rng = random.Random(seed)

    # Tag decision points (action differs from previous frame)
    for i, frame in enumerate(frames):
        if i == 0 or frame["episode_id"] != frames[i - 1]["episode_id"]:
            frame["is_decision"] = True  # First frame of episode is always a decision
        else:
            frame["is_decision"] = frame["action_label"] != frames[i - 1]["action_label"]

    # Separate NONE vs non-NONE frames
    none_frames = [f for f in frames if f["action_label"] == "NONE"]
    action_frames = [f for f in frames if f["action_label"] != "NONE"]

    # Downsample NONE if it exceeds max_noop_ratio
    total = len(frames)
    max_none = int(total * max_noop_ratio)
    if len(none_frames) > max_none:
        # Prefer keeping decision-point NONE frames
        none_decisions = [f for f in none_frames if f["is_decision"]]
        none_others = [f for f in none_frames if not f["is_decision"]]
        rng.shuffle(none_others)
        keep = max(0, max_none - len(none_decisions))
        none_frames = none_decisions + none_others[:keep]
        log.info(f"  NONE downsampled: {total - len(action_frames):,} -> {len(none_frames):,}")

    # Oversample decision points in action frames
    decision_frames = [f for f in action_frames if f["is_decision"]]
    non_decision_frames = [f for f in action_frames if not f["is_decision"]]

    oversampled = non_decision_frames + none_frames
    for _ in range(decision_oversample):
        oversampled.extend(decision_frames)

    rng.shuffle(oversampled)
    log.info(f"  Decision points oversampled {decision_oversample}x: "
             f"{len(decision_frames):,} decision frames -> {len(decision_frames) * decision_oversample:,}")
    log.info(f"  Balanced dataset: {len(oversampled):,} frames")

    return oversampled


# ============================================================================
# Conversation Formatting
# ============================================================================

def frame_to_conversation(
    frame: dict,
    system_prompt: str,
    frames_base_dir: str,
    rng: random.Random,
) -> dict:
    """Convert a frame record to a Qwen3.5-9B VLM conversation dict."""
    action = frame["action_label"]
    reasoning = get_reasoning(action, rng)

    # Build image path relative to training data location
    image_path = frame["frame_path"]

    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What action should Pac-Man take?"},
                ],
            },
            {
                "role": "assistant",
                "content": reasoning,
            },
        ]
    }


# ============================================================================
# Train/Test Split
# ============================================================================

def split_by_episode(
    frames: list[dict],
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split frames into train/test by episode boundaries.

    Splitting by episode prevents data leakage from consecutive similar frames.
    """
    rng = random.Random(seed)

    # Group by episode
    episodes: dict[int, list[dict]] = {}
    for f in frames:
        ep = f["episode_id"]
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(f)

    episode_ids = sorted(episodes.keys())
    rng.shuffle(episode_ids)

    # Assign episodes to test until we reach target ratio
    total_frames = len(frames)
    test_target = int(total_frames * test_ratio)
    test_count = 0
    test_episodes = set()

    for ep_id in episode_ids:
        if test_count >= test_target:
            break
        test_episodes.add(ep_id)
        test_count += len(episodes[ep_id])

    train = [f for f in frames if f["episode_id"] not in test_episodes]
    test = [f for f in frames if f["episode_id"] in test_episodes]

    return train, test


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export game frames to VLM training JSONL")
    parser.add_argument("--game", required=True, help="Game name (e.g. pacman)")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/games/<game>/training)")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="Fraction of data for test set (default: 0.10)")
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip action balancing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    game = args.game
    output_dir = Path(args.output_dir) if args.output_dir else _project_root / "data" / "games" / game / "training"

    # Load game config
    system_prompt = (
        f"You are TARS, playing {game.title()}. Analyze the game frame and choose an action.\n"
        f"ACTIONS: NONE (continue current direction), UP, DOWN, LEFT, RIGHT\n"
        f"Respond with brief reasoning followed by: Action: <ACTION>"
    )
    max_noop_ratio = 0.25
    decision_oversample = 3

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        game_cfg = cfg.get("game_adapters", {}).get("games", {}).get(game, {})
        if game_cfg.get("system_prompt"):
            system_prompt = game_cfg["system_prompt"]
        balance_cfg = game_cfg.get("action_balance", {})
        max_noop_ratio = balance_cfg.get("max_noop_ratio", max_noop_ratio)
        decision_oversample = balance_cfg.get("decision_point_oversample", decision_oversample)

    t0 = time.time()
    log.info("=" * 60)
    log.info("  Game Training Data Export")
    log.info("=" * 60)
    log.info(f"  Game: {game}")
    log.info(f"  Test ratio: {args.test_ratio}")
    log.info(f"  Balance: {'no' if args.no_balance else 'yes'}")
    log.info(f"  Output: {output_dir}")
    log.info("")

    # Fetch frames
    frames = fetch_game_frames(args.db, game)
    if not frames:
        log.error(f"No frames found for game '{game}' in database")
        sys.exit(1)
    log.info(f"Total frames in database: {len(frames):,}")

    # Action distribution before balancing
    action_counts: dict[str, int] = {}
    for f in frames:
        action_counts[f["action_label"]] = action_counts.get(f["action_label"], 0) + 1
    log.info("Action distribution (raw):")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / len(frames) * 100
        log.info(f"  {action:<10} {count:>8,}  ({pct:.1f}%)")
    log.info("")

    # Apply action balancing
    if not args.no_balance:
        log.info("Applying action balancing:")
        frames = apply_action_balancing(
            frames,
            max_noop_ratio=max_noop_ratio,
            decision_oversample=decision_oversample,
            seed=args.seed,
        )
        log.info("")

    # Split by episode
    train_frames, test_frames = split_by_episode(frames, args.test_ratio, args.seed)
    log.info(f"Split: train={len(train_frames):,}, test={len(test_frames):,}")

    # Get episode counts
    train_episodes = len(set(f["episode_id"] for f in train_frames))
    test_episodes = len(set(f["episode_id"] for f in test_frames))
    log.info(f"Episodes: train={train_episodes}, test={test_episodes}")

    # Convert to conversations and write JSONL
    rng = random.Random(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_base = str(_project_root / "data" / "games" / game)

    for split_name, split_frames in [("train", train_frames), ("test", test_frames)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for frame in split_frames:
                conv = frame_to_conversation(frame, system_prompt, frames_base, rng)
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(split_frames):,} examples to {out_path}")

    # Write export manifest
    manifest = {
        "game": game,
        "total_raw_frames": len(fetch_game_frames(args.db, game)),
        "train_examples": len(train_frames),
        "test_examples": len(test_frames),
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "action_balanced": not args.no_balance,
        "max_noop_ratio": max_noop_ratio,
        "decision_oversample": decision_oversample,
        "reasoning_strategy": "template_v1",
        "seed": args.seed,
    }
    manifest_path = output_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\nExport complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
