#!/usr/bin/env python3
"""
Fetch JAT Ms. Pac-Man dataset from HuggingFace and convert to our training format.

Downloads full episodes of Ms. Pac-Man gameplay (84x84 RGBA frames + actions + rewards)
from jat-project/jat-dataset, maps 9-action Ms. Pac-Man to 5-action Pac-Man space,
filters to decision-point frames, and exports as JSONL + PNG frames.

Usage:
    python -m scripts.sources.fetch_jat_mspacman
    python -m scripts.sources.fetch_jat_mspacman --max-episodes 100 --max-frames 50000
"""

import argparse
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Ms. Pac-Man ALE action space → Pac-Man 5-action mapping
MSPACMAN_TO_PACMAN = {
    0: "NONE",      # NOOP
    1: "UP",         # UP
    2: "RIGHT",      # RIGHT
    3: "LEFT",       # LEFT
    4: "DOWN",       # DOWN
    5: "UP",         # UPRIGHT → UP (dominant direction)
    6: "UP",         # UPLEFT → UP
    7: "DOWN",       # DOWNRIGHT → DOWN
    8: "DOWN",       # DOWNLEFT → DOWN
}

SYSTEM_PROMPT = (
    "You are TARS, playing Pac-Man. Analyze the game frame and choose an action. "
    "ACTIONS: NONE (continue current direction), UP, DOWN, LEFT, RIGHT "
    "Respond with brief reasoning followed by: Action: <ACTION>\n"
)

# Reasoning templates keyed by action + context
# More diverse than the original 25 templates
REASONING_TEMPLATES = {
    "NONE": [
        "The current path is clear with pellets ahead. Continue forward.",
        "No threats nearby — keep moving in the current direction.",
        "Pac-Man is mid-corridor with pellets to collect. No change needed.",
        "Safe to continue — ghosts are in other sections of the maze.",
        "Staying on course to clear this corridor of pellets.",
    ],
    "UP": [
        "There's an opening upward with pellets to collect. Head up.",
        "Moving up to reach the upper corridor where pellets remain.",
        "A ghost is approaching from below — move up to escape.",
        "The upper path has uncollected pellets. Turn up at this junction.",
        "Heading up to avoid the ghost and collect pellets above.",
        "Moving upward through the maze to clear this section.",
    ],
    "DOWN": [
        "The lower corridor has pellets remaining. Move down.",
        "Heading down to collect pellets in the lower section.",
        "A ghost is visible above — move down to create distance.",
        "The downward path is clear. Turn down at this intersection.",
        "Moving down to reach uncollected pellets below.",
        "Dropping down to avoid the approaching ghost.",
    ],
    "LEFT": [
        "The left corridor has uncollected pellets. Turn left.",
        "Moving left to reach more pellets in the western section.",
        "A ghost is approaching from the right. Turn left to escape.",
        "The left path is clear with pellets ahead. Go left.",
        "Heading left at this junction to collect remaining pellets.",
        "Turning left to avoid danger and collect pellets.",
    ],
    "RIGHT": [
        "The right corridor is clear with pellets ahead. Move right.",
        "Heading right to collect pellets in the eastern section.",
        "A ghost is visible to the left. Turn right to get away.",
        "Moving right at this intersection to reach more pellets.",
        "The rightward path has uncollected pellets. Go right.",
        "Turning right to escape the ghost and pick up pellets.",
    ],
}

# Context-aware templates for high-reward frames (eating ghost/fruit)
REWARD_TEMPLATES = {
    "UP": "A power pellet effect is active — moving up aggressively to chase ghosts.",
    "DOWN": "Chasing a vulnerable ghost downward for bonus points.",
    "LEFT": "A blue ghost is nearby to the left — go eat it for points.",
    "RIGHT": "Pursuing a vulnerable ghost to the right for bonus points.",
    "NONE": "Just scored points — continuing in the same direction.",
}


def generate_reasoning(action: str, reward: float, prev_action: str) -> str:
    """Generate context-aware reasoning text for a frame."""
    if reward > 100:
        return REWARD_TEMPLATES.get(action, random.choice(REASONING_TEMPLATES[action]))
    if action != prev_action and prev_action:
        # Direction change — decision point
        templates = REASONING_TEMPLATES[action]
        return random.choice(templates)
    return random.choice(REASONING_TEMPLATES[action])


def is_decision_frame(idx: int, actions: list, rewards: list, sample_rate: int = 4) -> bool:
    """Determine if a frame is worth keeping for training.

    Keeps: direction changes, reward events, and every Nth frame.
    Skips: repetitive mid-corridor frames.
    """
    if idx == 0:
        return True
    # Direction change
    if actions[idx] != actions[idx - 1]:
        return True
    # Reward event (eating pellet, ghost, fruit)
    if rewards[idx] > 0:
        return True
    # Regular sampling
    if idx % sample_rate == 0:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Fetch JAT Ms. Pac-Man dataset")
    parser.add_argument("--max-episodes", type=int, default=0, help="Max episodes (0=all)")
    parser.add_argument("--max-frames", type=int, default=80000, help="Max total frames")
    parser.add_argument("--sample-rate", type=int, default=4, help="Keep every Nth frame (besides decision frames)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test set fraction")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "games" / "pacman" / "jat-mspacman"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading JAT Ms. Pac-Man dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("jat-project/jat-dataset", "atari-mspacman", split="train", streaming=True)

    all_examples = []
    action_counts = Counter()
    total_raw_frames = 0
    episode_count = 0
    total_reward = 0

    for ep_idx, episode in enumerate(ds):
        if args.max_episodes > 0 and ep_idx >= args.max_episodes:
            break

        images = episode["image_observations"]
        actions = episode["discrete_actions"]
        rewards = episode["rewards"]
        n_frames = len(actions)
        total_raw_frames += n_frames
        total_reward += sum(rewards)
        episode_count += 1

        prev_action = None
        for frame_idx in range(n_frames):
            if len(all_examples) >= args.max_frames:
                break

            mapped_action = MSPACMAN_TO_PACMAN[actions[frame_idx]]

            if not is_decision_frame(frame_idx, actions, rewards, args.sample_rate):
                prev_action = mapped_action
                continue

            # Save frame image
            img = images[frame_idx]
            if img.mode == "RGBA":
                img = img.convert("RGB")
            frame_filename = f"jat_ep{ep_idx:04d}_f{frame_idx:05d}.png"
            frame_path = frames_dir / frame_filename
            img.save(frame_path)

            # Generate reasoning
            reasoning = generate_reasoning(
                mapped_action,
                rewards[frame_idx],
                prev_action,
            )

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": f"jat-mspacman/frames/{frame_filename}"},
                        {"type": "text", "text": "What action should Pac-Man take?"},
                    ]},
                    {"role": "assistant", "content": f"{reasoning}\n\nAction: {mapped_action}"},
                ],
                "metadata": {
                    "source": "jat-mspacman",
                    "episode": ep_idx,
                    "frame": frame_idx,
                    "reward": rewards[frame_idx],
                    "original_action": actions[frame_idx],
                },
            }
            all_examples.append(example)
            action_counts[mapped_action] += 1
            prev_action = mapped_action

        if len(all_examples) >= args.max_frames:
            log.info(f"Reached {args.max_frames} frame limit at episode {ep_idx}")
            break

        if (ep_idx + 1) % 10 == 0:
            log.info(f"  Episode {ep_idx + 1}: {len(all_examples):,} frames collected, "
                     f"{total_raw_frames:,} raw frames processed")

    log.info(f"\nDone. {episode_count} episodes, {total_raw_frames:,} raw frames → "
             f"{len(all_examples):,} training frames")
    log.info(f"Avg reward/episode: {total_reward / max(episode_count, 1):.0f}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * (1 - args.test_split))
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]

    # Write JSONL files
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    train_path = training_dir / "train.jsonl"
    test_path = training_dir / "test.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex) + "\n")

    log.info(f"\nOutput:")
    log.info(f"  Train: {len(train_examples):,} examples → {train_path}")
    log.info(f"  Test:  {len(test_examples):,} examples → {test_path}")
    log.info(f"  Frames: {frames_dir}")

    log.info(f"\nAction distribution:")
    total = sum(action_counts.values())
    for action in ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]:
        c = action_counts[action]
        log.info(f"  {action:<8} {c:>6} ({c/total*100:.1f}%)")

    log.info(f"\nTo train with this data, update config.yaml dataset_dir or merge with existing data.")


if __name__ == "__main__":
    main()
