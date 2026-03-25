#!/usr/bin/env python3
"""
Fetch Tahahah/PacmanDataset_2 from HuggingFace and convert to VLM training format.

Downloads 22K Pac-Man gameplay frames (552x456 RGB) with actions from a trained
Rainbow DQN agent. Converts to chat-template JSONL for Qwen3.5-9B LoRA training.

Usage:
    python -m scripts.sources.fetch_tahahah_pacman
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ACTION_MAP = {0: "NONE", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}

SYSTEM_PROMPT = (
    "You are TARS, playing Pac-Man. Analyze the game frame and choose an action. "
    "ACTIONS: NONE (continue current direction), UP, DOWN, LEFT, RIGHT "
    "Respond with brief reasoning followed by: Action: <ACTION>\n"
)

REASONING = {
    "NONE": [
        "The current path is clear with pellets ahead. Continue forward.",
        "No threats nearby — keep moving in the current direction.",
        "Pac-Man is mid-corridor with pellets to collect. No change needed.",
        "Safe to continue — ghosts are in other sections of the maze.",
        "Staying on course to clear this corridor of pellets.",
        "The path ahead is open. No reason to change direction.",
        "Continuing straight to collect the remaining pellets in this lane.",
        "No ghost threat detected. Maintain current heading.",
    ],
    "UP": [
        "There's an opening upward with pellets to collect. Head up.",
        "Moving up to reach the upper corridor where pellets remain.",
        "A ghost is approaching from below — move up to escape.",
        "The upper path has uncollected pellets. Turn up at this junction.",
        "Heading up to avoid the ghost and collect pellets above.",
        "The upward corridor is clear. Turn up to collect more pellets.",
        "Ghost below — escape upward through this intersection.",
        "Upper section has more pellets remaining. Go up.",
    ],
    "DOWN": [
        "The lower corridor has pellets remaining. Move down.",
        "Heading down to collect pellets in the lower section.",
        "A ghost is visible above — move down to create distance.",
        "The downward path is clear. Turn down at this intersection.",
        "Moving down to reach uncollected pellets below.",
        "Ghost approaching from above — drop down to escape.",
        "Lower maze section has pellets. Head down.",
        "Turning down at the junction to clear the bottom corridor.",
    ],
    "LEFT": [
        "The left corridor has uncollected pellets. Turn left.",
        "Moving left to reach more pellets in the western section.",
        "A ghost is approaching from the right. Turn left to escape.",
        "The left path is clear with pellets ahead. Go left.",
        "Heading left at this junction to collect remaining pellets.",
        "Ghost to the right — turn left to avoid it.",
        "The western corridor has more pellets. Go left.",
        "Turning left at this intersection to clear the corridor.",
    ],
    "RIGHT": [
        "The right corridor is clear with pellets ahead. Move right.",
        "Heading right to collect pellets in the eastern section.",
        "A ghost is visible to the left. Turn right to get away.",
        "Moving right at this intersection to reach more pellets.",
        "The rightward path has uncollected pellets. Go right.",
        "Ghost approaching from the left — turn right to escape.",
        "Eastern corridor has pellets remaining. Head right.",
        "Turning right to collect pellets and avoid the nearby ghost.",
    ],
}


def main():
    from datasets import load_dataset

    output_dir = PROJECT_ROOT / "data" / "games" / "pacman" / "tahahah"
    frames_dir = output_dir / "frames"
    training_dir = output_dir / "training"
    frames_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading Tahahah/PacmanDataset_2...")
    ds = load_dataset("Tahahah/PacmanDataset_2", split="train", verification_mode="no_checks")
    log.info(f"Downloaded {len(ds):,} frames")

    examples = []
    action_counts = Counter()
    prev_action = None

    for i, ex in enumerate(ds):
        action_id = ex["action"]
        action = ACTION_MAP[action_id]
        action_counts[action] += 1

        # Save frame
        img = ex["frame_image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        frame_name = f"tahahah_ep{ex['episode']:04d}_f{i:06d}.png"
        img.save(frames_dir / frame_name)

        # Generate reasoning
        reasoning = random.choice(REASONING[action])

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": f"tahahah/frames/{frame_name}"},
                    {"type": "text", "text": "What action should Pac-Man take?"},
                ]},
                {"role": "assistant", "content": f"{reasoning}\n\nAction: {action}"},
            ],
        }
        examples.append(example)
        prev_action = action

        if (i + 1) % 5000 == 0:
            log.info(f"  Processed {i+1:,}/{len(ds):,} frames")

    log.info(f"Processed all {len(examples):,} frames")

    # Shuffle and split
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    test = examples[split_idx:]

    with open(training_dir / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(training_dir / "test.jsonl", "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    log.info(f"\nTrain: {len(train):,} → {training_dir / 'train.jsonl'}")
    log.info(f"Test:  {len(test):,} → {training_dir / 'test.jsonl'}")
    log.info(f"Frames: {frames_dir}")

    total = sum(action_counts.values())
    log.info(f"\nAction distribution:")
    for a in ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]:
        c = action_counts[a]
        log.info(f"  {a:<8} {c:>6} ({c/total*100:.1f}%)")


if __name__ == "__main__":
    main()
