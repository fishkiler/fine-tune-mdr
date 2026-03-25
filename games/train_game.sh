#!/usr/bin/env bash
# train_game.sh — Run game adapter training with a memory ceiling (cgroup guard)
# Caps training at 110GB so OOM kills the process, not the system.
# Disables swap for training entirely to prevent the death spiral.
#
# Usage:
#   bash train_game.sh pacman
#   bash train_game.sh pacman --fresh

set -euo pipefail
cd "$(dirname "$0")"

GAME="${1:?Usage: train_game.sh <game_name> [extra args...]}"
shift

exec systemd-run --user --scope -p MemoryMax=110G -p MemorySwapMax=0 \
    python train_game_adapter.py --game "$GAME" "$@"
