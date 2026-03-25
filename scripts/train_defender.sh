#!/usr/bin/env bash
# train_defender.sh — Run Defender RL training with a memory ceiling (cgroup guard)
# Same pattern as train.sh: caps at 110GB so OOM kills the process, not the system.

set -euo pipefail
cd "$(dirname "$0")/.."

exec systemd-run --user --scope -p MemoryMax=110G -p MemorySwapMax=0 \
    python defender.py train "$@"
