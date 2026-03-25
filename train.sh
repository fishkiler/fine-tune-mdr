#!/usr/bin/env bash
# train.sh — Run training with a memory ceiling (cgroup guard)
# Caps training at 110GB so OOM kills the process, not the system.
# Disables swap for training entirely to prevent the death spiral.

set -euo pipefail
cd "$(dirname "$0")"

exec systemd-run --user --scope -p MemoryMax=110G -p MemorySwapMax=0 \
    python train_native.py "$@"
