#!/usr/bin/env bash
# ============================================================================
# upload_checkpoint.sh — Copy latest checkpoint to shared AI_projects drive
# ============================================================================
# Copies the most recent outputs/checkpoint-NNNN/ to the shared SMB mount
# at /mnt/ai_projects/checkpoints/ so the other server (or any machine
# mounting //mvsraid.local/ai_projects) can access it.
#
# For Colab: upload from /mnt/ai_projects/checkpoints/ to Google Drive,
# or use the other server to push it up.
#
# Usage:
#   ./upload_checkpoint.sh              # copy latest checkpoint
#   ./upload_checkpoint.sh checkpoint-4250  # copy specific checkpoint
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
SHARED_DIR="/mnt/ai_projects/checkpoints"

# --- Find checkpoint to upload ---
if [[ -n "${1:-}" ]]; then
    CKPT_DIR="${OUTPUT_DIR}/${1}"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
        exit 1
    fi
else
    # Find latest checkpoint by modification time
    CKPT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -z "$CKPT_DIR" ]]; then
        echo "ERROR: No checkpoint-* directories found in $OUTPUT_DIR"
        exit 1
    fi
fi

CKPT_NAME=$(basename "$CKPT_DIR")
DEST="${SHARED_DIR}/${CKPT_NAME}"

echo "============================================"
echo "Uploading checkpoint to shared drive"
echo "============================================"
echo "Source:      $CKPT_DIR"
echo "Destination: $DEST"
echo ""

# --- Verify checkpoint integrity ---
REQUIRED_FILES=("adapter_model.safetensors" "trainer_state.json" "optimizer.pt" "scheduler.pt")
MISSING=()
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${CKPT_DIR}/${f}" ]]; then
        MISSING+=("$f")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Checkpoint is incomplete! Missing files:"
    printf '  - %s\n' "${MISSING[@]}"
    echo "Training may still be saving. Wait and try again."
    exit 1
fi

# --- Print checkpoint info ---
if command -v python3 &>/dev/null; then
    python3 -c "
import json
with open('${CKPT_DIR}/trainer_state.json') as f:
    s = json.load(f)
last = s.get('log_history', [{}])[-1]
print(f\"  Step:  {s['global_step']}\")
print(f\"  Epoch: {s.get('epoch', 'N/A')}\")
print(f\"  Loss:  {last.get('loss', 'N/A')}\")
print(f\"  LR:    {last.get('learning_rate', 'N/A')}\")
"
fi

# --- Check shared drive is mounted ---
if [[ ! -d "$SHARED_DIR" ]]; then
    echo "ERROR: Shared drive not mounted at $SHARED_DIR"
    echo "Mount it with: sudo mount -t cifs //mvsraid.local/ai_projects /mnt/ai_projects"
    exit 1
fi

# --- Calculate size ---
SIZE=$(du -sh "$CKPT_DIR" | cut -f1)
echo ""
echo "Checkpoint size: $SIZE"
echo ""

# --- Copy with rsync (shows progress, handles interrupts) ---
echo "Copying..."
if command -v rsync &>/dev/null; then
    rsync -ah --progress "$CKPT_DIR/" "$DEST/"
else
    mkdir -p "$DEST"
    cp -rv "$CKPT_DIR/"* "$DEST/"
fi

echo ""
echo "============================================"
echo "Upload complete!"
echo "============================================"
echo "Shared path: $DEST"
echo ""
echo "For Colab, upload this checkpoint to Google Drive:"
echo "  My Drive/fine-tune-mdr/outputs/${CKPT_NAME}/"
echo ""
echo "Or from the other server:"
echo "  ls ${DEST}/"
