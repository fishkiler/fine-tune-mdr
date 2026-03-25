#!/bin/bash
# ============================================================================
# Record a Pac-Man gameplay session in MAME with frame + input capture
# ============================================================================
# Records:
#   1. Input recording (.inp) — every joystick input with frame-precise timing
#   2. AVI video — every frame at native resolution (288x224)
#   3. Snapshots — can be triggered manually with F12
#
# Usage:
#   bash scripts/mame_record_pacman.sh
#
# Controls:
#   Arrow keys  — Move Pac-Man
#   5           — Insert coin
#   1           — Start (1 player)
#   Escape      — Quit (saves recording automatically)
#   F12         — Take snapshot
#
# After recording, run the frame extraction script to convert to training data.
# ============================================================================

set -euo pipefail

MAME_DIR="$HOME/mame"
ROM_PATH="$MAME_DIR/roms"
RECORD_DIR="/home/jayoung/Documents/dgx-code-bank/fine-tune-mdr/data/games/pacman/mame-recording"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INP_FILE="$RECORD_DIR/pacman_${TIMESTAMP}.inp"
SNAP_DIR="$RECORD_DIR/snaps_${TIMESTAMP}"

mkdir -p "$RECORD_DIR" "$SNAP_DIR"

echo "============================================"
echo "  MAME Pac-Man Recording Session"
echo "============================================"
echo ""
echo "  Output:"
echo "    Input recording: $INP_FILE"
echo "    Snapshots:       $SNAP_DIR"
echo ""
echo "  Controls:"
echo "    Arrow keys  — Move Pac-Man"
echo "    5           — Insert coin"
echo "    1           — Start (1 player)"
echo "    Escape      — Quit & save"
echo ""
echo "  Starting MAME..."
echo ""

# Run MAME with input recording
# -rompath:    where to find ROMs
# -record:     save all inputs to .inp file
# -snapshot_directory: where F12 snapshots go
# -skip_gameinfo: skip the info screen at startup
mame pacman \
    -rompath "$ROM_PATH" \
    -record "$INP_FILE" \
    -snapshot_directory "$SNAP_DIR" \
    -skip_gameinfo \
    -window \
    -nomaximize \
    -resolution 576x448

echo ""
echo "Recording saved to: $INP_FILE"
echo "To replay: mame pacman -rompath $ROM_PATH -playback $INP_FILE"
echo ""
echo "Next step: Extract frames from the recording"
echo "  python scripts/extract_mame_frames.py --inp $INP_FILE"
