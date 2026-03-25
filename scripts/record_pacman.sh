#!/bin/bash
# ============================================================================
# Record Pac-Man with auto frame + action capture
# ============================================================================
# Launches MAME with a Lua script that captures every 4th frame (~15fps)
# as a PNG screenshot + logs the joystick action for each frame.
#
# Just play the game! Frames and actions are saved automatically.
#
# Controls:
#   5           — Insert coin
#   1           — Start (1 player)
#   Arrow keys  — Move Pac-Man
#   Escape      — Quit & save
#
# Output:
#   data/games/pacman/mame-recording/capture/frames/   — PNG screenshots
#   data/games/pacman/mame-recording/capture/actions.jsonl — frame-action log
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROM_PATH="$HOME/mame/roms"
LUA_SCRIPT="$SCRIPT_DIR/mame_capture_frames.lua"
CAPTURE_DIR="$PROJECT_DIR/data/games/pacman/mame-recording/capture"

mkdir -p "$CAPTURE_DIR/frames"

echo "============================================"
echo "  Pac-Man Recording + Frame Capture"
echo "============================================"
echo ""
echo "  Controls:"
echo "    5           — Insert coin"
echo "    1           — Start (1 player)"
echo "    Arrow keys  — Move Pac-Man"
echo "    Escape      — Quit & save"
echo ""
echo "  Frames saved to: $CAPTURE_DIR"
echo "  Starting MAME..."
echo ""

mame pacman \
    -rompath "$ROM_PATH" \
    -autoboot_script "$LUA_SCRIPT" \
    -skip_gameinfo \
    -window \
    -nomaximize \
    -resolution 576x448

# Count results
FRAME_COUNT=$(ls "$CAPTURE_DIR/frames/"*.png 2>/dev/null | wc -l)
echo ""
echo "============================================"
echo "  Recording Complete!"
echo "  Frames captured: $FRAME_COUNT"
echo "  Actions log: $CAPTURE_DIR/actions.jsonl"
echo "============================================"
