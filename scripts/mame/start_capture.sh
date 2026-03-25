#!/usr/bin/env bash
# ============================================================================
# start_capture.sh — Launch MAME Pac-Man with frame capture
# ============================================================================
# Convenience wrapper that sets up directories and launches MAME with the
# Lua capture script attached.
#
# Usage:
#   bash scripts/mame/start_capture.sh
#   bash scripts/mame/start_capture.sh --capture-every 6   # 10fps instead of 20fps
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LUA_SCRIPT="$SCRIPT_DIR/capture_gameplay.lua"
OUTPUT_DIR="$PROJECT_ROOT/data/games/pacman"

# Create output directories
mkdir -p "$OUTPUT_DIR/frames"
mkdir -p "$OUTPUT_DIR/training"

# Detect next episode number
EPISODE=1
while [ -d "$OUTPUT_DIR/frames/ep${EPISODE}" ] && [ "$(ls -A "$OUTPUT_DIR/frames/ep${EPISODE}" 2>/dev/null)" ]; do
    EPISODE=$((EPISODE + 1))
done

echo "============================================"
echo "  MAME Pac-Man Capture"
echo "  Episode: $EPISODE"
echo "  Output:  $OUTPUT_DIR/frames/ep${EPISODE}/"
echo "  Actions: $OUTPUT_DIR/actions.csv"
echo "============================================"
echo ""
echo "Controls:"
echo "  Arrow keys = Move Pac-Man"
echo "  5          = Insert coin"
echo "  1          = Start (1 player)"
echo "  Esc        = Quit MAME"
echo ""
echo "Launching MAME..."

# Launch MAME with the Lua capture script
# -autoboot_delay gives MAME time to initialize before the script hooks in
exec mame pacman \
    -autoboot_script "$LUA_SCRIPT" \
    -autoboot_delay 2 \
    -rompath "$HOME/mame/roms" \
    "$@"
