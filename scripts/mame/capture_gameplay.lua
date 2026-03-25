-- ============================================================================
-- MAME Pac-Man Frame + Input Capture Script
-- ============================================================================
-- Autoboots with MAME and records every Nth frame alongside joystick state.
--
-- Usage:
--   mame pacman -autoboot_script scripts/mame/capture_gameplay.lua \
--       -autoboot_delay 2
--
-- Output:
--   data/games/pacman/frames/ep{N}/frame_{NNNNNN}.png
--   data/games/pacman/actions.csv
--
-- MAME 0.264 Lua API notes:
--   - emu.register_start is deprecated; use emu.add_machine_reset_notifier
--   - video:snapshot() saves to MAME's snapshot_directory (no path arg)
--   - IN0 port is active-low: 0xFF = nothing pressed, bit cleared = pressed
--   - Field names: "P1 Up", "P1 Down", "P1 Left", "P1 Right"
-- ============================================================================

-- Configuration
local CAPTURE_EVERY = 3        -- Capture every Nth frame (60fps / 3 = 20fps)
local OUTPUT_BASE = nil        -- Set in init() to absolute path
local EPISODE = nil

-- State
local frame_count = 0
local capture_index = 0
local actions_file = nil
local episode_dir = nil
local initialized = false

-- ── Helpers ──

local function ensure_dir(path)
    os.execute('mkdir -p "' .. path .. '"')
end

local function get_project_root()
    -- Get the absolute path to the project root from this script's location
    -- The autoboot_script path is relative to MAME's CWD, so we need to
    -- figure out where the project root is.
    -- Try: the script is at scripts/mame/capture_gameplay.lua relative to project root
    -- MAME CWD might be anywhere, so we use an environment variable or fixed path.

    -- Check if we're being run from the project directory
    local f = io.open("config.yaml", "r")
    if f then
        f:close()
        -- CWD is the project root
        local p = io.popen("pwd")
        local cwd = p:read("*l")
        p:close()
        return cwd
    end

    -- Fallback: hardcoded project path
    return "/home/jayoung/Documents/dgx-code-bank/fine-tune-mdr"
end

-- ── Joystick Reading ──
-- Pac-Man IN0 port (active-low, 0xFF = idle):
--   Bit 0 (0x01): P1 Up
--   Bit 1 (0x02): P1 Left
--   Bit 2 (0x04): P1 Right
--   Bit 3 (0x08): P1 Down
-- When a direction is pressed, its bit goes LOW (0).

local function get_joystick_action()
    local ok, val = pcall(function()
        return manager.machine.ioport.ports[":IN0"]:read()
    end)

    if not ok then return "NONE" end

    -- Active-low: invert to get active-high
    local pressed = 0xFF ~ val  -- XOR with 0xFF (bitwise NOT for lower 8 bits)

    if (pressed & 0x01) ~= 0 then return "UP" end
    if (pressed & 0x08) ~= 0 then return "DOWN" end
    if (pressed & 0x02) ~= 0 then return "LEFT" end
    if (pressed & 0x04) ~= 0 then return "RIGHT" end

    return "NONE"
end

-- ── Frame Saving ──
-- MAME's video:snapshot() saves to the configured snapshot_directory.
-- We use it to trigger a snapshot, then move/copy the file to our output dir.
-- Alternative: use screen:pixels() to get raw pixel data (slower but more control).
-- For simplicity, we use the PPM approach via screen:pixels().

local function save_frame_ppm(path)
    -- Get screen pixels and write as PPM (simple image format)
    -- Then convert to PNG with an external tool, or just keep as PPM
    local ok, err = pcall(function()
        local screen = manager.machine.screens[":screen"]
        local pixels, width, height = screen:pixels()

        local f = io.open(path, "wb")
        if not f then
            print("ERROR: Cannot open " .. path)
            return
        end

        -- PPM header
        f:write(string.format("P6\n%d %d\n255\n", width, height))

        -- pixels is a binary string of uint32 ARGB values
        -- Each pixel is 4 bytes: BBGGRRAA (little-endian) or AARRGGBB (big-endian)
        -- MAME returns packed uint32 in native byte order
        for i = 1, #pixels, 4 do
            local b1, b2, b3, b4 = pixels:byte(i, i + 3)
            -- MAME pixel format: 0x00RRGGBB (packed uint32, native endian)
            -- On little-endian (aarch64): bytes are BB GG RR 00
            local blue = b1
            local green = b2
            local red = b3
            f:write(string.char(red, green, blue))
        end

        f:close()
    end)

    if not ok then
        print("Frame save error: " .. tostring(err))
    end

    return ok
end

-- ── Episode Detection ──

local function detect_episode(base)
    local ep = 1
    while true do
        local dir = base .. "/frames/ep" .. ep
        local f = io.open(dir .. "/frame_000000.ppm", "r")
        if not f then
            f = io.open(dir .. "/frame_000000.png", "r")
        end
        if f then
            f:close()
            ep = ep + 1
        else
            break
        end
    end
    return ep
end

-- ── Init ──

local function init()
    if initialized then return end
    initialized = true

    local project_root = get_project_root()
    OUTPUT_BASE = project_root .. "/data/games/pacman"

    EPISODE = detect_episode(OUTPUT_BASE)
    episode_dir = OUTPUT_BASE .. "/frames/ep" .. EPISODE
    ensure_dir(episode_dir)

    -- Open or create actions CSV
    local csv_path = OUTPUT_BASE .. "/actions.csv"
    local csv_exists = io.open(csv_path, "r")
    if csv_exists then
        csv_exists:close()
        actions_file = io.open(csv_path, "a")
    else
        ensure_dir(OUTPUT_BASE)
        actions_file = io.open(csv_path, "w")
        actions_file:write("episode,frame_index,frame_path,action,timestamp\n")
    end

    -- Ensure difficulty is set to Normal (bit 5 of DSW1 = 1)
    local dsw1 = manager.machine.ioport.ports[":DSW1"]
    if dsw1 then
        local val = dsw1:read()
        local difficulty_bit = (val >> 6) & 0x01
        if difficulty_bit == 0 then
            -- Set to Normal by writing bit 6 high
            local fields = dsw1.fields
            for name, field in pairs(fields) do
                if name == "Difficulty" then
                    pcall(function() field:set_value(64) end)
                    print("  ** Difficulty set to NORMAL **")
                end
            end
        else
            print("  Difficulty: Normal")
        end
    end

    print("============================================")
    print("  MAME Pac-Man Capture Script Active")
    print("  Episode: " .. EPISODE)
    print("  Capture rate: every " .. CAPTURE_EVERY .. " frames (~" ..
          math.floor(60 / CAPTURE_EVERY) .. " fps)")
    print("  Output: " .. episode_dir)
    print("============================================")
end

-- ── Frame Callback ──

local function on_frame_done()
    frame_count = frame_count + 1

    -- Lazy init on first frame (ensures machine is fully ready)
    if frame_count == 3 then
        local ok, err = pcall(init)
        if not ok then
            print("Init error: " .. tostring(err))
            return
        end
    end

    if not initialized then return end

    -- Only capture every Nth frame
    if frame_count % CAPTURE_EVERY ~= 0 then
        return
    end

    -- Get current joystick action
    local action = get_joystick_action()

    -- Build frame filename (PPM format — lightweight, no external deps)
    local frame_name = string.format("frame_%06d.ppm", capture_index)
    local frame_path = episode_dir .. "/" .. frame_name
    local relative_path = string.format("frames/ep%d/%s", EPISODE, frame_name)

    -- Save frame
    local saved = save_frame_ppm(frame_path)
    if not saved then return end

    -- Write CSV row
    local timestamp = string.format("%.3f", capture_index * CAPTURE_EVERY / 60.0)
    if actions_file then
        actions_file:write(string.format("%d,%d,%s,%s,%s\n",
            EPISODE, capture_index, relative_path, action, timestamp))
        -- Flush every 100 frames to avoid data loss
        if capture_index % 100 == 0 then
            actions_file:flush()
        end
    end

    capture_index = capture_index + 1

    -- Progress update every 500 captures
    if capture_index % 500 == 0 then
        print(string.format("  Captured %d frames (episode %d)", capture_index, EPISODE))
    end
end

-- ── Shutdown ──

local function on_stop()
    if actions_file then
        actions_file:flush()
        actions_file:close()
        actions_file = nil
    end
    if initialized then
        print(string.format("\nCapture complete: %d frames saved for episode %d",
            capture_index, EPISODE))
    end
end

-- ── Register Callbacks ──
-- Use new API (add_machine_reset_notifier) if available, fall back to deprecated register_start

emu.register_frame_done(on_frame_done, "capture")

pcall(function() emu.register_stop(on_stop) end)
pcall(function() emu.add_machine_stop_notifier(on_stop) end)
