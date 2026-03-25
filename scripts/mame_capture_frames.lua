-- ============================================================================
-- MAME Lua Script: Capture Pac-Man frames + actions every N frames
-- ============================================================================
-- Saves PNG screenshots and logs the joystick input for each captured frame.
-- Run with: mame pacman -autoboot_script scripts/mame_capture_frames.lua
--
-- Output:
--   frames/frame_NNNNNN.png  — screenshot
--   actions.jsonl             — {frame, action, score} per captured frame
-- ============================================================================

local capture_every = 4      -- capture every Nth frame (4 = ~15 fps from 60fps game)
local output_dir = "/home/jayoung/Documents/dgx-code-bank/fine-tune-mdr/data/games/pacman/mame-recording/capture"
local frame_count = 0
local capture_count = 0

-- Create output dirs
os.execute("mkdir -p " .. output_dir .. "/frames")
local log_file = io.open(output_dir .. "/actions.jsonl", "w")

-- Action mapping: MAME input port values to our action names
-- Pac-Man joystick uses "P1 Up", "P1 Down", "P1 Left", "P1 Right"
local function get_action()
    local port = manager.machine.ioport.ports[":IN1"]
    if not port then return "NONE" end

    local fields = port.fields
    for name, field in pairs(fields) do
        local n = tostring(name)
        if field.pressed then
            if n:find("Up") then return "UP"
            elseif n:find("Down") then return "DOWN"
            elseif n:find("Left") then return "LEFT"
            elseif n:find("Right") then return "RIGHT"
            end
        end
    end
    return "NONE"
end

-- Read score from game RAM (Pac-Man specific addresses)
local function get_score()
    local cpu = manager.machine.devices[":maincpu"]
    if not cpu then return 0 end
    local mem = cpu.spaces["program"]
    -- Pac-Man score is BCD at 0x4E80-0x4E83 (tens, hundreds, thousands, ten-thousands)
    local s = 0
    pcall(function()
        s = mem:read_u8(0x4E80) * 10
          + mem:read_u8(0x4E81) * 100
          + mem:read_u8(0x4E82) * 1000
          + mem:read_u8(0x4E83) * 10000
    end)
    return s
end

-- Register frame callback
emu.register_frame_done(function()
    frame_count = frame_count + 1

    if frame_count % capture_every ~= 0 then return end

    capture_count = capture_count + 1
    local fname = string.format("frame_%06d.png", capture_count)
    local fpath = output_dir .. "/frames/" .. fname

    -- Save screenshot
    local screen = manager.machine.screens[":screen"]
    if screen then
        screen:snapshot(fpath)
    end

    -- Log action
    local action = get_action()
    local score = get_score()

    if log_file then
        log_file:write(string.format(
            '{"frame":%d,"capture":%d,"action":"%s","score":%d,"file":"%s"}\n',
            frame_count, capture_count, action, score, fname
        ))
        log_file:flush()
    end

    if capture_count % 100 == 0 then
        print(string.format("[Capture] %d frames saved, score: %d, action: %s",
            capture_count, score, action))
    end
end)

print("[Capture] Pac-Man frame capture active — saving every " .. capture_every .. " frames to " .. output_dir)
print("[Capture] Play the game! Press Escape when done.")
