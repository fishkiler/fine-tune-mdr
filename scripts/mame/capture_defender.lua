-- ============================================================================
-- MAME Defender Frame + Input Capture Script
-- ============================================================================
-- Autoboots with MAME and records every Nth frame alongside joystick/button state.
--
-- Usage:
--   mame defender -rompath ~/mame/roms \
--       -autoboot_script scripts/mame/capture_defender.lua \
--       -autoboot_delay 2
--
-- Output:
--   data/games/defender/frames/ep{N}/frame_{NNNNNN}.ppm
--   data/games/defender/actions.csv
--
-- Defender controls:
--   Joystick: UP / DOWN (2-way)
--   Buttons:  FIRE, THRUST, REVERSE, SMART_BOMB, HYPERSPACE
--
-- Combined action encoding:
--   Each frame records a comma-separated set of active inputs.
--   e.g., "UP+FIRE+THRUST" or "NONE" if nothing pressed.
-- ============================================================================

-- Configuration
local CAPTURE_EVERY = 3        -- 60fps / 3 = 20fps capture rate
local OUTPUT_BASE = nil        -- Set in init()
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
    local f = io.open("config.yaml", "r")
    if f then
        f:close()
        local p = io.popen("pwd")
        local cwd = p:read("*l")
        p:close()
        return cwd
    end
    return "/home/jayoung/Documents/dgx-code-bank/fine-tune-mdr"
end

-- ── Input Reading ──
-- Defender uses several IO ports for its controls.
-- We read individual input fields by name via the MAME ioport API.

local function read_field(port_tag, field_name)
    local ok, val = pcall(function()
        local port = manager.machine.ioport.ports[port_tag]
        if not port then return false end
        local field = port.fields[field_name]
        if not field then return false end
        -- For digital inputs: check if the field's current value differs from defvalue
        return field.live_value ~= field.defvalue
    end)
    if ok then return val end
    return false
end

local function get_defender_actions()
    local actions = {}

    -- Joystick directions
    if read_field(":IN0", "P1 Up") then table.insert(actions, "UP") end
    if read_field(":IN0", "P1 Down") then table.insert(actions, "DOWN") end

    -- Buttons
    if read_field(":IN0", "P1 Button 1") then table.insert(actions, "FIRE") end
    if read_field(":IN0", "P1 Button 2") then table.insert(actions, "THRUST") end
    if read_field(":IN0", "P1 Button 3") then table.insert(actions, "SMART_BOMB") end
    if read_field(":IN0", "P1 Button 4") then table.insert(actions, "HYPERSPACE") end
    if read_field(":IN0", "P1 Button 5") then table.insert(actions, "REVERSE") end

    if #actions == 0 then
        return "NONE"
    end
    return table.concat(actions, "+")
end

-- Fallback: try reading raw port value if field names don't work
local port_names_discovered = false
local discovered_ports = {}

local function discover_ports()
    if port_names_discovered then return end
    port_names_discovered = true

    print("\n  Discovering IO ports...")
    local ok, _ = pcall(function()
        for tag, port in pairs(manager.machine.ioport.ports) do
            local fields_list = {}
            for name, field in pairs(port.fields) do
                table.insert(fields_list, name)
            end
            if #fields_list > 0 then
                print(string.format("    Port %s: %s", tag, table.concat(fields_list, ", ")))
                discovered_ports[tag] = fields_list
            end
        end
    end)
    if not ok then
        print("  Warning: Could not discover ports")
    end
end

-- ── Frame Saving ──

local function save_frame_ppm(path)
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

        -- MAME pixel format: 0x00RRGGBB (packed uint32, native endian)
        -- On little-endian (aarch64): bytes are BB GG RR 00
        for i = 1, #pixels, 4 do
            local b1, b2, b3, b4 = pixels:byte(i, i + 3)
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
    OUTPUT_BASE = project_root .. "/data/games/defender"

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

    -- Discover IO ports for debugging
    discover_ports()

    print("============================================")
    print("  MAME Defender Capture Script Active")
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

    -- Get current actions
    local action = get_defender_actions()

    -- Build frame filename
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

emu.register_frame_done(on_frame_done, "capture")

pcall(function() emu.register_stop(on_stop) end)
pcall(function() emu.add_machine_stop_notifier(on_stop) end)
