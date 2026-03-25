-- Quick diagnostic: test what MAME Lua API is available
-- Run: mame pacman -autoboot_script scripts/mame/test_lua_api.lua -autoboot_delay 2 -str 3

print("=== MAME Lua API Diagnostic ===")
print("MAME version: " .. emu.app_name() .. " " .. emu.app_version())

emu.register_start(function()
    print("--- register_start fired ---")

    local machine = manager.machine
    print("machine: " .. tostring(machine))

    -- Check screens
    local screens = machine.screens
    print("screens: " .. tostring(screens))
    for tag, screen in pairs(screens) do
        print("  screen tag: " .. tag)
        print("  screen type: " .. tostring(screen))

        -- Check what methods the screen has
        local has_snapshot = pcall(function() return screen.snapshot end)
        print("  has snapshot: " .. tostring(has_snapshot))

        local has_pixels = pcall(function() return screen.pixels end)
        print("  has pixels: " .. tostring(has_pixels))

        -- Try to get screen dimensions
        local ok, w = pcall(function() return screen.width end)
        if ok then print("  width: " .. tostring(w)) end
        local ok2, h = pcall(function() return screen.height end)
        if ok2 then print("  height: " .. tostring(h)) end
    end

    -- Check I/O ports
    local ports = machine.ioport.ports
    print("\n--- I/O Ports ---")
    for tag, port in pairs(ports) do
        print("  port: " .. tag)
        for name, field in pairs(port.fields) do
            print("    field: " .. name .. " (type=" .. tostring(field.type) .. ")")
        end
    end

    -- Check video
    print("\n--- Video ---")
    local video = machine.video
    print("video: " .. tostring(video))
    local has_snap = pcall(function() return video.snapshot end)
    print("video has snapshot: " .. tostring(has_snap))

    -- Try video:snapshot()
    local ok3, err3 = pcall(function()
        video:snapshot()
    end)
    print("video:snapshot() result: ok=" .. tostring(ok3) .. " err=" .. tostring(err3))

    print("\n=== Diagnostic complete ===")
end)

-- Also try frame_done
local frame_test_count = 0
emu.register_frame_done(function()
    frame_test_count = frame_test_count + 1
    if frame_test_count == 1 then
        print("frame_done callback fired (frame " .. frame_test_count .. ")")
    end
    if frame_test_count == 10 then
        print("frame_done: reached 10 frames OK")
    end
end, "test")
