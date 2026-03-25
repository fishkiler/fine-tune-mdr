-- Diagnostic v2: use correct API and test each feature with pcall
print("=== MAME Lua API Diagnostic v2 ===")

local init_done = false

local function do_init()
    if init_done then return end
    init_done = true

    print("--- Initializing ---")

    -- Test manager.machine
    local ok, machine = pcall(function() return manager.machine end)
    print("manager.machine: ok=" .. tostring(ok))
    if not ok then print("  error: " .. tostring(machine)); return end

    -- Test screens
    local ok2, screens = pcall(function() return manager.machine.screens end)
    print("screens: ok=" .. tostring(ok2))
    if ok2 then
        for tag, screen in pairs(screens) do
            print("  screen tag: " .. tag)

            -- Test methods
            for _, method in ipairs({"snapshot", "pixels", "pixel", "width", "height"}) do
                local ok3 = pcall(function() local _ = screen[method] end)
                print("    screen." .. method .. ": " .. tostring(ok3))
            end
        end
    end

    -- Test video
    local ok4, video = pcall(function() return manager.machine.video end)
    print("video: ok=" .. tostring(ok4))
    if ok4 then
        -- Try video:snapshot()
        local ok5, err5 = pcall(function() video:snapshot() end)
        print("  video:snapshot(): ok=" .. tostring(ok5) .. " err=" .. tostring(err5))
    end

    -- Test I/O ports
    local ok6, ports = pcall(function() return manager.machine.ioport.ports end)
    print("ioport.ports: ok=" .. tostring(ok6))
    if ok6 then
        for tag, port in pairs(ports) do
            print("  port: " .. tag)
            local ok7, fields = pcall(function() return port.fields end)
            if ok7 then
                for name, field in pairs(fields) do
                    -- Try getting pressed state
                    local ok8, pressed = pcall(function() return field.pressed end)
                    -- Try getting value
                    local ok9, val = pcall(function() return field.value end)
                    print("    " .. name .. " pressed_ok=" .. tostring(ok8) .. " value_ok=" .. tostring(ok9))
                    if ok8 then print("      pressed=" .. tostring(pressed)) end
                    if ok9 then print("      value=" .. tostring(val)) end
                end
            end
            -- Try port:read()
            local ok10, val10 = pcall(function() return port:read() end)
            print("  port:read(): ok=" .. tostring(ok10) .. " val=" .. tostring(val10))
        end
    end

    -- Test os.execute for mkdir
    local ok11, err11 = pcall(function() os.execute('mkdir -p /tmp/mame_lua_test') end)
    print("os.execute mkdir: ok=" .. tostring(ok11))

    -- Test io.open for file writing
    local ok12, f = pcall(function() return io.open("/tmp/mame_lua_test/test.txt", "w") end)
    print("io.open: ok=" .. tostring(ok12))
    if ok12 and f then
        f:write("test")
        f:close()
        print("  file write OK")
    end

    print("=== Diagnostic v2 complete ===")
end

-- Use both old and new API
pcall(function() emu.register_start(do_init) end)
pcall(function() emu.add_machine_reset_notifier(do_init) end)

-- Also try from frame_done as backup
local frame_count = 0
emu.register_frame_done(function()
    frame_count = frame_count + 1
    if frame_count == 5 then
        do_init()
    end
end, "diag")
