-- Check Pac-Man DIP switch settings
print("=== Pac-Man DIP Switch Check ===")

local done = false
emu.register_frame_done(function()
    if done then return end
    done = true

    local ports = manager.machine.ioport.ports
    local dsw1 = ports[":DSW1"]
    if dsw1 then
        local val = dsw1:read()
        print(string.format("DSW1 raw value: 0x%02X (%d)", val, val))

        -- Pac-Man DSW1 bit layout:
        -- Bits 0-1: Coinage
        -- Bits 2-3: Lives (00=1, 01=2, 10=3, 11=5)
        -- Bit 4: Bonus Life (0=10000, 1=15000)
        -- Bit 5: Difficulty (0=Hard, 1=Normal)
        -- Bit 6: Ghost Names (0=Normal, 1=Alternate)

        local lives_bits = (val >> 2) & 0x03
        local lives_map = {[0]=1, [1]=2, [2]=3, [3]=5}
        local lives = lives_map[lives_bits] or "?"

        local bonus = ((val >> 4) & 0x01) == 0 and "10000" or "15000"
        local difficulty = ((val >> 5) & 0x01) == 0 and "HARD" or "Normal"
        local ghost_names = ((val >> 6) & 0x01) == 0 and "Normal" or "Alternate"

        print("  Lives: " .. lives)
        print("  Bonus Life: " .. bonus .. " pts")
        print("  Difficulty: " .. difficulty)
        print("  Ghost Names: " .. ghost_names)

        for name, field in pairs(dsw1.fields) do
            local ok, defval = pcall(function() return field.defvalue end)
            local ok2, curval = pcall(function() return field.value end)
            print(string.format("  Field '%s': defvalue=%s value=%s",
                name, tostring(defval), tostring(curval)))
        end
    end
end, "dips")
