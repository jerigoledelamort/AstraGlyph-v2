-- AstraGlyph demo script.
--
-- Edit this file while the engine is running: it is reloaded within a frame of
-- being saved, and `state` below survives the reload, so an accumulating value
-- keeps accumulating across edits.
--
-- Available to scripts:
--   engine.time, engine.dt, engine.frame, engine.fps, engine.entity_count
--   engine.camera_x, engine.camera_y, engine.camera_z
--   set_position(entity, x, y, z)   translate(entity, dx, dy, dz)
--   set_scale(entity, s)            log(...)
--   play_sound(index, x, y, z)      set_physics(on)   set_tracing(on)
-- Plus print, type, tostring, tonumber, pairs, ipairs, math, string, table.

-- `or {}` rather than `= {}`: on a reload this line runs again, and a plain
-- assignment would wipe whatever the script had accumulated. This is the idiom
-- that makes hot-reload useful rather than just fast.
state = state or {
  announced = false,
  beeps = 0,
}

-- The satellite sphere in material_spheres.json. Entity ids are assigned in load
-- order, and this one is the fourth entity: ground, red, mirror, satellite.
local SATELLITE = 4

--- Called once per frame with the frame delta in seconds.
function update(dt)
  if not state.announced then
    log("demo.lua loaded — " .. engine.entity_count .. " entities, editing this file reloads it")
    state.announced = true
  end

  -- Lift the satellite in a slow sine, relative to its resting height. The
  -- position is local to its parent (the mirror sphere), which is why a small
  -- number moves it visibly: the parent's 1.5x scale multiplies it.
  local lift = math.sin(engine.time * 1.5) * 0.4
  set_position(SATELLITE, 0, 1.6 + lift, 0)

  -- A beep every two seconds, positioned where the satellite is, so the
  -- spatialisation has something to act on.
  local beep = math.floor(engine.time / 2)
  if beep > state.beeps then
    state.beeps = beep
    play_sound(1, 2.2, 1.9 + lift, -2.0)
  end
end
