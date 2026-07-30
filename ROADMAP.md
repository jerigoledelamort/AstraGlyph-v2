# AstraGlyph — Roadmap

> **English** | [Русский](README_RU.md#дорожная-карта)

The journey from an ASCII rendering experiment to a full-featured game engine.

---

## ✅ Already Done

### Stage 1-8: MVP Core
- [x] Project scaffold (Cargo, Git, CI structure)
- [x] Core engine (Vec2/3/4, Mat4, Transform, error handling, async runner)
- [x] Graphics layer (wgpu 30 device, buffers, pipelines, textures)
- [x] ECS-like scene system (Entity, Component, Scene)
- [x] Camera controller (WASD + mouse look, LMB)
- [x] ASCII renderer (procedural glyph atlas, cell grid, scene pass, composite pass)
- [x] Cornell Box demo (walls, boxes, lighting, colors)
- [x] 58 unit tests, 0 compiler warnings
- [x] winit 0.30 `ApplicationHandler` integration

---

## ✅ Phase 1: Solid Renderer 🧱
*Make the renderer high-quality and performant.*

### 1.1 Depth & Sorting
- [x] Add depth buffer to scene pass
- [x] Correct face sorting (back-to-front for transparent, depth test for opaque)
- [x] Fix z-fighting for Cornell Box walls

### 1.2 Performance
- [x] Cache GPU buffers (don't re-upload every frame)
- [x] Async GPU readback (double-buffered, no CPU stalls)
- [x] Measure and log FPS, GPU frame time, CPU frame time

### 1.3 Lighting
- [x] Phong shading (ambient + diffuse + specular)
- [x] Multiple light sources (directional, point)
- [x] Shadow maps → ASCII (simplified — one shadow-casting light)
- [x] Per-mesh material colors (Matte / Mirror / Glass via a material storage buffer)

---

## ✅ Phase 2: Scene System 🎬
*Transform raw meshes into a manageable scene graph.*

### 2.1 Transforms
- [x] Model matrix in scene vertex shader
- [x] TransformComponent with position, rotation, scale
- [x] Parent/child hierarchy (world matrix = parent_world × local)

### 2.2 Scene Management
- [x] Load scenes from JSON files (hand-written parser, no serde)
- [x] Material registry (shared materials, no duplication)
- [x] Frustum culling (skip meshes outside camera FOV)

### 2.3 Camera
- [x] Orthographic projection support
- [x] Camera presets (first-person, third-person, orbit)
- [x] Smooth camera transitions (dampening)

---

## 🎨 Phase 3: Advanced ASCII 🎨
*Push ASCII art to its visual limits.*

### 3.1 Glyph System
- [x] Dynamic cell grid (variable cell sizes — depth-driven merging, toggle: G)
- [x] Procedural text font: hand-coded 5x7 bitmaps for printable ASCII
- [x] Block Elements (U+2580–U+259F) with 2x supersampling: each cell resolves a
      2x2 subpixel block into a quadrant glyph, doubling effective resolution
      (toggle: B switches to the classic brightness ramp)
- [x] Box Drawing set (U+2500–U+257F) + geometric arrows, for UI frames
- [ ] CJK / emoji — needs a larger glyph cell (8x8 cannot hold a legible
      ideograph), so it waits on a second atlas geometry
- [x] Color modes: ANSI 256, TrueColor (16M), ANSI 16, grayscale (cycle: M)
- [ ] TTF font loading → bitmap atlas (optional — the procedural font above is
      the primary path, per this item's original "fallback" framing)

### 3.2 Post-Processing
- [x] Bloom effect in ASCII space
- [x] SSAO (Screen-Space Ambient Occlusion) → ASCII
- [x] Gamma correction
- [x] Chromatic aberration (for retro CRT feel)

### 3.3 UI Layer
- [x] 2D ASCII overlay on top of 3D scene (buffer + compositing)
- [x] HUD elements (FPS counter, debug info, crosshair — toggle: H)
- [x] Menu system (buttons, toggles, choices, submenus — Tab)
- [x] Console / debug command line (history, scrollback, commands — `)

---

## ✅ Phase 4: Hardware Ray Tracing (RTX) 🔷
*The engine's stated goal — "quality lighting on ASCII graphics" — cashed out.*

Until now lighting has been rasterised: one simplified shadow map, an analytic
sky for "reflections", and a refraction that bends the view ray but samples that
same procedural sky. Mirrors do not reflect the scene and glass does not show
what is actually behind it, because a fragment shader cannot know what lies along
a reflected ray. Tracing is what closes that gap, and it is the reason the rest
of the renderer exists.

Hardware path: wgpu 30 exposes `EXPERIMENTAL_RAY_QUERY` (and
`EXPERIMENTAL_RAY_TRACING_PIPELINES`) behind `ExperimentalFeatures`, giving WGSL
ray queries against a GPU acceleration structure on Vulkan/DX12 — i.e. real RTX
cores on the target hardware.

### 4.1 Acceleration Structure
- [x] Feature detection: request `EXPERIMENTAL_RAY_QUERY`, report what the
      adapter actually supports, and fall back cleanly when it does not
      (`ASTRAGLYPH_NO_RAYTRACING=1` forces the fallback on capable hardware, so
      that path stays exercised)
- [x] Build BLAS per mesh from a shared vertex/index heap — *not* from the
      rasteriser's cached buffers: a hit shader has to read the same geometry
      back, and it cannot index an array of bindings without `BINDING_ARRAY`
- [x] Build and update a TLAS from the scene's per-object model matrices
- [x] Rebuild only what moved. Measured: 5 BLAS builds and 1 TLAS build at
      startup, both unchanged 400 frames later, TLAS → 2 after one object moved

### 4.2 Traced Lighting
- [x] Ray-traced shadows: one shadow ray per light, replacing the single
      simplified shadow map (and its self-shadowing bias problems)
- [x] Ray-traced reflections: mirrors reflect actual scene geometry, with
      reflectivity and Fresnel already present in the material
- [x] Ray-traced refraction: glass shows the geometry behind it, bent by its IOR,
      with total internal reflection
- [x] Bounded depth, configurable (`depth 0..4` in the console). Iterative, not
      recursive — WGSL has no recursion — and single-path, so a mirror reflecting
      glass reflecting a mirror costs one ray per bounce, not two to the power

### 4.3 Quality & Integration
- [x] Soft shadows from area/point lights (`shadow_samples`, default 2, jittered
      inside a cone of the light's apparent angular radius; `samples shadow <n>`)
- [x] Ambient occlusion from traced rays (`ao_samples`, default 4), replacing the
      screen-space approximation — which is suppressed while tracing, because
      stacking the two darkens every crease twice
- [x] Toggle between rasterised and traced lighting at runtime (R, the menu, or
      `trace on|off`) for A/B comparison
- [x] HUD/console report which path is active and the ray budget per frame
      (`rays` prints the acceleration-structure and sampler state too)

### 4.4 CPU Fallback
- [x] Analytic CPU tracer over spheres/planes/boxes for machines without ray
      query. The intersection maths lives in `engine/geometry/`, shared with
      gameplay raycasting (5.1), and the shapes come from `ColliderComponent`,
      the same source physics uses — so a reflection can never disagree with a
      collision
- [x] Same visual features (reflections, refraction with TIR, soft shadows,
      traced AO) at half resolution, multithreaded over rows with
      `std::thread::scope`. Measured 452-480 FPS against 1320 for the hardware
      path; its camera basis matches the rasteriser's to within 0.31 px on a
      240-pixel-wide target

---

## 🎮 Phase 5: Game Engine Foundations 🎮 (5.1–5.4, less OGG and the scheduler)
*Graphics engine → general-purpose game engine.*

### 5.1 Physics
- [x] Collision detection: sphere/sphere, sphere/box, sphere/plane, box/box
      (separating-axis over all 15 axes), box/plane. Every test returns a contact
      normal and depth rather than a bool — a solver that only knows "they
      overlap" has to guess a direction, and guesses differently each frame
- [x] Rigid body simulation, self-written: semi-implicit Euler, impulses with
      restitution and clamped Coulomb friction, positional correction split by
      inverse mass. **Linear only** — no angular velocity or inertia tensor,
      because the output is a character grid where a rolling sphere and a sliding
      one are the same glyphs
- [x] Raycasting for gameplay (X picks through the crosshair, line of sight for
      visibility), sharing `engine/geometry` with the CPU tracer so what the
      player can click is what the renderer draws

### 5.2 Audio
- [x] WAV playback: PCM 8/16/24/32-bit and IEEE float 32/64, including
      `WAVE_FORMAT_EXTENSIBLE`, with a chunk walker rather than fixed offsets
- [ ] OGG/Vorbis — not implemented. It needs a codebook decoder, floor and residue
      decoders and an MDCT: several thousand lines for a second container format,
      weighed against the rest of the phase and deferred rather than half-built
- [x] 3D spatial audio: constant-power panning from the listener's own basis,
      inverse and linear distance falloff, and Doppler from the velocity
      components along the line of sight only — so an orbiting source has no
      pitch shift, which a naive speed-based formula gets wrong
- [x] Music streaming — as an ordinary voice, seeked into, rather than a second
      code path. Duplicating the resampler and panner for "music" would buy
      nothing; `Voice::seek` is the whole mechanism

### 5.3 Scripting
- [x] Lua integration, self-written: lexer, recursive-descent parser with
      precedence climbing, tree-walking interpreter, and a standard library
      subset (print, type, tostring, tonumber, pairs, ipairs, math, string,
      table, `setmetatable` for `__index`). No coroutines — they need a resumable
      interpreter, which is a different design rather than an addition
- [x] Game logic in Lua: scripts queue commands into a mailbox the engine drains
      once per frame, so a script cannot mutate state mid-frame and what it did
      is inspectable from Rust. `assets/scripts/demo.lua` drives an entity
- [x] Hot-reload: watched files are reloaded when their **contents** change.
      Timestamps are not consulted — measured on NTFS, consecutive writes can
      report identical modification times, and using them as a pre-check missed
      about one edit in twenty

### 5.4 ECS Overhaul
- [x] Archetype-based ECS: entities sharing a component-type set keep each type in
      one contiguous `Vec`, replacing
      `HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>>` — which boxed every
      component individually behind two hash lookups. `Scene`'s public API is
      byte-for-byte unchanged, so no call site needed editing and the pre-existing
      tests became a regression suite for the new storage
- [x] Component queries: `entities_with`, `entities_with_both`, and
      `component_columns` for the packed read path. Results are ordered by creation,
      because the renderer's per-object GPU indices are derived from them
- [ ] System pipeline (update order, dependencies) — **not built**. The engine's
      update order is currently explicit and readable in `AppState::render`
      (input → scripts → physics → audio → render), about a dozen lines. A
      scheduler with declared dependencies would replace that with indirection and
      buy nothing until systems come from outside the engine — plugins, or scripts
      registering their own. Deferred rather than added as scaffolding

---

## 🛠️ Phase 6: Tooling 🛠️
*Developer tools for a better workflow.*

### 6.1 Scene Editor
- [ ] Separate window for scene editing
- [ ] Drag-and-drop entity placement
- [ ] Transform gizmo (move, rotate, scale widgets)
- [ ] Property inspector (edit component values)

### 6.2 Asset Pipeline
- [ ] Model loader (OBJ, glTF — parse manually)
- [ ] Texture loader (PNG, JPEG → wgpu textures)
- [ ] Asset hot-reload (edit assets without restarting)

### 6.3 Profiler
- [ ] FPS overlay with frame breakdown
- [ ] GPU timing (render pass, readback, composite)
- [ ] Memory usage tracking
- [ ] Draw call / instance count stats

---

## 🏁 Victory Condition

The roadmap is complete when:
1. A playable game exists, built entirely on AstraGlyph
2. The engine is declared "1.0 — Game Engine"
3. The game is released (or at least playable)

Until then, every feature is a step toward that goal.

---

## Contributing

This is a solo project, but ideas, issues, and discussions are welcome!
The engine is MIT licensed — use it, fork it, break it.
