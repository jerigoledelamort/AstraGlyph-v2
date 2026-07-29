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
- [ ] Extended Unicode set (CJK characters, emojis for fun)
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
- [ ] Menu system (text-based menus, buttons, navigation)
- [ ] Console / debug command line

---

## 🎮 Phase 4: Game Engine Foundations 🎮
*Graphics engine → general-purpose game engine.*

### 4.1 Physics
- [ ] Collision detection (AABB, OBB, sphere)
- [ ] Rigid body simulation (self-written, no physics library)
- [ ] Raycasting for gameplay (click-to-move, line of sight)

### 4.2 Audio
- [ ] Sound playback (WAV, OGG)
- [ ] 3D spatial audio
- [ ] Music streaming

### 4.3 Scripting
- [ ] Lua integration (self-written bindings)
- [ ] Game logic in Lua (entity behaviors, triggers)
- [ ] Hot-reload Lua scripts

### 4.4 ECS Overhaul
- [ ] Archetype-based ECS (for cache-friendly iteration)
- [ ] Component queries (filter by component types)
- [ ] System pipeline (update order, dependencies)

---

## 🛠️ Phase 5: Tooling 🛠️
*Developer tools for a better workflow.*

### 5.1 Scene Editor
- [ ] Separate window for scene editing
- [ ] Drag-and-drop entity placement
- [ ] Transform gizmo (move, rotate, scale widgets)
- [ ] Property inspector (edit component values)

### 5.2 Asset Pipeline
- [ ] Model loader (OBJ, glTF — parse manually)
- [ ] Texture loader (PNG, JPEG → wgpu textures)
- [ ] Asset hot-reload (edit assets without restarting)

### 5.3 Profiler
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
