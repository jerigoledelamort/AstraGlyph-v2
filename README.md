# AstraGlyph

[English](README.md) | [Русский](README_RU.md)

> **Rust + wgpu → ASCII Art. A rendering experiment with ambitions.**

AstraGlyph is a self-written 3D rendering engine in Rust that projects a 3D scene (Cornell Box) onto a low-resolution grid and renders it as **colored ASCII art** using a procedural glyph atlas.

It's a building block — a graphics layer that started as a proof-of-concept and is growing into something bigger.

## 🖥️ Quick Start

### Prerequisites
- **Rust** (stable, 2024 edition)
- **Windows** (DX12 / wgpu core) or **Linux** (Vulkan)
- **GPU** with wgpu support (RTX 5070 used for development)

### Build & Run

```bash
# Clone and build
git clone https://github.com/jerigoledelamort/AstraGlyph-v2.git
cd AstraGlyph-v2
cargo run --release

# Run tests
cargo test
```

That's it. A window opens. You see a 3D room rendered in colored ASCII characters.

## 🎮 Controls

### Camera

| Input | Action |
|---|---|
| **W / A / S / D** | Move forward, left, back, right |
| **Left Shift / Left Ctrl** | Move up / down |
| **Mouse (LMB held)** | Look around |
| **Mouse wheel** | Zoom — rig distance, or field of view in first person |
| **C** | Cycle camera preset (first person / third person / orbit) |

### Rendering

| Input | Action |
|---|---|
| **R** | Rasterised ↔ ray-traced lighting. The HUD's `LIGHT` line says which is live |
| **B** | Quadrant block glyphs ↔ brightness ramp |
| **M** | Colour mode (TrueColor / 256 / 16 / greyscale / mono) |
| **P** | Post-processing (bloom, SSAO, gamma, chromatic aberration) |
| **G** | Dynamic cell grid — merge distant, depth-flat regions into larger glyphs |

### Simulation and scripting

| Input | Action |
|---|---|
| **F** | Rigid-body physics |
| **X** | Raycast through the crosshair; reports distance and line of sight |
| **L** | Run `assets/scripts/demo.lua` — edit it while running, it reloads |
| **Space** | Play a procedural sound at the crosshair |
| **O** | A looping sound orbiting the listener, to hear the 3D panning |

### Tools and UI

| Input | Action |
|---|---|
| **H** | HUD |
| **F3** | Profiler: frame breakdown, per-pass GPU time, draw calls, memory |
| **F2** | Scene editor. Tab picks an entity, **G** cycles move/rotate/scale, **V** cycles the axis, `-`/`=` nudge, `[`/`]` change the step, **D** duplicates, **Delete** removes, **Ctrl-S** saves |
| **Tab** | Settings menu (when the editor is closed) |
| **`** | Console — `help` lists every command |
| **Escape** | Close the open panel, or exit |

Diagnostics: set `ASTRAGLYPH_INPUT_TRACE=1` to log input events, or
`ASTRAGLYPH_NO_RAYTRACING=1` to force the CPU fallback tracer on hardware that
supports ray query.

## 🏗️ Architecture

```
src/
├── main.rs                  ← Entry point, winit event loop (ApplicationHandler)
├── app/
│   ├── state.rs             ← AppState: ties every subsystem to the frame loop
│   ├── input.rs             ← InputState (key/mouse tracking)
│   └── metrics.rs           ← FPS, CPU and submit timing
├── engine/
│   ├── core/                ← EngineError, block_on, Pod, hand-written JSON parser
│   ├── math/                ← Vec2/3/4, Mat4, Transform
│   ├── geometry/            ← Analytic shapes, ray intersection, shape collision.
│   │                          Shared by the CPU tracer and physics, so a reflection
│   │                          cannot disagree with a collision
│   └── platform/            ← winit 0.30 EventLoop + window creation
├── graphics/                ← wgpu abstractions
│   ├── device.rs            ← Instance, Adapter, Surface, Device, feature negotiation
│   ├── capabilities.rs      ← Ray-query detection and the fallback decision
│   ├── timing.rs            ← QuerySet timestamp queries: real per-pass GPU time
│   └── shaders/             ← WGSL. scene_shading is shared by both lighting paths
├── renderer/
│   ├── scene_pass.rs        ← 3D scene → offscreen texture, rasterised or traced
│   ├── raytrace.rs          ← BLAS/TLAS, geometry heap, ray budget
│   ├── cpu_trace.rs         ← Analytic CPU tracer for hardware without ray query
│   ├── post_process.rs      ← Bloom, SSAO, gamma, chromatic aberration
│   ├── ascii_pass.rs        ← Subpixels → glyph quads
│   └── composite_pass.rs    ← Glyph atlas → screen surface
├── ascii/
│   ├── glyph_atlas.rs       ← Procedural 8×8 bitmaps
│   ├── font5x7.rs           ← Hand-coded font for printable ASCII
│   ├── blocks.rs            ← Quadrant block elements (2× effective resolution)
│   ├── box_drawing.rs       ← Box-drawing set for UI frames
│   ├── grid_layout.rs       ← Depth-driven merging of distant cells
│   ├── color.rs             ← TrueColor / 256 / 16 / greyscale quantisation
│   └── overlay.rs           ← 2D UI layer composited over the scene
├── scene/
│   ├── archetype.rs         ← Archetype component storage (packed columns)
│   ├── scene.rs             ← Scene container over it; API unchanged by the rewrite
│   ├── loader.rs            ← Scene files → Scene, via the hand-written JSON parser
│   ├── writer.rs            ← Scene → scene file, round-trip tested
│   ├── hierarchy.rs         ← Parent/child with cycle detection and memoisation
│   ├── frustum.rs           ← Aabb/Plane/Frustum culling
│   └── camera.rs, camera_rig.rs, primitives.rs, material_registry.rs
├── physics/
│   ├── body.rs              ← Rigid bodies, semi-implicit Euler (linear only)
│   └── world.rs             ← Contact solver, raycasting, line of sight
├── audio/
│   ├── wav.rs               ← WAV decoder (PCM 8/16/24/32, float, EXTENSIBLE)
│   ├── mixer.rs             ← Software mixer: panning, attenuation, Doppler
│   ├── device.rs            ← winmm waveOut via raw FFI
│   └── synth.rs             ← Procedural sounds, so the demo needs no audio assets
├── scripting/               ← Lua, self-implemented
│   ├── lexer.rs, parser.rs  ← Tokens, then recursive descent with precedence climbing
│   ├── value.rs             ← Values and tables (array + hash parts)
│   ├── interp.rs            ← Tree-walking interpreter, bounded steps and depth
│   ├── stdlib.rs            ← print, type, pairs, math, string, table, __index
│   └── bindings.rs          ← Engine mailbox + hot-reload
├── assets/
│   ├── obj.rs               ← Wavefront OBJ parser
│   ├── png.rs               ← PNG decoder, including a DEFLATE inflater
│   └── hot_reload.rs        ← Watches by contents, not timestamps
├── ui/
│   ├── menu.rs, console.rs  ← Settings menu and debug console
│   └── editor.rs            ← Scene editor overlay: select, gizmo, save
└── demo/                    ← Demo scenes
```

### Rendering Pipeline

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  3D Scene (spheres) │────▶│  Offscreen Tex   │────▶│  ASCII Grid     │
│  (240×136 subpixels)│     │  (Rgba8Unorm)    │     │  (8160 cells)   │
└─────────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Screen Surface     │◀────│  Composite Pass  │◀────│  GPU Readback   │
│  (1280×720 ASCII)   │     │  (Glyph atlas)   │     │  (CPU async)    │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Scene Pass**: 3D meshes rendered to a 240×136 RGBA texture — twice the glyph grid
   on each axis, so every cell owns a 2×2 block of subpixels. Lighting is either
   rasterised (Phong, shadow map, analytic environment) or ray-traced against a GPU
   acceleration structure; **R** switches between them, and a third path traces
   analytically on the CPU for hardware without ray query.
2. **GPU Readback**: the texture is copied to CPU memory double-buffered and
   non-blocking — it may lag the current frame by one, and never stalls the CPU on the
   GPU. (It *was* synchronous in the MVP; that was Phase 1.2's whole point.)
3. **Post-processing** (optional, **P**): bloom, SSAO, gamma and chromatic aberration run
   on the subpixel buffer *before* glyphs are chosen, so they influence which character
   each cell gets rather than only its colour.
4. **ASCII Converter**: each 2×2 subpixel block → a quadrant block glyph (doubling the
   effective resolution) or a brightness-ramp character, then an `InstanceData`. The cell
   grid can merge distant, depth-flat regions into larger glyphs (**G**).
5. **Composite Pass**: instances go to the GPU as a storage buffer. The vertex shader maps
   each to a quad sampling the right glyph from the atlas; the fragment shader colours it.
6. **Screen**: composited to the winit surface.

## 🧪 Testing

1022 unit tests. Every architectural function is covered, per the project's own rule, and
the emphasis is on the properties that break silently rather than on line coverage:

- **Math**: Vec2/3/4, Mat4, `inverse_affine`, Transform composition
- **Geometry**: ray/sphere/box/plane/triangle intersection, shape-shape collision with
  contact normals and depths, oriented boxes via the separating-axis theorem
- **Physics**: energy is never manufactured, a resting body does not jitter, a fast body
  does not tunnel through a wall, positional correction splits by inverse mass
- **Audio**: WAV decoding for every supported depth, constant-power panning, Doppler from
  the line-of-sight velocity component only, 16-bit conversion that does not wrap
- **Scripting**: the Lua lexer, parser precedence and associativity, interpreter semantics
  (Lua's `%`, coercion in arithmetic but not comparison, lexical scope, closures sharing
  state), the standard library, and hot-reload
- **Assets**: OBJ 1-based *and* negative indices, per-triple vertex deduplication, n-gon
  triangulation; PNG for all five filters, Adam7 interlacing, and a real zlib-compressed
  fixture so the Huffman path is genuinely exercised
- **Scene**: archetype storage (swap-remove bookkeeping, creation-order queries),
  serialisation round-trips, JSON parsing
- **Renderer**: TLAS transform conversion, ray budgets, glyph and grid layout, colour
  quantisation, post-processing

```bash
cargo test
# 1022 tests, 0 failures
```

GPU-dependent behaviour (shader output, acceleration-structure builds, readback) has no
unit tests — there is no wgpu device in the test environment. It is verified by running
the engine and measuring: pixel differences in the render target, build counters,
timestamp queries. Those measurements are recorded in the commit messages.

## 📚 Tech Stack

| Component | Technology |
|---|---|
| **Language** | Rust (edition 2024) |
| **Graphics API** | wgpu 30 (Vulkan / DX12 abstraction) |
| **Windowing** | winit 0.30 |
| **Math** | Self-written (Vec2, Vec3, Vec4, Mat4, Transform) |
| **Shaders** | WGSL |
| **Testing** | Rust built-in test framework |
| **Build** | Cargo |

**No external math libraries** (no glam, no nalgebra). **No bytemuck, no pollster.** Everything is self-written — this is a learning project.

## 🗺️ Roadmap

See **[ROADMAP.md](ROADMAP.md)** for the full development plan.

Current status: **MVP Pre-Alpha** (tag: `mvp-pre-alpha`)

### Highlights
- **Phase 1**: Depth buffer, GPU buffer caching, async readback, Phong lighting
- **Phase 2**: Model transforms, scene graph, frustum culling
- **Phase 3**: Dynamic cell grid, Unicode glyphs, post-processing, ASCII UI
- **Phase 4**: Physics, audio, Lua scripting, ECS overhaul
- **Phase 5**: Scene editor, asset pipeline, profiler

**Victory condition**: A playable game built on AstraGlyph, then the engine is declared "1.0 — Game Engine".

## 📜 License

MIT License — see [LICENSE](LICENSE).

Use it, fork it, break it. Attribution appreciated but not required.

---

*Built with wgpu, winit, and a healthy dose of stubbornness.*
