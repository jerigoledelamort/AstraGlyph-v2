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

| Input | Action |
|---|---|
| **W / A / S / D** | Move forward, left, back, right |
| **Space** | Move up |
| **Left Ctrl** | Move down |
| **Mouse (LMB held)** | Look around (pitch + yaw) |
| **Escape** | Exit |

Movement speed: 300 units/sec. Look sensitivity: 0.0025 rad/pixel.

## 🏗️ Architecture

```
src/
├── main.rs                  ← Entry point, winit event loop (ApplicationHandler)
├── app/                     ← Application state & input handling
│   ├── state.rs             ← AppState, CameraController, render loop
│   └── input.rs             ← InputState (key/mouse tracking, tests)
├── engine/
│   ├── core/                ← EngineError, block_on (async→sync), Pod trait
│   ├── math/                ← Vec2, Vec3, Vec4, Mat4, Transform (all tested)
│   └── platform/            ← winit 0.30 EventLoop + window creation
├── graphics/                ← wgpu abstractions
│   ├── device.rs            ← Instance, Adapter, Surface, Device
│   ├── buffer.rs            ← Buffer utilities
│   ├── pipeline.rs          ← Shader compilation, render pipeline builder
│   ├── texture.rs           ← Render target, texture upload
│   └── shaders/             ← WGSL shaders (scene + composite)
├── renderer/                ← Rendering passes
│   ├── scene_pass.rs        ← 3D scene → offscreen texture (120×68)
│   ├── ascii_pass.rs        ← GPU readback → ASCII glyph quads
│   └── composite_pass.rs    ← Glyph atlas → screen surface
├── ascii/                   ← ASCII-specific systems
│   ├── glyph_atlas.rs       ← Procedural 8×8 bitmap glyphs (14 chars)
│   └── cell_grid.rs         ← Dynamic cell grid for variable-size rendering
├── scene/                   ← Scene graph
│   ├── entity.rs            ← Entity (handle-based)
│   ├── component.rs         ← Component system (TypeId-based storage)
│   ├── scene.rs             ← Scene container (ECS-like)
│   └── camera.rs            ← Camera + Frustum + Projection
└── demo/                    ← Demo scenes
    └── cornell_box.rs       ← Cornell Box (walls, boxes, lighting)
```

### Rendering Pipeline

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  3D Scene (Cornell) │────▶│  Offscreen Tex   │────▶│  ASCII Grid     │
│  (120×68 pixels)    │     │  (Rgba8Unorm)    │     │  (8160 cells)   │
└─────────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Screen Surface     │◀────│  Composite Pass  │◀────│  GPU Readback   │
│  (1280×720 ASCII)   │     │  (Glyph atlas)   │     │  (CPU sync)     │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Scene Pass**: 3D meshes rendered to a 120×68 RGBA texture with simple lighting (ambient + diffuse).
2. **GPU Readback**: Texture copied to CPU memory (synchronous — blocks frame).
3. **ASCII Converter**: Each pixel → brightness index → glyph index → InstanceData.
4. **Composite Pass**: InstanceData sent to GPU as storage buffer. Vertex shader maps each instance to a quad sampling the correct glyph from the atlas texture. Fragment shader colors the glyph.
5. **Screen**: Composite output rendered to the winit surface.

## 🧪 Testing

58 unit tests covering:
- **Math**: Vec2, Vec3, Vec4, Mat4 (add, mul, perspective, rotation, etc.)
- **Input**: Key press/release, mouse delta accumulation, mouse button state
- **ASCII**: Glyph atlas size, brightness mapping, cell grid creation
- **Scene**: Entity creation, component storage, destruction, filtering
- **Camera**: View/projection matrices, forward/right vectors

```bash
cargo test
# 58 tests, 0 failures
```

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
