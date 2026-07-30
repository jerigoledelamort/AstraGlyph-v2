# AstraGlyph

[English](README.md) | **Русский**

> **Rust + wgpu → ASCII-арт. Эксперимент с рендерингом, который растёт в движок.**

AstraGlyph — это самописный 3D-движок на Rust, который проецирует 3D-сцену (Cornell Box) на низкоразрешающую сетку и рендерит её как **цветной ASCII-арт** с использованием процедурного атласа глифов.

Это строительный блок — графический слой, начавшийся как proof-of-concept и растущий во что-то большее.

## 🖥️ Быстрый старт

### Требования
- **Rust** (stable, edition 2024)
- **Windows** (DX12 / wgpu core) или **Linux** (Vulkan)
- **Видеокарта** с поддержкой wgpu (RTX 5070 для разработки)

### Сборка и запуск

```bash
# Клонируйте и соберите
git clone https://github.com/jerigoledelamort/AstraGlyph-v2.git
cd AstraGlyph-v2
cargo run --release

# Запустите тесты
cargo test
```

Всё. Откроется окно. Вы увидите 3D-комнату, отрендеренную цветными ASCII-символами.

## 🎮 Управление

### Камера

| Клавиши | Действие |
|---|---|
| **W / A / S / D** | Движение вперёд, влево, назад, вправо |
| **Left Shift / Left Ctrl** | Вверх / вниз |
| **Мышь (ЛКМ зажата)** | Обзор |
| **Колесо** | Зум — дистанция рига или FOV в первом лице |
| **C** | Пресет камеры (first person / third person / orbit) |

### Рендер

| Клавиши | Действие |
|---|---|
| **R** | Растеризация ↔ трассировка лучей. Строка `LIGHT` в HUD показывает активный путь |
| **B** | Квадрантные блоки ↔ градиентная рампа |
| **M** | Цветовой режим (TrueColor / 256 / 16 / grayscale / mono) |
| **P** | Пост-обработка (bloom, SSAO, гамма, хроматическая аберрация) |
| **G** | Динамическая сетка — далёкие плоские по глубине области сливаются в крупные глифы |

### Симуляция и скриптинг

| Клавиши | Действие |
|---|---|
| **F** | Физика твёрдых тел |
| **X** | Луч через перекрестье: сообщает дистанцию и line of sight |
| **L** | Запуск `assets/scripts/demo.lua` — правь на ходу, файл перезагрузится |
| **Space** | Процедурный звук в точке перекрестья |
| **O** | Зацикленный источник, вращающийся вокруг слушателя — слышно 3D-панорамирование |

### Инструменты и UI

| Клавиши | Действие |
|---|---|
| **H** | HUD |
| **F3** | Профайлер: разбивка кадра, GPU-время по проходам, draw calls, память |
| **F2** | Редактор сцены. Tab выбирает сущность, **G** — move/rotate/scale, **V** — ось, `-`/`=` — сдвиг, `[`/`]` — шаг, **D** — дублировать, **Delete** — удалить, **Ctrl-S** — сохранить |
| **Tab** | Меню настроек (когда редактор закрыт) |
| **`** | Консоль — `help` перечисляет все команды |
| **Escape** | Закрыть открытую панель или выйти |

Диагностика: `ASTRAGLYPH_INPUT_TRACE=1` логирует события ввода,
`ASTRAGLYPH_NO_RAYTRACING=1` принудительно включает CPU-трассировщик на железе,
которое поддерживает ray query.

## 🏗️ Архитектура

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

### Пайплайн рендеринга

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  3D Сцена (Cornell) │────▶│  Offscreen Tex   │────▶│  ASCII Сетка    │
│  (240×136 субпикс.) │     │  (Rgba8Unorm)    │     │  (8160 ячеек)   │
└─────────────────────┘     └──────────────────┘     └────────┬────────┘
                                                              │
                                                              ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Поверхность экрана │◀────│  Composite Pass  │◀────│  GPU Readback   │
│  (1280×720 ASCII)   │     │  (Атлас глифов)  │     │  (CPU async)    │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Scene Pass**: меши рендерятся в RGBA-текстуру 240×136 — вдвое больше сетки глифов по
   каждой оси, так что каждой ячейке достаётся блок 2×2 субпикселя. Освещение либо
   растеризованное (Phong, shadow map, аналитическое окружение), либо трассированное по
   ускоряющей структуре GPU; **R** переключает, а третий путь трассирует аналитически на
   CPU для железа без ray query.
2. **GPU Readback**: текстура копируется в память CPU двойной буферизацией и без
   блокировки — может отставать на кадр, но никогда не останавливает CPU в ожидании GPU.
   (В MVP было синхронно; ровно это и правила Phase 1.2.)
3. **Пост-обработка** (по желанию, **P**): bloom, SSAO, гамма и хроматическая аберрация
   работают по субпиксельному буферу *до* выбора глифов, поэтому влияют на то, какой
   символ получит ячейка, а не только на его цвет.
4. **ASCII Converter**: каждый блок 2×2 → квадрантный блочный глиф (удваивая эффективное
   разрешение) или символ градиентной рампы, затем `InstanceData`. Сетка умеет сливать
   далёкие плоские по глубине области в крупные глифы (**G**).
5. **Composite Pass**: инстансы уходят на GPU storage-буфером. Вершинный шейдер
   разворачивает каждый в квад, выбирающий нужный глиф из атласа; фрагментный красит.
6. **Экран**: композит выводится на surface winit.

## 🧪 Тестирование

1022 юнит-теста. Каждая архитектурная функция покрыта, как требуют правила проекта, и
акцент на свойствах, которые ломаются молча, а не на покрытии строк:

- **Математика**: Vec2/3/4, Mat4, `inverse_affine`, композиция Transform
- **Геометрия**: пересечения луча со сферой/боксом/плоскостью/треугольником, коллизии
  фигур с нормалями и глубинами контакта, ориентированные боксы через теорему о
  разделяющей оси
- **Физика**: энергия не берётся из ниоткуда, покоящееся тело не дрожит, быстрое не
  проходит сквозь стену, позиционная коррекция делится по обратной массе
- **Аудио**: декодирование WAV для каждой поддержанной битности, constant-power
  панорамирование, Doppler только по компоненте скорости вдоль линии визирования,
  16-битная конверсия без переполнения
- **Скриптинг**: лексер Lua, приоритеты и ассоциативность парсера, семантика интерпретатора
  (`%` как в Lua, приведение типов в арифметике но не в сравнении, лексическая область,
  замыкания с общим состоянием), стандартная библиотека, hot-reload
- **Ассеты**: OBJ с 1-based *и* отрицательными индексами, дедупликация вершин по тройкам,
  триангуляция n-угольников; PNG со всеми пятью фильтрами, Adam7 и настоящим
  zlib-сжатым фикстуром, чтобы путь Huffman'а реально проверялся
- **Сцена**: archetype-хранение (учёт swap-remove, запросы в порядке создания),
  round-trip сериализации, разбор JSON
- **Рендерер**: конверсия трансформа TLAS, бюджеты лучей, раскладка глифов и сетки,
  квантование цвета, пост-обработка

```bash
cargo test
# 1022 теста, 0 падений
```

Зависящее от GPU поведение (вывод шейдеров, сборка ускоряющих структур, readback) юнит-
тестами не покрыто — в тестовом окружении нет wgpu-устройства. Оно проверяется запуском и
измерением: разницей пикселей в render target, счётчиками сборок, timestamp queries. Эти
измерения записаны в сообщениях коммитов.

## 📚 Технологический стек

| Компонент | Технология |
|---|---|
| **Язык** | Rust (edition 2024) |
| **Графика** | wgpu 30 (абстракция над Vulkan / DX12) |
| **Окна** | winit 0.30 |
| **Математика** | Самописная (Vec2, Vec3, Vec4, Mat4, Transform) |
| **Шейдеры** | WGSL |
| **Тесты** | Встроенная система тестирования Rust |
| **Сборка** | Cargo |

**Нет внешних библиотек математики** (no glam, no nalgebra). **Нет bytemuck, нет pollster.** Всё самописное — это обучающий проект.

## 🗺️ Дорожная карта

Полный план разработки — в **[ROADMAP.md](ROADMAP.md)**.

Текущий статус: **MVP Pre-Alpha** (тег: `mvp-pre-alpha`)

### Основные вехи
- **Фаза 1**: Depth buffer, кэширование GPU-буферов, асинхронный readback, Phong-освещение
- **Фаза 2**: Model-трансформации, граф сцены, frustum culling
- **Фаза 3**: Динамическая сетка ячеек, Unicode-глифы, постобработка, ASCII UI
- **Фаза 4**: Физика, аудио, скриптинг на Lua, переработка ECS
- **Фаза 5**: Редактор сцен, пайплайн ассетов, профайлер

**Условие победы**: Игра, собранная на AstraGlyph, после чего движок объявляется «1.0 — Game Engine».

## 📜 Лицензия

MIT License — см. [LICENSE](LICENSE).

Используйте, форкайте, ломайте. Упоминание автора приветствуется, но не обязательно.

---

*Собрано на wgpu, winit и здоровой доле упрямства.*
