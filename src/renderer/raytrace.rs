// Hardware ray tracing resources: one BLAS per mesh, one TLAS over the scene,
// and the side tables a hit shader needs to rebuild a surface from a hit record.
//
// Why the side tables exist: a WGSL `RayIntersection` reports *where* a ray hit
// (instance, primitive index, barycentrics) but carries no surface data — no
// normal, no material, not even the triangle's vertices unless
// `EXPERIMENTAL_RAY_HIT_VERTEX_RETURN` is enabled. So the geometry has to be
// readable from the shader as well as from the acceleration-structure builder,
// which is why every mesh is copied into one shared *heap* of vertices and
// indices rather than kept in per-mesh buffers: a fragment shader cannot index
// an array of bindings without `BINDING_ARRAY`, but it can index one big array.
//
// Rebuild policy (Phase 4.1's "rebuild only what moved"):
// - A BLAS is built exactly once per mesh, when the mesh is first seen. Mesh
//   vertex data is immutable in this engine, so a rebuild could never change it.
// - The TLAS is rebuilt only when the instance set actually differs from the
//   previous frame — same transforms and same materials means no build at all.
// Both are counted (`blas_builds`, `tlas_builds`) so the policy is observable
// from outside instead of being a claim in a comment.

use std::collections::HashMap;

use crate::engine::core::{cast_slice, Pod};
use crate::engine::math::Mat4;
use crate::scene::component::MeshComponent;
use wgpu::{Device, Queue};

/// Floats per vertex in the geometry heap:
/// position(3) + normal(3) + color(3) + uv(2).
///
/// The heap is typed `array<f32>` in WGSL rather than `array<Vertex>` on
/// purpose: a WGSL struct of vec3s is padded to 16-byte columns under std430
/// alignment rules, which would silently disagree with the 44-byte
/// `MeshVertex` the acceleration-structure builder reads from the same memory.
/// A flat float array has no alignment opinion, so both readers agree.
///
/// Mirrored in WGSL as `HEAP_STRIDE` in `scene_traced_fragment.wgsl` — the two
/// must change together, and `heap_stride_matches_the_mesh_vertex_layout`
/// below pins this side to the actual struct size.
pub const HEAP_FLOATS_PER_VERTEX: usize = 11;

/// Byte stride of one heap vertex, as the BLAS builder sees it.
pub const HEAP_VERTEX_STRIDE: u64 = (HEAP_FLOATS_PER_VERTEX * 4) as u64;

/// Upper bound on TLAS instances the engine asks for. The device may grant less,
/// in which case `RayTracer` uses what it got — see `max_instances`.
const WANTED_INSTANCES: u32 = crate::graphics::capabilities::REQUESTED_TLAS_INSTANCES;

/// Initial heap capacity in vertices; grows by doubling when exceeded.
const INITIAL_VERTEX_CAPACITY: usize = 1 << 15;

/// Initial heap capacity in indices; grows by doubling when exceeded.
const INITIAL_INDEX_CAPACITY: usize = 1 << 16;

/// Where one mesh's data lives inside the shared geometry heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometrySlice {
    /// First vertex of the mesh, in vertices (not bytes).
    pub vertex_offset: u32,
    /// Number of vertices.
    pub vertex_count: u32,
    /// First index of the mesh, in indices (not bytes).
    pub index_offset: u32,
    /// Number of indices; always a multiple of three.
    pub index_count: u32,
}

/// Per-instance record the traced shader reads after a hit, indexed by the
/// intersection's `instance_custom_data`.
///
/// The object-to-world matrix is deliberately *not* stored here: the hit record
/// already carries it as `object_to_world`, and duplicating it would let the two
/// drift apart.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TracedInstance {
    /// First vertex of this instance's mesh in the heap.
    pub vertex_offset: u32,
    /// First index of this instance's mesh in the heap.
    pub index_offset: u32,
    /// Material slot in the scene pass's material storage buffer.
    pub material_index: u32,
    /// Padding to a 16-byte stride.
    pub _padding: u32,
}

unsafe impl Pod for TracedInstance {}

/// Runtime knobs for the traced path, mirrored in WGSL as `TraceSettings`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceSettings {
    /// Maximum number of secondary bounces a reflection/refraction path takes.
    /// Zero means "shade the primary hit only".
    pub max_depth: u32,
    /// Shadow rays per light. 1 gives hard shadows; more gives a penumbra.
    pub shadow_samples: u32,
    /// Ambient-occlusion rays per fragment. Zero disables traced AO.
    pub ao_samples: u32,
    /// Bit flags, see `TraceFlags`.
    pub flags: u32,
    /// Apparent radius of a light, in world units, for soft shadows.
    pub light_radius: f32,
    /// Maximum distance an AO ray travels before it counts as unoccluded.
    pub ao_radius: f32,
    /// Padding to a 32-byte uniform block.
    ///
    /// A frame counter used to live here, seeding the shader's sampler so
    /// successive frames drew different samples. It was removed: with only a
    /// handful of rays per pixel and no temporal accumulation, re-seeding per
    /// frame made 19% of the render target change between two otherwise
    /// identical frames, which the ASCII quantizer turns into a shimmering
    /// image and which drowns any attempt to measure a real change. The sampler
    /// now depends on pixel position alone, so the traced image is stable and
    /// reproducible; the price is fixed banding instead of moving noise, which
    /// is the right trade for output made of characters.
    pub _padding: [u32; 2],
}

unsafe impl Pod for TraceSettings {}

/// Feature bits inside `TraceSettings::flags`.
pub mod trace_flags {
    /// Cast shadow rays instead of trusting the shadow map.
    pub const SHADOWS: u32 = 1 << 0;
    /// Trace reflection rays for mirror materials.
    pub const REFLECTIONS: u32 = 1 << 1;
    /// Trace refraction rays for glass materials.
    pub const REFRACTION: u32 = 1 << 2;
    /// Trace ambient occlusion rays.
    pub const AMBIENT_OCCLUSION: u32 = 1 << 3;
}

impl Default for TraceSettings {
    /// The shipped preset: every traced feature on, a bounce budget of two, and
    /// sample counts chosen to stay far above 30 FPS on the target GPU.
    fn default() -> Self {
        Self {
            max_depth: 2,
            shadow_samples: 2,
            ao_samples: 4,
            flags: trace_flags::SHADOWS
                | trace_flags::REFLECTIONS
                | trace_flags::REFRACTION
                | trace_flags::AMBIENT_OCCLUSION,
            light_radius: 0.35,
            ao_radius: 2.0,
            _padding: [0, 0],
        }
    }
}

impl TraceSettings {
    /// Whether a feature bit is set.
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }

    /// Set or clear a feature bit.
    pub fn set(&mut self, flag: u32, on: bool) {
        if on {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }

    /// Worst-case rays cast per shaded fragment, for the HUD's ray budget line.
    ///
    /// One primary ray is *not* counted: the primary hit still comes from the
    /// rasteriser. Each bounce re-pays the shadow cost, which is why depth
    /// multiplies rather than adds.
    pub fn rays_per_fragment(&self, light_count: u32) -> u32 {
        let shadows = if self.has(trace_flags::SHADOWS) {
            light_count * self.shadow_samples.max(1)
        } else {
            0
        };
        let ao = if self.has(trace_flags::AMBIENT_OCCLUSION) {
            self.ao_samples
        } else {
            0
        };
        let bounces = if self.has(trace_flags::REFLECTIONS) || self.has(trace_flags::REFRACTION) {
            self.max_depth
        } else {
            0
        };
        // Primary fragment: shadows + AO. Each bounce: one bounce ray + shadows.
        shadows + ao + bounces * (1 + shadows)
    }
}

/// Convert a column-major `Mat4` into the row-major 3x4 affine matrix a TLAS
/// instance expects.
///
/// This is the single most error-prone conversion in the whole acceleration
/// structure: a transposed instance transform does not fail validation, it just
/// puts the traced geometry somewhere other than where the rasteriser drew it,
/// which reads as "reflections are broken" rather than "the matrix is wrong".
pub fn tlas_transform(model: &Mat4) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    for row in 0..3 {
        for col in 0..4 {
            // Column-major source: element (row, col) is at m[col * 4 + row].
            out[row * 4 + col] = model.m[col * 4 + row];
        }
    }
    out
}

/// One frame's snapshot of an instance, used to decide whether the TLAS needs
/// rebuilding. Compared bit-for-bit, so "the same matrix recomputed" counts as
/// unchanged only when it really produced the same bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct InstanceSnapshot {
    entity_id: u64,
    transform: [u32; 12],
    material_index: u32,
}

/// Description of one instance to place in the TLAS this frame.
#[derive(Clone, Copy, Debug)]
pub struct InstanceRequest {
    /// Entity owning the mesh; the BLAS is keyed by this.
    pub entity_id: u64,
    /// World matrix of the instance.
    pub model: Mat4,
    /// Material slot for the instance.
    pub material_index: u32,
}

/// A mesh's BLAS plus the size descriptor it was created with (the builder needs
/// the descriptor again on every build, and it must match the creation sizes).
struct MeshBlas {
    blas: wgpu::Blas,
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    slice: GeometrySlice,
    /// Whether the BLAS has been built at least once. An unbuilt BLAS
    /// referenced by a TLAS is a validation error, not a silent miss.
    built: bool,
}

/// GPU acceleration structures and geometry side tables for the traced path.
pub struct RayTracer {
    tlas: wgpu::Tlas,
    blas: HashMap<u64, MeshBlas>,
    /// CPU mirror of the vertex heap, kept so the GPU buffer can be reallocated
    /// and refilled when a scene outgrows it without re-walking the scene graph.
    vertices: Vec<f32>,
    indices: Vec<u32>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    index_capacity: usize,
    instance_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
    /// Instance slots the device granted; the TLAS is exactly this big.
    max_instances: u32,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// Previous frame's instance set; an identical set skips the TLAS build.
    last_snapshot: Vec<InstanceSnapshot>,
    blas_builds: u64,
    tlas_builds: u64,
    /// Instances actually placed in the TLAS on the most recent build.
    instance_count: u32,
}

impl RayTracer {
    /// Create the acceleration structures and their bind group.
    ///
    /// The caller must have verified that the device has
    /// `EXPERIMENTAL_RAY_QUERY`; without it `create_tlas` is a validation error.
    pub fn new(device: &Device) -> Self {
        // What the device actually granted, clamped to what the engine wants. A
        // device is free to hand back a smaller instance budget than requested,
        // and sizing the TLAS above it is a validation error rather than a
        // graceful degradation.
        let max_instances = WANTED_INSTANCES.min(device.limits().max_tlas_instance_count);
        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("scene_tlas"),
            max_instances,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });

        let vertex_buffer = Self::create_heap_buffer(
            device,
            "traced_vertex_heap",
            (INITIAL_VERTEX_CAPACITY * HEAP_FLOATS_PER_VERTEX * 4) as u64,
        );
        let index_buffer =
            Self::create_heap_buffer(device, "traced_index_heap", (INITIAL_INDEX_CAPACITY * 4) as u64);

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traced_instances"),
            size: (max_instances as usize * std::mem::size_of::<TracedInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let settings_buffer = crate::graphics::buffer::uniform_buffer(
            device,
            "traced_settings",
            std::mem::size_of::<TraceSettings>() as u64,
        );

        let layout = Self::create_layout(device);
        let bind_group = Self::create_bind_group(
            device,
            &layout,
            &tlas,
            &instance_buffer,
            &vertex_buffer,
            &index_buffer,
            &settings_buffer,
        );

        Self {
            tlas,
            blas: HashMap::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer,
            index_buffer,
            vertex_capacity: INITIAL_VERTEX_CAPACITY,
            index_capacity: INITIAL_INDEX_CAPACITY,
            instance_buffer,
            settings_buffer,
            max_instances,
            layout,
            bind_group,
            last_snapshot: Vec::new(),
            blas_builds: 0,
            tlas_builds: 0,
            instance_count: 0,
        }
    }

    fn create_heap_buffer(device: &Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            // BLAS_INPUT so the builder can read it, STORAGE so the hit shader
            // can read the same bytes back.
            usage: wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            size,
            mapped_at_creation: false,
        })
    }

    fn create_layout(device: &Device) -> wgpu::BindGroupLayout {
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("traced_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_bind_group(
        device: &Device,
        layout: &wgpu::BindGroupLayout,
        tlas: &wgpu::Tlas,
        instances: &wgpu::Buffer,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        settings: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("traced_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: settings.as_entire_binding(),
                },
            ],
        })
    }

    /// Bind group layout for the traced pipelines' group 1.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Bind group holding the TLAS and the geometry side tables.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// How many BLAS builds have been issued since startup. Static geometry must
    /// make this stop growing after the first frame.
    pub fn blas_builds(&self) -> u64 {
        self.blas_builds
    }

    /// How many TLAS builds have been issued since startup. A still scene must
    /// make this stop growing too.
    pub fn tlas_builds(&self) -> u64 {
        self.tlas_builds
    }

    /// Instances in the TLAS as of the last build.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Triangles reachable by a ray, i.e. the size of the traced scene.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Upload the current trace settings.
    pub fn upload_settings(&self, queue: &Queue, settings: &TraceSettings) {
        crate::graphics::buffer::write_buffer(queue, &self.settings_buffer, 0, &[*settings]);
    }

    /// Append a mesh to the geometry heap, returning where it landed. Returns
    /// `None` for a mesh that cannot be traced (empty, or an index count that is
    /// not a whole number of triangles).
    fn push_geometry(&mut self, mesh: &MeshComponent) -> Option<GeometrySlice> {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() || mesh.indices.len() % 3 != 0 {
            return None;
        }
        let slice = GeometrySlice {
            vertex_offset: (self.vertices.len() / HEAP_FLOATS_PER_VERTEX) as u32,
            vertex_count: mesh.vertices.len() as u32,
            index_offset: self.indices.len() as u32,
            index_count: mesh.indices.len() as u32,
        };
        for v in &mesh.vertices {
            self.vertices.extend_from_slice(&[
                v.position.x,
                v.position.y,
                v.position.z,
                v.normal.x,
                v.normal.y,
                v.normal.z,
                v.color.x,
                v.color.y,
                v.color.z,
                v.uv.x,
                v.uv.y,
            ]);
        }
        // Indices are stored mesh-relative. The BLAS builder is told where the
        // mesh starts via `first_vertex`/`first_index`, and the hit shader adds
        // the instance's `vertex_offset` itself, so rebasing them here would
        // double-count the offset.
        self.indices.extend_from_slice(&mesh.indices);
        Some(slice)
    }

    /// Grow the GPU heaps to fit the CPU mirror, if needed. Returns true when a
    /// buffer was reallocated (and therefore the bind group was rebuilt).
    ///
    /// Reallocation does not invalidate existing BLASes: a build copies the
    /// geometry into the structure's own memory, so the input buffer is only
    /// needed while the build is recorded.
    fn ensure_capacity(&mut self, device: &Device) -> bool {
        let needed_vertices = self.vertices.len() / HEAP_FLOATS_PER_VERTEX;
        let needed_indices = self.indices.len();
        let mut grew = false;
        while self.vertex_capacity < needed_vertices {
            self.vertex_capacity *= 2;
            grew = true;
        }
        while self.index_capacity < needed_indices {
            self.index_capacity *= 2;
            grew = true;
        }
        if !grew {
            return false;
        }
        self.vertex_buffer = Self::create_heap_buffer(
            device,
            "traced_vertex_heap",
            (self.vertex_capacity * HEAP_FLOATS_PER_VERTEX * 4) as u64,
        );
        self.index_buffer =
            Self::create_heap_buffer(device, "traced_index_heap", (self.index_capacity * 4) as u64);
        self.bind_group = Self::create_bind_group(
            device,
            &self.layout,
            &self.tlas,
            &self.instance_buffer,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.settings_buffer,
        );
        true
    }

    /// Bring the acceleration structures in line with this frame's draw list.
    ///
    /// `requests` must be ordered so that request `i` is the instance the scene
    /// pass will shade with object index `i` — the traced shader looks up its
    /// instance record by the hit's `instance_custom_data`, which is set to that
    /// same index here.
    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        requests: &[(InstanceRequest, &MeshComponent)],
    ) {
        // 1. Make sure every referenced mesh has geometry in the heap and a BLAS.
        let mut new_geometry = false;
        let mut pending: Vec<u64> = Vec::new();
        for (request, mesh) in requests {
            if self.blas.contains_key(&request.entity_id) {
                continue;
            }
            let Some(slice) = self.push_geometry(mesh) else {
                continue;
            };
            new_geometry = true;
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: slice.vertex_count,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(slice.index_count),
                // OPAQUE is mandatory in practice: naga has no candidate
                // intersection support yet, so a non-opaque BLAS records no
                // hits at all and every ray would silently miss.
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("mesh_blas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            );
            self.blas.insert(
                request.entity_id,
                MeshBlas {
                    blas,
                    size,
                    slice,
                    built: false,
                },
            );
            pending.push(request.entity_id);
        }

        if new_geometry {
            self.ensure_capacity(device);
            queue.write_buffer(&self.vertex_buffer, 0, cast_slice(&self.vertices));
            queue.write_buffer(&self.index_buffer, 0, cast_slice(&self.indices));
        }

        // 2. Decide whether the TLAS needs rebuilding at all.
        let snapshot: Vec<InstanceSnapshot> = requests
            .iter()
            .filter(|(r, _)| self.blas.contains_key(&r.entity_id))
            .map(|(r, _)| {
                let t = tlas_transform(&r.model);
                let mut bits = [0u32; 12];
                for (dst, src) in bits.iter_mut().zip(t.iter()) {
                    *dst = src.to_bits();
                }
                InstanceSnapshot {
                    entity_id: r.entity_id,
                    transform: bits,
                    material_index: r.material_index,
                }
            })
            .collect();

        let needs_tlas_build = !pending.is_empty() || snapshot != self.last_snapshot;
        if !needs_tlas_build {
            return;
        }

        // 3. Fill the TLAS instance slots and the shader-side instance records.
        //
        // Slot i must correspond to draw i: `instance_custom_data` is what the
        // hit shader uses to find the material and geometry offsets, and the
        // TLAS's own instance ordering is what determines the reported index.
        let mut records: Vec<TracedInstance> = Vec::with_capacity(snapshot.len());
        let mut slot = 0usize;
        for (request, _) in requests {
            let Some(entry) = self.blas.get(&request.entity_id) else {
                continue;
            };
            if slot >= self.max_instances as usize {
                break;
            }
            self.tlas[slot] = Some(wgpu::TlasInstance::new(
                &entry.blas,
                tlas_transform(&request.model),
                slot as u32,
                // Mask 0xff: every instance is visible to every ray. Culling
                // happens per-ray through the ray flags, not per-instance.
                0xff,
            ));
            records.push(TracedInstance {
                vertex_offset: entry.slice.vertex_offset,
                index_offset: entry.slice.index_offset,
                material_index: request.material_index,
                _padding: 0,
            });
            slot += 1;
        }
        // Clear any slots the previous frame used and this one does not, so a
        // removed object stops casting shadows and reflections.
        for stale in slot..self.instance_count as usize {
            if stale < self.max_instances as usize {
                self.tlas[stale] = None;
            }
        }
        self.instance_count = slot as u32;

        if !records.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, cast_slice(&records));
        }

        // 4. Record the builds. Only unbuilt BLASes are included, which is what
        // keeps static geometry from being rebuilt every frame.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("acceleration_structure_build"),
        });
        let entries: Vec<wgpu::BlasBuildEntry<'_>> = pending
            .iter()
            .filter_map(|id| self.blas.get(id))
            .map(|entry| wgpu::BlasBuildEntry {
                blas: &entry.blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &entry.size,
                        vertex_buffer: &self.vertex_buffer,
                        first_vertex: entry.slice.vertex_offset,
                        vertex_stride: HEAP_VERTEX_STRIDE,
                        index_buffer: Some(&self.index_buffer),
                        first_index: Some(entry.slice.index_offset),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
            .collect();
        encoder.build_acceleration_structures(entries.iter(), std::iter::once(&self.tlas));
        queue.submit(std::iter::once(encoder.finish()));

        self.blas_builds += entries.len() as u64;
        self.tlas_builds += 1;
        drop(entries);
        for id in &pending {
            if let Some(entry) = self.blas.get_mut(id) {
                entry.built = true;
            }
        }
        self.last_snapshot = snapshot;
    }

    /// Whether every BLAS referenced by the TLAS has been built. A false here
    /// means the next traced frame would hit a validation error.
    pub fn all_blas_built(&self) -> bool {
        self.blas.values().all(|e| e.built)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::Vec3;

    #[test]
    fn heap_stride_matches_the_mesh_vertex_layout() {
        // The BLAS builder walks the heap at HEAP_VERTEX_STRIDE bytes per
        // vertex. If MeshVertex ever grows a field, the stride must follow or
        // the acceleration structure reads garbage positions.
        assert_eq!(
            HEAP_VERTEX_STRIDE as usize,
            std::mem::size_of::<crate::scene::component::MeshVertex>()
        );
        // Float32x3 requires a stride that is a multiple of 4 and at least 12.
        assert_eq!(HEAP_VERTEX_STRIDE % 4, 0);
        assert!(HEAP_VERTEX_STRIDE >= 12);
    }

    #[test]
    fn traced_instance_has_a_16_byte_stride() {
        assert_eq!(std::mem::size_of::<TracedInstance>(), 16);
        assert_eq!(std::mem::size_of::<TracedInstance>() % 16, 0);
    }

    #[test]
    fn trace_settings_is_a_valid_uniform_block() {
        assert_eq!(std::mem::size_of::<TraceSettings>(), 32);
        assert_eq!(std::mem::size_of::<TraceSettings>() % 16, 0);
    }

    /// A transposed instance transform is the classic silent failure: it passes
    /// validation and simply puts traced geometry in the wrong place. Pin the
    /// conversion by checking it against the point it must transform.
    #[test]
    fn tlas_transform_is_row_major_and_preserves_translation() {
        let model = Mat4::translation(1.0, 2.0, 3.0);
        let t = tlas_transform(&model);
        // Row-major 3x4: the translation column is the last entry of each row.
        assert_eq!(t[3], 1.0);
        assert_eq!(t[7], 2.0);
        assert_eq!(t[11], 3.0);
        // The rotational part is identity.
        assert_eq!([t[0], t[1], t[2]], [1.0, 0.0, 0.0]);
        assert_eq!([t[4], t[5], t[6]], [0.0, 1.0, 0.0]);
        assert_eq!([t[8], t[9], t[10]], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn tlas_transform_agrees_with_mat4_on_every_basis_vector() {
        // Rotation composed with a non-uniform scale and a translation: enough
        // asymmetry that a transposed conversion cannot pass by accident.
        let model = Mat4::translation(1.0, -2.0, 0.5)
            .mul(Mat4::rotation_y(0.7))
            .mul(Mat4::scaling(2.0, 3.0, 0.5));
        let t = tlas_transform(&model);
        let apply = |p: Vec3| {
            Vec3::new(
                t[0] * p.x + t[1] * p.y + t[2] * p.z + t[3],
                t[4] * p.x + t[5] * p.y + t[6] * p.z + t[7],
                t[8] * p.x + t[9] * p.y + t[10] * p.z + t[11],
            )
        };
        for p in [
            Vec3::ZERO,
            Vec3::UNIT_X,
            Vec3::UNIT_Y,
            Vec3::UNIT_Z,
            Vec3::new(-1.5, 2.5, 3.5),
        ] {
            let want = model.transform_point(p);
            let got = apply(p);
            assert!(
                (want - got).length() < 1e-5,
                "transform disagreed at {p}: mat4 gave {want}, tlas form gave {got}"
            );
        }
    }

    #[test]
    fn trace_flags_round_trip() {
        let mut s = TraceSettings::default();
        assert!(s.has(trace_flags::SHADOWS));
        s.set(trace_flags::SHADOWS, false);
        assert!(!s.has(trace_flags::SHADOWS));
        assert!(
            s.has(trace_flags::REFLECTIONS),
            "clearing one flag must not disturb the others"
        );
        s.set(trace_flags::SHADOWS, true);
        assert!(s.has(trace_flags::SHADOWS));
    }

    /// The ray budget shown in the HUD has to react to the settings, otherwise
    /// it is decoration rather than a measurement.
    #[test]
    fn ray_budget_tracks_the_settings() {
        let mut s = TraceSettings {
            max_depth: 0,
            shadow_samples: 1,
            ao_samples: 0,
            flags: trace_flags::SHADOWS,
            ..TraceSettings::default()
        };
        assert_eq!(s.rays_per_fragment(2), 2);

        s.ao_samples = 4;
        s.set(trace_flags::AMBIENT_OCCLUSION, true);
        assert_eq!(s.rays_per_fragment(2), 6);

        // Each bounce costs its own ray plus a fresh set of shadow rays.
        s.max_depth = 2;
        s.set(trace_flags::REFLECTIONS, true);
        assert_eq!(s.rays_per_fragment(2), 6 + 2 * (1 + 2));

        // Turning everything off must cost nothing.
        s.flags = 0;
        assert_eq!(s.rays_per_fragment(2), 0);
    }

    #[test]
    fn disabled_shadows_are_not_billed_even_with_samples_set() {
        let s = TraceSettings {
            max_depth: 0,
            shadow_samples: 8,
            ao_samples: 0,
            flags: 0,
            ..TraceSettings::default()
        };
        assert_eq!(s.rays_per_fragment(4), 0);
    }

    #[test]
    fn default_settings_enable_every_traced_feature() {
        let s = TraceSettings::default();
        for flag in [
            trace_flags::SHADOWS,
            trace_flags::REFLECTIONS,
            trace_flags::REFRACTION,
            trace_flags::AMBIENT_OCCLUSION,
        ] {
            assert!(s.has(flag), "flag {flag} should default to on");
        }
        assert!(s.max_depth >= 1, "reflections need at least one bounce");
    }
}
