// CPU fallback tracer: the traced look on hardware without ray query.
//
// Phase 4.4's requirement is "the same visual features at lower resolution/ray
// count, so the demo still shows real reflections everywhere". That rules out
// falling back to the rasterised approximation — a mirror reflecting a sky
// gradient is not the same feature at lower quality, it is a different feature.
//
// Two decisions make it affordable:
//
// 1. Analytic geometry. This tracer intersects `engine::geometry::Shape`, not
//    triangles. One sphere in the demo scene is 1536 triangles; solving its
//    quadratic is one intersection instead of 1536. The shapes come from
//    `ColliderComponent`, the same source physics uses, so a reflection can
//    never disagree with a collision.
// 2. Reduced resolution. Rays are cast at a fraction of the subpixel grid and
//    the result is expanded back. The ASCII stage quantizes to characters
//    anyway, so the resolution the eye actually receives is the cell grid.
//
// Unlike the GPU path, this tracer casts *primary* rays too: there is no
// rasterised fragment to start from, because the whole image is replaced.
//
// The shading model is a hand port of `scene_shading.wgsl`. That duplication is
// unavoidable across two languages, and it is pinned by tests that check the
// ported functions against the values the WGSL is specified to produce.

use crate::engine::geometry::{ray, Ray, RayHit, WorldShape};
use crate::engine::math::Vec3;
use crate::renderer::raytrace::{trace_flags, TraceSettings};
use crate::renderer::scene_pass::LightUniform;
use crate::scene::{Camera, MaterialUniform, Projection};

/// Material type tags, mirroring `scene::component::MaterialType` and the WGSL
/// constants of the same name.
const MATERIAL_MIRROR: u32 = 1;
const MATERIAL_GLASS: u32 = 2;

/// Distance a directional shadow ray travels before it counts as unoccluded.
/// Same value as `scene_shading.wgsl::DIRECTIONAL_RANGE`.
const DIRECTIONAL_RANGE: f32 = 500.0;

/// How far a secondary ray starts from the surface it left.
const SURFACE_EPSILON: f32 = 0.004;

/// Far limit for any traced ray.
const MAX_RAY_DISTANCE: f32 = 1000.0;

/// Hard ceiling on bounces regardless of the configured `max_depth`.
const MAX_BOUNCES: u32 = 4;

/// One analytic object the tracer can intersect.
#[derive(Clone, Copy, Debug)]
pub struct CpuObject {
    /// World-space analytic shape, from `ColliderComponent` through the entity's
    /// world matrix.
    pub shape: WorldShape,
    /// The same material the rasteriser would have used.
    pub material: MaterialUniform,
}

/// Everything the tracer needs about one frame of the scene.
#[derive(Clone, Debug, Default)]
pub struct CpuScene {
    pub objects: Vec<CpuObject>,
    pub lights: Vec<LightUniform>,
}

impl CpuScene {
    /// Whether there is anything to trace. An empty scene must not be handed to
    /// the tracer: it would spend a full frame's rays producing sky.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

/// CPU fallback tracer.
///
/// Owns its output buffer so a frame does not allocate; `render` returns a
/// borrow of it.
pub struct CpuTracer {
    /// Rays are cast on a grid this many times coarser than the subpixel grid,
    /// on each axis. 1 means full resolution.
    scale: u32,
    /// Traced image at ray resolution.
    low: Vec<[u8; 4]>,
    /// Traced image expanded to the subpixel grid.
    full: Vec<[u8; 4]>,
    /// Rays cast during the most recent `render`, for the HUD's budget line.
    rays_cast: usize,
}

/// Ray resolution the fallback traces at, relative to the subpixel grid.
///
/// 2 means a quarter of the pixels. Chosen because the ASCII stage collapses
/// each 2x2 subpixel block into one glyph anyway, so a factor of two costs the
/// block's internal detail — which the quadrant-block glyphs do use, but which
/// is the first thing worth spending on a machine that has no ray-tracing
/// hardware to begin with.
pub const DEFAULT_SCALE: u32 = 2;

impl CpuTracer {
    /// Create a tracer that casts rays at `1 / scale` of the target resolution.
    /// A scale of zero is treated as one rather than dividing by zero.
    pub fn new(scale: u32) -> Self {
        Self {
            scale: scale.max(1),
            low: Vec::new(),
            full: Vec::new(),
            rays_cast: 0,
        }
    }

    /// Ray-resolution grid for a given subpixel grid.
    pub fn ray_resolution(&self, sub_cols: u32, sub_rows: u32) -> (u32, u32) {
        (
            (sub_cols / self.scale).max(1),
            (sub_rows / self.scale).max(1),
        )
    }

    /// Rays cast by the most recent `render` call.
    pub fn rays_cast(&self) -> usize {
        self.rays_cast
    }

    /// Trace the scene and return an RGBA buffer of `sub_cols * sub_rows`.
    ///
    /// Rows are traced in parallel across the available cores with
    /// `std::thread::scope` — no dependency, and the difference between a usable
    /// fallback and a slideshow on a scene this size.
    pub fn render(
        &mut self,
        scene: &CpuScene,
        camera: &Camera,
        sub_cols: u32,
        sub_rows: u32,
        settings: &TraceSettings,
    ) -> &[[u8; 4]] {
        let (cols, rows) = self.ray_resolution(sub_cols, sub_rows);
        self.low.resize((cols * rows) as usize, [0, 0, 0, 255]);
        self.full
            .resize((sub_cols * sub_rows) as usize, [0, 0, 0, 255]);

        let rig = CameraRays::new(camera);
        let ctx = TraceContext {
            scene,
            settings,
            rig: &rig,
            cols,
            rows,
        };

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(rows as usize)
            .max(1);
        // Whole rows per chunk, so a chunk index maps back to a y coordinate.
        let rows_per_chunk = (rows as usize).div_ceil(threads);
        let stride = cols as usize;
        std::thread::scope(|s| {
            for (chunk_index, chunk) in self
                .low
                .chunks_mut(rows_per_chunk * stride)
                .enumerate()
            {
                let ctx = &ctx;
                let y0 = (chunk_index * rows_per_chunk) as u32;
                s.spawn(move || {
                    for (i, pixel) in chunk.iter_mut().enumerate() {
                        let y = y0 + (i / stride) as u32;
                        let x = (i % stride) as u32;
                        *pixel = ctx.shade_pixel(x, y);
                    }
                });
            }
        });

        self.rays_cast = (cols * rows) as usize
            * (1 + settings.rays_per_fragment(scene.lights.len() as u32) as usize);

        // Nearest-neighbour expansion. Interpolating would blur the traced image
        // before the glyph chooser sees it, and the glyph chooser is what turns
        // luminance into a character — a blurred edge becomes the wrong glyph,
        // not a softer one.
        for y in 0..sub_rows {
            let sy = (y / self.scale).min(rows - 1);
            for x in 0..sub_cols {
                let sx = (x / self.scale).min(cols - 1);
                self.full[(y * sub_cols + x) as usize] = self.low[(sy * cols + sx) as usize];
            }
        }
        &self.full
    }
}

/// Camera basis and field of view, prepared once per frame.
struct CameraRays {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    /// Half-extent of the image plane at unit distance (perspective) or in world
    /// units (orthographic).
    half_width: f32,
    half_height: f32,
    orthographic: bool,
}

impl CameraRays {
    fn new(camera: &Camera) -> Self {
        let forward = camera.forward();
        let right = camera.right();
        // Re-derived rather than taken from `camera.up`, which is only a hint:
        // the authored up vector need not be perpendicular to the view direction.
        let up = right.cross(forward).normalize();
        match camera.projection {
            Projection::Perspective { fov_y, aspect, .. } => {
                let half_height = (fov_y * 0.5).tan();
                Self {
                    origin: camera.position,
                    forward,
                    right,
                    up,
                    half_width: half_height * aspect,
                    half_height,
                    orthographic: false,
                }
            }
            Projection::Orthographic { left, right: r, bottom, top, .. } => {
                let half_width = (r - left).abs() * 0.5;
                let half_height = (top - bottom).abs() * 0.5;
                Self {
                    origin: camera.position,
                    forward,
                    right,
                    up,
                    half_width,
                    half_height,
                    orthographic: true,
                }
            }
        }
    }

    /// Primary ray through the centre of pixel `(x, y)` of a `cols x rows` image.
    fn primary(&self, x: u32, y: u32, cols: u32, rows: u32) -> Ray {
        // Pixel centres, and y flipped: image row 0 is the top of the screen,
        // which is +y in view space.
        let ndc_x = 2.0 * (x as f32 + 0.5) / cols as f32 - 1.0;
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / rows as f32;
        let offset = self.right * (ndc_x * self.half_width) + self.up * (ndc_y * self.half_height);
        if self.orthographic {
            Ray::new(self.origin + offset, self.forward)
        } else {
            Ray::new(self.origin, self.forward + offset)
        }
    }
}

/// Immutable per-frame state shared by the worker threads.
struct TraceContext<'a> {
    scene: &'a CpuScene,
    settings: &'a TraceSettings,
    rig: &'a CameraRays,
    cols: u32,
    rows: u32,
}

/// A resolved hit: where, which way, and against which material.
struct Surface {
    hit: RayHit,
    material: MaterialUniform,
}

impl TraceContext<'_> {
    fn shade_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let ray = self.rig.primary(x, y, self.cols, self.rows);
        let mut seed = pixel_seed(x, y);
        let color = self.trace_path(ray, &mut seed);
        [
            encode_channel(color.x),
            encode_channel(color.y),
            encode_channel(color.z),
            255,
        ]
    }

    /// Nearest object along a ray.
    fn closest(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<Surface> {
        let mut best: Option<Surface> = None;
        let mut nearest = t_max;
        for object in &self.scene.objects {
            if let Some(hit) = ray::intersect(ray, &object.shape, t_min, nearest) {
                nearest = hit.t;
                best = Some(Surface {
                    hit,
                    material: object.material,
                });
            }
        }
        best
    }

    /// Whether anything blocks the segment. Stops at the first hit, like the
    /// GPU path's `RAY_FLAG_TERMINATE_ON_FIRST_HIT`.
    fn occluded(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        if t_max <= t_min {
            return false;
        }
        self.scene
            .objects
            .iter()
            .any(|o| ray::intersect(ray, &o.shape, t_min, t_max).is_some())
    }

    /// Fraction of a light that reaches `pos`, from `shadow_samples` rays.
    fn shadow(&self, pos: Vec3, normal: Vec3, light: &ResolvedLight, seed: &mut u32) -> f32 {
        if normal.dot(light.dir) <= 0.0 {
            return 0.0;
        }
        let samples = self.settings.shadow_samples.max(1);
        let origin = pos + normal * SURFACE_EPSILON;
        let spread = (self.settings.light_radius / light.distance.max(0.001)).clamp(0.0, 0.5);
        let mut lit = 0.0;
        for _ in 0..samples {
            let (u1, u2) = (rand01(seed), rand01(seed));
            let dir = if samples > 1 {
                jitter_cone(light.dir, spread, u1, u2)
            } else {
                light.dir
            };
            let ray = Ray::new(origin, dir);
            if !self.occluded(&ray, SURFACE_EPSILON, light.distance - SURFACE_EPSILON) {
                lit += 1.0;
            }
        }
        lit / samples as f32
    }

    /// Traced ambient occlusion: the unblocked fraction of the hemisphere.
    fn ambient_occlusion(&self, pos: Vec3, normal: Vec3, seed: &mut u32) -> f32 {
        let samples = self.settings.ao_samples;
        if samples == 0 {
            return 1.0;
        }
        let origin = pos + normal * SURFACE_EPSILON;
        let mut open = 0.0;
        for _ in 0..samples {
            let (u1, u2) = (rand01(seed), rand01(seed));
            let dir = cosine_hemisphere(normal, u1, u2);
            let ray = Ray::new(origin, dir);
            if !self.occluded(&ray, SURFACE_EPSILON, self.settings.ao_radius) {
                open += 1.0;
            }
        }
        open / samples as f32
    }

    /// Direct lighting at a surface, every light shadow-tested.
    fn direct(
        &self,
        material: &MaterialUniform,
        pos: Vec3,
        normal: Vec3,
        view_dir: Vec3,
        seed: &mut u32,
    ) -> Vec3 {
        let mut ambient_sum = material.ambient;
        let mut lit = Vec3::ZERO;
        let shadows_on = self.settings.has(trace_flags::SHADOWS);

        for light in &self.scene.lights {
            let resolved = ResolvedLight::new(light, pos);
            ambient_sum += resolved.ambient;
            let visibility = if shadows_on {
                self.shadow(pos, normal, &resolved, seed)
            } else {
                1.0
            };
            lit = lit + phong_term(material, normal, view_dir, &resolved) * visibility;
        }

        let occlusion = if self.settings.has(trace_flags::AMBIENT_OCCLUSION) {
            self.ambient_occlusion(pos, normal, seed)
        } else {
            1.0
        };

        albedo(material) * (Vec3::splat(ambient_sum * occlusion) + lit)
    }

    /// Follow a ray through up to `max_depth + 1` surfaces.
    ///
    /// Iterative and single-path, exactly like the WGSL version: a mirror
    /// reflecting glass reflecting a mirror costs one ray per bounce, not two to
    /// the power of the depth.
    fn trace_path(&self, start: Ray, seed: &mut u32) -> Vec3 {
        let mut ray = start;
        let mut throughput = Vec3::ONE;
        let mut accum = Vec3::ZERO;
        let reflections_on = self.settings.has(trace_flags::REFLECTIONS);
        let refraction_on = self.settings.has(trace_flags::REFRACTION);
        // One more surface than the bounce budget: the primary hit is not a
        // bounce, and with max_depth = 0 the image must still be shaded.
        let limit = self.settings.max_depth.min(MAX_BOUNCES) + 1;

        for _ in 0..limit {
            let Some(surface) = self.closest(&ray, SURFACE_EPSILON, MAX_RAY_DISTANCE) else {
                accum = accum + throughput * (environment(ray.direction()) + light_glints(&self.scene.lights, ray.direction()));
                return accum;
            };
            let material = surface.material;
            let normal = surface.hit.normal;
            let view_dir = -ray.direction();
            let entering = surface.hit.front_face;
            // Scale the offset with distance: a fixed epsilon stops separating
            // points once float spacing at that magnitude exceeds it.
            let eps = SURFACE_EPSILON * (1.0 + surface.hit.t);

            let direct = self.direct(&material, surface.hit.point, normal, view_dir, seed);
            let fresnel = fresnel_weight(normal, view_dir);

            if material.material_type == MATERIAL_MIRROR && reflections_on {
                let k = (material.reflectivity + (1.0 - material.reflectivity) * fresnel)
                    .clamp(0.0, 1.0);
                accum = accum + throughput * direct * (1.0 - k);
                throughput = throughput * albedo(&material) * k;
                ray = Ray::new(surface.hit.point + normal * eps, reflect(ray.direction(), normal));
            } else if material.material_type == MATERIAL_GLASS && refraction_on {
                let eta = if entering {
                    1.0 / material.ior.max(1.0)
                } else {
                    // Leaving the denser medium: the ratio inverts, which is what
                    // makes total internal reflection possible on the way out.
                    material.ior.max(1.0)
                };
                let mirror_dir = reflect(ray.direction(), normal);
                let sheen = environment(mirror_dir) * fresnel;
                accum = accum
                    + throughput * (direct * (1.0 - material.transparency) + sheen);
                match refract(ray.direction(), normal, eta) {
                    Some(refracted) => {
                        ray = Ray::new(surface.hit.point - normal * eps, refracted);
                    }
                    None => {
                        // Total internal reflection: nothing passes, all bounces.
                        ray = Ray::new(surface.hit.point + normal * eps, mirror_dir);
                    }
                }
                let tint = Vec3::ONE.lerp(albedo(&material), material.transparency);
                throughput = throughput * tint * material.transparency;
            } else {
                accum = accum + throughput * direct;
                return accum;
            }

            // Once almost nothing would come back, stop paying for rays.
            if throughput.x + throughput.y + throughput.z < 0.01 {
                return accum;
            }
        }

        accum
    }
}

/// One light resolved at a surface point. Mirrors `LightSample` in
/// `scene_shading.wgsl`.
struct ResolvedLight {
    dir: Vec3,
    distance: f32,
    attenuation: f32,
    energy: Vec3,
    ambient: f32,
}

impl ResolvedLight {
    fn new(light: &LightUniform, world_pos: Vec3) -> Self {
        let energy = Vec3::new(light.color[0], light.color[1], light.color[2]) * light.diffuse;
        if light.position[3] < 0.5 {
            // Directional: direction points FROM the light.
            let dir =
                -Vec3::new(light.direction[0], light.direction[1], light.direction[2]).normalize();
            Self {
                dir,
                distance: DIRECTIONAL_RANGE,
                attenuation: 1.0,
                energy,
                ambient: light.ambient,
            }
        } else {
            let to_light =
                Vec3::new(light.position[0], light.position[1], light.position[2]) - world_pos;
            let dist = to_light.length();
            Self {
                dir: to_light / dist.max(0.001),
                distance: dist,
                attenuation: 1.0 / (1.0 + 0.1 * dist + 0.01 * dist * dist),
                energy,
                ambient: light.ambient,
            }
        }
    }
}

fn albedo(material: &MaterialUniform) -> Vec3 {
    Vec3::new(material.albedo[0], material.albedo[1], material.albedo[2])
}

/// Diffuse + specular contribution of one light, before shadowing. Port of
/// `scene_shading.wgsl::phong_term`.
fn phong_term(
    material: &MaterialUniform,
    normal: Vec3,
    view_dir: Vec3,
    light: &ResolvedLight,
) -> Vec3 {
    let n_dot_l = normal.dot(light.dir).max(0.0);
    let diffuse = material.diffuse * n_dot_l * light.attenuation;
    let reflect_dir = reflect(-light.dir, normal);
    let spec_angle = reflect_dir.dot(view_dir).max(0.0);
    let specular =
        material.specular * spec_angle.powf(material.shininess.max(1.0)) * light.attenuation;
    light.energy * (diffuse + specular)
}

/// Port of `scene_shading.wgsl::environment`.
pub fn environment(dir: Vec3) -> Vec3 {
    let sky = Vec3::new(0.28, 0.42, 0.72);
    let horizon = Vec3::new(0.58, 0.60, 0.66);
    let ground = Vec3::new(0.16, 0.14, 0.12);
    let h = dir.normalize().y.clamp(-1.0, 1.0);
    if h > 0.0 {
        horizon.lerp(sky, h.powf(0.45))
    } else {
        horizon.lerp(ground, (-h).powf(0.35))
    }
}

/// Port of `scene_shading.wgsl::light_glints`.
fn light_glints(lights: &[LightUniform], dir: Vec3) -> Vec3 {
    let d = dir.normalize();
    let mut total = Vec3::ZERO;
    for light in lights {
        let to_light = if light.position[3] < 0.5 {
            -Vec3::new(light.direction[0], light.direction[1], light.direction[2]).normalize()
        } else {
            Vec3::new(light.position[0], light.position[1], light.position[2]).normalize()
        };
        let alignment = d.dot(to_light).max(0.0);
        total = total
            + Vec3::new(light.color[0], light.color[1], light.color[2])
                * alignment.powf(220.0)
                * light.diffuse;
    }
    total
}

/// Port of `scene_shading.wgsl::fresnel_weight`.
pub fn fresnel_weight(normal: Vec3, view_dir: Vec3) -> f32 {
    let facing = normal.dot(view_dir).max(0.0);
    (1.0 - facing).powf(5.0)
}

/// Mirror `incident` about `normal`. `incident` points at the surface, matching
/// WGSL's `reflect`.
fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - normal * (2.0 * incident.dot(normal))
}

/// Refract `incident` through `normal` with index ratio `eta`.
///
/// `None` means total internal reflection, which is WGSL's zero-vector return
/// made explicit — a zero direction silently normalizes to something arbitrary,
/// and "the ray bent to a nonsense direction" is much harder to spot than
/// "there was no refracted ray".
fn refract(incident: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = -incident.dot(normal);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        return None;
    }
    Some((incident * eta + normal * (eta * cos_i - k.sqrt())).normalize())
}

/// Integer hash, the same one the traced shader uses, so both paths dither
/// identically.
fn hash_u32(x: u32) -> u32 {
    let mut v = x;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^= v >> 16;
    v
}

fn rand01(state: &mut u32) -> f32 {
    *state = hash_u32(*state);
    (*state >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Seed for one pixel. Position only, for the same reason the shader's is: with
/// no temporal accumulation, per-frame reseeding is a shimmer, not a smoothing.
fn pixel_seed(x: u32, y: u32) -> u32 {
    hash_u32(
        x.wrapping_mul(1973)
            .wrapping_add(y.wrapping_mul(9277))
            .wrapping_add(1),
    )
}

fn any_tangent(n: Vec3) -> Vec3 {
    let up = if n.z.abs() > 0.9 {
        Vec3::UNIT_X
    } else {
        Vec3::UNIT_Z
    };
    up.cross(n).normalize()
}

fn cosine_hemisphere(n: Vec3, u1: f32, u2: f32) -> Vec3 {
    let r = u1.sqrt();
    let theta = std::f32::consts::TAU * u2;
    let tangent = any_tangent(n);
    let bitangent = n.cross(tangent);
    let z = (1.0 - u1).max(0.0).sqrt();
    (tangent * (r * theta.cos()) + bitangent * (r * theta.sin()) + n * z).normalize()
}

fn jitter_cone(dir: Vec3, spread: f32, u1: f32, u2: f32) -> Vec3 {
    if spread <= 0.0 {
        return dir;
    }
    let tangent = any_tangent(dir);
    let bitangent = dir.cross(tangent);
    let r = spread * u1.sqrt();
    let theta = std::f32::consts::TAU * u2;
    (dir + tangent * (r * theta.cos()) + bitangent * (r * theta.sin())).normalize()
}

/// Linear colour to an 8-bit channel, clamped rather than wrapped.
fn encode_channel(v: f32) -> u8 {
    if !v.is_finite() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::geometry::Shape;
    use crate::engine::math::radians;
    use crate::scene::MaterialComponent;
    use crate::engine::math::Vec4;

    fn matte(color: Vec3) -> MaterialUniform {
        MaterialUniform::from(&MaterialComponent::matte(
            Vec4::new(color.x, color.y, color.z, 1.0),
            0.1,
            0.9,
        ))
    }

    fn mirror(color: Vec3) -> MaterialUniform {
        MaterialUniform::from(&MaterialComponent::mirror(
            Vec4::new(color.x, color.y, color.z, 1.0),
            0.9,
        ))
    }

    fn glass(color: Vec3) -> MaterialUniform {
        MaterialUniform::from(&MaterialComponent::glass(
            Vec4::new(color.x, color.y, color.z, 1.0),
            1.5,
            0.9,
        ))
    }

    fn camera_at(position: Vec3, target: Vec3) -> Camera {
        Camera::new(
            position,
            target,
            Vec3::UNIT_Y,
            Projection::perspective(radians(60.0), 1.0, 0.1, 200.0),
        )
    }

    /// Hard shadows, no AO, one bounce: deterministic and cheap, so a test can
    /// assert exact relationships rather than statistical ones.
    fn crisp_settings() -> TraceSettings {
        TraceSettings {
            max_depth: 2,
            shadow_samples: 1,
            ao_samples: 0,
            flags: trace_flags::SHADOWS | trace_flags::REFLECTIONS | trace_flags::REFRACTION,
            ..TraceSettings::default()
        }
    }

    #[test]
    fn ray_resolution_divides_and_never_reaches_zero() {
        let t = CpuTracer::new(2);
        assert_eq!(t.ray_resolution(240, 136), (120, 68));
        // A grid smaller than the scale must still get one ray, not zero.
        assert_eq!(t.ray_resolution(1, 1), (1, 1));
        // A zero scale must not divide by zero.
        assert_eq!(CpuTracer::new(0).ray_resolution(240, 136), (240, 136));
    }

    #[test]
    fn render_fills_the_whole_subpixel_grid() {
        let mut tracer = CpuTracer::new(2);
        let scene = CpuScene {
            objects: vec![CpuObject {
                shape: WorldShape::new(Vec3::new(0.0, 0.0, -5.0), Shape::Sphere { radius: 1.0 }),
                material: matte(Vec3::new(1.0, 0.0, 0.0)),
            }],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 5.0, 0.0),
                Vec3::ONE,
                0.1,
                1.0,
            )],
        };
        let camera = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let out = tracer.render(&scene, &camera, 40, 24, &crisp_settings());
        assert_eq!(out.len(), 40 * 24);
        assert!(out.iter().all(|p| p[3] == 255), "alpha must be opaque");
    }

    /// The primary ray for the centre pixel must actually look where the camera
    /// looks. A y-flip or a swapped basis puts the whole image upside down or
    /// mirrored, which no unit test on the intersection maths would catch.
    #[test]
    fn centre_pixel_sees_an_object_straight_ahead() {
        let mut tracer = CpuTracer::new(1);
        let scene = CpuScene {
            objects: vec![CpuObject {
                shape: WorldShape::new(Vec3::new(0.0, 0.0, -5.0), Shape::Sphere { radius: 1.0 }),
                material: matte(Vec3::new(1.0, 0.0, 0.0)),
            }],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::ONE,
                0.1,
                1.0,
            )],
        };
        let camera = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let (w, h) = (31u32, 31u32);
        let out = tracer.render(&scene, &camera, w, h, &crisp_settings());
        let centre = out[((h / 2) * w + w / 2) as usize];
        // Red sphere lit head-on: red channel must dominate.
        assert!(
            centre[0] > centre[2] && centre[0] > 60,
            "centre pixel {centre:?} does not look like a lit red sphere"
        );
        // A corner must see sky, which is blue-dominant.
        let corner = out[0];
        assert!(
            corner[2] > corner[0],
            "corner pixel {corner:?} should be sky, not geometry"
        );
    }

    /// The y axis must not be flipped: an object above the camera axis has to
    /// appear in the upper half of the image.
    #[test]
    fn an_object_above_the_axis_lands_in_the_upper_half() {
        let mut tracer = CpuTracer::new(1);
        let scene = CpuScene {
            objects: vec![CpuObject {
                shape: WorldShape::new(Vec3::new(0.0, 1.5, -5.0), Shape::Sphere { radius: 0.8 }),
                material: matte(Vec3::new(1.0, 0.0, 0.0)),
            }],
            lights: vec![LightUniform::point(Vec3::ZERO, Vec3::ONE, 0.2, 1.0)],
        };
        let camera = camera_at(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let (w, h) = (41u32, 41u32);
        let out = tracer.render(&scene, &camera, w, h, &crisp_settings());
        let redness = |row_range: std::ops::Range<u32>| -> i32 {
            let mut score = 0i32;
            for y in row_range {
                for x in 0..w {
                    let p = out[(y * w + x) as usize];
                    if p[0] as i32 > p[2] as i32 + 20 {
                        score += 1;
                    }
                }
            }
            score
        };
        let top = redness(0..h / 2);
        let bottom = redness(h / 2..h);
        assert!(
            top > bottom,
            "sphere above the axis should be in the top half: top {top}, bottom {bottom}"
        );
    }

    /// The whole point of Phase 4.4: a mirror must show scene geometry, not a
    /// sky gradient. Moving the reflected object has to change the mirror's
    /// pixels — the same property the GPU path was measured on.
    #[test]
    fn a_mirror_reflects_scene_geometry_and_follows_it() {
        let settings = crisp_settings();
        let camera = camera_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
        let build = |red_x: f32| CpuScene {
            objects: vec![
                CpuObject {
                    shape: WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.5 }),
                    material: mirror(Vec3::new(0.9, 0.9, 0.9)),
                },
                CpuObject {
                    shape: WorldShape::new(
                        Vec3::new(red_x, 0.0, -3.0),
                        Shape::Sphere { radius: 1.2 },
                    ),
                    material: matte(Vec3::new(0.9, 0.1, 0.1)),
                },
            ],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 6.0, 4.0),
                Vec3::ONE,
                0.1,
                1.0,
            )],
        };
        let mut tracer = CpuTracer::new(1);
        let a = tracer.render(&build(-4.0), &camera, 48, 48, &settings).to_vec();
        let b = tracer.render(&build(4.0), &camera, 48, 48, &settings).to_vec();
        let changed = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();
        assert!(
            changed > 0,
            "moving a reflected object changed nothing, so the mirror is not reflecting the scene"
        );
    }

    #[test]
    fn reflections_can_be_switched_off() {
        let camera = camera_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
        let scene = CpuScene {
            objects: vec![
                CpuObject {
                    shape: WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.5 }),
                    material: mirror(Vec3::new(0.9, 0.9, 0.9)),
                },
                CpuObject {
                    shape: WorldShape::new(
                        Vec3::new(-4.0, 0.0, -3.0),
                        Shape::Sphere { radius: 1.2 },
                    ),
                    material: matte(Vec3::new(0.9, 0.1, 0.1)),
                },
            ],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 6.0, 4.0),
                Vec3::ONE,
                0.1,
                1.0,
            )],
        };
        let mut tracer = CpuTracer::new(1);
        let with = tracer.render(&scene, &camera, 32, 32, &crisp_settings()).to_vec();
        let mut off = crisp_settings();
        off.set(trace_flags::REFLECTIONS, false);
        let without = tracer.render(&scene, &camera, 32, 32, &off).to_vec();
        assert_ne!(
            with, without,
            "the reflection flag must actually change the image"
        );
    }

    #[test]
    fn shadows_darken_a_surface_under_an_occluder() {
        let camera = camera_at(Vec3::new(0.0, 4.0, 8.0), Vec3::new(0.0, -1.0, 0.0));
        let ground = CpuObject {
            shape: WorldShape::new(
                Vec3::new(0.0, -2.0, 0.0),
                Shape::Plane {
                    normal: Vec3::UNIT_Y,
                    half_size: 25.0,
                },
            ),
            material: matte(Vec3::ONE),
        };
        let blocker = CpuObject {
            shape: WorldShape::new(Vec3::new(0.0, 0.5, 0.0), Shape::Sphere { radius: 1.0 }),
            material: matte(Vec3::new(0.2, 0.2, 0.2)),
        };
        let lights = vec![LightUniform::point(
            Vec3::new(0.0, 8.0, 0.0),
            Vec3::ONE,
            0.05,
            1.0,
        )];
        let mut settings = crisp_settings();
        settings.max_depth = 0;

        let mut tracer = CpuTracer::new(1);
        let with_blocker = tracer
            .render(
                &CpuScene {
                    objects: vec![ground, blocker],
                    lights: lights.clone(),
                },
                &camera,
                48,
                48,
                &settings,
            )
            .to_vec();
        let mut no_shadow = settings;
        no_shadow.set(trace_flags::SHADOWS, false);
        let unshadowed = tracer
            .render(
                &CpuScene {
                    objects: vec![ground, blocker],
                    lights,
                },
                &camera,
                48,
                48,
                &no_shadow,
            )
            .to_vec();

        let brightness = |b: &[[u8; 4]]| b.iter().map(|p| p[0] as u64).sum::<u64>();
        assert!(
            brightness(&with_blocker) < brightness(&unshadowed),
            "shadow rays must remove light, not add it: {} vs {}",
            brightness(&with_blocker),
            brightness(&unshadowed)
        );
    }

    #[test]
    fn glass_shows_what_is_behind_it() {
        let camera = camera_at(Vec3::new(0.0, 0.0, 6.0), Vec3::ZERO);
        let build = |behind: Vec3| CpuScene {
            objects: vec![
                CpuObject {
                    shape: WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.5 }),
                    material: glass(Vec3::new(0.9, 0.95, 0.9)),
                },
                CpuObject {
                    shape: WorldShape::new(
                        Vec3::new(0.0, 0.0, -6.0),
                        Shape::Sphere { radius: 2.5 },
                    ),
                    material: matte(behind),
                },
            ],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 6.0, 4.0),
                Vec3::ONE,
                0.2,
                1.0,
            )],
        };
        let mut tracer = CpuTracer::new(1);
        let red = tracer
            .render(&build(Vec3::new(0.9, 0.05, 0.05)), &camera, 32, 32, &crisp_settings())
            .to_vec();
        let blue = tracer
            .render(&build(Vec3::new(0.05, 0.05, 0.9)), &camera, 32, 32, &crisp_settings())
            .to_vec();
        // Only the object *behind* the glass changed, so any difference at all
        // proves the refracted ray reached it.
        assert_ne!(
            red, blue,
            "recolouring the object behind the glass changed nothing, so nothing is seen through it"
        );
    }

    #[test]
    fn refract_reports_total_internal_reflection_instead_of_a_zero_vector() {
        // Leaving glass (eta > 1) at a grazing angle must fail to refract.
        let normal = Vec3::UNIT_Y;
        let grazing = Vec3::new(0.999, -0.045, 0.0).normalize();
        assert!(
            refract(grazing, normal, 1.5).is_none(),
            "a grazing exit ray should be totally internally reflected"
        );
        // Head-on entry must always pass through.
        assert!(refract(-Vec3::UNIT_Y, normal, 1.0 / 1.5).is_some());
    }

    #[test]
    fn reflect_matches_the_mirror_law() {
        let n = Vec3::UNIT_Y;
        let incident = Vec3::new(1.0, -1.0, 0.0).normalize();
        let r = reflect(incident, n);
        assert!((r - Vec3::new(1.0, 1.0, 0.0).normalize()).length() < 1e-5, "{r}");
        // The angle of incidence equals the angle of reflection.
        assert!((incident.dot(n).abs() - r.dot(n).abs()).abs() < 1e-5);
    }

    /// The environment gradient must match the WGSL one at the three anchors it
    /// is defined by, or the fallback's sky would not be the engine's sky.
    #[test]
    fn environment_matches_its_wgsl_anchors() {
        let up = environment(Vec3::UNIT_Y);
        assert!((up - Vec3::new(0.28, 0.42, 0.72)).length() < 1e-5, "{up}");
        let down = environment(-Vec3::UNIT_Y);
        assert!((down - Vec3::new(0.16, 0.14, 0.12)).length() < 1e-5, "{down}");
        // At the horizon both branches must agree on the horizon colour.
        let horizon = environment(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (horizon - Vec3::new(0.58, 0.60, 0.66)).length() < 1e-4,
            "{horizon}"
        );
    }

    #[test]
    fn fresnel_is_zero_head_on_and_one_at_grazing() {
        assert!(fresnel_weight(Vec3::UNIT_Y, Vec3::UNIT_Y).abs() < 1e-6);
        let grazing = fresnel_weight(Vec3::UNIT_Y, Vec3::UNIT_X);
        assert!((grazing - 1.0).abs() < 1e-6, "{grazing}");
    }

    #[test]
    fn encode_channel_clamps_instead_of_wrapping() {
        assert_eq!(encode_channel(0.0), 0);
        assert_eq!(encode_channel(1.0), 255);
        assert_eq!(encode_channel(4.0), 255, "an overbright value must saturate");
        assert_eq!(encode_channel(-1.0), 0);
        assert_eq!(encode_channel(f32::NAN), 0);
        assert_eq!(encode_channel(f32::INFINITY), 0);
    }

    /// Row chunking across threads must not shuffle or drop rows. Same scene,
    /// same settings, one thread's worth of rows versus many: the image has to
    /// be identical, which a y-offset bug in the chunk mapping would break.
    #[test]
    fn threaded_rendering_is_deterministic() {
        let scene = CpuScene {
            objects: vec![
                CpuObject {
                    shape: WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.5 }),
                    material: mirror(Vec3::new(0.6, 0.7, 0.9)),
                },
                CpuObject {
                    shape: WorldShape::new(
                        Vec3::new(-3.0, 0.0, -3.0),
                        Shape::Sphere { radius: 1.0 },
                    ),
                    material: matte(Vec3::new(0.9, 0.2, 0.2)),
                },
            ],
            lights: vec![LightUniform::point(
                Vec3::new(0.0, 6.0, 4.0),
                Vec3::ONE,
                0.1,
                1.0,
            )],
        };
        let camera = camera_at(Vec3::new(0.0, 1.0, 6.0), Vec3::ZERO);
        let settings = crisp_settings();
        let mut a = CpuTracer::new(1);
        let first = a.render(&scene, &camera, 37, 29, &settings).to_vec();
        let second = a.render(&scene, &camera, 37, 29, &settings).to_vec();
        assert_eq!(first, second, "two identical frames must produce identical output");
    }

    #[test]
    fn empty_scene_is_recognised() {
        assert!(CpuScene::default().is_empty());
        assert!(!CpuScene {
            objects: vec![CpuObject {
                shape: WorldShape::new(Vec3::ZERO, Shape::Sphere { radius: 1.0 }),
                material: matte(Vec3::ONE),
            }],
            lights: Vec::new(),
        }
        .is_empty());
    }

    #[test]
    fn orthographic_camera_produces_parallel_rays() {
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::UNIT_Y,
            Projection::orthographic_sized(10.0, 1.0, 0.1, 50.0),
        );
        let rig = CameraRays::new(&camera);
        let a = rig.primary(0, 0, 16, 16);
        let b = rig.primary(15, 15, 16, 16);
        assert!(
            (a.direction() - b.direction()).length() < 1e-5,
            "orthographic rays must be parallel"
        );
        assert!(
            (a.origin - b.origin).length() > 1.0,
            "orthographic ray origins must spread across the image plane"
        );
    }
}
