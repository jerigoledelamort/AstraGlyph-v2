// Scene pass: ray-traced fragment shader.
//
// Concatenated after `scene_shading.wgsl`, which supplies the group-0 bindings
// and the surface model. This file adds group 1 — the acceleration structure and
// the geometry side tables — and replaces three approximations with real rays:
//
//   shadows      one shadow map for light[0]  ->  a shadow ray per light
//   reflections  an analytic sky gradient     ->  the actual scene geometry
//   refraction   the same gradient, bent      ->  what is really behind the glass
//
// The primary hit is still rasterised: this shader runs per fragment and already
// knows its own position, normal and material, so a primary ray would be pure
// duplicated work. Only secondary rays are traced.
//
// There is no recursion — WGSL has none. Reflection and refraction are followed
// as a single iterative path with a hard bounce cap, which is what makes the
// per-frame cost predictable (see `TraceSettings::rays_per_fragment`).

@group(1) @binding(0) var scene_tlas: acceleration_structure;

/// Per-instance record, indexed by an intersection's `instance_custom_data`.
/// Mirrors `renderer::raytrace::TracedInstance`.
struct TracedInstance {
    vertex_offset: u32,
    index_offset: u32,
    material_index: u32,
    padding: u32,
};

@group(1) @binding(1) var<storage, read> traced_instances: array<TracedInstance>;

// The geometry heap. Typed as a flat float array rather than an array of vertex
// structs because a WGSL struct of vec3s is padded to 16-byte columns, which
// would disagree with the 44-byte stride the acceleration-structure builder
// reads from the very same memory.
@group(1) @binding(2) var<storage, read> heap_vertices: array<f32>;
@group(1) @binding(3) var<storage, read> heap_indices: array<u32>;

/// Mirrors `renderer::raytrace::TraceSettings`.
struct TraceSettings {
    max_depth: u32,
    shadow_samples: u32,
    ao_samples: u32,
    flags: u32,
    light_radius: f32,
    ao_radius: f32,
    padding: vec2<u32>,
};

@group(1) @binding(4) var<uniform> trace: TraceSettings;

// Feature bits, mirroring `renderer::raytrace::trace_flags`.
const FLAG_SHADOWS: u32 = 1u;
const FLAG_REFLECTIONS: u32 = 2u;
const FLAG_REFRACTION: u32 = 4u;
const FLAG_AO: u32 = 8u;

/// Floats per heap vertex: position(3) + normal(3) + colour(3) + uv(2).
/// Mirrors `renderer::raytrace::HEAP_FLOATS_PER_VERTEX` — change together.
const HEAP_STRIDE: u32 = 11u;
/// f32 offsets of the fields inside one heap vertex.
const HEAP_NORMAL_OFFSET: u32 = 3u;
const HEAP_UV_OFFSET: u32 = 9u;

/// How far a secondary ray starts from the surface it left.
///
/// Too small and a surface shadows itself; too large and thin geometry leaks.
/// This value is scaled by the hit distance for bounce rays, because absolute
/// offsets stop working once a ray has travelled far enough for float spacing to
/// exceed them.
const SURFACE_EPSILON: f32 = 0.004;

/// Far limit for any traced ray. The demo scene is tens of units across.
const MAX_RAY_DISTANCE: f32 = 1000.0;

/// Hard ceiling on bounces, independent of the configured `max_depth`. A shader
/// loop with a runtime bound still needs a compile-time worst case for the cost
/// of a frame to be knowable.
const MAX_BOUNCES: u32 = 4u;

const TAU: f32 = 6.283185307;

// --- Sampling ---------------------------------------------------------------

/// Integer hash (a variant of the Murmur3 finaliser). Self-implemented, like the
/// rest of the maths in this engine, and chosen because it decorrelates adjacent
/// pixel indices well enough that soft shadows do not band.
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

/// Next uniform sample in [0, 1), advancing the state.
fn rand01(state: ptr<function, u32>) -> f32 {
    let next = hash_u32(*state);
    *state = next;
    return f32(next >> 8u) * (1.0 / 16777216.0);
}

/// Seed for one fragment: a function of pixel position only.
///
/// Deliberately *not* seeded by a frame counter. Re-seeding per frame is the
/// textbook move, because it lets noise average out over time — but that needs
/// temporal accumulation to average into, and there is none here. Measured on
/// the demo scene, per-frame seeding changed 19% of the render target between
/// two frames with nothing moving at all, which the ASCII stage renders as a
/// crawling shimmer and which makes any pixel-difference measurement useless.
/// Position-only seeding trades that for a fixed dither pattern.
fn pixel_seed(frag: vec2<f32>) -> u32 {
    let x = u32(frag.x);
    let y = u32(frag.y);
    return hash_u32(x * 1973u + y * 9277u + 1u);
}

/// Any unit vector perpendicular to `n`.
fn any_tangent(n: vec3<f32>) -> vec3<f32> {
    // Picking the axis furthest from n keeps the cross product well conditioned.
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(n.z) > 0.9) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    return normalize(cross(up, n));
}

/// Cosine-weighted direction in the hemisphere around `n`.
///
/// Cosine weighting rather than uniform because the ambient term it feeds is
/// itself cosine weighted; uniform samples would need an extra dot() factor and
/// waste rays near the horizon.
fn cosine_hemisphere(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = TAU * u2;
    let tangent = any_tangent(n);
    let bitangent = cross(n, tangent);
    let z = sqrt(max(0.0, 1.0 - u1));
    return normalize(tangent * (r * cos(theta)) + bitangent * (r * sin(theta)) + n * z);
}

/// Perturb `dir` inside a cone whose angular radius is `spread` radians.
///
/// This is how an area light becomes a penumbra: each shadow ray aims at a
/// slightly different point on the light's disc, and averaging the hits gives a
/// soft edge instead of a hard one.
fn jitter_cone(dir: vec3<f32>, spread: f32, u1: f32, u2: f32) -> vec3<f32> {
    if (spread <= 0.0) {
        return dir;
    }
    let tangent = any_tangent(dir);
    let bitangent = cross(dir, tangent);
    let r = spread * sqrt(u1);
    let theta = TAU * u2;
    return normalize(dir + tangent * (r * cos(theta)) + bitangent * (r * sin(theta)));
}

// --- Ray casting ------------------------------------------------------------

/// A resolved surface hit, reconstructed from an intersection record.
struct TracedHit {
    hit: bool,
    position: vec3<f32>,
    normal: vec3<f32>,
    material_index: u32,
    distance: f32,
    uv: vec2<f32>,
};

fn heap_vec3(base: u32) -> vec3<f32> {
    return vec3<f32>(heap_vertices[base], heap_vertices[base + 1u], heap_vertices[base + 2u]);
}

fn heap_vec2(base: u32) -> vec2<f32> {
    return vec2<f32>(heap_vertices[base], heap_vertices[base + 1u]);
}

/// Closest hit along a ray, with the surface normal interpolated from the
/// triangle's vertex normals.
///
/// A `RayIntersection` carries no surface data at all — no normal, not even the
/// triangle's vertices without `EXPERIMENTAL_RAY_HIT_VERTEX_RETURN`. Everything
/// below the `kind` check is the cost of that: look up the instance record, walk
/// the index heap to the primitive, read three vertex normals, and blend them by
/// the reported barycentrics.
/// How many alpha-tested surfaces a single ray may pass through before the
/// next one is treated as opaque. Two layers of fence read correctly; a stack
/// of ten is spending rays on something the glyph grid cannot resolve anyway.
const MAX_ALPHA_SKIPS: u32 = 4u;

fn trace_closest(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> TracedHit {
    // Alpha-tested geometry is still OPAQUE to the ray query (naga has no
    // candidate-intersection support to do this in-traversal), so a cutout is
    // handled by *re-casting*: hit an alpha-test surface, sample its texture,
    // and if the texel is transparent, start a new ray just past the hit.
    var start = origin;
    var range = tmax;
    for (var skip = 0u; skip <= MAX_ALPHA_SKIPS; skip = skip + 1u) {
        let h = trace_closest_raw(start, dir, tmin, range);
        if (!h.hit) {
            return h;
        }
        let mat = materials[h.material_index];
        if ((mat.flags & MATERIAL_FLAG_ALPHA_TEST) != 0u) {
            let texel = surface_color_level(mat, h.uv, ray_mip_level(h.distance));
            if (texel.a < 0.5) {
                // Continue from just past the hole. tmin stays as given; the
                // remaining range shrinks by the distance already travelled.
                let step = h.distance + SURFACE_EPSILON * (1.0 + h.distance);
                start = start + normalize(dir) * step;
                range = range - step;
                if (range <= tmin) {
                    var miss: TracedHit;
                    miss.hit = false;
                    return miss;
                }
                continue;
            }
        }
        return h;
    }
    // Skip budget exhausted: return the last surface as-is (opaque). Wrong
    // only for >4 stacked cutouts, and bounded.
    return trace_closest_raw(start, dir, tmin, range);
}

fn trace_closest_raw(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> TracedHit {
    var out: TracedHit;
    out.hit = false;
    out.position = origin;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.material_index = 0u;
    out.distance = tmax;
    out.uv = vec2<f32>(0.0, 0.0);

    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene_tlas,
        RayDesc(RAY_FLAG_NONE, 0xFFu, tmin, tmax, origin, normalize(dir)),
    );
    // Every geometry is flagged OPAQUE, so no candidate intersections are
    // produced and the loop settles immediately. It is still a loop because
    // that is the contract of rayQueryProceed, not because it iterates.
    while (rayQueryProceed(&rq)) {}
    let isect = rayQueryGetCommittedIntersection(&rq);
    if (isect.kind == RAY_QUERY_INTERSECTION_NONE) {
        return out;
    }

    let inst = traced_instances[isect.instance_custom_data];
    let base = inst.index_offset + isect.primitive_index * 3u;
    let i0 = inst.vertex_offset + heap_indices[base];
    let i1 = inst.vertex_offset + heap_indices[base + 1u];
    let i2 = inst.vertex_offset + heap_indices[base + 2u];

    let n0 = heap_vec3(i0 * HEAP_STRIDE + HEAP_NORMAL_OFFSET);
    let n1 = heap_vec3(i1 * HEAP_STRIDE + HEAP_NORMAL_OFFSET);
    let n2 = heap_vec3(i2 * HEAP_STRIDE + HEAP_NORMAL_OFFSET);

    // Barycentrics report the second and third weights; the first is implied.
    let b = isect.barycentrics;
    let w0 = 1.0 - b.x - b.y;
    let local_normal = n0 * w0 + n1 * b.x + n2 * b.y;

    // UV interpolates by exactly the same weights as the normal.
    let uv0 = heap_vec2(i0 * HEAP_STRIDE + HEAP_UV_OFFSET);
    let uv1 = heap_vec2(i1 * HEAP_STRIDE + HEAP_UV_OFFSET);
    let uv2 = heap_vec2(i2 * HEAP_STRIDE + HEAP_UV_OFFSET);
    out.uv = uv0 * w0 + uv1 * b.x + uv2 * b.y;

    // object_to_world is a mat4x3: three basis columns then a translation.
    // Rotating the normal by the basis is correct for rotation and uniform
    // scale only. The rasterised path (and the primary hit of the traced one,
    // which shades the rasterised fragment) uses a per-object inverse-transpose
    // and gets non-uniform scale right; here that would mean carrying a normal
    // matrix in every TracedInstance. Until an instance record grows one, a
    // non-uniformly scaled mesh seen in a reflection or through glass keeps
    // slightly skewed normals — said plainly rather than hidden.
    let o2w = isect.object_to_world;
    let basis = mat3x3<f32>(o2w[0], o2w[1], o2w[2]);

    out.hit = true;
    out.distance = isect.t;
    out.position = origin + normalize(dir) * isect.t;
    out.normal = normalize(basis * local_normal);
    out.material_index = inst.material_index;
    return out;
}

/// Whether anything blocks the segment from `origin` along `dir`.
///
/// Uses TERMINATE_ON_FIRST_HIT: a shadow ray does not care which surface blocks
/// it, only that one does, and letting the traversal stop early is most of the
/// reason shadow rays are affordable at all.
///
/// Alpha-tested surfaces cast *opaque* shadows here: honouring the cutout
/// would forfeit TERMINATE_ON_FIRST_HIT and re-cast per hole per sample per
/// light. A documented first-iteration limitation — a fence shadows as a wall.
fn trace_occluded(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    if (tmax <= tmin) {
        return false;
    }
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        scene_tlas,
        RayDesc(
            RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0xFFu,
            tmin,
            tmax,
            origin,
            normalize(dir),
        ),
    );
    while (rayQueryProceed(&rq)) {}
    let isect = rayQueryGetCommittedIntersection(&rq);
    return isect.kind != RAY_QUERY_INTERSECTION_NONE;
}

// --- Traced lighting terms --------------------------------------------------

/// Fraction of a light that reaches `pos`, from `shadow_samples` shadow rays.
///
/// The rays are jittered inside a cone whose angular radius is the light's
/// apparent size, so a sample count above one produces a genuine penumbra rather
/// than the same hard edge computed repeatedly. `tmax` stops at the light: a ray
/// allowed to run past it would be blocked by geometry *behind* the light.
fn traced_shadow(
    pos: vec3<f32>,
    normal: vec3<f32>,
    s: LightSample,
    seed: ptr<function, u32>,
) -> f32 {
    // Facing away from the light: no ray needed, the surface shadows itself.
    if (dot(normal, s.dir) <= 0.0) {
        return 0.0;
    }
    let samples = max(trace.shadow_samples, 1u);
    let origin = pos + normal * SURFACE_EPSILON;
    // Angular radius of the light disc as seen from here.
    let spread = clamp(trace.light_radius / max(s.distance, 0.001), 0.0, 0.5);
    var lit = 0.0;
    for (var i = 0u; i < samples; i = i + 1u) {
        let u1 = rand01(seed);
        let u2 = rand01(seed);
        var dir = s.dir;
        if (samples > 1u) {
            dir = jitter_cone(s.dir, spread, u1, u2);
        }
        if (!trace_occluded(origin, dir, SURFACE_EPSILON, s.distance - SURFACE_EPSILON)) {
            lit = lit + 1.0;
        }
    }
    return lit / f32(samples);
}

/// Traced ambient occlusion: the fraction of the hemisphere above `pos` that is
/// not blocked within `ao_radius`.
///
/// This replaces the screen-space approximation in `renderer::post_process`,
/// which could only see what the depth buffer happened to contain — geometry
/// off-screen or behind a nearer surface contributed nothing. A ray does not
/// have that limitation.
fn traced_ao(pos: vec3<f32>, normal: vec3<f32>, seed: ptr<function, u32>) -> f32 {
    let samples = trace.ao_samples;
    if (samples == 0u) {
        return 1.0;
    }
    let origin = pos + normal * SURFACE_EPSILON;
    var open = 0.0;
    for (var i = 0u; i < samples; i = i + 1u) {
        let u1 = rand01(seed);
        let u2 = rand01(seed);
        let dir = cosine_hemisphere(normal, u1, u2);
        if (!trace_occluded(origin, dir, SURFACE_EPSILON, trace.ao_radius)) {
            open = open + 1.0;
        }
    }
    return open / f32(samples);
}

/// Direct lighting at a surface: every light, every one of them shadow-tested.
///
/// This is where the traced path earns its name. The rasterised version can only
/// shadow light[0], because there is only one shadow map; here the cost of one
/// more shadowed light is one more ray.
// `albedo` is passed in rather than read from the material because it may be
// texture-modulated, and the sampling strategy differs by caller: the primary
// hit has screen derivatives (textureSample), a ray hit does not
// (textureSampleLevel with a distance-based level).
fn traced_direct(
    mat: Material,
    albedo: vec3<f32>,
    pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    seed: ptr<function, u32>,
) -> vec3<f32> {
    var ambient_sum = mat.ambient;
    var lit = vec3<f32>(0.0, 0.0, 0.0);
    let shadows_on = (trace.flags & FLAG_SHADOWS) != 0u;

    for (var i = 0u; i < lights_meta.count; i = i + 1u) {
        let s = sample_light(i, pos);
        ambient_sum = ambient_sum + s.ambient;
        var visibility = 1.0;
        if (shadows_on) {
            visibility = traced_shadow(pos, normal, s, seed);
        }
        lit = lit + phong_term(mat, normal, view_dir, s) * visibility;
    }

    // Occlusion attenuates the ambient term only. Ambient stands in for light
    // arriving from everywhere; occlusion is precisely the measure of how much
    // of "everywhere" is actually visible. Applying it to the direct term as
    // well — as a screen-space filter over the final image must — double-counts
    // shadowing that the shadow rays already resolved.
    var occlusion = 1.0;
    if ((trace.flags & FLAG_AO) != 0u) {
        occlusion = traced_ao(pos, normal, seed);
    }

    return albedo * (ambient_sum * occlusion + lit);
}

/// Mip level for a ray-hit sample: no screen derivatives exist along a ray, so
/// approximate from the hit distance. The constant is tuned for the subpixel
/// target — at 240x136 a surface a few units away is already minified several
/// levels. The output is quantized to glyphs, so a coarse heuristic suffices;
/// what matters is *not* sampling mip 0 at distance, which aliases into noise
/// the ASCII stage amplifies.
fn ray_mip_level(distance: f32) -> f32 {
    return max(0.0, log2(max(distance, 1.0)));
}

/// Follow a secondary ray through up to `max_depth` bounces and return the
/// radiance arriving back along it.
///
/// Iterative rather than recursive, and single-path rather than branching: a
/// mirror reflecting glass reflecting a mirror costs `max_depth` rays, not
/// `2^max_depth`. That is the whole point of the bounce cap being a budget.
fn trace_path(start_origin: vec3<f32>, start_dir: vec3<f32>, seed_in: u32) -> vec3<f32> {
    var seed = seed_in;
    var origin = start_origin;
    var dir = normalize(start_dir);
    let limit = min(trace.max_depth, MAX_BOUNCES);

    // A zero bounce budget still has to answer with something plausible, and the
    // analytic environment is exactly the rasterised path's answer.
    if (limit == 0u) {
        return environment(dir) + light_glints(dir);
    }

    var throughput = vec3<f32>(1.0, 1.0, 1.0);
    var accum = vec3<f32>(0.0, 0.0, 0.0);
    let reflections_on = (trace.flags & FLAG_REFLECTIONS) != 0u;
    let refraction_on = (trace.flags & FLAG_REFRACTION) != 0u;

    for (var bounce = 0u; bounce < limit; bounce = bounce + 1u) {
        let h = trace_closest(origin, dir, SURFACE_EPSILON, MAX_RAY_DISTANCE);
        if (!h.hit) {
            // The ray left the scene: it sees the sky, and the lights in it.
            accum = accum + throughput * (environment(dir) + light_glints(dir));
            break;
        }

        let mat = materials[h.material_index];
        let view_dir = -dir;
        // Flip the normal toward the incoming ray so a back face (the inside of
        // a glass shell, for instance) shades as a surface rather than as a hole.
        let entering = dot(h.normal, view_dir) > 0.0;
        var normal = h.normal;
        if (!entering) {
            normal = -normal;
        }
        // Scale the offset with distance: a fixed epsilon stops separating
        // points once float spacing at that magnitude exceeds it.
        let eps = SURFACE_EPSILON * (1.0 + h.distance);

        // Texture-modulated albedo at the hit point. No screen derivatives on
        // a ray, so the mip comes from the hit distance.
        let hit_albedo = surface_color_level(mat, h.uv, ray_mip_level(h.distance)).rgb;
        let direct = traced_direct(mat, hit_albedo, h.position, normal, view_dir, &seed);
        let fres = fresnel_weight(normal, view_dir);

        if (mat.material_type == MATERIAL_MIRROR && reflections_on) {
            let k = clamp(mat.reflectivity + (1.0 - mat.reflectivity) * fres, 0.0, 1.0);
            accum = accum + throughput * direct * (1.0 - k);
            throughput = throughput * k * hit_albedo;
            origin = h.position + normal * eps;
            dir = reflect(dir, normal);
        } else if (mat.material_type == MATERIAL_GLASS && refraction_on) {
            var eta = 1.0 / max(mat.ior, 1.0);
            if (!entering) {
                // Leaving the denser medium: the ratio inverts, which is what
                // makes total internal reflection possible on the way out.
                eta = max(mat.ior, 1.0);
            }
            let sheen = environment(reflect(dir, normal)) * fres;
            accum = accum + throughput * (direct * (1.0 - mat.transparency) + sheen);
            let refracted = refract(dir, normal, eta);
            if (dot(refracted, refracted) < 1e-6) {
                // Total internal reflection: nothing passes, everything bounces.
                dir = reflect(dir, normal);
                origin = h.position + normal * eps;
            } else {
                dir = normalize(refracted);
                origin = h.position - normal * eps;
            }
            // Coloured glass filters what passes through it.
            throughput = throughput
                * mix(vec3<f32>(1.0, 1.0, 1.0), hit_albedo, mat.transparency)
                * mat.transparency;
        } else {
            // Matte, or a bouncing material with its bounce disabled: the path
            // ends here.
            accum = accum + throughput * direct;
            break;
        }

        // Once almost nothing would come back, stop paying for rays.
        if (throughput.x + throughput.y + throughput.z < 0.01) {
            break;
        }
    }

    return accum;
}

// --- Entry point ------------------------------------------------------------

/// Traced counterpart of `scene_fragment.wgsl::main`.
///
/// Alpha is always 1.0, unlike the rasterised path. Glass no longer needs to be
/// blended over the background, because the refraction ray *is* the background —
/// blending on top of it would show the scene behind the glass twice, once
/// straight and once bent. This is why the traced path draws every material
/// through a single opaque pipeline and skips the tint pass entirely.
@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = input.world_pos;
    let mat = materials[input.material_index];

    // Primary hit is rasterised, so it has real derivatives: textureSample.
    // Must run before the discard below (uniform control flow).
    let surface = surface_color(mat, input.uv);
    if ((mat.flags & MATERIAL_FLAG_ALPHA_TEST) != 0u && surface.a < 0.5) {
        discard;
    }

    let view_dir = normalize(camera.camera_pos - world_pos);

    var normal = normalize(input.world_normal);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    var seed = pixel_seed(input.clip_pos.xy);

    var color = traced_direct(mat, surface.rgb, world_pos, normal, view_dir, &seed);
    let fres = fresnel_weight(normal, view_dir);
    let reflections_on = (trace.flags & FLAG_REFLECTIONS) != 0u;
    let refraction_on = (trace.flags & FLAG_REFRACTION) != 0u;

    if (mat.material_type == MATERIAL_MIRROR && reflections_on) {
        let refl = reflect(-view_dir, normal);
        // Metal tints its reflection with its own (texture-modulated) albedo,
        // same as the raster path.
        let reflected = trace_path(world_pos + normal * SURFACE_EPSILON, refl, seed)
            * surface.rgb;
        let k = clamp(mat.reflectivity + (1.0 - mat.reflectivity) * fres, 0.0, 1.0);
        color = mix(color, reflected, k);
    } else if (mat.material_type == MATERIAL_GLASS && refraction_on) {
        let mirror_dir = reflect(-view_dir, normal);
        let eta = 1.0 / max(mat.ior, 1.0);
        let refracted = refract(-view_dir, normal, eta);

        var through: vec3<f32>;
        if (dot(refracted, refracted) < 1e-6) {
            // Total internal reflection at the entry face.
            through = trace_path(world_pos + normal * SURFACE_EPSILON, mirror_dir, seed);
        } else {
            through = trace_path(world_pos - normal * SURFACE_EPSILON, refracted, seed);
        }
        // A distinct seed, so the two paths do not share a jitter sequence and
        // correlate their noise.
        let sheen = trace_path(
            world_pos + normal * SURFACE_EPSILON,
            mirror_dir,
            seed ^ 0x9e3779b9u,
        );

        color = color * (1.0 - mat.transparency)
            + through * mix(vec3<f32>(1.0, 1.0, 1.0), surface.rgb, mat.transparency)
                * mat.transparency
            + sheen * fres;
    }

    return vec4<f32>(color, 1.0);
}
