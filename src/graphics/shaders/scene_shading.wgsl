// Shared shading code for the scene pass: group-0 bindings, material and light
// structs, the analytic environment, and the per-light Phong term.
//
// This file is not a standalone shader. It is concatenated in front of exactly
// one of two entry-point files — `scene_fragment.wgsl` (rasterised lighting) or
// `scene_traced_fragment.wgsl` (ray-traced lighting) — because WGSL has no
// include directive and the two paths must agree on the surface model down to
// the last coefficient. Duplicating it is how "the traced sphere looks slightly
// different" bugs get born.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct Light {
    // xyz = position (point lights only). w = light type: 0.0 = directional, 1.0 = point.
    position: vec4<f32>,
    // xyz = direction the light travels (directional lights only, points FROM the source).
    direction: vec4<f32>,
    color: vec4<f32>,
    ambient: f32,
    diffuse: f32,
};

struct LightsMeta {
    count: u32,
};

@group(0) @binding(1) var<uniform> lights_meta: LightsMeta;
@group(0) @binding(2) var<storage, read> lights: array<Light>;

// Mirrors `scene::component::MaterialUniform` field for field (64 bytes).
struct Material {
    albedo: vec4<f32>,
    material_type: u32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    ior: f32,
    reflectivity: f32,
    transparency: f32,
    texture_index: u32,
    flags: u32,
    uv_scale: vec2<f32>,
};

@group(0) @binding(3) var<storage, read> materials: array<Material>;

// Every texture the scene samples, as layers of one array (see
// `graphics::texture_array` for why an array and not an atlas or bindings).
@group(0) @binding(8) var scene_textures: texture_2d_array<f32>;
@group(0) @binding(9) var scene_texture_sampler: sampler;

/// Mirrors `scene::component::NO_TEXTURE` / `texture_array::NO_TEXTURE`.
const NO_TEXTURE: u32 = 0xFFFFFFFFu;
/// Mirrors `scene::component::material_flags::ALPHA_TEST`.
const MATERIAL_FLAG_ALPHA_TEST: u32 = 1u;

/// Map a mesh UV into a padded array layer: tile first (fract), then squeeze
/// into the fraction of the layer the real texels cover. The order matters —
/// scaling before wrapping would tile across the padding.
fn layer_uv(mat: Material, uv: vec2<f32>) -> vec2<f32> {
    return fract(uv) * mat.uv_scale;
}

/// Surface colour+alpha of a material at `uv`, for the rasterised path:
/// albedo times the sampled texel, or the albedo alone when untextured.
/// `textureSample` needs uniform control flow for its implicit derivatives,
/// so the sample is unconditional and the untextured case selects afterwards.
fn surface_color(mat: Material, uv: vec2<f32>) -> vec4<f32> {
    // Layer 0 for the untextured case: the sample result is discarded below,
    // but the *index* must stay in bounds rather than relying on robustness
    // clamping to make an out-of-range layer harmless.
    let layer = select(i32(mat.texture_index), 0, mat.texture_index == NO_TEXTURE);
    let texel = textureSample(
        scene_textures,
        scene_texture_sampler,
        layer_uv(mat, uv),
        layer,
    );
    if (mat.texture_index == NO_TEXTURE) {
        return vec4<f32>(mat.albedo.rgb, 1.0);
    }
    return vec4<f32>(mat.albedo.rgb * texel.rgb, texel.a);
}

/// Same, for contexts with no screen-space derivatives (ray hits): the mip
/// level is chosen by the caller. Free of uniformity requirements, so the
/// untextured branch can skip the sample entirely.
fn surface_color_level(mat: Material, uv: vec2<f32>, level: f32) -> vec4<f32> {
    if (mat.texture_index == NO_TEXTURE) {
        return vec4<f32>(mat.albedo.rgb, 1.0);
    }
    let texel = textureSampleLevel(
        scene_textures,
        scene_texture_sampler,
        layer_uv(mat, uv),
        i32(mat.texture_index),
        level,
    );
    return vec4<f32>(mat.albedo.rgb * texel.rgb, texel.a);
}

// Simplified shadow map: depth rendered from the point of view of light[0]
// only. Used by the rasterised path; the traced path casts shadow rays for
// every light instead and never samples this.
@group(0) @binding(4) var shadow_map: texture_depth_2d;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;

struct LightSpace {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(6) var<uniform> light_space: LightSpace;

/// Visibility factor for light[0] at `world_pos`: 1.0 = fully lit,
/// 0.0 = fully shadowed. Points outside the shadow frustum are treated as lit.
fn shadow_visibility(world_pos: vec3<f32>) -> f32 {
    let clip = light_space.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    return textureSampleCompare(shadow_map, shadow_sampler, uv, ndc.z);
}

// Material type tags, mirroring scene::component::MaterialType.
const MATERIAL_MATTE: u32 = 0u;
const MATERIAL_MIRROR: u32 = 1u;
const MATERIAL_GLASS: u32 = 2u;

/// Analytic environment used where no traced geometry was hit.
///
/// In the rasterised path this stands in for reflections entirely: a mirror
/// needs *something* direction-dependent to reflect, and a sky/horizon/ground
/// gradient gives exactly the cue the eye reads as "this surface is reflective".
/// In the traced path it is the sky itself — what a ray that leaves the scene
/// without hitting anything returns.
fn environment(dir: vec3<f32>) -> vec3<f32> {
    let sky = vec3<f32>(0.28, 0.42, 0.72);
    let horizon = vec3<f32>(0.58, 0.60, 0.66);
    let ground = vec3<f32>(0.16, 0.14, 0.12);
    let h = clamp(normalize(dir).y, -1.0, 1.0);
    if (h > 0.0) {
        // Sharpen the gradient near the horizon rather than spreading it evenly.
        return mix(horizon, sky, pow(h, 0.45));
    }
    return mix(horizon, ground, pow(-h, 0.35));
}

/// Bright spots where a reflection direction points at a light source.
///
/// Without this, reflective surfaces miss the single most recognisable feature of
/// a mirror: a sharp image of the lights themselves.
fn light_glints(dir: vec3<f32>) -> vec3<f32> {
    var total = vec3<f32>(0.0, 0.0, 0.0);
    let d = normalize(dir);
    for (var i = 0u; i < lights_meta.count; i = i + 1u) {
        let l = lights[i];
        var to_light: vec3<f32>;
        if (l.position.w < 0.5) {
            to_light = normalize(-l.direction.xyz);
        } else {
            // Direction only — the glint is an angular feature, so distance
            // attenuation would wash it out.
            to_light = normalize(l.position.xyz);
        }
        let alignment = max(dot(d, to_light), 0.0);
        total = total + l.color.rgb * pow(alignment, 220.0) * l.diffuse;
    }
    return total;
}

/// One light resolved at a surface point: which way it lies, how far, and how
/// much of it arrives.
///
/// Both lighting paths need exactly this, and the traced path additionally needs
/// `distance` as the shadow ray's `tmax` — a shadow ray that runs past the light
/// would be occluded by geometry *behind* the light.
struct LightSample {
    /// Unit vector from the surface toward the light.
    dir: vec3<f32>,
    /// Distance to the light; a large finite value for directional lights.
    distance: f32,
    /// Distance falloff, already applied to nothing — callers multiply by it.
    attenuation: f32,
    /// Light colour times its intensity.
    energy: vec3<f32>,
    /// This light's ambient contribution.
    ambient: f32,
};

/// Distance a directional shadow ray travels before it counts as unoccluded.
/// The demo scene is a few tens of units across, so this clears it comfortably
/// without inviting float precision problems.
const DIRECTIONAL_RANGE: f32 = 500.0;

fn sample_light(index: u32, world_pos: vec3<f32>) -> LightSample {
    let l = lights[index];
    var s: LightSample;
    s.ambient = l.ambient;
    s.energy = l.color.rgb * l.diffuse;
    if (l.position.w < 0.5) {
        // Directional: direction points FROM the light, surface faces -direction.
        s.dir = normalize(-l.direction.xyz);
        s.distance = DIRECTIONAL_RANGE;
        s.attenuation = 1.0;
    } else {
        let to_light = l.position.xyz - world_pos;
        let dist = length(to_light);
        s.dir = to_light / max(dist, 0.001);
        s.distance = dist;
        // Distance attenuation (inverse-square approximation).
        s.attenuation = 1.0 / (1.0 + 0.1 * dist + 0.01 * dist * dist);
    }
    return s;
}

/// Diffuse + specular contribution of one light at a surface, before shadowing.
///
/// `view_dir` points from the surface toward the eye.
fn phong_term(
    mat: Material,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    s: LightSample,
) -> vec3<f32> {
    // Diffuse (Lambertian).
    let n_dot_l = max(dot(normal, s.dir), 0.0);
    let diffuse = mat.diffuse * n_dot_l * s.attenuation;

    // Specular (Phong: reflect light dir around normal, compare with view dir).
    let reflect_dir = reflect(-s.dir, normal);
    let spec_angle = max(dot(reflect_dir, view_dir), 0.0);
    let specular = mat.specular * pow(spec_angle, max(mat.shininess, 1.0)) * s.attenuation;

    return (diffuse + specular) * s.energy;
}

/// Schlick-style Fresnel weight: any smooth surface reflects far more at grazing
/// angles than head-on. This is what makes a mirror read as a mirror and gives
/// glass its characteristic bright rim.
fn fresnel_weight(normal: vec3<f32>, view_dir: vec3<f32>) -> f32 {
    let facing = max(dot(normal, view_dir), 0.0);
    return pow(1.0 - facing, 5.0);
}

// Must match the VertexOutput in scene_vertex.wgsl location for location.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
    @location(3) uv: vec2<f32>,
};

/// Filter colour for the glass tint pass.
///
/// Alpha blending can only fade the background toward the glass colour, never
/// *tint* it — so what is behind a green sphere stays its own colour. This entry
/// point is drawn with a multiply blend (result = destination * source) so the
/// background is actually filtered through the glass, which is how a coloured
/// transparent body really behaves.
///
/// The filter is white where the glass is opaque-thin and approaches the albedo
/// as transparency rises; `1.0` would leave the background untouched.
///
/// Shared by both lighting paths: the tint is a property of the glass, not of
/// how the light behind it was computed.
@fragment
fn tint(input: VertexOutput) -> @location(0) vec4<f32> {
    let mat = materials[input.material_index];
    if (mat.material_type != MATERIAL_GLASS) {
        // Not glass: multiply by white, i.e. leave the background alone.
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let filter_strength = clamp(mat.transparency, 0.0, 1.0);
    let filter_color = mix(vec3<f32>(1.0, 1.0, 1.0), mat.albedo.rgb, filter_strength);
    return vec4<f32>(filter_color, 1.0);
}
