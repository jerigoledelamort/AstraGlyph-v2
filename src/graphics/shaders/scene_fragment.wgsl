// Scene pass: fragment shader — Phong lighting (ambient + diffuse + specular)
// summed over multiple light sources (directional or point), with alpha
// output for transparent materials.

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
};

@group(0) @binding(3) var<storage, read> materials: array<Material>;

// Simplified shadow map: depth rendered from the point of view of light[0]
// only (see ScenePipeline::compute_shadow_view_proj). Other lights are
// treated as unshadowed — a deliberate simplification for Phase 1.
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

/// Analytic environment used by reflective and refractive materials.
///
/// There is no environment map and no ray tracing here: a mirror needs
/// *something* direction-dependent to reflect, and a sky/horizon/ground gradient
/// gives exactly the cue the eye reads as "this surface is reflective" — the
/// horizon line sweeping across a sphere as the camera moves. It costs three
/// mixes and stays stable under the heavy quantization the ASCII stage applies.
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
};

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    let world_pos = input.world_pos;

    let mat = materials[input.material_index];
    let albedo = mat.albedo.rgb;
    let view_dir = normalize(camera.camera_pos - world_pos);

    var ambient_sum = mat.ambient;
    var lit = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < lights_meta.count; i = i + 1u) {
        let l = lights[i];
        ambient_sum = ambient_sum + l.ambient;

        var light_dir: vec3<f32>;
        var attenuation = 1.0;
        if (l.position.w < 0.5) {
            // Directional: direction points FROM the light, surface faces -direction.
            light_dir = normalize(-l.direction.xyz);
        } else {
            let to_light = l.position.xyz - world_pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.001);
            // Distance attenuation (inverse-square approximation).
            attenuation = 1.0 / (1.0 + 0.1 * dist + 0.01 * dist * dist);
        }

        // Diffuse (Lambertian).
        let n_dot_l = max(dot(normal, light_dir), 0.0);
        let diffuse = mat.diffuse * n_dot_l * attenuation * l.diffuse;

        // Specular (Phong: reflect light dir around normal, compare with view dir).
        let reflect_dir = reflect(-light_dir, normal);
        let spec_angle = max(dot(reflect_dir, view_dir), 0.0);
        let specular = mat.specular * pow(spec_angle, max(mat.shininess, 1.0)) * attenuation * l.diffuse;

        var shadow = 1.0;
        if (i == 0u) {
            shadow = shadow_visibility(world_pos);
        }

        lit = lit + (diffuse + specular) * l.color.rgb * shadow;
    }

    var color = albedo * (ambient_sum + lit);
    var alpha = 1.0 - mat.transparency;

    // Fresnel: any smooth surface reflects far more at grazing angles than
    // head-on. This is what makes a mirror read as a mirror and gives glass its
    // characteristic bright rim, so both material types use it.
    let facing = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - facing, 5.0);

    if (mat.material_type == MATERIAL_MIRROR) {
        let refl = reflect(-view_dir, normal);
        // Metal tints its reflection with its own albedo.
        let reflected = environment(refl) * albedo + light_glints(refl);
        // Schlick: reflectivity at normal incidence, rising to 1.0 at the edges.
        let k = clamp(mat.reflectivity + (1.0 - mat.reflectivity) * fresnel, 0.0, 1.0);
        color = mix(color, reflected, k);
    } else if (mat.material_type == MATERIAL_GLASS) {
        // Refraction through a thin shell: bend the view ray and reflect the
        // environment seen through it, plus a reflective sheen off the surface.
        let eta = 1.0 / max(mat.ior, 1.0);
        let refracted = refract(-view_dir, normal, eta);
        var through = environment(refracted);
        if (dot(refracted, refracted) < 0.0001) {
            // Total internal reflection — nothing gets through at this angle.
            through = environment(reflect(-view_dir, normal));
        }
        let sheen = environment(reflect(-view_dir, normal)) + light_glints(reflect(-view_dir, normal));
        color = color + through * albedo * mat.transparency + sheen * fresnel;
        // Glass is denser at the rim, so it hides more of the background there.
        alpha = clamp(alpha + fresnel * (1.0 - alpha), 0.0, 1.0);
    }

    return vec4<f32>(color, alpha);
}

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
