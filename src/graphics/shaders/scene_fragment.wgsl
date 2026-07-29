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

    let intensity = ambient_sum + lit;
    return vec4<f32>(albedo * intensity, 1.0 - mat.transparency);
}
