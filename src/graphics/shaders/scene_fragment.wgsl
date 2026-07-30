// Scene pass: rasterised fragment shader — Phong lighting summed over multiple
// light sources, with the shadow map for light[0] and the analytic environment
// standing in for reflection and refraction.
//
// Concatenated after `scene_shading.wgsl`, which supplies the bindings, the
// material/light structs, `sample_light`, `phong_term`, `environment`,
// `light_glints`, `fresnel_weight` and the shared `tint` entry point.
//
// This is the A side of the runtime raster/traced comparison. It is kept intact
// on purpose: it is the only reference for what the traced path is supposed to
// improve on, and the only thing that still runs on hardware without ray query.

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.world_normal);
    let world_pos = input.world_pos;

    let mat = materials[input.material_index];
    // Sampled before any branching: textureSample requires uniform control
    // flow, and the alpha-test discard below would break it if sampled later.
    let surface = surface_color(mat, input.uv);
    // Binary cutout, distinct from glass blending: below the threshold the
    // fragment simply does not exist — no depth write, no shading cost.
    if ((mat.flags & MATERIAL_FLAG_ALPHA_TEST) != 0u && surface.a < 0.5) {
        discard;
    }
    let albedo = surface.rgb;
    let view_dir = normalize(camera.camera_pos - world_pos);

    var ambient_sum = mat.ambient;
    var lit = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < lights_meta.count; i = i + 1u) {
        let s = sample_light(i, world_pos);
        ambient_sum = ambient_sum + s.ambient;

        // Only light[0] has a shadow map; the rest are treated as unshadowed.
        var shadow = 1.0;
        if (i == 0u) {
            shadow = shadow_visibility(world_pos);
        }

        lit = lit + phong_term(mat, normal, view_dir, s) * shadow;
    }

    var color = albedo * (ambient_sum + lit);
    var alpha = 1.0 - mat.transparency;

    let fresnel = fresnel_weight(normal, view_dir);

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
