// Scene pass: fragment shader — simple ambient + diffuse lighting,
// output to a render target (one pixel per ASCII cell).

struct Light {
    direction: vec3<f32>,
    ambient: f32,
    diffuse: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Light;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Normalize interpolated normal.
    let normal = normalize(input.world_normal);
    // Diffuse lighting.
    let light_dir = normalize(uniforms.direction);
    let n_dot_l = max(dot(normal, -light_dir), 0.0);
    let intensity = uniforms.ambient + uniforms.diffuse * n_dot_l;
    return vec4<f32>(intensity, intensity, intensity, 1.0);
}