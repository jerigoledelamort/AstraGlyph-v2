// Scene pass: vertex shader — transforms vertices with view-projection matrix.
// Passes material index (via instance_index) and world position to fragment shader.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
};

@vertex
fn main(
    input: VertexInput,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var output: VertexOutput;
    output.clip_pos = camera.view_proj * vec4<f32>(input.position, 1.0);
    output.world_normal = input.normal;
    output.world_pos = input.position;
    output.material_index = instance_id;
    return output;
}