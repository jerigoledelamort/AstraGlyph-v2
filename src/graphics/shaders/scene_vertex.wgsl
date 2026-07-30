// Scene pass: vertex shader — transforms vertices by their object's model matrix,
// then by the camera view-projection. Passes world position/normal and the
// object's material index to the fragment shader.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Per-object data, indexed by instance_index (which the renderer sets to the
// object's slot via draw_indexed's first_instance).
// Must match `renderer::scene_pass::ObjectUniform` (and the copy of this
// struct in shadow_vertex.wgsl) field for field: both shaders read the same
// storage buffer, so a mismatched stride silently shears every draw.
struct Object {
    model: mat4x4<f32>,
    // Inverse-transpose of model's upper 3x3, computed on the CPU (WGSL has no
    // inverse). Stored as a full mat4x4 for layout parity with the Rust side.
    normal: mat4x4<f32>,
    material_index: u32,
};

@group(0) @binding(7) var<storage, read> objects: array<Object>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) @interpolate(flat) material_index: u32,
    @location(3) uv: vec2<f32>,
};

@vertex
fn main(
    input: VertexInput,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var output: VertexOutput;

    let object = objects[instance_id];
    let world_pos = object.model * vec4<f32>(input.position, 1.0);

    output.clip_pos = camera.view_proj * world_pos;
    output.world_pos = world_pos.xyz;
    // The normal matrix (inverse-transpose of the model's upper 3x3) keeps
    // normals perpendicular to the surface under non-uniform scale, where the
    // model matrix would skew them. The fragment shader still renormalizes,
    // since interpolation shortens normals regardless of how correct they are.
    output.world_normal = (object.normal * vec4<f32>(input.normal, 0.0)).xyz;
    output.material_index = object.material_index;
    output.uv = input.uv;
    return output;
}
