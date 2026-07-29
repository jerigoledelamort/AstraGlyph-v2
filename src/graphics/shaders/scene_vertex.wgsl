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
struct Object {
    model: mat4x4<f32>,
    material_index: u32,
};

@group(0) @binding(7) var<storage, read> objects: array<Object>;

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

    let object = objects[instance_id];
    let world_pos = object.model * vec4<f32>(input.position, 1.0);

    output.clip_pos = camera.view_proj * world_pos;
    output.world_pos = world_pos.xyz;
    // Rotating the normal by the model matrix is correct for rotation and
    // uniform scale. Non-uniform scale would need the inverse-transpose;
    // renormalizing in the fragment shader keeps it usable meanwhile.
    output.world_normal = (object.model * vec4<f32>(input.normal, 0.0)).xyz;
    output.material_index = object.material_index;
    return output;
}
