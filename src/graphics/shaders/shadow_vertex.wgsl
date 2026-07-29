// Shadow pass: depth-only vertex shader, renders scene geometry from the
// point of view of a single light to build a shadow map. No fragment stage —
// only depth is written.
//
// Uses the same per-object model matrices as the scene pass, so a transformed
// mesh casts its shadow from where it actually is.

struct LightSpace {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light_space: LightSpace;

struct Object {
    model: mat4x4<f32>,
    material_index: u32,
};

@group(0) @binding(1) var<storage, read> objects: array<Object>;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn main(
    input: VertexInput,
    @builtin(instance_index) instance_id: u32,
) -> @builtin(position) vec4<f32> {
    let model = objects[instance_id].model;
    return light_space.view_proj * model * vec4<f32>(input.position, 1.0);
}
