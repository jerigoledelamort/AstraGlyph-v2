// Shadow pass: depth-only vertex shader, renders scene geometry from the
// point of view of a single light to build a shadow map. No fragment stage —
// only depth is written.

struct LightSpace {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> light_space: LightSpace;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return light_space.view_proj * vec4<f32>(input.position, 1.0);
}
