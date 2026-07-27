// Composite pass: vertex shader — renders ASCII glyphs as textured quads.
// Uses storage buffer with per-instance data (position, uv offset, color).
// Quad is defined as a unit square [0,1]x[0,1] with the glyph atlas as a
// texture array — each glyph index maps to a layer.

struct InstanceData {
    ndc_x: f32,
    ndc_y: f32,
    width: f32,
    height: f32,
    glyph_index: u32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
};

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

// 6 vertices for a quad (two triangles)
const QUAD_POSITIONS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

const QUAD_UVS = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

@vertex
fn main(
    @builtin(vertex_index) vertex_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    let data = instances[instance_id];
    let local = QUAD_POSITIONS[vertex_id];

    // Convert local [0,1]x[0,1] to NDC with cell position and size.
    let x = data.ndc_x + local.x * data.width;
    let y = data.ndc_y - local.y * data.height; // y goes down, ndc_y is top

    var output: VertexOutput;
    output.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = QUAD_UVS[vertex_id];
    output.color = vec3<f32>(data.color_r, data.color_g, data.color_b);
    return output;
}