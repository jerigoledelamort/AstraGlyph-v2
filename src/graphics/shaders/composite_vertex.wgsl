// Composite pass: vertex shader — renders ASCII glyphs as textured quads.
// Uses storage buffer with per-instance data (position, uv offset, color).
// Each glyph is a horizontal slice of the atlas texture.

struct InstanceData {
    ndc_x: f32,
    ndc_y: f32,
    width: f32,
    height: f32,
    glyph_index: u32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
};

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
// Bound here only to read its dimensions — the glyph count is derived from the
// atlas width instead of being hardcoded, so adding glyphs cannot desynchronize
// the UV maths from the texture. textureDimensions is not a sampling operation,
// so it is legal in a vertex shader.
@group(0) @binding(1) var glyph_atlas: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) bg: vec3<f32>,
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

// Atlas layout: N glyphs in a single row, each GLYPH_SIZE px wide.
const GLYPH_SIZE: f32 = 8.0;

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

    // Map UV [0,1] to the glyph's slice in the atlas row. The number of glyphs
    // comes from the atlas texture itself.
    let glyphs_per_row = max(f32(textureDimensions(glyph_atlas).x) / GLYPH_SIZE, 1.0);
    let glyph_start = f32(data.glyph_index) / glyphs_per_row;
    let glyph_size = 1.0 / glyphs_per_row;
    let base_uv = QUAD_UVS[vertex_id];
    let atlas_uv = vec2<f32>(glyph_start + base_uv.x * glyph_size, base_uv.y);

    var output: VertexOutput;
    output.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = atlas_uv;
    output.color = vec3<f32>(data.color_r, data.color_g, data.color_b);
    output.bg = vec3<f32>(data.bg_r, data.bg_g, data.bg_b);
    return output;
}