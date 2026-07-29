// Composite pass: fragment shader — sample glyph atlas and tint with cell color.
// The atlas is a 2D texture where each glyph is packed in a grid.
// We use the glyph index to select the region in the atlas.

@group(0) @binding(1) var glyph_atlas: texture_2d<f32>;
@group(0) @binding(2) var glyph_sampler: sampler;

// The atlas is a single row of 8px glyph cells; the vertex stage already mapped
// each instance's UV into its own slice, so nothing here needs the glyph count.

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@fragment
fn main(input: VertexOutput) -> @location(0) vec4<f32> {
    let glyph_mask = textureSample(glyph_atlas, glyph_sampler, input.uv).r;
    return vec4<f32>(input.color * glyph_mask, 1.0);
}