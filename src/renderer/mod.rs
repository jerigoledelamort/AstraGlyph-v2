// Renderer module: scene pass, ascii pass, composite pass.

pub mod ascii_pass;
pub mod composite_pass;
pub mod scene_pass;

pub use ascii_pass::AsciiProcessor;
pub use composite_pass::CompositePipeline;
pub use scene_pass::{LightUniform, ScenePipeline};