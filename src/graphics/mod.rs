// Graphics module: wgpu device, buffer, pipeline, texture abstractions.

pub mod buffer;
pub mod capabilities;
pub mod device;
pub mod pipeline;
pub mod texture;

pub use device::{FrameOutcome, GraphicsContext};