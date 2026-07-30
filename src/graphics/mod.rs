// Graphics module: wgpu device, buffer, pipeline, texture abstractions.

pub mod buffer;
pub mod capabilities;
pub mod device;
pub mod pipeline;
pub mod texture;
pub mod texture_array;
pub mod timing;

pub use device::{FrameOutcome, GraphicsContext};