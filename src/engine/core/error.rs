// Engine error types and Result alias.

use std::fmt;

/// All engine errors flow through this type.
#[derive(Debug)]
pub enum EngineError {
    /// Graphics/GPU related error (wgpu).
    Graphics(String),
    /// Window/platform related error.
    Platform(String),
    /// Invalid configuration or state.
    InvalidState(String),
    /// I/O error.
    Io(std::io::Error),
    /// Shader compilation error.
    Shader(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Graphics(msg) => write!(f, "graphics error: {msg}"),
            EngineError::Platform(msg) => write!(f, "platform error: {msg}"),
            EngineError::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            EngineError::Io(err) => write!(f, "io error: {err}"),
            EngineError::Shader(msg) => write!(f, "shader error: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EngineError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EngineError {
    fn from(err: std::io::Error) -> Self {
        EngineError::Io(err)
    }
}

impl From<wgpu::RequestDeviceError> for EngineError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        EngineError::Graphics(err.to_string())
    }
}

/// Convenience Result alias used across the engine.
pub type Result<T> = std::result::Result<T, EngineError>;
