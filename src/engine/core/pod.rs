// Minimal Pod (Plain Old Data) trait — replaces the `bytemuck` crate.
// Used to safely cast structured data to byte slices for GPU buffers.

/// Marker trait for types that are safe to reinterpret as raw bytes.
///
/// # Safety
/// The type must:
/// - Be `#[repr(C)]` or `#[repr(transparent)]`
/// - Have no padding (or padding is acceptable for GPU layout)
/// - Be `Copy`
/// - Have no uninitialized memory
pub unsafe trait Pod: Copy + 'static {}

/// Cast a slice of `Pod` values to a byte slice.
pub fn cast_slice<T: Pod>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(data);
    // SAFETY: Pod types are safe to reinterpret as bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }
}

/// Cast a single `Pod` value to a byte slice.
pub fn cast_bytes<T: Pod>(data: &T) -> &[u8] {
    cast_slice(std::slice::from_ref(data))
}

// Implement Pod for common primitive types used in vertex/index data.
unsafe impl Pod for f32 {}
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for i32 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TestVertex {
        x: f32,
        y: f32,
        z: f32,
    }

    unsafe impl Pod for TestVertex {}

    #[test]
    fn cast_slice_vertex() {
        let vertices = [
            TestVertex { x: 1.0, y: 2.0, z: 3.0 },
            TestVertex { x: 4.0, y: 5.0, z: 6.0 },
        ];
        let bytes = cast_slice(&vertices);
        assert_eq!(bytes.len(), 2 * 12); // 2 vertices * 3 floats * 4 bytes
    }

    #[test]
    fn cast_slice_f32() {
        let data = [1.0f32, 2.0, 3.0];
        let bytes = cast_slice(&data);
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn cast_bytes_single() {
        let val = 42u32;
        let bytes = cast_bytes(&val);
        assert_eq!(bytes.len(), 4);
    }
}
