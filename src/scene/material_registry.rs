// Material registry: deduplicating store of GPU-ready materials.
//
// The scene pass uploads a storage buffer of `MaterialUniform` and indexes it once per draw
// call. Without dedup, N meshes sharing the same material burn N slots in a buffer that only
// holds MAX_MATERIALS entries. The registry collapses byte-identical materials into a single
// slot and hands back a stable index.
//
// Design notes:
// - `MaterialComponent` holds f32 fields, so it is neither Eq nor Hash and cannot key a HashMap.
//   Instead of loosening that type (it mirrors a WGSL struct and is not ours to change), we
//   derive the key from the *converted* `MaterialUniform` and compare it bit-for-bit via
//   `f32::to_bits`. The dedup question here is literally "are these the same bytes on the GPU?",
//   so bitwise identity is the semantically correct relation — not `==`.
// - Deliberate consequences of bitwise keys: `+0.0` and `-0.0` hash and compare as DIFFERENT
//   materials (different bit patterns, even though `0.0 == -0.0`), and NaN is self-consistent
//   (two identical NaN bit patterns dedup, even though `NaN != NaN`). Both are intended: the
//   registry mirrors buffer contents, not numeric equivalence.
// - Pure logic, no GPU types. The renderer only ever consumes `uniforms()`.

use std::collections::HashMap;

use super::component::{MaterialComponent, MaterialUniform};

/// Maximum number of distinct materials the renderer's storage buffer can hold.
///
/// Intended as the single canonical value: `renderer::scene_pass` still carries a private
/// constant of its own, and the two must stay equal (both 256) until the pass is switched over
/// to reference this one. Must be >= 1 — a zero-capacity buffer has no slot to saturate to.
pub const MAX_MATERIALS: usize = 256;

/// Number of 32-bit words in a material dedup key.
///
/// 4 (albedo) + 1 (material_type) + 7 (ambient, diffuse, specular, shininess, ior,
/// reflectivity, transparency) + 1 (texture_index) + 1 (flags) + 2 (uv_scale)
/// = 16 words, covering every field of `MaterialUniform`.
const KEY_WORDS: usize = 16;

/// Bitwise identity key for a `MaterialUniform`.
///
/// Every f32 is stored as its raw `to_bits()` pattern, so the key is `Eq + Hash` while
/// remaining an exact witness of the bytes that would be uploaded.
type MaterialKey = [u32; KEY_WORDS];

/// Build the bitwise dedup key for a uniform.
fn material_key(u: &MaterialUniform) -> MaterialKey {
    [
        u.albedo[0].to_bits(),
        u.albedo[1].to_bits(),
        u.albedo[2].to_bits(),
        u.albedo[3].to_bits(),
        // material_type is already an integer discriminant — no bit reinterpretation needed.
        u.material_type,
        u.ambient.to_bits(),
        u.diffuse.to_bits(),
        u.specular.to_bits(),
        u.shininess.to_bits(),
        u.ior.to_bits(),
        u.reflectivity.to_bits(),
        u.transparency.to_bits(),
        // Integer fields enter the key as-is.
        u.texture_index,
        u.flags,
        u.uv_scale[0].to_bits(),
        u.uv_scale[1].to_bits(),
    ]
}

/// A deduplicating registry of materials, rebuilt every frame.
///
/// Registering the same material twice yields the same index, so the GPU buffer contains one
/// entry per *distinct* material rather than one per mesh.
pub struct MaterialRegistry {
    /// Dense list of unique uniforms, in index order — exactly the buffer contents.
    uniforms: Vec<MaterialUniform>,
    /// Bitwise key -> index into `uniforms`, for O(1) dedup lookup.
    lookup: HashMap<MaterialKey, u32>,
}

impl MaterialRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            uniforms: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    /// Register a material and return its index in the uniform buffer.
    ///
    /// If a byte-identical material was already registered, its existing index is returned and
    /// nothing is appended. Otherwise the material is appended and the new index returned.
    ///
    /// Saturation: the renderer's buffer holds at most [`MAX_MATERIALS`] entries. Once the
    /// registry is full, a *new* material is not stored and `MAX_MATERIALS - 1` (the last
    /// usable slot) is returned instead. Such a mesh is then shaded with whatever material
    /// legitimately occupies that slot — visually wrong, but bounded and never a buffer
    /// overrun or a panic. Materials that already have an index keep resolving correctly even
    /// when the registry is full.
    pub fn register(&mut self, material: &MaterialComponent) -> u32 {
        let uniform = MaterialUniform::from(material);
        let key = material_key(&uniform);

        if let Some(&index) = self.lookup.get(&key) {
            return index;
        }

        if self.uniforms.len() >= MAX_MATERIALS {
            // Full: clamp to the last real slot instead of growing past the GPU buffer.
            // `saturating_sub` keeps this from underflowing if MAX_MATERIALS is ever retuned
            // to 0, which would otherwise be a compile-time arithmetic overflow.
            return MAX_MATERIALS.saturating_sub(1) as u32;
        }

        let index = self.uniforms.len() as u32;
        self.uniforms.push(uniform);
        self.lookup.insert(key, index);
        index
    }

    /// Look up a registered uniform by index. Returns `None` if the index is out of range.
    pub fn get(&self, index: u32) -> Option<&MaterialUniform> {
        self.uniforms.get(index as usize)
    }

    /// The uniform slice to upload to the GPU, ordered so that slot `i` is the material whose
    /// index `register` returned as `i`.
    pub fn uniforms(&self) -> &[MaterialUniform] {
        &self.uniforms
    }

    /// Number of distinct materials currently registered.
    pub fn len(&self) -> usize {
        self.uniforms.len()
    }

    /// True when no material has been registered yet.
    pub fn is_empty(&self) -> bool {
        self.uniforms.is_empty()
    }

    /// Drop all materials, restarting index assignment from 0.
    ///
    /// Called once per frame before collecting the scene, so both the uniform list and the
    /// dedup map must be reset — a stale key would otherwise alias a freed index.
    pub fn clear(&mut self) {
        self.uniforms.clear();
        self.lookup.clear();
    }
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::Vec4;
    use crate::scene::component::MaterialType;

    /// Bitwise comparison helper: `MaterialUniform` is not `PartialEq`, and bitwise equality is
    /// the relation the registry actually promises.
    fn same_bits(a: &MaterialUniform, b: &MaterialUniform) -> bool {
        material_key(a) == material_key(b)
    }

    fn base() -> MaterialComponent {
        MaterialComponent::matte(Vec4::new(0.2, 0.4, 0.6, 1.0), 0.1, 0.9)
    }

    #[test]
    fn new_registry_is_empty() {
        let reg = MaterialRegistry::new();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
        assert!(reg.uniforms().is_empty());
        assert!(reg.get(0).is_none());
    }

    #[test]
    fn default_matches_new() {
        let reg = MaterialRegistry::default();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    #[test]
    fn identical_materials_share_one_slot() {
        let mut reg = MaterialRegistry::new();
        let a = base();
        let b = base();

        let ia = reg.register(&a);
        let ib = reg.register(&b);

        assert_eq!(ia, 0);
        assert_eq!(ib, ia, "byte-identical materials must dedup to one index");
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        assert_eq!(reg.uniforms().len(), 1);
    }

    #[test]
    fn repeated_registration_is_stable() {
        let mut reg = MaterialRegistry::new();
        let a = base();
        let b = MaterialComponent::mirror(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.9);

        let ia = reg.register(&a);
        let ib = reg.register(&b);
        // Interleaved re-registration must keep handing back the original indices.
        assert_eq!(reg.register(&a), ia);
        assert_eq!(reg.register(&b), ib);
        assert_eq!(reg.register(&a), ia);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn many_meshes_collapse_to_few_slots() {
        // The whole point of the registry: 90 draw calls sharing 3 materials must occupy 3 slots,
        // not 90. Mirrors how the frame loop registers once per mesh.
        let palette = [
            MaterialComponent::matte(Vec4::new(0.9, 0.1, 0.1, 1.0), 0.1, 0.9),
            MaterialComponent::mirror(Vec4::new(0.9, 0.9, 0.9, 1.0), 0.8),
            MaterialComponent::glass(Vec4::new(0.5, 0.7, 1.0, 1.0), 1.5, 0.8),
        ];

        let mut reg = MaterialRegistry::new();
        for mesh in 0..90 {
            let picked = mesh % palette.len();
            // The index must depend only on the material, never on the draw order.
            assert_eq!(reg.register(&palette[picked]), picked as u32);
        }

        assert_eq!(reg.len(), palette.len());
        assert_eq!(reg.uniforms().len(), palette.len());
        for (i, m) in palette.iter().enumerate() {
            assert!(same_bits(reg.get(i as u32).unwrap(), &MaterialUniform::from(m)));
        }
    }

    #[test]
    fn differing_single_field_creates_new_slot() {
        // Each variant differs from `base()` in exactly one field.
        let mut variants: Vec<MaterialComponent> = Vec::new();

        let mut v = base();
        v.color.x = 0.21;
        variants.push(v);

        let mut v = base();
        v.color.y = 0.41;
        variants.push(v);

        let mut v = base();
        v.color.z = 0.61;
        variants.push(v);

        // .w is padding on the GPU but still part of the uploaded bytes.
        let mut v = base();
        v.color.w = 0.5;
        variants.push(v);

        let mut v = base();
        v.material_type = MaterialType::Glass as u32;
        variants.push(v);

        let mut v = base();
        v.ambient = 0.11;
        variants.push(v);

        let mut v = base();
        v.diffuse = 0.91;
        variants.push(v);

        let mut v = base();
        v.specular = 0.25;
        variants.push(v);

        let mut v = base();
        v.shininess = 64.0;
        variants.push(v);

        let mut v = base();
        v.ior = 1.5;
        variants.push(v);

        let mut v = base();
        v.reflectivity = 0.3;
        variants.push(v);

        let mut v = base();
        v.transparency = 0.7;
        variants.push(v);

        let mut reg = MaterialRegistry::new();
        let base_index = reg.register(&base());
        assert_eq!(base_index, 0);

        let mut seen: Vec<u32> = vec![base_index];
        for (i, variant) in variants.iter().enumerate() {
            let index = reg.register(variant);
            assert!(
                !seen.contains(&index),
                "variant {i} collided with an existing material index {index}"
            );
            seen.push(index);
        }

        // 1 base + 12 one-field variants, all distinct.
        assert_eq!(reg.len(), 1 + variants.len());
        assert_eq!(seen.len(), reg.len());
    }

    #[test]
    fn matte_mirror_glass_are_distinct() {
        let color = Vec4::new(0.8, 0.8, 0.8, 1.0);
        let matte = MaterialComponent::matte(color, 0.1, 0.9);
        let mirror = MaterialComponent::mirror(color, 0.9);
        let glass = MaterialComponent::glass(color, 1.5, 0.85);

        let mut reg = MaterialRegistry::new();
        let im = reg.register(&matte);
        let ir = reg.register(&mirror);
        let ig = reg.register(&glass);

        assert_eq!((im, ir, ig), (0, 1, 2));
        assert_eq!(reg.len(), 3);
        assert_eq!(reg.get(im).unwrap().material_type, MaterialType::Matte as u32);
        assert_eq!(reg.get(ir).unwrap().material_type, MaterialType::Mirror as u32);
        assert_eq!(reg.get(ig).unwrap().material_type, MaterialType::Glass as u32);
    }

    #[test]
    fn uniforms_order_matches_returned_indices() {
        let mut reg = MaterialRegistry::new();
        let a = MaterialComponent::matte(Vec4::new(1.0, 0.0, 0.0, 1.0), 0.1, 0.9);
        let b = MaterialComponent::mirror(Vec4::new(0.0, 1.0, 0.0, 1.0), 0.75);
        let c = MaterialComponent::glass(Vec4::new(0.0, 0.0, 1.0, 1.0), 1.33, 0.6);

        let ia = reg.register(&a);
        let ib = reg.register(&b);
        // A duplicate in the middle must not disturb the ordering.
        assert_eq!(reg.register(&a), ia);
        let ic = reg.register(&c);

        let expected = [
            MaterialUniform::from(&a),
            MaterialUniform::from(&b),
            MaterialUniform::from(&c),
        ];
        let slice = reg.uniforms();
        assert_eq!(slice.len(), 3);
        for (i, exp) in expected.iter().enumerate() {
            assert!(same_bits(&slice[i], exp), "slot {i} content mismatch");
        }

        // get() agrees with the slice, and both agree with the returned indices.
        assert!(same_bits(reg.get(ia).unwrap(), &expected[0]));
        assert!(same_bits(reg.get(ib).unwrap(), &expected[1]));
        assert!(same_bits(reg.get(ic).unwrap(), &expected[2]));
        assert_eq!(reg.get(ia).unwrap().albedo, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(reg.get(ic).unwrap().ior, 1.33);
    }

    #[test]
    fn get_out_of_range_is_none() {
        let mut reg = MaterialRegistry::new();
        reg.register(&base());
        assert!(reg.get(0).is_some());
        assert!(reg.get(1).is_none());
        assert!(reg.get(u32::MAX).is_none());
    }

    #[test]
    fn clear_resets_len_and_index_assignment() {
        let mut reg = MaterialRegistry::new();
        let a = base();
        let b = MaterialComponent::mirror(Vec4::ONE, 0.5);
        reg.register(&a);
        let ib = reg.register(&b);
        assert_eq!(ib, 1);
        assert_eq!(reg.len(), 2);

        reg.clear();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
        assert!(reg.uniforms().is_empty());
        assert!(reg.get(0).is_none());

        // After clearing, index assignment restarts — a stale dedup key would wrongly
        // report index 1 for `b` here.
        assert_eq!(reg.register(&b), 0);
        assert_eq!(reg.register(&a), 1);
        assert_eq!(reg.len(), 2);
        assert!(same_bits(reg.get(0).unwrap(), &MaterialUniform::from(&b)));
    }

    #[test]
    fn saturates_at_max_materials() {
        let mut reg = MaterialRegistry::new();

        // Fill every slot with a distinct material.
        for i in 0..MAX_MATERIALS {
            let m = MaterialComponent::matte(Vec4::new(i as f32, 0.0, 0.0, 1.0), 0.1, 0.9);
            assert_eq!(reg.register(&m), i as u32);
        }
        assert_eq!(reg.len(), MAX_MATERIALS);

        let last = (MAX_MATERIALS - 1) as u32;
        let last_bits = *reg.get(last).unwrap();

        // Overflowing materials clamp to the last usable slot and are not stored.
        for i in 0..4 {
            let extra =
                MaterialComponent::glass(Vec4::new(-1.0 - i as f32, 0.0, 0.0, 1.0), 1.5, 0.5);
            assert_eq!(reg.register(&extra), last);
        }
        assert_eq!(reg.len(), MAX_MATERIALS, "registry must not grow past the buffer");
        assert_eq!(reg.uniforms().len(), MAX_MATERIALS);

        // The occupant of the clamped slot is untouched.
        assert!(same_bits(reg.get(last).unwrap(), &last_bits));
        assert!(reg.get(MAX_MATERIALS as u32).is_none());

        // Already-registered materials still resolve to their own index while full.
        let first = MaterialComponent::matte(Vec4::new(0.0, 0.0, 0.0, 1.0), 0.1, 0.9);
        assert_eq!(reg.register(&first), 0);

        // Clearing recovers full capacity.
        reg.clear();
        assert_eq!(reg.register(&first), 0);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn texture_and_flags_participate_in_dedup() {
        let mut reg = MaterialRegistry::new();
        let plain = base();
        let textured = base().with_texture(0);
        let other_texture = base().with_texture(1);
        let cutout = base().with_texture(0).with_alpha_test();

        let ip = reg.register(&plain);
        let it = reg.register(&textured);
        let io = reg.register(&other_texture);
        let ic = reg.register(&cutout);
        assert_eq!(reg.len(), 4, "texture index and flags must each split a slot");
        assert_eq!(reg.register(&textured), it, "and still dedup with themselves");
        assert!(ip != it && it != io && io != ic);
    }

    #[test]
    fn positive_and_negative_zero_are_distinct() {
        // Bitwise keys mean signed zeros do not dedup, even though 0.0 == -0.0.
        let mut reg = MaterialRegistry::new();
        let pos = MaterialComponent::matte(Vec4::new(0.0, 0.5, 0.5, 1.0), 0.1, 0.9);
        let neg = MaterialComponent::matte(Vec4::new(-0.0, 0.5, 0.5, 1.0), 0.1, 0.9);

        let ip = reg.register(&pos);
        let in_ = reg.register(&neg);
        assert_ne!(ip, in_, "+0.0 and -0.0 have different bit patterns");
        assert_eq!(reg.len(), 2);

        // Same story for a scalar field.
        let mut a = base();
        a.specular = 0.0;
        let mut b = base();
        b.specular = -0.0;
        let ia = reg.register(&a);
        let ib = reg.register(&b);
        assert_ne!(ia, ib);
        assert_eq!(reg.len(), 4);
    }

    #[test]
    fn nan_keys_are_self_consistent() {
        // Unlike f32 ==, identical NaN bit patterns dedup — the uploaded bytes are the same.
        let mut reg = MaterialRegistry::new();
        let mut a = base();
        a.shininess = f32::NAN;
        let mut b = base();
        b.shininess = f32::NAN;

        assert_eq!(reg.register(&a), 0);
        assert_eq!(reg.register(&b), 0);
        assert_eq!(reg.len(), 1);
        assert!(reg.get(0).unwrap().shininess.is_nan());
    }

    #[test]
    fn key_covers_every_uniform_field() {
        // Guards against a field being forgotten in material_key: mutating any single field of
        // the uniform must change the key.
        let uniform = MaterialUniform::from(&base());
        let reference = material_key(&uniform);

        let mut mutated: Vec<MaterialUniform> = Vec::new();
        for i in 0..4 {
            let mut u = uniform;
            u.albedo[i] += 1.0;
            mutated.push(u);
        }
        let mut u = uniform;
        u.material_type += 1;
        mutated.push(u);
        let mut u = uniform;
        u.ambient += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.diffuse += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.specular += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.shininess += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.ior += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.reflectivity += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.transparency += 1.0;
        mutated.push(u);
        let mut u = uniform;
        u.texture_index = 3;
        mutated.push(u);
        let mut u = uniform;
        u.flags |= 1;
        mutated.push(u);
        let mut u = uniform;
        u.uv_scale[0] += 0.5;
        mutated.push(u);
        let mut u = uniform;
        u.uv_scale[1] += 0.5;
        mutated.push(u);

        assert_eq!(mutated.len(), KEY_WORDS);
        for (i, m) in mutated.iter().enumerate() {
            assert_ne!(material_key(m), reference, "field {i} is not part of the key");
        }
    }
}
