// Parent/child transform hierarchy (Phase 2.1): world = parent_world * local.
//
// Design notes:
// - Kept OUT of `Scene`. Scene is a flat ECS store; the graph is a separate concern, so
//   hierarchy lives here and only ever speaks in terms of `Entity` handles. It owns no
//   transforms either: local matrices are supplied by the caller through a closure, which
//   keeps this module free of any component/GPU knowledge and trivially testable.
// - Two mirrored maps (child -> parent, parent -> children). The redundancy is deliberate:
//   `world_matrix` needs the upward link, traversal/`roots` need the downward one. Every
//   mutation updates both, so no stale child entries can survive a re-parent or a remove.
// - Every traversal is depth-bounded so that no graph state can hang the render loop. The two
//   bounds are deliberately different: cycle detection uses an exact bound derived from the
//   number of stored links (so it is never fooled by a deep-but-valid chain), while the
//   matrix walks use the fixed `MAX_DEPTH` budget the renderer is willing to pay per frame.

use std::collections::HashMap;

use super::entity::Entity;
use crate::engine::core::{EngineError, Result};
use crate::engine::math::Mat4;

/// Hard cap on the number of parent links a world-matrix walk will follow.
///
/// Real scenes are far shallower, but `set_parent` accepts any acyclic chain, so the cap is a
/// contract rather than an impossibility: past `MAX_DEPTH` links the walk stops and the
/// root-most levels are simply not applied. The result is then wrong but finite and NaN-free —
/// the render loop keeps running instead of stalling. Because the truncation is anchored at the
/// queried entity, a batch run that has already memoized deeper ancestors can be more accurate
/// than a cold single lookup; do not rely on the two agreeing beyond the cap.
pub const MAX_DEPTH: usize = 256;

/// Directed acyclic parent/child graph over entities.
///
/// Stores only relationships — no transforms, no components. Combine it with a local-matrix
/// lookup (e.g. `TransformComponent::world_matrix`) to obtain world matrices.
#[derive(Clone, Debug)]
pub struct Hierarchy {
    /// child -> parent.
    parents: HashMap<Entity, Entity>,
    /// parent -> children, in insertion order (stable iteration for the renderer).
    children: HashMap<Entity, Vec<Entity>>,
}

impl Hierarchy {
    /// Create an empty hierarchy.
    pub fn new() -> Self {
        Self {
            parents: HashMap::new(),
            children: HashMap::new(),
        }
    }

    /// Attach `child` under `parent`, replacing any previous parent link.
    ///
    /// Returns `EngineError::InvalidState` when the link would create a cycle — including
    /// `child == parent` and longer loops such as `a -> b -> c -> a`, at any chain length. On
    /// error the graph is left completely untouched.
    pub fn set_parent(&mut self, child: Entity, parent: Entity) -> Result<()> {
        if child == parent {
            return Err(EngineError::InvalidState(format!(
                "entity {} cannot be its own parent",
                child.id()
            )));
        }
        // A cycle appears exactly when the new parent already sits below the child.
        if self.is_descendant_of(parent, child) {
            return Err(EngineError::InvalidState(format!(
                "parenting entity {} to {} would create a cycle",
                child.id(),
                parent.id()
            )));
        }

        // Drop the old link first so the previous parent keeps no stale child entry.
        self.clear_parent(child);
        self.parents.insert(child, parent);
        self.children.entry(parent).or_default().push(child);
        Ok(())
    }

    /// Detach `child` from its parent, making it a root. Its own children are unaffected.
    pub fn clear_parent(&mut self, child: Entity) {
        if let Some(old_parent) = self.parents.remove(&child) {
            if let Some(siblings) = self.children.get_mut(&old_parent) {
                siblings.retain(|&e| e != child);
                // Prune empty vectors so `contains` stays a pure membership question.
                if siblings.is_empty() {
                    self.children.remove(&old_parent);
                }
            }
        }
    }

    /// Parent of `child`, or `None` if it is a root / unknown.
    pub fn parent(&self, child: Entity) -> Option<Entity> {
        self.parents.get(&child).copied()
    }

    /// Direct children of `parent` in insertion order; an empty slice if there are none.
    pub fn children(&self, parent: Entity) -> &[Entity] {
        match self.children.get(&parent) {
            Some(list) => list.as_slice(),
            None => &[],
        }
    }

    /// Remove `entity` from the graph entirely.
    ///
    /// Chosen semantics: **orphan, do not cascade**. The entity is detached from its parent and
    /// each of its children becomes a root, keeping its own local transform. Destroying an
    /// entity is `Scene`'s job, so silently deleting a whole subtree here would delete
    /// relationships for entities that still exist and still need to be drawn. Callers wanting a
    /// recursive delete walk `children()` themselves before calling this.
    pub fn remove(&mut self, entity: Entity) {
        self.clear_parent(entity);
        if let Some(kids) = self.children.remove(&entity) {
            for kid in kids {
                self.parents.remove(&kid);
            }
        }
    }

    /// Filter `entities` down to those without a parent, preserving the input order.
    pub fn roots(&self, entities: &[Entity]) -> Vec<Entity> {
        entities
            .iter()
            .copied()
            .filter(|e| !self.parents.contains_key(e))
            .collect()
    }

    /// Whether the graph holds any link for `entity` (as a child or as a parent).
    pub fn contains(&self, entity: Entity) -> bool {
        self.parents.contains_key(&entity) || self.children.contains_key(&entity)
    }

    /// Number of parent links stored (i.e. non-root entities).
    pub fn link_count(&self) -> usize {
        self.parents.len()
    }

    /// World matrix of a single entity: `parent_world * local`, composed root-first.
    ///
    /// `local_of` returns the entity's LOCAL matrix; entities unknown to the caller should map
    /// to `Mat4::IDENTITY`. Cost is O(depth) — use [`Hierarchy::world_matrices`] when computing
    /// many at once. The upward walk stops after [`MAX_DEPTH`] links.
    pub fn world_matrix(&self, entity: Entity, local_of: &impl Fn(Entity) -> Mat4) -> Mat4 {
        let mut chain: Vec<Entity> = Vec::new();
        let mut cursor = Some(entity);
        while let Some(current) = cursor {
            chain.push(current);
            if chain.len() >= MAX_DEPTH {
                break; // bounded walk: pathological graphs terminate
            }
            cursor = self.parent(current);
        }

        // chain is leaf-first, so fold in reverse: world = root * ... * leaf.
        let mut world = Mat4::IDENTITY;
        for current in chain.iter().rev() {
            world = world.mul(local_of(*current));
        }
        world
    }

    /// World matrices for many entities at once, memoized.
    ///
    /// Each entity's world matrix is computed exactly once and reused by its descendants, so the
    /// whole batch costs O(n) matrix multiplications for n distinct entities (plus hash lookups)
    /// regardless of tree depth — versus O(n * depth) for repeated [`Hierarchy::world_matrix`]
    /// calls. Input order is irrelevant: ancestors are resolved on demand, so a child listed
    /// before its parent is still correct.
    ///
    /// Returns one `(entity, world)` pair per input entry, in input order.
    pub fn world_matrices(
        &self,
        entities: &[Entity],
        local_of: &impl Fn(Entity) -> Mat4,
    ) -> Vec<(Entity, Mat4)> {
        let mut cache: HashMap<Entity, Mat4> = HashMap::with_capacity(entities.len());
        let mut out = Vec::with_capacity(entities.len());
        for &entity in entities {
            let world = self.resolve_cached(entity, local_of, &mut cache);
            out.push((entity, world));
        }
        out
    }

    /// Resolve one world matrix, filling `cache` for every ancestor visited on the way.
    ///
    /// Iterative (not recursive) so that a deep chain cannot blow the stack.
    fn resolve_cached(
        &self,
        entity: Entity,
        local_of: &impl Fn(Entity) -> Mat4,
        cache: &mut HashMap<Entity, Mat4>,
    ) -> Mat4 {
        if let Some(world) = cache.get(&entity) {
            return *world;
        }

        // Walk up until a cached ancestor (or a root) is found, collecting the uncached part.
        let mut pending: Vec<Entity> = Vec::new();
        let mut base = Mat4::IDENTITY;
        let mut cursor = Some(entity);
        while let Some(current) = cursor {
            if let Some(world) = cache.get(&current) {
                base = *world;
                break;
            }
            pending.push(current);
            if pending.len() >= MAX_DEPTH {
                break; // bounded walk
            }
            cursor = self.parent(current);
        }

        // Compose downwards from the known base, memoizing every level.
        let mut world = base;
        for current in pending.iter().rev() {
            world = world.mul(local_of(*current));
            cache.insert(*current, world);
        }
        world
    }

    /// Whether `candidate` is `ancestor` itself or sits below it in the graph.
    ///
    /// The walk is bounded by `parents.len() + 1`, not by [`MAX_DEPTH`]: an acyclic chain follows
    /// a distinct stored link at every step, so it can never visit more nodes than that. Using
    /// the exact bound keeps cycle detection reliable for arbitrarily deep chains (a fixed cap
    /// would silently accept the edge that closes a loop longer than the cap), while still
    /// terminating on an already-cyclic graph — where exceeding the bound *is* the proof of a
    /// cycle, so we answer conservatively and let the caller reject the edit.
    fn is_descendant_of(&self, candidate: Entity, ancestor: Entity) -> bool {
        let max_nodes = self.parents.len() + 1;
        let mut cursor = Some(candidate);
        let mut visited = 0usize;
        while let Some(current) = cursor {
            if current == ancestor {
                return true;
            }
            visited += 1;
            if visited > max_nodes {
                return true;
            }
            cursor = self.parent(current);
        }
        false
    }
}

impl Default for Hierarchy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::math::{radians, Transform, Vec3};

    /// Build a local-matrix lookup from an (id, matrix) table; unknown entities are identity.
    fn locals(table: Vec<(Entity, Mat4)>) -> impl Fn(Entity) -> Mat4 {
        let map: HashMap<u64, Mat4> = table.into_iter().map(|(e, m)| (e.id(), m)).collect();
        move |e: Entity| map.get(&e.id()).copied().unwrap_or(Mat4::IDENTITY)
    }

    fn assert_mat_eq(a: Mat4, b: Mat4) {
        for i in 0..16 {
            assert!(
                (a.m[i] - b.m[i]).abs() < 1e-5,
                "matrices differ at {i}: {} vs {}",
                a.m[i],
                b.m[i]
            );
        }
    }

    fn assert_vec_eq(a: Vec3, b: Vec3) {
        assert!(
            (a - b).length() < 1e-4,
            "expected {:?}, got {:?}",
            b,
            a
        );
    }

    #[test]
    fn root_world_matrix_is_its_own_local() {
        let h = Hierarchy::new();
        let a = Entity::new(1);
        let local = Mat4::translation(3.0, -2.0, 1.0);
        let f = locals(vec![(a, local)]);
        assert_mat_eq(h.world_matrix(a, &f), local);
    }

    #[test]
    fn child_world_matrix_is_parent_times_local() {
        let mut h = Hierarchy::new();
        let parent = Entity::new(1);
        let child = Entity::new(2);
        h.set_parent(child, parent).unwrap();

        let p_local = Mat4::translation(5.0, 0.0, 0.0);
        let c_local = Mat4::scaling(2.0, 2.0, 2.0);
        let f = locals(vec![(parent, p_local), (child, c_local)]);

        assert_mat_eq(h.world_matrix(child, &f), p_local.mul(c_local));
        assert_mat_eq(h.world_matrix(parent, &f), p_local);
        assert_eq!(h.parent(child), Some(parent));
        assert_eq!(h.children(parent), &[child][..]);
    }

    #[test]
    fn three_deep_chain_composes_in_order() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let mid = Entity::new(2);
        let leaf = Entity::new(3);
        h.set_parent(mid, root).unwrap();
        h.set_parent(leaf, mid).unwrap();

        // root translates +10x, mid yaws 90 deg, leaf offsets +5z in its own space.
        let f = locals(vec![
            (root, Mat4::translation(10.0, 0.0, 0.0)),
            (mid, Mat4::rotation_y(std::f32::consts::FRAC_PI_2)),
            (leaf, Mat4::translation(0.0, 0.0, 5.0)),
        ]);

        // Order matters: rotate the +5z offset into +5x, then translate by +10x.
        let p = h.world_matrix(leaf, &f).transform_point(Vec3::ZERO);
        assert_vec_eq(p, Vec3::new(15.0, 0.0, 0.0));

        // Guard against the reversed (leaf-first) composition, which yields (0, 0, -5).
        let reversed = Mat4::translation(0.0, 0.0, 5.0)
            .mul(Mat4::rotation_y(std::f32::consts::FRAC_PI_2))
            .mul(Mat4::translation(10.0, 0.0, 0.0))
            .transform_point(Vec3::ZERO);
        assert!(
            (p - reversed).length() > 1.0,
            "composition order is not being tested: {p:?} vs {reversed:?}"
        );
    }

    #[test]
    fn parent_scale_and_rotation_apply_to_child_offset() {
        let mut h = Hierarchy::new();
        let parent = Entity::new(1);
        let child = Entity::new(2);
        h.set_parent(child, parent).unwrap();

        let p = Transform::new(
            Vec3::ZERO,
            Vec3::new(0.0, radians(90.0), 0.0),
            Vec3::splat(2.0),
        );
        let c = Transform::new(Vec3::UNIT_X, Vec3::ZERO, Vec3::ONE);
        let f = locals(vec![(parent, p.to_matrix()), (child, c.to_matrix())]);

        // +1x local, doubled by the parent scale, then yawed 90 deg -> (0, 0, -2).
        let world_origin = h.world_matrix(child, &f).transform_point(Vec3::ZERO);
        assert_vec_eq(world_origin, Vec3::new(0.0, 0.0, -2.0));
    }

    #[test]
    fn moving_the_parent_moves_the_child() {
        let mut h = Hierarchy::new();
        let parent = Entity::new(1);
        let child = Entity::new(2);
        h.set_parent(child, parent).unwrap();

        let c_local = Mat4::translation(1.0, 0.0, 0.0);
        let before = locals(vec![(parent, Mat4::IDENTITY), (child, c_local)]);
        assert_vec_eq(
            h.world_matrix(child, &before).transform_point(Vec3::ZERO),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let after = locals(vec![
            (parent, Mat4::translation(0.0, 7.0, 0.0)),
            (child, c_local),
        ]);
        assert_vec_eq(
            h.world_matrix(child, &after).transform_point(Vec3::ZERO),
            Vec3::new(1.0, 7.0, 0.0),
        );
    }

    #[test]
    fn self_parenting_is_rejected() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        assert!(h.set_parent(a, a).is_err());
        assert_eq!(h.parent(a), None);
        assert!(!h.contains(a));
    }

    #[test]
    fn two_cycle_is_rejected() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        h.set_parent(a, b).unwrap();
        assert!(h.set_parent(b, a).is_err());
        // The rejected call must not have disturbed the existing link.
        assert_eq!(h.parent(a), Some(b));
        assert_eq!(h.parent(b), None);
        assert_eq!(h.children(b), &[a][..]);
    }

    #[test]
    fn three_cycle_is_rejected() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        let c = Entity::new(3);
        h.set_parent(b, a).unwrap(); // a -> b
        h.set_parent(c, b).unwrap(); // b -> c
        assert!(h.set_parent(a, c).is_err()); // would close a -> b -> c -> a
        assert_eq!(h.parent(a), None);
        assert_eq!(h.children(c), &[] as &[Entity]);
    }

    #[test]
    fn reparenting_removes_child_from_old_parent() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        let c = Entity::new(3);
        h.set_parent(c, a).unwrap();
        h.set_parent(c, b).unwrap();

        assert_eq!(h.parent(c), Some(b));
        assert_eq!(h.children(b), &[c][..]);
        assert!(h.children(a).is_empty(), "stale child entry left on old parent");
        assert!(!h.contains(a));
        assert_eq!(h.link_count(), 1, "duplicate parent link stored");
    }

    #[test]
    fn reparenting_to_the_same_parent_does_not_duplicate() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        h.set_parent(b, a).unwrap();
        h.set_parent(b, a).unwrap();
        assert_eq!(h.children(a), &[b][..]);
        assert_eq!(h.link_count(), 1);
    }

    #[test]
    fn clear_parent_makes_a_root_and_keeps_grandchildren() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let mid = Entity::new(2);
        let leaf = Entity::new(3);
        h.set_parent(mid, root).unwrap();
        h.set_parent(leaf, mid).unwrap();

        h.clear_parent(mid);
        assert_eq!(h.parent(mid), None);
        assert!(h.children(root).is_empty());
        assert_eq!(h.parent(leaf), Some(mid), "clear_parent must not cascade");

        // Detached subtree no longer inherits the root transform.
        let f = locals(vec![
            (root, Mat4::translation(100.0, 0.0, 0.0)),
            (mid, Mat4::IDENTITY),
            (leaf, Mat4::translation(1.0, 0.0, 0.0)),
        ]);
        assert_vec_eq(
            h.world_matrix(leaf, &f).transform_point(Vec3::ZERO),
            Vec3::new(1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn remove_orphans_children_and_detaches_from_parent() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let mid = Entity::new(2);
        let leaf_a = Entity::new(3);
        let leaf_b = Entity::new(4);
        h.set_parent(mid, root).unwrap();
        h.set_parent(leaf_a, mid).unwrap();
        h.set_parent(leaf_b, mid).unwrap();

        h.remove(mid);

        assert!(!h.contains(mid));
        assert_eq!(h.parent(mid), None);
        assert!(h.children(root).is_empty());
        assert!(h.children(mid).is_empty());
        assert_eq!(h.parent(leaf_a), None, "children must become roots");
        assert_eq!(h.parent(leaf_b), None);
        assert_eq!(h.link_count(), 0);

        // The orphans keep their own locals as world matrices.
        let f = locals(vec![(leaf_a, Mat4::translation(2.0, 0.0, 0.0))]);
        assert_mat_eq(h.world_matrix(leaf_a, &f), Mat4::translation(2.0, 0.0, 0.0));
    }

    #[test]
    fn children_of_unknown_entity_is_empty_slice() {
        let mut h = Hierarchy::new();
        h.set_parent(Entity::new(2), Entity::new(1)).unwrap();
        let unknown = Entity::new(9999);
        assert!(h.children(unknown).is_empty());
        assert_eq!(h.parent(unknown), None);
        assert!(!h.contains(unknown));
    }

    #[test]
    fn roots_filters_parented_entities() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        let c = Entity::new(3);
        let d = Entity::new(4);
        h.set_parent(b, a).unwrap();
        h.set_parent(c, b).unwrap();

        let all = [a, b, c, d];
        assert_eq!(h.roots(&all), vec![a, d]);
        // Order follows the input, not the map.
        assert_eq!(h.roots(&[d, c, a]), vec![d, a]);
        assert!(h.roots(&[]).is_empty());
    }

    #[test]
    fn world_matrices_agree_with_per_entity_on_a_branching_tree() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let child_a = Entity::new(2);
        let child_b = Entity::new(3);
        let grandchild = Entity::new(4);
        h.set_parent(child_a, root).unwrap();
        h.set_parent(child_b, root).unwrap();
        h.set_parent(grandchild, child_a).unwrap();

        let f = locals(vec![
            (root, Mat4::translation(1.0, 2.0, 3.0)),
            (child_a, Mat4::rotation_z(radians(30.0))),
            (child_b, Mat4::scaling(2.0, 1.0, 0.5)),
            (grandchild, Mat4::translation(0.0, 4.0, 0.0)),
        ]);

        let entities = [root, child_a, child_b, grandchild];
        let batch = h.world_matrices(&entities, &f);
        assert_eq!(batch.len(), 4);
        for (entity, world) in batch {
            assert_mat_eq(world, h.world_matrix(entity, &f));
        }
    }

    #[test]
    fn world_matrices_is_order_independent() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let mid = Entity::new(2);
        let leaf = Entity::new(3);
        h.set_parent(mid, root).unwrap();
        h.set_parent(leaf, mid).unwrap();

        let f = locals(vec![
            (root, Mat4::translation(1.0, 0.0, 0.0)),
            (mid, Mat4::translation(0.0, 1.0, 0.0)),
            (leaf, Mat4::translation(0.0, 0.0, 1.0)),
        ]);

        // Children before parents must still resolve correctly (memoized upward walk).
        let leaf_first = h.world_matrices(&[leaf, mid, root], &f);
        assert_eq!(leaf_first[0].0, leaf);
        assert_vec_eq(
            leaf_first[0].1.transform_point(Vec3::ZERO),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert_vec_eq(
            leaf_first[1].1.transform_point(Vec3::ZERO),
            Vec3::new(1.0, 1.0, 0.0),
        );
        assert_vec_eq(
            leaf_first[2].1.transform_point(Vec3::ZERO),
            Vec3::new(1.0, 0.0, 0.0),
        );
    }

    #[test]
    fn world_matrices_handles_repeated_entities() {
        let mut h = Hierarchy::new();
        let root = Entity::new(1);
        let leaf = Entity::new(2);
        h.set_parent(leaf, root).unwrap();
        let f = locals(vec![
            (root, Mat4::translation(1.0, 0.0, 0.0)),
            (leaf, Mat4::translation(1.0, 0.0, 0.0)),
        ]);
        let out = h.world_matrices(&[leaf, leaf, root], &f);
        assert_eq!(out.len(), 3);
        assert_mat_eq(out[0].1, out[1].1);
    }

    #[test]
    fn deep_chain_is_bounded_and_terminates() {
        // 400 levels, each translating +1x, is deeper than MAX_DEPTH on purpose.
        let mut h = Hierarchy::new();
        let count = 400u64;
        let mut table = Vec::new();
        for id in 1..=count {
            let e = Entity::new(id);
            table.push((e, Mat4::translation(1.0, 0.0, 0.0)));
            if id > 1 {
                h.set_parent(e, Entity::new(id - 1)).unwrap();
            }
        }
        let f = locals(table);

        // The walk stops after MAX_DEPTH links instead of hanging or overflowing.
        let leaf = Entity::new(count);
        let world = h.world_matrix(leaf, &f);
        assert_vec_eq(
            world.transform_point(Vec3::ZERO),
            Vec3::new(MAX_DEPTH as f32, 0.0, 0.0),
        );

        // Within the bound the result is exact.
        let inside = Entity::new(100);
        assert_vec_eq(
            h.world_matrix(inside, &f).transform_point(Vec3::ZERO),
            Vec3::new(100.0, 0.0, 0.0),
        );

        // Batch mode over the whole chain must also terminate, and the memoized walk resolves
        // each level from its already-cached parent, so it stays exact past MAX_DEPTH.
        let all: Vec<Entity> = (1..=count).map(Entity::new).collect();
        let batch = h.world_matrices(&all, &f);
        assert_eq!(batch.len(), count as usize);
        assert_vec_eq(
            batch[count as usize - 1].1.transform_point(Vec3::ZERO),
            Vec3::new(count as f32, 0.0, 0.0),
        );
    }

    /// Build a straight chain `1 -> 2 -> ... -> len`, returning the leaf.
    fn chain(h: &mut Hierarchy, len: u64) -> Entity {
        for id in 2..=len {
            h.set_parent(Entity::new(id), Entity::new(id - 1)).unwrap();
        }
        Entity::new(len)
    }

    #[test]
    fn cycle_is_rejected_on_a_chain_longer_than_max_depth() {
        // Regression: with a fixed MAX_DEPTH bound the ancestor walk gives up before reaching the
        // root of an over-deep chain and happily closes the loop, corrupting the graph for good.
        let mut h = Hierarchy::new();
        let leaf = chain(&mut h, MAX_DEPTH as u64 * 2);
        let root = Entity::new(1);

        assert!(h.set_parent(root, leaf).is_err());
        assert_eq!(h.parent(root), None, "graph must be untouched on error");
        assert!(h.children(leaf).is_empty());

        // A non-cyclic link on the same oversized graph must still be accepted.
        let outsider = Entity::new(999_999);
        assert!(h.set_parent(root, outsider).is_ok());
        assert_eq!(h.parent(root), Some(outsider));
    }

    #[test]
    fn cycle_errors_use_invalid_state() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        h.set_parent(b, a).unwrap();
        let self_err = h.set_parent(a, a).unwrap_err();
        let cycle_err = h.set_parent(a, b).unwrap_err();
        for err in [self_err, cycle_err] {
            assert!(
                matches!(err, EngineError::InvalidState(_)),
                "unexpected variant: {err:?}"
            );
        }
    }

    #[test]
    fn unknown_entities_resolve_to_their_local_only() {
        let h = Hierarchy::new();
        let ghost = Entity::new(77);
        // No graph entry and no local -> identity, never NaN.
        assert_mat_eq(h.world_matrix(ghost, &locals(vec![])), Mat4::IDENTITY);
        let f = locals(vec![(ghost, Mat4::scaling(3.0, 3.0, 3.0))]);
        assert_mat_eq(h.world_matrix(ghost, &f), Mat4::scaling(3.0, 3.0, 3.0));
        assert_eq!(h.world_matrices(&[ghost], &f).len(), 1);
    }

    #[test]
    fn mutating_unknown_entities_is_a_no_op() {
        let mut h = Hierarchy::new();
        let a = Entity::new(1);
        let b = Entity::new(2);
        h.set_parent(b, a).unwrap();

        h.clear_parent(Entity::new(50));
        h.remove(Entity::new(51));
        assert_eq!(h.parent(b), Some(a));
        assert_eq!(h.children(a), &[b][..]);
        assert_eq!(h.link_count(), 1);
    }

    #[test]
    fn contains_covers_both_ends_of_a_link() {
        let mut h = Hierarchy::new();
        let parent = Entity::new(1);
        let child = Entity::new(2);
        h.set_parent(child, parent).unwrap();
        assert!(h.contains(parent), "a parent-only entity is still in the graph");
        assert!(h.contains(child));
        h.clear_parent(child);
        assert!(!h.contains(parent));
        assert!(!h.contains(child));
    }

    #[test]
    fn default_is_empty() {
        let h = Hierarchy::default();
        assert_eq!(h.link_count(), 0);
        assert_eq!(h.roots(&[Entity::new(1)]), vec![Entity::new(1)]);
        assert!(!h.contains(Entity::new(1)));
    }
}
