// Scene: a container of entities and their components.
//
// Storage is archetype-based (`scene::archetype`): entities sharing the same set of
// component types keep each type in one contiguous column, so iterating everything
// with a given component is a linear walk over packed memory.
//
// This file is deliberately thin. It owns entity-id allocation and nothing else;
// every storage decision lives in `archetype`. The public API here is unchanged
// from the previous `HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>>` version —
// same names, same signatures, same ordering guarantees — which is why `demo/`,
// `app/state.rs`, `scene/loader.rs` and `physics/` needed no edits for the rewrite,
// and why the tests written against the old storage are a regression suite for the
// new one.

use super::archetype::Archetypes;
use super::component::Component;
use super::entity::Entity;

/// A scene holds entities and their components.
pub struct Scene {
    next_id: u64,
    storage: Archetypes,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            storage: Archetypes::new(),
        }
    }

    /// Create a new entity in the scene.
    ///
    /// Ids are handed out in increasing order and never reused. Scene files and
    /// scripts both address entities by id, so a recycled id would make a stale
    /// reference silently point at a different object.
    pub fn create_entity(&mut self) -> Entity {
        let id = self.next_id;
        self.next_id += 1;
        let entity = Entity::new(id);
        self.storage.insert_entity(entity);
        entity
    }

    /// Remove an entity and all its components.
    pub fn destroy_entity(&mut self, entity: Entity) {
        self.storage.remove_entity(entity);
    }

    /// Add a component to an entity, replacing one of the same type if present.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        self.storage.set(entity, component);
    }

    /// Remove a component, returning whether the entity had one.
    pub fn remove_component<T: Component>(&mut self, entity: Entity) -> bool {
        self.storage.remove::<T>(entity)
    }

    /// Get a component reference from an entity.
    pub fn get_component<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.storage.get::<T>(entity)
    }

    /// Get a mutable component reference from an entity.
    pub fn get_component_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        self.storage.get_mut::<T>(entity)
    }

    /// Every entity that has a specific component type, in creation order.
    ///
    /// The ordering is part of the contract, not an accident of the storage:
    /// `app/state.rs` builds its per-object GPU indices from this list, so a
    /// shuffled order would reshuffle every object's material and transform lookup
    /// whenever a component was added anywhere in the scene.
    pub fn entities_with<T: Component>(&self) -> Vec<Entity> {
        self.storage.entities_with::<T>()
    }

    /// Every entity that has both component types, in creation order.
    pub fn entities_with_both<T: Component, U: Component>(&self) -> Vec<Entity> {
        self.storage.entities_with_both::<T, U>()
    }

    /// Packed slices of one component type, one per archetype containing it.
    ///
    /// The cache-friendly read path for a system that needs the values but not the
    /// entity ids. Not expressible over the previous storage at all.
    pub fn component_columns<T: Component>(&self) -> Vec<&[T]> {
        self.storage.columns::<T>()
    }

    /// Return all entities.
    pub fn all_entities(&self) -> &[Entity] {
        self.storage.all_entities()
    }

    /// How many archetypes the scene has fragmented into, and how many structural
    /// moves it has cost. Exposed for the profiler and the console: an archetype
    /// layout that fragments per entity has lost its advantage, and the only way to
    /// notice is to look.
    pub fn storage_stats(&self) -> (usize, u64) {
        (self.storage.archetype_count(), self.storage.migrations())
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct Health {
        hp: f32,
    }
    impl Component for Health {}

    #[derive(Clone, Debug)]
    struct Name(String);
    impl Component for Name {}

    #[test]
    fn create_entity_add_component() {
        let mut scene = Scene::new();
        let entity = scene.create_entity();
        scene.add_component(entity, Health { hp: 100.0 });

        let health = scene.get_component::<Health>(entity);
        assert!(health.is_some());
        assert_eq!(health.unwrap().hp, 100.0);
    }

    #[test]
    fn entity_without_component_returns_none() {
        let mut scene = Scene::new();
        let entity = scene.create_entity();
        assert!(scene.get_component::<Health>(entity).is_none());
    }

    #[test]
    fn destroy_entity_removes_all_components() {
        let mut scene = Scene::new();
        let entity = scene.create_entity();
        scene.add_component(entity, Health { hp: 50.0 });
        scene.destroy_entity(entity);
        assert!(scene.get_component::<Health>(entity).is_none());
        assert!(!scene.all_entities().iter().any(|e| e.id() == entity.id()));
    }

    #[test]
    fn multiple_components_on_one_entity() {
        let mut scene = Scene::new();
        let entity = scene.create_entity();
        scene.add_component(entity, Health { hp: 100.0 });
        scene.add_component(entity, Name("hero".to_string()));

        assert_eq!(scene.get_component::<Health>(entity).unwrap().hp, 100.0);
        assert_eq!(scene.get_component::<Name>(entity).unwrap().0, "hero");
    }

    #[test]
    fn entities_with_filter() {
        let mut scene = Scene::new();
        let a = scene.create_entity();
        let _b = scene.create_entity();
        let c = scene.create_entity();

        scene.add_component(a, Health { hp: 10.0 });
        scene.add_component(c, Health { hp: 20.0 });

        let healthy = scene.entities_with::<Health>();
        assert_eq!(healthy.len(), 2);
    }

    // --- properties the archetype rewrite had to preserve ---

    /// Ids must be handed out in increasing order and never reused. Scene files and
    /// scripts address entities by id, so recycling one would make a stale
    /// reference silently point at a different object.
    #[test]
    fn entity_ids_increase_and_are_never_reused() {
        let mut scene = Scene::new();
        let a = scene.create_entity();
        let b = scene.create_entity();
        assert!(b.id() > a.id());
        scene.destroy_entity(a);
        let c = scene.create_entity();
        assert!(
            c.id() > b.id(),
            "a destroyed entity's id must not come back: {} after {}",
            c.id(),
            b.id()
        );
    }

    /// `all_entities` is in creation order, and stays that way across a destroy —
    /// the archetype storage reorders its own rows on removal, so this is exactly
    /// the guarantee the rewrite could have broken.
    #[test]
    fn all_entities_stays_in_creation_order_across_a_destroy() {
        let mut scene = Scene::new();
        let entities: Vec<_> = (0..5).map(|_| scene.create_entity()).collect();
        for e in &entities {
            scene.add_component(*e, Health { hp: 1.0 });
        }
        scene.destroy_entity(entities[1]);
        let ids: Vec<u64> = scene.all_entities().iter().map(|e| e.id()).collect();
        assert_eq!(ids, vec![1, 3, 4, 5]);
    }

    /// The same guarantee for `entities_with`, which the renderer's object indices
    /// depend on: a shuffled order would reshuffle every material and transform
    /// lookup whenever a component was added anywhere.
    #[test]
    fn entities_with_stays_in_creation_order_across_archetypes() {
        let mut scene = Scene::new();
        let entities: Vec<_> = (0..6).map(|_| scene.create_entity()).collect();
        for (i, e) in entities.iter().enumerate() {
            scene.add_component(*e, Health { hp: i as f32 });
            // Alternating second component, so they split across archetypes.
            if i % 2 == 0 {
                scene.add_component(*e, Name(format!("n{i}")));
            }
        }
        let ids: Vec<u64> = scene
            .entities_with::<Health>()
            .iter()
            .map(|e| e.id())
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
    }

    /// Adding a component must not disturb any other entity's components. This is
    /// the failure mode a swap-remove storage introduces and the map storage could
    /// not have.
    #[test]
    fn adding_components_never_disturbs_other_entities() {
        let mut scene = Scene::new();
        let entities: Vec<_> = (0..10).map(|_| scene.create_entity()).collect();
        for (i, e) in entities.iter().enumerate() {
            scene.add_component(*e, Health { hp: i as f32 });
            scene.add_component(*e, Name(format!("n{i}")));
        }
        // Interleave more structural changes.
        for (i, e) in entities.iter().enumerate() {
            if i % 2 == 0 {
                scene.remove_component::<Name>(*e);
            }
        }
        for (i, e) in entities.iter().enumerate() {
            assert_eq!(
                scene.get_component::<Health>(*e).map(|h| h.hp),
                Some(i as f32),
                "entity {i} lost its own Health"
            );
            let name = scene.get_component::<Name>(*e).map(|n| n.0.clone());
            if i % 2 == 0 {
                assert_eq!(name, None);
            } else {
                assert_eq!(name, Some(format!("n{i}")));
            }
        }
    }

    #[test]
    fn a_component_can_be_replaced_in_place() {
        let mut scene = Scene::new();
        let e = scene.create_entity();
        scene.add_component(e, Health { hp: 1.0 });
        scene.add_component(e, Health { hp: 2.0 });
        assert_eq!(scene.get_component::<Health>(e).unwrap().hp, 2.0);
        // And replacing costs no structural change, which is what makes writing a
        // component every frame affordable.
        let (_, migrations_before) = scene.storage_stats();
        for i in 0..50 {
            scene.add_component(e, Health { hp: i as f32 });
        }
        let (_, migrations_after) = scene.storage_stats();
        assert_eq!(migrations_before, migrations_after);
    }

    #[test]
    fn a_two_type_query_requires_both_components() {
        let mut scene = Scene::new();
        let both = scene.create_entity();
        let one = scene.create_entity();
        scene.add_component(both, Health { hp: 1.0 });
        scene.add_component(both, Name("x".into()));
        scene.add_component(one, Health { hp: 2.0 });
        assert_eq!(scene.entities_with_both::<Health, Name>(), vec![both]);
    }

    /// The point of the rewrite: components of one archetype are contiguous.
    #[test]
    fn components_are_stored_in_packed_columns() {
        let mut scene = Scene::new();
        for _ in 0..64 {
            let e = scene.create_entity();
            scene.add_component(e, Health { hp: 1.0 });
        }
        let columns = scene.component_columns::<Health>();
        assert_eq!(columns.len(), 1, "identical entities should form one archetype");
        assert_eq!(columns[0].len(), 64);
    }

    /// A uniform scene must not fragment into an archetype per entity — that would
    /// leave the layout no better than the map it replaced.
    #[test]
    fn a_uniform_scene_does_not_fragment() {
        let mut scene = Scene::new();
        for i in 0..40 {
            let e = scene.create_entity();
            scene.add_component(e, Health { hp: i as f32 });
            scene.add_component(e, Name(format!("n{i}")));
        }
        let (archetypes, _) = scene.storage_stats();
        assert!(
            archetypes <= 3,
            "40 identical entities produced {archetypes} archetypes"
        );
    }

    #[test]
    fn removing_a_component_reports_whether_there_was_one() {
        let mut scene = Scene::new();
        let e = scene.create_entity();
        assert!(!scene.remove_component::<Health>(e));
        scene.add_component(e, Health { hp: 1.0 });
        assert!(scene.remove_component::<Health>(e));
        assert!(scene.get_component::<Health>(e).is_none());
    }

    /// An owned component has to be moved across archetypes, not dropped and
    /// default-constructed.
    #[test]
    fn an_owned_component_survives_a_structural_change() {
        let mut scene = Scene::new();
        let e = scene.create_entity();
        scene.add_component(e, Name("keep me".into()));
        scene.add_component(e, Health { hp: 1.0 });
        scene.remove_component::<Health>(e);
        assert_eq!(
            scene.get_component::<Name>(e).map(|n| n.0.clone()),
            Some("keep me".to_string())
        );
    }

    #[test]
    fn operations_on_an_unknown_entity_are_harmless() {
        let mut scene = Scene::new();
        let ghost = Entity::new(9999);
        scene.add_component(ghost, Health { hp: 1.0 });
        assert!(scene.get_component::<Health>(ghost).is_none());
        assert!(!scene.remove_component::<Health>(ghost));
        scene.destroy_entity(ghost);
        assert!(scene.all_entities().is_empty());
    }

    #[test]
    fn an_empty_scene_answers_every_query_without_panicking() {
        let scene = Scene::new();
        assert!(scene.all_entities().is_empty());
        assert!(scene.entities_with::<Health>().is_empty());
        assert!(scene.entities_with_both::<Health, Name>().is_empty());
        assert!(scene.component_columns::<Health>().is_empty());
    }
}