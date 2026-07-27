// Scene: a container of entities and their components.
// Uses a simple arena-based ECS-like storage.

use std::collections::HashMap;
use std::any::TypeId;
use crate::engine::core::Result;

use super::component::Component;
use super::entity::Entity;

/// Type-erased storage for components of a single type.
///
/// Each entity ID maps to a boxed component.
type ComponentStorage = HashMap<u64, Box<dyn std::any::Any + Send>>;

/// A scene holds entities and their components.
pub struct Scene {
    next_id: u64,
    entities: Vec<Entity>,
    components: HashMap<TypeId, ComponentStorage>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entities: Vec::new(),
            components: HashMap::new(),
        }
    }

    /// Create a new entity in the scene.
    pub fn create_entity(&mut self) -> Entity {
        let id = self.next_id;
        self.next_id += 1;
        let entity = Entity::new(id);
        self.entities.push(entity);
        entity
    }

    /// Remove an entity and all its components.
    pub fn destroy_entity(&mut self, entity: Entity) {
        let id = entity.id();
        self.entities.retain(|e| e.id() != id);
        for storage in self.components.values_mut() {
            storage.remove(&id);
        }
    }

    /// Add a component to an entity.
    pub fn add_component<T: Component>(&mut self, entity: Entity, component: T) {
        let tid = TypeId::of::<T>();
        let storage = self
            .components
            .entry(tid)
            .or_insert_with(HashMap::new);
        storage.insert(entity.id(), Box::new(component));
    }

    /// Get a component reference from an entity.
    pub fn get_component<T: Component>(&self, entity: Entity) -> Option<&T> {
        let tid = TypeId::of::<T>();
        let storage = self.components.get(&tid)?;
        let boxed = storage.get(&entity.id())?;
        boxed.downcast_ref::<T>()
    }

    /// Get a mutable component reference from an entity.
    pub fn get_component_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let tid = TypeId::of::<T>();
        let storage = self.components.get_mut(&tid)?;
        let boxed = storage.get_mut(&entity.id())?;
        boxed.downcast_mut::<T>()
    }

    /// Iterate over all entities that have a specific component type.
    pub fn entities_with<T: Component>(&self) -> Vec<Entity> {
        let tid = TypeId::of::<T>();
        let Some(storage) = self.components.get(&tid) else {
            return Vec::new();
        };
        self.entities
            .iter()
            .filter(|e| storage.contains_key(&e.id()))
            .copied()
            .collect()
    }

    /// Return all entities.
    pub fn all_entities(&self) -> &[Entity] {
        &self.entities
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
        let b = scene.create_entity();
        let c = scene.create_entity();

        scene.add_component(a, Health { hp: 10.0 });
        scene.add_component(c, Health { hp: 20.0 });

        let healthy = scene.entities_with::<Health>();
        assert_eq!(healthy.len(), 2);
    }
}