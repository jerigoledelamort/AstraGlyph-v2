// Archetype-based component storage.
//
// Phase 5.4's stated goal is "cache-friendly iteration". The old storage was
// `HashMap<TypeId, HashMap<EntityId, Box<dyn Any>>>`: every component individually
// boxed, reached through two hash lookups, scattered across the heap. Iterating a
// hundred of them touched a hundred cache lines in whatever order the allocator
// chose.
//
// An archetype is a set of entities that share exactly the same component types.
// Each archetype stores each of its component types in one contiguous `Vec`, so
// iterating every entity with a given component is a linear walk over packed
// memory. `entities_with::<T>` becomes "list the archetypes containing T, walk
// each one's rows" instead of a filtered scan over every entity in the scene.
//
// What this costs, honestly: adding or removing a component *moves* an entity to a
// different archetype, which copies its components. That is the archetype
// trade-off — cheap iteration paid for with expensive structural change — and it is
// the right side of the trade here because a scene is built once and iterated every
// frame.
//
// The public API of `Scene` is unchanged, deliberately. `create_entity`,
// `add_component`, `get_component`, `entities_with` and `all_entities` behave
// exactly as before, so `demo/`, `app/state.rs` and `scene/loader.rs` needed no
// edits — which is also what makes the old tests a regression suite for the new
// storage rather than tests of it.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use super::component::Component;
use super::entity::Entity;

/// A type-erased column: all values of one component type, packed.
///
/// `Box<dyn Any>` wraps the *column*, not each element, so a column of 1000
/// components is one allocation rather than 1000. That is the whole point of the
/// rewrite — the indirection moves from per-component to per-type-per-archetype.
trait Column: Any {
    /// Number of rows.
    fn len(&self) -> usize;
    /// Remove row `index` by swapping the last row into it, and return the removed
    /// value so a migrating entity can carry it to its new archetype.
    ///
    /// Swap-remove rather than shift: a shift is O(n) and reorders every row after
    /// the hole, which would invalidate every other column's row indices. Swapping
    /// touches exactly two rows, and the caller fixes up the one entity whose index
    /// moved.
    fn swap_remove_boxed(&mut self, index: usize) -> Box<dyn Any + Send>;
    /// Push a value that was moved out of another archetype's matching column.
    ///
    /// Returns false if the boxed value was not of this column's type, which can
    /// only happen through a bug in the migration code — reported rather than
    /// silently dropping the component.
    fn push_boxed(&mut self, value: Box<dyn Any + Send>) -> bool;
    /// An empty column of the same type, for building a new archetype.
    fn empty_clone(&self) -> Box<dyn Column + Send>;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A concrete column of `T`.
struct TypedColumn<T> {
    values: Vec<T>,
}

impl<T: Component> Column for TypedColumn<T> {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn swap_remove_boxed(&mut self, index: usize) -> Box<dyn Any + Send> {
        Box::new(self.values.swap_remove(index))
    }

    fn push_boxed(&mut self, value: Box<dyn Any + Send>) -> bool {
        match value.downcast::<T>() {
            Ok(typed) => {
                self.values.push(*typed);
                true
            }
            Err(_) => false,
        }
    }

    fn empty_clone(&self) -> Box<dyn Column + Send> {
        Box::new(TypedColumn::<T> { values: Vec::new() })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A set of entities sharing exactly one set of component types.
struct Archetype {
    /// The component types every entity here has, sorted so a type set has one
    /// canonical form and two archetypes with the same types are recognised as the
    /// same archetype regardless of insertion order.
    types: Vec<TypeId>,
    /// Entities in row order: `entities[i]` owns row `i` of every column.
    entities: Vec<Entity>,
    /// One packed column per type in `types`.
    columns: HashMap<TypeId, Box<dyn Column + Send>>,
}

impl Archetype {
    fn new(types: Vec<TypeId>) -> Self {
        Self {
            types,
            entities: Vec::new(),
            columns: HashMap::new(),
        }
    }

    fn has(&self, type_id: TypeId) -> bool {
        self.types.contains(&type_id)
    }
}

/// Where an entity lives: which archetype, and which row within it.
#[derive(Clone, Copy, Debug)]
struct Location {
    archetype: usize,
    row: usize,
}

/// Archetype-based component storage.
pub struct Archetypes {
    archetypes: Vec<Archetype>,
    /// Which archetype holds each type set, keyed by the sorted type list. Avoids a
    /// linear scan over archetypes on every structural change.
    by_types: HashMap<Vec<TypeId>, usize>,
    /// Where each entity is.
    locations: HashMap<u64, Location>,
    /// Every live entity, in creation order.
    ///
    /// Kept separately because row order within an archetype is *not* creation
    /// order — swap-remove shuffles it — and `all_entities` is documented to be
    /// stable. Scene files rely on that: entity ids are assigned in load order and
    /// scripts address them by id.
    entities: Vec<Entity>,
    /// Structural moves performed, so the archetype trade-off is measurable rather
    /// than asserted. A scene that migrates every frame is misusing the storage.
    migrations: u64,
}

impl Default for Archetypes {
    fn default() -> Self {
        Self::new()
    }
}

impl Archetypes {
    pub fn new() -> Self {
        Self {
            archetypes: Vec::new(),
            by_types: HashMap::new(),
            locations: HashMap::new(),
            entities: Vec::new(),
            migrations: 0,
        }
    }

    /// Register a new entity, with no components.
    pub fn insert_entity(&mut self, entity: Entity) {
        // The empty archetype is real: an entity with no components still has to be
        // somewhere, or `all_entities` and `destroy` would need a special case.
        let archetype = self.archetype_for(&[]);
        let row = self.archetypes[archetype].entities.len();
        self.archetypes[archetype].entities.push(entity);
        self.locations.insert(entity.id(), Location { archetype, row });
        self.entities.push(entity);
    }

    /// Remove an entity and all its components.
    pub fn remove_entity(&mut self, entity: Entity) {
        let Some(location) = self.locations.remove(&entity.id()) else {
            return;
        };
        self.entities.retain(|e| e.id() != entity.id());
        self.remove_row(location);
    }

    /// Every live entity, in creation order.
    pub fn all_entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Structural moves since construction.
    pub fn migrations(&self) -> u64 {
        self.migrations
    }

    /// How many archetypes exist. Exposed because it is the measurable shape of the
    /// storage: a scene of uniform entities should have very few.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Add or replace a component.
    pub fn set<T: Component>(&mut self, entity: Entity, component: T) {
        let type_id = TypeId::of::<T>();
        let Some(location) = self.locations.get(&entity.id()).copied() else {
            return;
        };

        // Already has this type: overwrite in place. No migration, which is what
        // makes repeatedly setting a component (every frame, from a script) cheap.
        if self.archetypes[location.archetype].has(type_id) {
            if let Some(column) = self.archetypes[location.archetype]
                .columns
                .get_mut(&type_id)
                .and_then(|c| c.as_any_mut().downcast_mut::<TypedColumn<T>>())
            {
                column.values[location.row] = component;
            }
            return;
        }

        // New type: the entity moves to the archetype with this type added.
        let mut types = self.archetypes[location.archetype].types.clone();
        types.push(type_id);
        types.sort();
        let target = self.archetype_for(&types);
        let row = self.migrate(entity, location, target);
        // Then append the new component, keeping the column lengths equal.
        let column = self.archetypes[target]
            .columns
            .entry(type_id)
            .or_insert_with(|| Box::new(TypedColumn::<T> { values: Vec::new() }));
        if let Some(typed) = column.as_any_mut().downcast_mut::<TypedColumn<T>>() {
            // Rows before this entity's may not have the column yet if the
            // archetype was just created; the migration filled them, so a push
            // lands at exactly `row`.
            debug_assert_eq!(typed.values.len(), row);
            typed.values.push(component);
        }
    }

    /// Remove a component, returning whether the entity had it.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> bool {
        let type_id = TypeId::of::<T>();
        let Some(location) = self.locations.get(&entity.id()).copied() else {
            return false;
        };
        if !self.archetypes[location.archetype].has(type_id) {
            return false;
        }
        let mut types = self.archetypes[location.archetype].types.clone();
        types.retain(|t| *t != type_id);
        let target = self.archetype_for(&types);
        self.migrate(entity, location, target);
        true
    }

    /// Read a component.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let location = self.locations.get(&entity.id())?;
        let column = self.archetypes[location.archetype]
            .columns
            .get(&TypeId::of::<T>())?;
        column
            .as_any()
            .downcast_ref::<TypedColumn<T>>()?
            .values
            .get(location.row)
    }

    /// Read a component mutably.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let location = *self.locations.get(&entity.id())?;
        let column = self.archetypes[location.archetype]
            .columns
            .get_mut(&TypeId::of::<T>())?;
        column
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()?
            .values
            .get_mut(location.row)
    }

    /// Every entity with component `T`.
    ///
    /// This is the operation the rewrite is for: it visits only the archetypes that
    /// contain `T`, and walks their rows directly. The old storage filtered the
    /// whole entity list through a hash lookup per entity.
    ///
    /// The result is ordered by creation, not by row, because callers depend on it:
    /// `app/state.rs` builds its draw list from this and a shuffled order would
    /// reshuffle object indices every time a component was added anywhere.
    pub fn entities_with<T: Component>(&self) -> Vec<Entity> {
        let type_id = TypeId::of::<T>();
        let mut matching: Vec<Entity> = Vec::new();
        for archetype in &self.archetypes {
            if archetype.has(type_id) {
                matching.extend_from_slice(&archetype.entities);
            }
        }
        if matching.len() > 1 {
            // Creation order. Sorting by id is equivalent and cheaper than
            // intersecting with `self.entities`, since ids are handed out in
            // increasing order.
            matching.sort_by_key(|e| e.id());
        }
        matching
    }

    /// Every entity with all of `T` and `U`.
    ///
    /// A two-type query rather than a generic variadic one: two covers what the
    /// engine actually asks (mesh + transform, body + collider), and a variadic
    /// version in stable Rust means either a macro or a tuple trait, both of which
    /// are more machinery than the callers justify.
    pub fn entities_with_both<T: Component, U: Component>(&self) -> Vec<Entity> {
        let (a, b) = (TypeId::of::<T>(), TypeId::of::<U>());
        let mut matching: Vec<Entity> = Vec::new();
        for archetype in &self.archetypes {
            if archetype.has(a) && archetype.has(b) {
                matching.extend_from_slice(&archetype.entities);
            }
        }
        if matching.len() > 1 {
            matching.sort_by_key(|e| e.id());
        }
        matching
    }

    /// Packed slices of `T`, one per archetype containing it.
    ///
    /// The cache-friendly read path: each slice is contiguous, so a system that only
    /// needs the component values (not the entity ids) walks memory linearly. This
    /// is what an archetype layout buys, and it is not expressible over the old
    /// storage at all.
    pub fn columns<T: Component>(&self) -> Vec<&[T]> {
        let type_id = TypeId::of::<T>();
        self.archetypes
            .iter()
            .filter_map(|archetype| {
                archetype
                    .columns
                    .get(&type_id)?
                    .as_any()
                    .downcast_ref::<TypedColumn<T>>()
                    .map(|c| c.values.as_slice())
            })
            .filter(|slice| !slice.is_empty())
            .collect()
    }

    /// Find or create the archetype for a sorted type set.
    fn archetype_for(&mut self, types: &[TypeId]) -> usize {
        let key = types.to_vec();
        if let Some(index) = self.by_types.get(&key) {
            return *index;
        }
        let index = self.archetypes.len();
        self.archetypes.push(Archetype::new(key.clone()));
        self.by_types.insert(key, index);
        index
    }

    /// Move an entity from `from` to archetype `target`, carrying every component
    /// both archetypes share. Returns the entity's new row.
    fn migrate(&mut self, entity: Entity, from: Location, target: usize) -> usize {
        if from.archetype == target {
            return from.row;
        }
        self.migrations += 1;

        // Pull the entity's components out of the source archetype.
        let source_types = self.archetypes[from.archetype].types.clone();
        let mut carried: Vec<(TypeId, Box<dyn Any + Send>)> = Vec::new();
        for type_id in &source_types {
            if let Some(column) = self.archetypes[from.archetype].columns.get_mut(type_id) {
                carried.push((*type_id, column.swap_remove_boxed(from.row)));
            }
        }
        // Fix the entity whose row the swap-remove moved. Doing this before the
        // insert keeps the two archetypes' bookkeeping independent.
        self.finish_swap_remove(from);

        // Put them into the target, whose columns may not exist yet.
        let row = self.archetypes[target].entities.len();
        self.archetypes[target].entities.push(entity);
        let target_types = self.archetypes[target].types.clone();
        for (type_id, value) in carried {
            if !target_types.contains(&type_id) {
                // Dropped on purpose: this is the `remove::<T>` path, where the
                // target archetype deliberately lacks the type.
                continue;
            }
            let column = match self.archetypes[target].columns.get_mut(&type_id) {
                Some(column) => column,
                None => {
                    // The target archetype is new and has no column of this type
                    // yet. Build one from any existing archetype that has it, since
                    // only a column can produce an empty column of its own type
                    // without knowing the type statically.
                    let Some(template) = self
                        .archetypes
                        .iter()
                        .find_map(|a| a.columns.get(&type_id))
                        .map(|c| c.empty_clone())
                    else {
                        continue;
                    };
                    self.archetypes[target]
                        .columns
                        .entry(type_id)
                        .or_insert(template)
                }
            };
            // A failed push means the boxed value did not match the column's type,
            // which is only reachable through a bug here. Assert in debug rather
            // than silently losing a component.
            let pushed = column.push_boxed(value);
            debug_assert!(pushed, "migrating a component into a mismatched column");
        }
        self.locations.insert(
            entity.id(),
            Location {
                archetype: target,
                row,
            },
        );
        row
    }

    /// Drop an entity's row entirely.
    fn remove_row(&mut self, location: Location) {
        let types = self.archetypes[location.archetype].types.clone();
        for type_id in &types {
            if let Some(column) = self.archetypes[location.archetype].columns.get_mut(type_id) {
                if location.row < column.len() {
                    let _ = column.swap_remove_boxed(location.row);
                }
            }
        }
        self.finish_swap_remove(location);
    }

    /// After a swap-remove at `location.row`, the archetype's last entity now
    /// occupies that row. Update its recorded location.
    ///
    /// Forgetting this is the defining bug of a swap-remove layout: one entity's
    /// components silently become another's, and it only shows up when the *last*
    /// entity is the one read next.
    fn finish_swap_remove(&mut self, location: Location) {
        let archetype = &mut self.archetypes[location.archetype];
        if location.row >= archetype.entities.len() {
            return;
        }
        archetype.entities.swap_remove(location.row);
        if let Some(moved) = archetype.entities.get(location.row).copied() {
            self.locations.insert(
                moved.id(),
                Location {
                    archetype: location.archetype,
                    row: location.row,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Position(f32, f32);
    impl Component for Position {}

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Velocity(f32, f32);
    impl Component for Velocity {}

    #[derive(Clone, Debug, PartialEq)]
    struct Label(String);
    impl Component for Label {}

    fn with_entities(count: u64) -> (Archetypes, Vec<Entity>) {
        let mut storage = Archetypes::new();
        let entities: Vec<Entity> = (1..=count).map(Entity::new).collect();
        for e in &entities {
            storage.insert_entity(*e);
        }
        (storage, entities)
    }

    // --- basics ---

    #[test]
    fn a_component_can_be_set_and_read() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(1.0, 2.0));
        assert_eq!(storage.get::<Position>(e[0]), Some(&Position(1.0, 2.0)));
    }

    #[test]
    fn a_missing_component_reads_as_none() {
        let (storage, e) = with_entities(1);
        assert_eq!(storage.get::<Position>(e[0]), None);
    }

    #[test]
    fn a_component_can_be_mutated_in_place() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(1.0, 2.0));
        if let Some(p) = storage.get_mut::<Position>(e[0]) {
            p.0 = 9.0;
        }
        assert_eq!(storage.get::<Position>(e[0]), Some(&Position(9.0, 2.0)));
    }

    /// Setting an existing component must not migrate. This is the common case — a
    /// script or a system writing every frame — and migrating each time would make
    /// the archetype layout slower than the map it replaced.
    #[test]
    fn overwriting_a_component_does_not_migrate() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(1.0, 1.0));
        let after_first = storage.migrations();
        for i in 0..100 {
            storage.set(e[0], Position(i as f32, 0.0));
        }
        assert_eq!(
            storage.migrations(),
            after_first,
            "overwriting should be free of structural change"
        );
        assert_eq!(storage.get::<Position>(e[0]), Some(&Position(99.0, 0.0)));
    }

    #[test]
    fn adding_a_second_component_keeps_the_first() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(1.0, 2.0));
        storage.set(e[0], Velocity(3.0, 4.0));
        assert_eq!(storage.get::<Position>(e[0]), Some(&Position(1.0, 2.0)));
        assert_eq!(storage.get::<Velocity>(e[0]), Some(&Velocity(3.0, 4.0)));
    }

    /// The order components are added in must not matter: `{Position, Velocity}` and
    /// `{Velocity, Position}` are the same type set and must land in one archetype,
    /// or a scene would fragment into an archetype per insertion order.
    #[test]
    fn component_insertion_order_does_not_create_extra_archetypes() {
        let (mut storage, e) = with_entities(2);
        storage.set(e[0], Position(0.0, 0.0));
        storage.set(e[0], Velocity(0.0, 0.0));
        storage.set(e[1], Velocity(0.0, 0.0));
        storage.set(e[1], Position(0.0, 0.0));
        // Empty, {Position}, {Position,Velocity}, {Velocity} — four, and crucially
        // both entities end in the same one.
        let both = storage.entities_with_both::<Position, Velocity>();
        assert_eq!(both.len(), 2);
        assert_eq!(
            storage.columns::<Position>().len(),
            1,
            "both entities' positions should be in one packed column, got {:?}",
            storage.columns::<Position>().iter().map(|s| s.len()).collect::<Vec<_>>()
        );
    }

    // --- removal ---

    #[test]
    fn removing_a_component_leaves_the_others() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(1.0, 2.0));
        storage.set(e[0], Velocity(3.0, 4.0));
        assert!(storage.remove::<Velocity>(e[0]));
        assert_eq!(storage.get::<Velocity>(e[0]), None);
        assert_eq!(
            storage.get::<Position>(e[0]),
            Some(&Position(1.0, 2.0)),
            "removing one component must not disturb another"
        );
    }

    #[test]
    fn removing_an_absent_component_reports_false() {
        let (mut storage, e) = with_entities(1);
        assert!(!storage.remove::<Position>(e[0]));
    }

    #[test]
    fn destroying_an_entity_removes_it_and_its_components() {
        let (mut storage, e) = with_entities(2);
        storage.set(e[0], Position(1.0, 1.0));
        storage.set(e[1], Position(2.0, 2.0));
        storage.remove_entity(e[0]);
        assert_eq!(storage.get::<Position>(e[0]), None);
        assert_eq!(storage.all_entities().len(), 1);
        assert_eq!(
            storage.get::<Position>(e[1]),
            Some(&Position(2.0, 2.0)),
            "the surviving entity must keep its own component"
        );
    }

    /// The defining bug of a swap-remove layout: removing a row moves the *last*
    /// entity into it, and forgetting to update that entity's recorded row makes
    /// one entity's components silently become another's. It only shows up when the
    /// moved entity is the one read next, which is why this is tested directly.
    #[test]
    fn swap_remove_updates_the_moved_entitys_location() {
        let (mut storage, e) = with_entities(4);
        for (i, entity) in e.iter().enumerate() {
            storage.set(*entity, Position(i as f32, 0.0));
        }
        // Remove the first: the last (index 3) swaps into row 0.
        storage.remove_entity(e[0]);
        for (i, entity) in e.iter().enumerate().skip(1) {
            assert_eq!(
                storage.get::<Position>(*entity),
                Some(&Position(i as f32, 0.0)),
                "entity {i} lost track of its own component after a swap-remove"
            );
        }
    }

    /// The same hazard on the component-removal path, which migrates rather than
    /// dropping the row.
    #[test]
    fn migration_updates_the_moved_entitys_location() {
        let (mut storage, e) = with_entities(4);
        for (i, entity) in e.iter().enumerate() {
            storage.set(*entity, Position(i as f32, 0.0));
            storage.set(*entity, Velocity(i as f32, 0.0));
        }
        // Removing Velocity from the first entity migrates it out, swapping the
        // last entity into its row in the {Position, Velocity} archetype.
        storage.remove::<Velocity>(e[0]);
        for (i, entity) in e.iter().enumerate() {
            assert_eq!(
                storage.get::<Position>(*entity),
                Some(&Position(i as f32, 0.0)),
                "entity {i} has the wrong Position after a migration"
            );
            if i > 0 {
                assert_eq!(
                    storage.get::<Velocity>(*entity),
                    Some(&Velocity(i as f32, 0.0)),
                    "entity {i} has the wrong Velocity after a migration"
                );
            }
        }
        assert_eq!(storage.get::<Velocity>(e[0]), None);
    }

    /// Repeated add/remove cycles are where a bookkeeping error compounds into
    /// visible corruption.
    #[test]
    fn many_add_remove_cycles_keep_every_entity_consistent() {
        let (mut storage, e) = with_entities(10);
        for (i, entity) in e.iter().enumerate() {
            storage.set(*entity, Position(i as f32, 0.0));
            storage.set(*entity, Label(format!("e{i}")));
        }
        for round in 0..20 {
            for (i, entity) in e.iter().enumerate() {
                if (i + round) % 3 == 0 {
                    storage.set(*entity, Velocity(i as f32, round as f32));
                } else {
                    storage.remove::<Velocity>(*entity);
                }
            }
            // Every entity must still own its own Position and Label.
            for (i, entity) in e.iter().enumerate() {
                assert_eq!(
                    storage.get::<Position>(*entity),
                    Some(&Position(i as f32, 0.0)),
                    "round {round}: entity {i} lost its Position"
                );
                assert_eq!(
                    storage.get::<Label>(*entity),
                    Some(&Label(format!("e{i}"))),
                    "round {round}: entity {i} lost its Label"
                );
            }
        }
    }

    // --- queries ---

    #[test]
    fn entities_with_finds_only_matching_entities() {
        let (mut storage, e) = with_entities(3);
        storage.set(e[0], Position(0.0, 0.0));
        storage.set(e[2], Position(0.0, 0.0));
        let found = storage.entities_with::<Position>();
        assert_eq!(found.len(), 2);
        assert!(found.contains(&e[0]) && found.contains(&e[2]));
        assert!(!found.contains(&e[1]));
    }

    /// The result must be in creation order. `app/state.rs` builds its draw list
    /// from this, and a shuffled order would reshuffle object indices — and with
    /// them the material and transform lookups — every time a component was added
    /// anywhere in the scene.
    #[test]
    fn entities_with_returns_creation_order_across_archetypes() {
        let (mut storage, e) = with_entities(6);
        // Deliberately give alternating entities a second component, so they end up
        // in different archetypes and the naive concatenation would interleave wrong.
        for (i, entity) in e.iter().enumerate() {
            storage.set(*entity, Position(i as f32, 0.0));
            if i % 2 == 0 {
                storage.set(*entity, Velocity(0.0, 0.0));
            }
        }
        let found = storage.entities_with::<Position>();
        let ids: Vec<u64> = found.iter().map(|x| x.id()).collect();
        assert_eq!(
            ids,
            vec![1, 2, 3, 4, 5, 6],
            "entities_with must be ordered by id regardless of archetype"
        );
    }

    /// Order must survive a removal too, which reorders rows within an archetype.
    #[test]
    fn entities_with_order_survives_a_swap_remove() {
        let (mut storage, e) = with_entities(5);
        for entity in &e {
            storage.set(*entity, Position(0.0, 0.0));
        }
        storage.remove_entity(e[1]);
        let ids: Vec<u64> = storage
            .entities_with::<Position>()
            .iter()
            .map(|x| x.id())
            .collect();
        assert_eq!(ids, vec![1, 3, 4, 5]);
    }

    #[test]
    fn a_two_type_query_requires_both() {
        let (mut storage, e) = with_entities(3);
        storage.set(e[0], Position(0.0, 0.0));
        storage.set(e[0], Velocity(0.0, 0.0));
        storage.set(e[1], Position(0.0, 0.0));
        storage.set(e[2], Velocity(0.0, 0.0));
        let both = storage.entities_with_both::<Position, Velocity>();
        assert_eq!(both, vec![e[0]]);
    }

    #[test]
    fn a_query_for_an_unused_type_is_empty() {
        let (mut storage, e) = with_entities(2);
        storage.set(e[0], Position(0.0, 0.0));
        assert!(storage.entities_with::<Label>().is_empty());
        assert!(storage.columns::<Label>().is_empty());
    }

    // --- packing ---

    /// The point of the rewrite: components of one type in one archetype are
    /// contiguous, so a system can walk them linearly.
    #[test]
    fn components_of_one_archetype_are_packed_contiguously() {
        let (mut storage, e) = with_entities(100);
        for (i, entity) in e.iter().enumerate() {
            storage.set(*entity, Position(i as f32, 0.0));
        }
        let columns = storage.columns::<Position>();
        assert_eq!(
            columns.len(),
            1,
            "100 identical entities should form exactly one archetype"
        );
        assert_eq!(columns[0].len(), 100, "and one column of 100 values");
        // The values are all there, whatever order the rows ended up in.
        let mut seen: Vec<f32> = columns[0].iter().map(|p| p.0).collect();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(seen, (0..100).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn columns_skips_empty_archetypes() {
        let (mut storage, e) = with_entities(2);
        storage.set(e[0], Position(1.0, 0.0));
        storage.set(e[1], Position(2.0, 0.0));
        // Give one a second component: it migrates out, leaving the {Position}
        // archetype with one row rather than an empty column in the result.
        storage.set(e[1], Velocity(0.0, 0.0));
        let columns = storage.columns::<Position>();
        assert_eq!(columns.len(), 2, "two archetypes now contain Position");
        assert!(
            columns.iter().all(|c| !c.is_empty()),
            "an empty column must not be reported"
        );
        assert_eq!(columns.iter().map(|c| c.len()).sum::<usize>(), 2);
    }

    /// A uniform scene must not fragment. Fragmentation is what turns an archetype
    /// layout back into a scattered one, so the count is worth asserting.
    #[test]
    fn a_uniform_scene_has_few_archetypes() {
        let (mut storage, e) = with_entities(50);
        for entity in &e {
            storage.set(*entity, Position(0.0, 0.0));
            storage.set(*entity, Velocity(0.0, 0.0));
        }
        // Empty, {Position}, {Position,Velocity} — the two intermediates are passed
        // through on the way, not fragmented per entity.
        assert!(
            storage.archetype_count() <= 3,
            "50 identical entities produced {} archetypes",
            storage.archetype_count()
        );
    }

    /// Building a scene costs one migration per component added, and that is the
    /// documented trade-off. Worth pinning so a change that migrates on *overwrite*
    /// shows up as a number rather than as a vague slowdown.
    #[test]
    fn migration_count_matches_the_number_of_structural_changes() {
        let (mut storage, e) = with_entities(10);
        for entity in &e {
            storage.set(*entity, Position(0.0, 0.0));
            storage.set(*entity, Velocity(0.0, 0.0));
        }
        assert_eq!(
            storage.migrations(),
            20,
            "two component additions per entity, ten entities"
        );
    }

    // --- edge cases ---

    #[test]
    fn operating_on_an_unknown_entity_is_a_no_op() {
        let mut storage = Archetypes::new();
        let ghost = Entity::new(999);
        storage.set(ghost, Position(1.0, 1.0));
        assert_eq!(storage.get::<Position>(ghost), None);
        assert!(!storage.remove::<Position>(ghost));
        storage.remove_entity(ghost);
        assert!(storage.all_entities().is_empty());
    }

    #[test]
    fn an_entity_with_no_components_still_exists() {
        let (storage, e) = with_entities(1);
        assert_eq!(storage.all_entities(), &[e[0]]);
        assert!(storage.entities_with::<Position>().is_empty());
    }

    #[test]
    fn destroying_the_same_entity_twice_is_harmless() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Position(0.0, 0.0));
        storage.remove_entity(e[0]);
        storage.remove_entity(e[0]);
        assert!(storage.all_entities().is_empty());
    }

    #[test]
    fn a_non_copy_component_survives_migration() {
        let (mut storage, e) = with_entities(1);
        storage.set(e[0], Label("hello".to_string()));
        storage.set(e[0], Position(0.0, 0.0));
        storage.set(e[0], Velocity(0.0, 0.0));
        storage.remove::<Position>(e[0]);
        assert_eq!(
            storage.get::<Label>(e[0]),
            Some(&Label("hello".to_string())),
            "an owned component must be moved, not dropped, across archetypes"
        );
    }
}
