// Engine bindings: the surface a script can act on, plus hot-reload.
//
// The design constraint is the one from `stdlib`: a native function is
// `Fn(&[Value])` with no path back to the engine. Handing a closure a reference to
// `AppState` is not possible (and would be a borrow cycle if it were), so the
// binding layer is a *mailbox*. Scripts write commands into a shared queue; the
// engine drains it once per frame and applies them.
//
// That indirection buys more than it costs:
//
// - A script cannot corrupt engine state mid-frame. Every change lands at one
//   known point in the frame, so "the scene changed under the renderer" is not a
//   failure mode that exists.
// - The queue is inspectable, so what a script did is testable without a GPU: the
//   tests below run scripts and assert on the commands produced.
// - A script error leaves a partial queue rather than partially-mutated state.
//
// Hot-reload watches file modification times. Not a filesystem notification API:
// that is per-platform (inotify, ReadDirectoryChangesW, FSEvents), and polling one
// `metadata()` call per file per frame costs microseconds against a frame budget
// of milliseconds.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

use crate::engine::core::Result;
use crate::engine::math::Vec3;
use crate::scripting::interp::Interpreter;
use crate::scripting::value::Value;

/// Something a script asked the engine to do.
///
/// An enum rather than a closure so the queue can be compared, counted and
/// asserted on in a test — which is the whole reason the mailbox exists.
#[derive(Clone, Debug, PartialEq)]
pub enum ScriptCommand {
    /// Move an entity to a world position.
    SetPosition { entity: u64, position: Vec3 },
    /// Offset an entity from where it is.
    Translate { entity: u64, delta: Vec3 },
    /// Set an entity's uniform scale.
    SetScale { entity: u64, scale: f32 },
    /// Print into the engine console.
    Log(String),
    /// Play one of the demo sounds at a world position.
    PlaySound { index: usize, position: Vec3 },
    /// Turn the rigid-body simulation on or off.
    SetPhysics(bool),
    /// Turn traced lighting on or off.
    SetTracing(bool),
}

/// The shared mailbox. `Rc<RefCell<...>>` because both the native closures and the
/// engine hold an end of it, and neither owns the other.
type Mailbox = Rc<RefCell<Vec<ScriptCommand>>>;

/// Largest number of commands one frame's scripts may queue.
///
/// A script in a loop can call `move_entity` a million times, and without a cap
/// the queue would grow until the process died. Dropping the excess and reporting
/// it is the recoverable failure.
const MAX_COMMANDS: usize = 4096;

/// Values the engine publishes to scripts each frame, as the `engine` table.
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineState {
    pub time: f32,
    pub dt: f32,
    pub frame: u32,
    pub fps: f32,
    pub camera_position: Vec3,
    pub entity_count: u32,
}

/// A script file being watched for changes.
struct WatchedFile {
    path: PathBuf,
    /// Last modification time seen. `None` until the file has been read once.
    modified: Option<SystemTime>,
    /// Length and checksum of the last loaded contents. This, not the timestamp,
    /// is what decides whether a reload is needed — see `poll_reloads`.
    fingerprint: Option<(u64, u64)>,
    /// Whether the last load succeeded, so a broken file is reported once rather
    /// than every frame — a per-frame error would bury the console it prints to.
    healthy: bool,
}

/// FNV-1a over the file's bytes. Self-implemented, like the rest; a cryptographic
/// hash would be pointless here since the only adversary is a coarse clock.
fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// The scripting host: an interpreter, the bindings, and the reload watcher.
pub struct ScriptHost {
    interpreter: Interpreter,
    commands: Mailbox,
    watched: Vec<WatchedFile>,
    /// Commands dropped because the queue was full, since startup.
    dropped: u64,
    /// Reloads performed, so the HUD can show that hot-reload is doing something.
    reloads: u64,
    /// The last error a script produced, kept so it can be shown rather than only
    /// logged once and lost.
    last_error: Option<String>,
}

impl Default for ScriptHost {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptHost {
    /// A host with the engine bindings installed.
    pub fn new() -> Self {
        let mut interpreter = Interpreter::new();
        let commands: Mailbox = Rc::new(RefCell::new(Vec::new()));
        install_bindings(&mut interpreter, commands.clone());
        Self {
            interpreter,
            commands,
            watched: Vec::new(),
            dropped: 0,
            reloads: 0,
            last_error: None,
        }
    }

    pub fn interpreter(&mut self) -> &mut Interpreter {
        &mut self.interpreter
    }

    pub fn reloads(&self) -> u64 {
        self.reloads
    }

    pub fn dropped_commands(&self) -> u64 {
        self.dropped
    }

    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Lines scripts printed.
    pub fn output(&self) -> &[String] {
        self.interpreter.output()
    }

    /// Run a chunk of source directly, as the console does.
    pub fn eval(&mut self, source: &str) -> Result<Vec<Value>> {
        let result = self.interpreter.run(source);
        match &result {
            Ok(_) => self.last_error = None,
            Err(e) => self.last_error = Some(e.to_string()),
        }
        result
    }

    /// Watch a file, loading it immediately.
    ///
    /// A missing file is not an error: the demo ships without scripts, and the
    /// point of watching is that dropping one in later starts working. It is
    /// recorded as watched so it will be picked up when it appears.
    pub fn watch(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        self.watched.push(WatchedFile {
            path,
            modified: None,
            fingerprint: None,
            healthy: true,
        });
        // Load whatever is already there.
        let index = self.watched.len() - 1;
        self.reload_index(index);
    }

    /// Reload any watched file whose modification time changed.
    ///
    /// Returns how many were reloaded, so a caller can log it rather than guess.
    pub fn poll_reloads(&mut self) -> usize {
        let mut reloaded = 0;
        for index in 0..self.watched.len() {
            // A deleted file is skipped by the read below: nothing to reload, and
            // the already-loaded definitions stay live — the useful behaviour while
            // a file is being moved or rewritten.

            // The file's contents are the only thing consulted. Timestamps are
            // read and stored for reporting, but are deliberately NOT used to skip
            // the read.
            //
            // Using them as a cheap pre-check is the obvious optimisation and it is
            // wrong. Measured on NTFS here: three writes in a row produced
            // timestamps 134298745089831770, 134298745089841769, 134298745089841769
            // — the last two identical, because the filesystem's effective update
            // granularity is coarser than its representation. All three writes were
            // the same length, so a size check did not save it either, and the
            // third edit was silently ignored. On FAT or a network mount the
            // granularity is 1-2 seconds and the problem is far worse.
            //
            // The cost of getting it right is one read of a few-kilobyte file per
            // watched script per frame: microseconds against a frame budget of
            // milliseconds. Cheap enough that trading correctness for it would be a
            // bad deal even if the trade worked.
            let Ok(bytes) = std::fs::read(&self.watched[index].path) else {
                continue;
            };
            let fingerprint = (bytes.len() as u64, checksum(&bytes));
            // An unchanged checksum means no reload even if the timestamp moved,
            // which is what an editor's "save all" does to every open file —
            // reloading each of them would reset script state for no reason.
            let changed = self.watched[index].fingerprint != Some(fingerprint);

            if changed && self.reload_index(index) {
                reloaded += 1;
            }
        }
        self.reloads += reloaded as u64;
        reloaded
    }

    /// Load one watched file. Returns whether it was read at all.
    fn reload_index(&mut self, index: usize) -> bool {
        let path = self.watched[index].path.clone();
        let Ok(meta) = std::fs::metadata(&path) else {
            return false;
        };
        let modified = meta.modified().ok();
        let Ok(source) = std::fs::read_to_string(&path) else {
            return false;
        };
        self.watched[index].modified = modified;
        self.watched[index].fingerprint =
            Some((source.len() as u64, checksum(source.as_bytes())));

        match self.interpreter.run(&source) {
            Ok(_) => {
                if !self.watched[index].healthy {
                    // Announce recovery: a file that was broken and now is not is
                    // exactly what the author is waiting to hear.
                    self.interpreter
                        .print_line(format!("script: {} loaded", path.display()));
                }
                self.watched[index].healthy = true;
                self.last_error = None;
            }
            Err(e) => {
                let message = format!("script {}: {e}", path.display());
                // Reported once per transition into failure, not once per frame:
                // a per-frame error would bury the console it prints to.
                if self.watched[index].healthy {
                    self.interpreter.print_line(message.clone());
                }
                self.watched[index].healthy = false;
                self.last_error = Some(message);
            }
        }
        true
    }

    /// Publish this frame's engine state as the `engine` global, then call the
    /// script's `update(dt)` if it defined one.
    pub fn update(&mut self, state: &EngineState) -> Result<()> {
        let table = Value::table();
        if let Value::Table(t) = &table {
            let mut borrowed = t.borrow_mut();
            let mut set = |name: &str, value: Value| {
                borrowed.set(
                    crate::scripting::value::Key::Str(Rc::from(name)),
                    value,
                );
            };
            set("time", Value::Number(state.time as f64));
            set("dt", Value::Number(state.dt as f64));
            set("frame", Value::Number(state.frame as f64));
            set("fps", Value::Number(state.fps as f64));
            set("entity_count", Value::Number(state.entity_count as f64));
            set("camera_x", Value::Number(state.camera_position.x as f64));
            set("camera_y", Value::Number(state.camera_position.y as f64));
            set("camera_z", Value::Number(state.camera_position.z as f64));
        }
        self.interpreter.set_global("engine", table);

        if !self.interpreter.has_function("update") {
            return Ok(());
        }
        let result = self
            .interpreter
            .call_global("update", &[Value::Number(state.dt as f64)]);
        match result {
            Ok(_) => {
                self.last_error = None;
                Ok(())
            }
            Err(e) => {
                let message = format!("script update: {e}");
                // Only on transition, for the same reason as reload errors: a
                // script that errors every frame would flood the console.
                if self.last_error.as_deref() != Some(message.as_str()) {
                    self.interpreter.print_line(message.clone());
                }
                self.last_error = Some(message);
                Ok(())
            }
        }
    }

    /// Take everything scripts queued, leaving the mailbox empty.
    ///
    /// Draining rather than reading means a command is applied exactly once even
    /// if the caller forgets to clear, which is the failure that would otherwise
    /// make an entity accelerate away.
    pub fn drain_commands(&mut self) -> Vec<ScriptCommand> {
        let mut queue = self.commands.borrow_mut();
        if queue.len() > MAX_COMMANDS {
            self.dropped += (queue.len() - MAX_COMMANDS) as u64;
            queue.truncate(MAX_COMMANDS);
        }
        queue.drain(..).collect()
    }

    /// Commands waiting to be drained.
    pub fn pending_commands(&self) -> usize {
        self.commands.borrow().len()
    }
}

/// Install the engine-facing functions.
fn install_bindings(interpreter: &mut Interpreter, commands: Mailbox) {
    /// Read three consecutive arguments as a vector, defaulting missing ones to 0.
    fn vec3_from(args: &[Value], start: usize) -> Vec3 {
        let component = |i: usize| {
            args.get(start + i)
                .and_then(|v| v.as_number())
                .unwrap_or(0.0) as f32
        };
        Vec3::new(component(0), component(1), component(2))
    }

    /// Read an entity handle. Rejects a non-integer or negative id rather than
    /// truncating it into a different entity's handle.
    fn entity_from(args: &[Value], index: usize, function: &str) -> std::result::Result<u64, String> {
        let raw = args
            .get(index)
            .and_then(|v| v.as_number())
            .ok_or_else(|| format!("{function}: entity id must be a number"))?;
        if raw < 1.0 || raw != raw.trunc() || !raw.is_finite() {
            return Err(format!(
                "{function}: {raw} is not a valid entity id"
            ));
        }
        Ok(raw as u64)
    }

    /// Push a command, or report that the queue is full.
    fn push(commands: &Mailbox, command: ScriptCommand) -> std::result::Result<Vec<Value>, String> {
        let mut queue = commands.borrow_mut();
        if queue.len() >= MAX_COMMANDS {
            return Err(format!(
                "script command queue is full ({MAX_COMMANDS}); is a loop calling this?"
            ));
        }
        queue.push(command);
        Ok(vec![])
    }

    let queue = commands.clone();
    interpreter.set_native("set_position", move |args| {
        let entity = entity_from(args, 0, "set_position")?;
        push(
            &queue,
            ScriptCommand::SetPosition {
                entity,
                position: vec3_from(args, 1),
            },
        )
    });

    let queue = commands.clone();
    interpreter.set_native("translate", move |args| {
        let entity = entity_from(args, 0, "translate")?;
        push(
            &queue,
            ScriptCommand::Translate {
                entity,
                delta: vec3_from(args, 1),
            },
        )
    });

    let queue = commands.clone();
    interpreter.set_native("set_scale", move |args| {
        let entity = entity_from(args, 0, "set_scale")?;
        let scale = args
            .get(1)
            .and_then(|v| v.as_number())
            .ok_or_else(|| "set_scale: scale must be a number".to_string())?;
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!("set_scale: {scale} is not a usable scale"));
        }
        push(
            &queue,
            ScriptCommand::SetScale {
                entity,
                scale: scale as f32,
            },
        )
    });

    let queue = commands.clone();
    interpreter.set_native("log", move |args| {
        let text = args
            .iter()
            .map(crate::scripting::interp::tostring)
            .collect::<Vec<_>>()
            .join(" ");
        push(&queue, ScriptCommand::Log(text))
    });

    let queue = commands.clone();
    interpreter.set_native("play_sound", move |args| {
        let index = args.first().and_then(|v| v.as_number()).unwrap_or(0.0);
        if !(index.is_finite() && index >= 0.0) {
            return Err(format!("play_sound: {index} is not a sound index"));
        }
        push(
            &queue,
            ScriptCommand::PlaySound {
                index: index as usize,
                position: vec3_from(args, 1),
            },
        )
    });

    let queue = commands.clone();
    interpreter.set_native("set_physics", move |args| {
        let on = args.first().map(|v| v.truthy()).unwrap_or(false);
        push(&queue, ScriptCommand::SetPhysics(on))
    });

    let queue = commands.clone();
    interpreter.set_native("set_tracing", move |args| {
        let on = args.first().map(|v| v.truthy()).unwrap_or(false);
        push(&queue, ScriptCommand::SetTracing(on))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_running(source: &str) -> ScriptHost {
        let mut host = ScriptHost::new();
        host.eval(source)
            .unwrap_or_else(|e| panic!("failed to run:\n{source}\nerror: {e}"));
        host
    }

    // --- the mailbox ---

    /// The reason the mailbox exists: what a script did is inspectable from Rust,
    /// so scripting is testable without a GPU.
    #[test]
    fn a_script_queues_commands_the_engine_can_read() {
        let mut host = host_running("set_position(1, 10, 20, 30)");
        let commands = host.drain_commands();
        assert_eq!(
            commands,
            vec![ScriptCommand::SetPosition {
                entity: 1,
                position: Vec3::new(10.0, 20.0, 30.0)
            }]
        );
    }

    #[test]
    fn every_binding_produces_its_command() {
        let mut host = host_running(
            "set_position(1, 1, 2, 3)\n\
             translate(2, 0, 1, 0)\n\
             set_scale(3, 2.5)\n\
             log(\"hello\", 42)\n\
             play_sound(1, 5, 0, 5)\n\
             set_physics(true)\n\
             set_tracing(false)",
        );
        let commands = host.drain_commands();
        assert_eq!(commands.len(), 7);
        assert!(matches!(commands[0], ScriptCommand::SetPosition { entity: 1, .. }));
        assert!(matches!(commands[1], ScriptCommand::Translate { entity: 2, .. }));
        assert!(matches!(
            commands[2],
            ScriptCommand::SetScale { entity: 3, scale } if (scale - 2.5).abs() < 1e-6
        ));
        assert_eq!(commands[3], ScriptCommand::Log("hello 42".to_string()));
        assert!(matches!(commands[4], ScriptCommand::PlaySound { index: 1, .. }));
        assert_eq!(commands[5], ScriptCommand::SetPhysics(true));
        assert_eq!(commands[6], ScriptCommand::SetTracing(false));
    }

    /// Draining rather than reading means a command is applied exactly once even
    /// if the caller forgets to clear — the failure that would make an entity
    /// accelerate away.
    #[test]
    fn draining_empties_the_queue() {
        let mut host = host_running("translate(1, 1, 0, 0)");
        assert_eq!(host.pending_commands(), 1);
        assert_eq!(host.drain_commands().len(), 1);
        assert_eq!(host.pending_commands(), 0);
        assert!(host.drain_commands().is_empty());
    }

    /// A script in a loop must not be able to grow the queue until the process
    /// dies.
    #[test]
    fn the_command_queue_is_capped() {
        let mut host = ScriptHost::new();
        // Deliberately more than the cap; the error stops the script.
        let result = host.eval("for i = 1, 100000 do translate(1, 1, 0, 0) end");
        assert!(result.is_err(), "the queue should refuse once full");
        assert!(
            host.pending_commands() <= MAX_COMMANDS,
            "queue grew to {}",
            host.pending_commands()
        );
    }

    // --- argument validation ---

    /// An entity id that is not a positive integer must be rejected rather than
    /// truncated into some other entity's handle.
    #[test]
    fn an_invalid_entity_id_is_rejected() {
        let mut host = ScriptHost::new();
        for bad in ["set_position(0, 1, 2, 3)", "set_position(-1, 1, 2, 3)",
                    "set_position(1.5, 1, 2, 3)", "set_position(\"x\", 1, 2, 3)"] {
            assert!(host.eval(bad).is_err(), "{bad} should have been rejected");
        }
        assert_eq!(
            host.pending_commands(),
            0,
            "a rejected call must not queue anything"
        );
    }

    #[test]
    fn missing_position_components_default_to_zero() {
        let mut host = host_running("set_position(1, 5)");
        assert_eq!(
            host.drain_commands()[0],
            ScriptCommand::SetPosition {
                entity: 1,
                position: Vec3::new(5.0, 0.0, 0.0)
            }
        );
    }

    #[test]
    fn a_nonsensical_scale_is_rejected() {
        let mut host = ScriptHost::new();
        assert!(host.eval("set_scale(1, 0)").is_err());
        assert!(host.eval("set_scale(1, -2)").is_err());
        assert!(host.eval("set_scale(1, 1/0)").is_err());
        assert!(host.eval("set_scale(1, 2)").is_ok());
    }

    // --- the update hook ---

    /// How the engine drives entity behaviour: a script defines `update(dt)` and
    /// Rust calls it every frame.
    #[test]
    fn the_update_hook_is_called_with_the_frame_delta() {
        let mut host = host_running(
            "accumulated = 0\n\
             function update(dt)\n\
               accumulated = accumulated + dt\n\
             end",
        );
        let state = EngineState {
            dt: 0.25,
            ..EngineState::default()
        };
        for _ in 0..4 {
            host.update(&state).unwrap();
        }
        assert_eq!(
            host.interpreter().get_global("accumulated"),
            Value::Number(1.0)
        );
    }

    #[test]
    fn a_script_without_update_is_not_an_error() {
        let mut host = host_running("x = 1");
        assert!(host.update(&EngineState::default()).is_ok());
    }

    /// A script that errors every frame must not take the engine down, and must
    /// not flood the console either.
    #[test]
    fn a_failing_update_is_reported_once_and_does_not_stop_the_engine() {
        let mut host = host_running("function update(dt) return nil + 1 end");
        for _ in 0..10 {
            assert!(
                host.update(&EngineState::default()).is_ok(),
                "a script error must not fail the frame"
            );
        }
        assert!(host.last_error().is_some());
        let error_lines = host
            .output()
            .iter()
            .filter(|l| l.contains("script update"))
            .count();
        assert_eq!(
            error_lines, 1,
            "the same error should be logged once, not ten times"
        );
    }

    /// The `engine` table is how a script reads the world. Published fresh every
    /// frame, so a script cannot see stale values.
    #[test]
    fn the_engine_table_carries_this_frames_state() {
        let mut host = host_running(
            "function update(dt)\n\
               seen_time = engine.time\n\
               seen_frame = engine.frame\n\
               seen_fps = engine.fps\n\
               seen_x = engine.camera_x\n\
             end",
        );
        host.update(&EngineState {
            time: 1.5,
            dt: 0.016,
            frame: 90,
            fps: 60.0,
            camera_position: Vec3::new(3.0, 0.0, 0.0),
            entity_count: 5,
        })
        .unwrap();
        assert_eq!(host.interpreter().get_global("seen_time"), Value::Number(1.5));
        assert_eq!(host.interpreter().get_global("seen_frame"), Value::Number(90.0));
        assert_eq!(host.interpreter().get_global("seen_fps"), Value::Number(60.0));
        assert_eq!(host.interpreter().get_global("seen_x"), Value::Number(3.0));
    }

    #[test]
    fn the_engine_table_is_refreshed_each_frame() {
        let mut host = host_running("function update(dt) seen = engine.frame end");
        host.update(&EngineState { frame: 1, ..Default::default() }).unwrap();
        assert_eq!(host.interpreter().get_global("seen"), Value::Number(1.0));
        host.update(&EngineState { frame: 2, ..Default::default() }).unwrap();
        assert_eq!(host.interpreter().get_global("seen"), Value::Number(2.0));
    }

    /// A realistic behaviour script: it reads the frame state, computes, and
    /// queues a command. This is the phase criterion — a Lua script driving an
    /// entity.
    #[test]
    fn a_behaviour_script_drives_an_entity_from_engine_state() {
        let mut host = host_running(
            "function update(dt)\n\
               local angle = engine.time\n\
               set_position(2, math.cos(angle) * 5, 0, math.sin(angle) * 5)\n\
             end",
        );
        host.update(&EngineState {
            time: 0.0,
            ..Default::default()
        })
        .unwrap();
        let commands = host.drain_commands();
        match &commands[0] {
            ScriptCommand::SetPosition { entity, position } => {
                assert_eq!(*entity, 2);
                // cos(0) * 5 = 5, sin(0) * 5 = 0.
                assert!((position.x - 5.0).abs() < 1e-4, "{position}");
                assert!(position.z.abs() < 1e-4, "{position}");
            }
            other => panic!("expected a SetPosition, got {other:?}"),
        }

        // A quarter turn later it should be somewhere else.
        host.update(&EngineState {
            time: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        })
        .unwrap();
        match &host.drain_commands()[0] {
            ScriptCommand::SetPosition { position, .. } => {
                assert!(position.x.abs() < 1e-3, "{position}");
                assert!((position.z - 5.0).abs() < 1e-3, "{position}");
            }
            other => panic!("expected a SetPosition, got {other:?}"),
        }
    }

    // --- hot reload ---

    /// The phase's hot-reload requirement, exercised on a real file.
    #[test]
    fn a_changed_file_is_reloaded_and_its_definitions_replaced() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_script_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("behaviour.lua");

        std::fs::write(&path, "function get() return 1 end").unwrap();
        let mut host = ScriptHost::new();
        host.watch(&path);
        assert_eq!(
            host.interpreter().call_global("get", &[]).unwrap()[0],
            Value::Number(1.0)
        );

        // Nothing changed: no reload.
        assert_eq!(host.poll_reloads(), 0);

        // Rewrite it with *no* sleep. Detection must not depend on the clock
        // having ticked: the contents differ, and that is what the fingerprint
        // catches. This test failed intermittently before the fingerprint existed.
        std::fs::write(&path, "function get() return 2 end").unwrap();
        assert_eq!(host.poll_reloads(), 1, "the change should be noticed");
        assert_eq!(
            host.interpreter().call_global("get", &[]).unwrap()[0],
            Value::Number(2.0),
            "the new definition must replace the old"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// State must survive a reload — otherwise hot-reload resets the game every
    /// time the author saves, which defeats the point.
    #[test]
    fn state_survives_a_reload() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_script_state_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("stateful.lua");

        std::fs::write(&path, "counter = counter or 0\nfunction bump() counter = counter + 1 end")
            .unwrap();
        let mut host = ScriptHost::new();
        host.watch(&path);
        host.interpreter().call_global("bump", &[]).unwrap();
        host.interpreter().call_global("bump", &[]).unwrap();
        assert_eq!(host.interpreter().get_global("counter"), Value::Number(2.0));

        std::fs::write(&path, "counter = counter or 0\nfunction bump() counter = counter + 10 end")
            .unwrap();
        host.poll_reloads();
        assert_eq!(
            host.interpreter().get_global("counter"),
            Value::Number(2.0),
            "the accumulated count must survive the reload"
        );
        host.interpreter().call_global("bump", &[]).unwrap();
        assert_eq!(host.interpreter().get_global("counter"), Value::Number(12.0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A broken file must not take the engine down, and the previously-loaded
    /// definitions must stay live — so the author can keep playing while fixing it.
    #[test]
    fn a_broken_reload_keeps_the_old_definitions_and_reports_once() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_script_broken_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("broken.lua");

        std::fs::write(&path, "function get() return 1 end").unwrap();
        let mut host = ScriptHost::new();
        host.watch(&path);

        std::fs::write(&path, "function get( -- unclosed").unwrap();
        host.poll_reloads();
        assert!(host.last_error().is_some(), "the error should be recorded");
        assert_eq!(
            host.interpreter().call_global("get", &[]).unwrap()[0],
            Value::Number(1.0),
            "the working definition must survive a broken reload"
        );
        let errors = host.output().iter().filter(|l| l.contains("broken.lua")).count();
        assert_eq!(errors, 1, "reported once, not once per poll");

        // Fixing it must be announced, so the author knows it took.
        std::fs::write(&path, "function get() return 3 end").unwrap();
        host.poll_reloads();
        assert!(host.last_error().is_none(), "recovery should clear the error");
        assert_eq!(
            host.interpreter().call_global("get", &[]).unwrap()[0],
            Value::Number(3.0)
        );
        assert!(
            host.output().iter().any(|l| l.contains("loaded")),
            "recovery should be announced: {:?}",
            host.output()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A save that rewrote identical bytes must NOT reload — which is what an
    /// editor's "save all" does to every open file, and reloading each of them
    /// would reset script state for no reason.
    #[test]
    fn an_identical_rewrite_does_not_trigger_a_reload() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_script_identical_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("same.lua");
        let source = "function get() return 1 end";

        std::fs::write(&path, source).unwrap();
        let mut host = ScriptHost::new();
        host.watch(&path);

        // Rewrite the same bytes, with a sleep so the modification time really
        // does change — the fingerprint has to be what decides, not the clock.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&path, source).unwrap();
        assert_eq!(
            host.poll_reloads(),
            0,
            "identical contents should not count as a change"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Saves inside one filesystem timestamp tick.
    ///
    /// This failed intermittently — about one run in twenty — while `poll_reloads`
    /// used the timestamp and size as a pre-check. Measured on NTFS: consecutive
    /// writes produced identical modification times, and being the same length too
    /// meant neither hint fired and the edit was ignored. Same-length writes are
    /// used deliberately here so the size cannot rescue it.
    #[test]
    fn a_change_within_one_timestamp_tick_is_still_detected() {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_script_tick_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("rapid.lua");

        std::fs::write(&path, "function get() return 1 end").unwrap();
        let mut host = ScriptHost::new();
        host.watch(&path);

        // Ten writes back to back, each different but all the same length, with no
        // sleeps. Ten rather than three so a single lucky timestamp cannot make the
        // test pass by accident.
        for expected in 2..=9 {
            std::fs::write(&path, format!("function get() return {expected} end")).unwrap();
            assert_eq!(
                host.poll_reloads(),
                1,
                "write {expected} should have been detected"
            );
            assert_eq!(
                host.interpreter().call_global("get", &[]).unwrap()[0],
                Value::Number(expected as f64)
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The demo ships without scripts, so watching a path that does not exist must
    /// be harmless — and must start working if a file appears there.
    #[test]
    fn watching_a_missing_file_is_harmless() {
        let mut host = ScriptHost::new();
        host.watch("definitely/not/a/real/path.lua");
        assert_eq!(host.poll_reloads(), 0);
        assert!(host.last_error().is_none());
        // And the interpreter still works.
        assert!(host.eval("x = 1").is_ok());
    }

    #[test]
    fn print_from_a_script_reaches_the_host_output() {
        let host = host_running("print(\"from lua\")");
        assert!(
            host.output().iter().any(|l| l == "from lua"),
            "{:?}",
            host.output()
        );
    }

    /// `log` goes through the command queue rather than the print buffer, because
    /// the engine console wants it interleaved with its own output at a known
    /// point in the frame.
    #[test]
    fn log_goes_through_the_command_queue_not_the_print_buffer() {
        let mut host = host_running("log(\"queued\")");
        assert!(!host.output().iter().any(|l| l.contains("queued")));
        assert_eq!(
            host.drain_commands()[0],
            ScriptCommand::Log("queued".to_string())
        );
    }
}
