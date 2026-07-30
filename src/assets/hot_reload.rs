// Asset hot-reload: notice when a file on disk changes and say which one.
//
// One watcher for every asset type rather than one per format, because watching is
// the same problem whatever the file contains — and because the hard part was already
// learned in `scripting::bindings`: **contents decide, not timestamps.**
//
// Measured there on NTFS, consecutive writes reported identical modification times
// (…831770, …841769, …841769), so using the timestamp as a pre-check missed about one
// edit in twenty. The same reasoning applies here and the same conclusion follows: one
// read of the file per poll. A texture is larger than a script, so the read is not
// free — but a missed reload is worse than a few hundred microseconds, and the poll
// interval is a knob for exactly that trade.
//
// This module deliberately does *not* decode anything. It reports "this path changed"
// and the caller decides what to do, which keeps the watcher testable without a GPU
// and means adding a format needs no changes here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// What changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReloadEvent {
    /// The file's contents changed.
    Changed(PathBuf),
    /// The file existed and now does not. Reported rather than ignored so a caller
    /// can keep the last good version and say why.
    Removed(PathBuf),
    /// A watched path that did not exist now does.
    Appeared(PathBuf),
}

impl ReloadEvent {
    /// The path this event concerns.
    pub fn path(&self) -> &Path {
        match self {
            Self::Changed(p) | Self::Removed(p) | Self::Appeared(p) => p,
        }
    }
}

/// How often the watcher will actually hit the disk.
///
/// A frame at 1300 FPS is 0.77 ms; hashing every watched texture that often would put
/// asset watching on the hot path for no benefit, since nobody edits a file more than
/// a few times a second. 250 ms is imperceptible to a human saving a file and three
/// orders of magnitude cheaper than per-frame.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Fingerprint of a file's contents.
///
/// Length and an FNV-1a hash. Not a cryptographic hash: the only adversary here is a
/// coarse filesystem clock, and length alone is not enough because an edit that
/// preserves the size is exactly what a texture tweak or a one-character script fix
/// looks like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fingerprint {
    length: u64,
    hash: u64,
}

fn fingerprint(bytes: &[u8]) -> Fingerprint {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    Fingerprint {
        length: bytes.len() as u64,
        hash,
    }
}

/// A watcher over a set of asset paths.
pub struct AssetWatcher {
    /// Path to its last-seen fingerprint. `None` means the path is watched but the
    /// file is absent.
    watched: HashMap<PathBuf, Option<Fingerprint>>,
    /// Insertion order, so `poll` reports deterministically rather than in whatever
    /// order the map iterates. A caller applying reloads in a varying order is a
    /// reproducibility problem waiting to happen.
    order: Vec<PathBuf>,
    interval: Duration,
    last_poll: Option<Instant>,
    /// Polls that actually hit the disk, and events emitted, so the cost and the
    /// benefit are both visible.
    disk_polls: u64,
    events: u64,
}

impl Default for AssetWatcher {
    fn default() -> Self {
        Self::new(DEFAULT_POLL_INTERVAL)
    }
}

impl AssetWatcher {
    pub fn new(interval: Duration) -> Self {
        Self {
            watched: HashMap::new(),
            order: Vec::new(),
            interval,
            last_poll: None,
            disk_polls: 0,
            events: 0,
        }
    }

    /// Start watching a path, recording its current state without emitting an event.
    ///
    /// A file present at registration is not "changed": the caller has presumably
    /// just loaded it, and reporting it would cause a redundant reload on the first
    /// poll of every run.
    pub fn watch(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        if self.watched.contains_key(&path) {
            return;
        }
        let current = std::fs::read(&path).ok().map(|bytes| fingerprint(&bytes));
        self.watched.insert(path.clone(), current);
        self.order.push(path);
    }

    /// Stop watching a path.
    pub fn unwatch(&mut self, path: impl AsRef<Path>) {
        let path = path.as_ref();
        self.watched.remove(path);
        self.order.retain(|p| p != path);
    }

    pub fn watched_count(&self) -> usize {
        self.order.len()
    }

    pub fn disk_polls(&self) -> u64 {
        self.disk_polls
    }

    pub fn events(&self) -> u64 {
        self.events
    }

    /// Whether enough time has passed to poll again.
    pub fn is_due(&self, now: Instant) -> bool {
        match self.last_poll {
            None => true,
            Some(last) => now.duration_since(last) >= self.interval,
        }
    }

    /// Check every watched path, respecting the poll interval.
    ///
    /// Takes `now` rather than reading the clock, so a test can drive the interval
    /// deterministically instead of sleeping.
    pub fn poll(&mut self, now: Instant) -> Vec<ReloadEvent> {
        if !self.is_due(now) {
            return Vec::new();
        }
        self.last_poll = Some(now);
        self.disk_polls += 1;

        let mut events = Vec::new();
        for path in &self.order {
            let current = std::fs::read(path).ok().map(|bytes| fingerprint(&bytes));
            let previous = self.watched.get(path).copied().flatten();
            let event = match (previous, current) {
                // Unchanged, or still absent.
                (Some(a), Some(b)) if a == b => None,
                (None, None) => None,
                (Some(_), Some(_)) => Some(ReloadEvent::Changed(path.clone())),
                (Some(_), None) => Some(ReloadEvent::Removed(path.clone())),
                (None, Some(_)) => Some(ReloadEvent::Appeared(path.clone())),
            };
            self.watched.insert(path.clone(), current);
            if let Some(event) = event {
                events.push(event);
            }
        }
        self.events += events.len() as u64;
        events
    }

    /// Poll immediately, ignoring the interval. Backs a manual `reload` command.
    pub fn poll_now(&mut self, now: Instant) -> Vec<ReloadEvent> {
        self.last_poll = None;
        self.poll(now)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory unique to each test: these run in parallel in one process, so a
    /// shared name would make them interfere.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "astraglyph_assets_{}_{}",
            name,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// A watcher with no interval, so tests exercise detection rather than timing.
    fn immediate() -> AssetWatcher {
        AssetWatcher::new(Duration::ZERO)
    }

    #[test]
    fn a_changed_file_is_reported() {
        let dir = temp_dir("changed");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"one").unwrap();

        let mut watcher = immediate();
        watcher.watch(&path);
        assert!(
            watcher.poll(Instant::now()).is_empty(),
            "an unchanged file must not be reported"
        );

        std::fs::write(&path, b"two").unwrap();
        assert_eq!(
            watcher.poll(Instant::now()),
            vec![ReloadEvent::Changed(path.clone())]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Registering a file must not report it: the caller has just loaded it, and an
    /// event here would cause a redundant reload on the first poll of every run.
    #[test]
    fn registering_a_file_does_not_report_it() {
        let dir = temp_dir("register");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"content").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);
        assert!(watcher.poll(Instant::now()).is_empty());
        assert_eq!(watcher.events(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Contents decide, not timestamps. An edit that preserves the size and lands in
    /// the same filesystem tick is exactly what a texture tweak looks like, and the
    /// same class of bug cost one edit in twenty in `scripting::bindings`.
    #[test]
    fn a_same_size_change_within_one_timestamp_tick_is_detected() {
        let dir = temp_dir("tick");
        let path = dir.join("a.bin");
        std::fs::write(&path, b"AAAA").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);

        // Eight writes back to back, each different but all four bytes, no sleeps.
        for i in 0..8u8 {
            std::fs::write(&path, [b'B' + i, b'B', b'B', b'B']).unwrap();
            assert_eq!(
                watcher.poll(Instant::now()).len(),
                1,
                "write {i} was missed, so the timestamp is being trusted"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A rewrite of identical bytes must not reload — that is what an editor's
    /// "save all" does to every open file, and reloading each would be pure cost.
    #[test]
    fn an_identical_rewrite_is_not_a_change() {
        let dir = temp_dir("identical");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"same").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);

        std::thread::sleep(Duration::from_millis(20)); // move the mtime
        std::fs::write(&path, b"same").unwrap();
        assert!(
            watcher.poll(Instant::now()).is_empty(),
            "identical contents must not count as a change"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A deletion is reported rather than ignored, so a caller can keep the last good
    /// version and say why it is stale.
    #[test]
    fn a_removal_and_a_reappearance_are_both_reported() {
        let dir = temp_dir("removal");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"here").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);

        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            watcher.poll(Instant::now()),
            vec![ReloadEvent::Removed(path.clone())]
        );
        // Still gone: reported once, not every poll.
        assert!(watcher.poll(Instant::now()).is_empty());

        std::fs::write(&path, b"back").unwrap();
        assert_eq!(
            watcher.poll(Instant::now()),
            vec![ReloadEvent::Appeared(path.clone())]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Watching a path that does not exist is harmless, and must start working when a
    /// file appears there — the demo ships without some assets.
    #[test]
    fn watching_a_missing_path_is_harmless_and_picks_it_up_later() {
        let dir = temp_dir("missing");
        let path = dir.join("later.txt");
        let mut watcher = immediate();
        watcher.watch(&path);
        assert!(watcher.poll(Instant::now()).is_empty());

        std::fs::write(&path, b"now here").unwrap();
        assert_eq!(
            watcher.poll(Instant::now()),
            vec![ReloadEvent::Appeared(path.clone())]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The interval is what keeps asset watching off the hot path: at 1300 FPS,
    /// hashing every texture per frame would cost more than the reload saves.
    #[test]
    fn the_poll_interval_is_respected() {
        let dir = temp_dir("interval");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"one").unwrap();

        let mut watcher = AssetWatcher::new(Duration::from_millis(500));
        watcher.watch(&path);
        let start = Instant::now();
        // First poll always runs.
        assert!(watcher.poll(start).is_empty());
        assert_eq!(watcher.disk_polls(), 1);

        std::fs::write(&path, b"two").unwrap();
        // Too soon: no disk access, no event, even though the file did change.
        assert!(watcher.poll(start + Duration::from_millis(100)).is_empty());
        assert_eq!(
            watcher.disk_polls(),
            1,
            "an early poll must not touch the disk"
        );

        // Past the interval.
        let events = watcher.poll(start + Duration::from_millis(600));
        assert_eq!(events.len(), 1);
        assert_eq!(watcher.disk_polls(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A manual reload has to work regardless of when the last poll was.
    #[test]
    fn poll_now_ignores_the_interval() {
        let dir = temp_dir("pollnow");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"one").unwrap();
        let mut watcher = AssetWatcher::new(Duration::from_secs(3600));
        watcher.watch(&path);
        let start = Instant::now();
        watcher.poll(start);

        std::fs::write(&path, b"two").unwrap();
        assert!(
            watcher.poll(start).is_empty(),
            "an hour-long interval should suppress the automatic poll"
        );
        assert_eq!(watcher.poll_now(start).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Events come out in registration order. A caller applying reloads in a varying
    /// order is a reproducibility problem waiting to happen.
    #[test]
    fn events_are_reported_in_registration_order() {
        let dir = temp_dir("order");
        let paths: Vec<PathBuf> = (0..5).map(|i| dir.join(format!("f{i}.txt"))).collect();
        for path in &paths {
            std::fs::write(path, b"one").unwrap();
        }
        let mut watcher = immediate();
        for path in &paths {
            watcher.watch(path);
        }
        for path in &paths {
            std::fs::write(path, b"two").unwrap();
        }
        let reported: Vec<PathBuf> = watcher
            .poll(Instant::now())
            .into_iter()
            .map(|e| e.path().to_path_buf())
            .collect();
        assert_eq!(reported, paths);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn watching_the_same_path_twice_registers_it_once() {
        let dir = temp_dir("dedup");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"x").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);
        watcher.watch(&path);
        assert_eq!(watcher.watched_count(), 1);

        std::fs::write(&path, b"y").unwrap();
        assert_eq!(
            watcher.poll(Instant::now()).len(),
            1,
            "a duplicated registration must not double the events"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unwatching_stops_the_reports() {
        let dir = temp_dir("unwatch");
        let path = dir.join("a.txt");
        std::fs::write(&path, b"one").unwrap();
        let mut watcher = immediate();
        watcher.watch(&path);
        watcher.unwatch(&path);
        assert_eq!(watcher.watched_count(), 0);

        std::fs::write(&path, b"two").unwrap();
        assert!(watcher.poll(Instant::now()).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_watcher_polls_without_incident() {
        let mut watcher = immediate();
        assert!(watcher.poll(Instant::now()).is_empty());
        assert_eq!(watcher.watched_count(), 0);
        assert_eq!(watcher.events(), 0);
    }

    /// Length alone is not enough, and neither is a hash alone in principle — the
    /// pair is what makes a same-length edit detectable.
    #[test]
    fn the_fingerprint_distinguishes_same_length_content() {
        assert_ne!(fingerprint(b"AAAA"), fingerprint(b"AAAB"));
        assert_ne!(fingerprint(b"AB"), fingerprint(b"BA"));
        assert_eq!(fingerprint(b"same"), fingerprint(b"same"));
        assert_ne!(fingerprint(b""), fingerprint(b"a"));
    }

    #[test]
    fn the_default_interval_is_imperceptible_but_not_per_frame() {
        // Fast enough that a human saving a file does not notice the lag, slow enough
        // that it is nowhere near the frame rate.
        assert!(DEFAULT_POLL_INTERVAL <= Duration::from_millis(500));
        assert!(DEFAULT_POLL_INTERVAL >= Duration::from_millis(50));
    }
}
