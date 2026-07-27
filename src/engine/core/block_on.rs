// Minimal future blocker — replaces the `pollster` crate.
// wgpu's async init functions need a way to run to completion synchronously.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

/// Block the current thread until the given future completes.
///
/// Panics if the future is already pinned or re-entrant.
pub fn block_on<F: Future>(mut future: F) -> F::Output {
    // A no-op waker: the futures we drive (wgpu init) never need wakeups
    // because they complete in a single poll or are internally synchronous.
    fn noop_clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }
    fn noop(_: *const ()) {}

    const VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);

    fn noop_raw_waker() -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);

    // SAFETY: we shadow `future` with a pinned reference that lives
    // for the duration of this function, never moving it afterwards.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    loop {
        if let Poll::Ready(val) = future.as_mut().poll(&mut cx) {
            return val;
        }
        // wgpu's init futures spin internally; yield to the OS.
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_on_immediate() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn block_on_string() {
        let result = block_on(async { "hello".to_string() });
        assert_eq!(result, "hello");
    }
}
