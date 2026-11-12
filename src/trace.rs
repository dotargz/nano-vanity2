//! Optional start-up timing trace (hidden `--trace` flag), for tuning.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static ON: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Vec<(&'static str, Instant)>> = Mutex::new(Vec::new());

pub fn enable() {
    ON.store(true, Ordering::Relaxed);
}

#[inline]
pub fn mark(label: &'static str) {
    if ON.load(Ordering::Relaxed) {
        LOG.lock().unwrap().push((label, Instant::now()));
    }
}

pub fn dump(start: Instant) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let mut prev = start;
    for (label, t) in LOG.lock().unwrap().iter() {
        eprintln!(
            "trace {:>9.1} us  (+{:>8.1})  {label}",
            t.duration_since(start).as_secs_f64() * 1e6,
            t.duration_since(prev).as_secs_f64() * 1e6
        );
        prev = *t;
    }
}
