//! Baseline for comparison: the conventional per-key pipeline that CPU
//! vanity generators use (BLAKE2b-512, curve25519-dalek fixed-base
//! multiplication, point compression), on all threads.
//!
//! Run with: cargo run --release --example baseline

use blake2::digest::{Update, VariableOutput};
use curve25519_dalek::edwards::EdwardsPoint;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn main() {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let stop = AtomicBool::new(false);
    let total = AtomicU64::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        for id in 0..threads {
            let (stop, total) = (&stop, &total);
            s.spawn(move || {
                let mut key = [id as u8 + 1; 32];
                let mut n = 0u64;
                let mut sink = 0u8;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..256 {
                        key[..8].copy_from_slice(&n.to_le_bytes());
                        n += 1;
                        let mut h = blake2::Blake2bVar::new(64).unwrap();
                        h.update(&key);
                        let mut out = [0u8; 64];
                        h.finalize_variable(&mut out).unwrap();
                        let mut scalar = [0u8; 32];
                        scalar.copy_from_slice(&out[..32]);
                        let pk = EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes();
                        sink ^= pk[0];
                    }
                    total.fetch_add(256, Ordering::Relaxed);
                }
                std::hint::black_box(sink);
            });
        }
        std::thread::sleep(Duration::from_secs(3));
        stop.store(true, Ordering::Relaxed);
    });
    let rate = total.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64();
    println!(
        "conventional pipeline (curve25519-dalek): {:.2} M keys/s on {threads} threads ({:.0} k per thread)",
        rate / 1e6,
        rate / 1e3 / threads as f64
    );
}
