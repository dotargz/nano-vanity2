//! Micro-benchmark of the vector engine's building blocks (tuning aid).
use std::hint::black_box;
use std::time::Instant;

/// Nanoseconds per core clock cycle, from a chain with known latency:
/// imul (3) + rol (1) + xor (1).
fn cycle_ns() -> f64 {
    const N: usize = 100_000_000;
    let k = black_box(0x9E37_79B9_7F4A_7C15u64);
    let y = black_box(0x1234_5678_9abc_def1u64);
    let mut x = black_box(1u64);
    let t = Instant::now();
    for _ in 0..N {
        x = x.wrapping_mul(k).rotate_left(7) ^ y;
    }
    black_box(x);
    t.elapsed().as_secs_f64() * 1e9 / N as f64 / 5.0
}

fn main() {
    let c = cycle_ns();
    println!("core clock ~{:.2} GHz", 1.0 / c);
    if nano_vanity2::engine::Backend::Ifma.available() {
        unsafe { nano_vanity2::engine::ifma::microbench(c) };
    }
}
