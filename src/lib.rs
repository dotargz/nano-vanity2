//! nano-vanity2: a very fast vanity address generator for Nano.
//!
//! Layout:
//! * `fe51`, `edwards`, `blake2b`, `keys`, `address` — small reference
//!   implementation of Nano key derivation; verifies every result.
//! * `tablegen`, `table` — fixed-base lookup tables.
//! * `engine` — the fast paths (AVX-512 IFMA and portable scalar).
//! * `pattern` — wildcard / lookalike / regex matching on address bits.
//! * `search` — threads, key generation, verification, progress.

pub mod address;
pub mod blake2b;
pub mod edwards;
pub mod engine;
pub mod fe51;
pub mod gpu;
pub mod keys;
pub mod pattern;
pub mod search;
pub mod selftest;
pub mod table;
pub mod tablegen;
pub mod trace;
