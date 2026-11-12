//! Search engines: turn a stream of candidate keys into public keys as fast
//! as possible and hand the few that pass a cheap bit filter to a `Handler`.
//!
//! Candidate `c` of a job is the 256-bit secret whose little-endian words are
//! `(key[0] + c, key[1], key[2], key[3])`. In seed mode the secret is a Nano
//! seed and the private key is `blake2b_256(seed || index)`; otherwise the
//! secret is the private key itself.

use crate::table::Table;

#[cfg(target_arch = "x86_64")]
pub mod ifma;
pub mod scalar;

/// Candidates per work unit (one AVX-512 vector).
pub const UNIT: usize = 8;
/// Largest number of units per `Engine::run` call.
pub const MAX_UNITS: usize = 64;

/// Cheap necessary condition evaluated on every candidate:
/// `(w0 & m0) == v0` on the first 8 public key bytes (as a little-endian
/// word) and, in full mode, `(checksum & mc) == vc`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Prefilter {
    pub m0: u64,
    pub v0: u64,
    pub mc: u64,
    pub vc: u64,
}

pub trait Handler {
    /// Full mode computes the complete public key and address checksum of
    /// every candidate; otherwise only the low 64 bits of the key.
    fn full(&self) -> bool;
    fn prefilter(&self) -> Prefilter;
    /// Quick mode: candidate `counter` passed the prefilter; `w0` holds the
    /// first 8 public key bytes.
    fn on_quick(&mut self, counter: u64, w0: u64);
    /// Full mode: candidate `counter` passed the prefilter; `pk` is the
    /// public key (little-endian words) and `cks` the 40-bit checksum.
    fn on_full(&mut self, counter: u64, pk: [u64; 4], cks: u64);
}

pub struct Job<'a> {
    pub table: &'a Table,
    pub key: [u64; 4],
    pub counter: u64,
    /// `Some(index)`: secrets are seeds; `None`: secrets are private keys.
    pub seed_index: Option<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// Portable 64-bit code.
    Scalar,
    /// AVX-512 IFMA, eight keys at a time.
    Ifma,
}

impl Backend {
    pub fn available(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Ifma => {
                #[cfg(target_arch = "x86_64")]
                {
                    std::is_x86_feature_detected!("avx512f")
                        && std::is_x86_feature_detected!("avx512dq")
                        && std::is_x86_feature_detected!("avx512ifma")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
        }
    }

    /// The fastest backend this CPU supports.
    pub fn detect() -> Backend {
        if Backend::Ifma.available() {
            Backend::Ifma
        } else {
            Backend::Scalar
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Ifma => "avx512-ifma",
        }
    }
}

/// Per-thread engine state.
pub struct Engine {
    backend: Backend,
    #[cfg(target_arch = "x86_64")]
    scratch: Option<Box<ifma::Scratch>>,
}

impl Engine {
    pub fn new(backend: Backend) -> Engine {
        assert!(backend.available(), "backend not supported by this CPU");
        Engine {
            backend,
            #[cfg(target_arch = "x86_64")]
            scratch: if backend == Backend::Ifma { Some(ifma::Scratch::new()) } else { None },
        }
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Process `units * UNIT` candidates starting at `job.counter`.
    /// Returns the number of candidates processed.
    #[inline]
    pub fn run<H: Handler>(&mut self, job: &Job, units: usize, h: &mut H) -> u64 {
        let units = units.clamp(1, MAX_UNITS);
        match self.backend {
            Backend::Scalar => scalar::run(job, units * UNIT, h),
            #[cfg(target_arch = "x86_64")]
            Backend::Ifma => {
                // Safety: `new` checked that the CPU supports the features.
                unsafe { ifma::run(job, units, self.scratch.as_mut().unwrap(), h) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Ifma => unreachable!(),
        }
        (units * UNIT) as u64
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        if let Some(s) = self.scratch.as_mut() {
            s.wipe();
        }
    }
}
