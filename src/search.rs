//! Search orchestration: worker threads, secret generation, result
//! verification, table upgrades, progress and time limits.

use crate::engine::{scalar, Backend, Engine, Handler, Job, Prefilter, MAX_UNITS};
use crate::pattern::{bits_from_words, Matcher};
use crate::table::{static_table, Builder, Table, STATIC_WINDOW};
use crate::{address, keys};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};
use zeroize::{Zeroize, Zeroizing};

/// Searches expected to need fewer keys than this stay on the calling
/// thread: a second thread could not start in time to help.
const SINGLE_THREAD_BELOW: f64 = 200.0;
/// Searches expected to need at least this many keys build a larger lookup
/// table in the background and switch to it when it is ready.
const BIG_TABLE_FROM: f64 = 400_000.0;
/// ... and from here on the largest table the L3 cache can hold.
const HUGE_TABLE_FROM: f64 = 20_000_000.0;
/// Searches expected to need at least this many keys also start the GPU;
/// shorter ones are over before it is ready.
const GPU_FROM: f64 = 12_000_000.0;
/// ... and from here on the GPU uses larger, slightly more efficient launches.
const GPU_LONG_FROM: f64 = 400_000_000.0;
/// Window width of the standard larger table (10.5 MB).
pub const BIG_WINDOW: u32 = 13;

/// Size of the level-3 cache in bytes, if the CPU reports it.
pub fn l3_cache_bytes() -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::__cpuid_count;
        // Intel: leaf 4. AMD: leaf 0x8000001D. Same layout.
        for leaf in [4u32, 0x8000_001D] {
            let max = __cpuid_count(leaf & 0x8000_0000, 0).eax;
            if max < leaf {
                continue;
            }
            for sub in 0..16 {
                let r = __cpuid_count(leaf, sub);
                if r.eax & 31 == 0 {
                    break;
                }
                if (r.eax >> 5) & 7 == 3 {
                    let ways = ((r.ebx >> 22) & 0x3ff) as usize + 1;
                    let parts = ((r.ebx >> 12) & 0x3ff) as usize + 1;
                    let line = (r.ebx & 0xfff) as usize + 1;
                    let sets = r.ecx as usize + 1;
                    return Some(ways * parts * line * sets);
                }
            }
        }
    }
    None
}

/// Table window for a search expected to need `expected` keys.
pub fn choose_window(expected: f64) -> Option<u32> {
    if expected < BIG_TABLE_FROM {
        return None;
    }
    // Keep the table comfortably inside the L3 cache; lookups that go to
    // main memory cost more than the saved additions.
    let budget = l3_cache_bytes().unwrap_or(16 << 20) * 7 / 10;
    let top = if expected >= HUGE_TABLE_FROM { 16 } else { BIG_WINDOW };
    (STATIC_WINDOW + 1..=top).rev().find(|&w| crate::tablegen::Layout::new(w).bytes() <= budget)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyKind {
    /// Generate wallet seeds; the address is account `index` of the seed.
    Seed { index: u32 },
    /// Generate raw private keys.
    Private,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum GpuMode {
    /// Use the GPU when the search is long enough to benefit.
    #[default]
    Auto,
    /// Use the GPU whenever the pattern allows it.
    On,
    Off,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuOptions {
    pub mode: GpuMode,
    /// Overrides for the launch shape and table window (tuning).
    pub threads: Option<u32>,
    pub batch: Option<u32>,
    pub window: Option<u32>,
}

pub struct Options<'a> {
    pub matcher: &'a Matcher,
    pub kind: KeyKind,
    pub threads: usize,
    /// Stop after this many matches.
    pub count: u64,
    pub timeout: Option<Duration>,
    pub backend: Backend,
    /// `None`: choose automatically; `Some(0)`: embedded table only;
    /// `Some(w)`: always build and use a table with window `w`.
    pub window: Option<u32>,
    /// When the program started; reported times are relative to this.
    pub start: Instant,
    /// Print a live status line to stderr.
    pub progress: bool,
    pub gpu: GpuOptions,
}

pub struct Found {
    /// The seed or private key (see `kind`).
    pub secret: Zeroizing<[u8; 32]>,
    pub kind: KeyKind,
    pub public: [u8; 32],
    pub address: String,
    /// Keys tried (all threads) when this match was found.
    pub attempts: u64,
    pub elapsed: Duration,
}

pub struct Summary {
    pub attempts: u64,
    /// Time since `Options::start`.
    pub elapsed: Duration,
    /// Time spent searching (excludes an up-front table build).
    pub search_elapsed: Duration,
    pub found: u64,
    pub timed_out: bool,
    pub threads: usize,
    pub backend: Backend,
    /// Window of the table in use at the end.
    pub window: u32,
    /// Candidates the fast path accepted but the reference derivation
    /// rejected. Always zero unless something is broken.
    pub mismatches: u64,
    /// Name of the GPU that took part, and the keys it tried.
    pub gpu: Option<String>,
    pub gpu_attempts: u64,
    /// Why the GPU could not be used although it was wanted.
    pub gpu_error: Option<String>,
}

struct Shared<'a> {
    opts: &'a Options<'a>,
    on_found: &'a (dyn Fn(&Found) + Sync),
    stop: AtomicBool,
    timed_out: AtomicBool,
    attempts: AtomicU64,
    found: Mutex<u64>,
    mismatches: AtomicU64,
    table: AtomicPtr<Table>,
    builder: Option<Builder>,
    aux: OnceLock<Thread>,
    /// GPU: what to run, the table for it if the CPU's does not fit, and
    /// the keys it has tried.
    gpu_cfg: Mutex<Option<crate::gpu::Config>>,
    gpu_builder: Option<Builder>,
    gpu_tried: Arc<AtomicU64>,
    gpu_name: Mutex<Option<String>>,
    gpu_error: Mutex<Option<String>>,
}

impl Shared<'_> {
    fn halt(&self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.aux.get() {
            t.unpark();
        }
    }

    fn help_build(&self) {
        if let Some(b) = &self.builder {
            if b.help() {
                self.table.store(b.table() as *const Table as *mut Table, Ordering::Release);
                crate::trace::mark("big table ready");
                self.wake_aux();
            }
        }
        if let Some(b) = &self.gpu_builder {
            if b.help() {
                self.wake_aux();
            }
        }
    }

    fn wake_aux(&self) {
        if let Some(t) = self.aux.get() {
            t.unpark();
        }
    }

    /// Keys tried so far by CPU and GPU together.
    fn total(&self) -> u64 {
        self.attempts.load(Ordering::Relaxed) + self.gpu_tried.load(Ordering::Relaxed)
    }

    /// The finished table the GPU should use, if any is ready.
    fn gpu_table(&self) -> Option<&Table> {
        let b = self.gpu_builder.as_ref().or(self.builder.as_ref())?;
        b.is_ready().then(|| b.table())
    }
}

/// 256 fresh bits from the operating system's CSPRNG.
pub(crate) fn fresh_key() -> [u64; 4] {
    let mut bytes = Zeroizing::new([0u8; 32]);
    getrandom::fill(&mut bytes[..]).expect("the operating system's random number generator failed");
    let mut w = [0u64; 4];
    for i in 0..4 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        w[i] = u64::from_le_bytes(t);
    }
    w
}

struct Hand<'a, 'b> {
    sh: &'a Shared<'b>,
    key: [u64; 4],
    seed_index: Option<u32>,
    batch_start: u64,
    hit: bool,
    /// For GPU hits: keys the GPU had tried before the launch in question.
    gpu_before: Option<u64>,
}

impl Hand<'_, '_> {
    fn candidate(&mut self, counter: u64, from_quick: bool) {
        let sh = self.sh;
        let m = sh.opts.matcher;
        let mut words = [self.key[0].wrapping_add(counter), self.key[1], self.key[2], self.key[3]];
        if from_quick && m.has_other {
            // Only the leading characters were checked so far. Settle the
            // rest with a cheap table-based derivation before paying for the
            // table-free one.
            let pk = scalar::derive(static_table(), &words, self.seed_index);
            if !m.check_bits(&bits_from_words(&pk, scalar::checksum_words(&pk))) {
                words.zeroize();
                return;
            }
        }
        let mut secret = Zeroizing::new([0u8; 32]);
        for i in 0..4 {
            secret[i * 8..i * 8 + 8].copy_from_slice(&words[i].to_le_bytes());
        }
        words.zeroize();
        crate::trace::mark("candidate passed fast checks");

        // Authoritative check: derive the address again with the simple
        // reference implementation, which shares no tables or vector code
        // with the search engines.
        let private = match sh.opts.kind {
            KeyKind::Seed { index } => Zeroizing::new(keys::seed_to_private(&secret, index)),
            KeyKind::Private => secret.clone(),
        };
        let public = keys::private_to_public(&private);
        let addr = address::encode(&public);
        if !m.check_public(&public) || address::decode(&addr) != Ok(public) {
            sh.mismatches.fetch_add(1, Ordering::Relaxed);
            return;
        }

        crate::trace::mark("candidate verified by reference");
        let mut found = sh.found.lock().unwrap();
        if *found >= sh.opts.count {
            return;
        }
        *found += 1;
        let others = match self.gpu_before {
            Some(b) => sh.attempts.load(Ordering::Relaxed) + b,
            None => sh.total(),
        };
        let attempts = others + (counter.wrapping_sub(self.batch_start) + 1);
        let f = Found {
            secret,
            kind: sh.opts.kind,
            public,
            address: addr,
            attempts,
            elapsed: sh.opts.start.elapsed(),
        };
        (sh.on_found)(&f);
        if *found >= sh.opts.count {
            sh.halt();
        }
        // The rest of this batch shares most of its secret with the key
        // just published, so it is abandoned and the thread reseeds.
        self.hit = true;
    }
}

impl Handler for Hand<'_, '_> {
    fn full(&self) -> bool {
        self.sh.opts.matcher.need_full
    }
    fn prefilter(&self) -> Prefilter {
        self.sh.opts.matcher.prefilter
    }
    fn on_quick(&mut self, counter: u64, w0: u64) {
        if !self.hit && self.sh.opts.matcher.check_quick(w0.swap_bytes()) {
            self.candidate(counter, true);
        }
    }
    fn on_full(&mut self, counter: u64, pk: [u64; 4], cks: u64) {
        if !self.hit && self.sh.opts.matcher.check_bits(&bits_from_words(&pk, cks)) {
            self.candidate(counter, false);
        }
    }
}

fn worker(sh: &Shared, may_build: bool) {
    if may_build {
        sh.help_build();
    }
    crate::trace::mark("worker start");
    let mut engine = Engine::new(sh.opts.backend);
    crate::trace::mark("engine created");
    let seed_index = match sh.opts.kind {
        KeyKind::Seed { index } => Some(index),
        KeyKind::Private => None,
    };
    let mut hand = Hand { sh, key: fresh_key(), seed_index, batch_start: 0, hit: false, gpu_before: None };
    crate::trace::mark("random key drawn");
    let mut counter = 0u64;
    // Small batches first so that easy patterns return quickly.
    let mut units = 4;
    while !sh.stop.load(Ordering::Relaxed) {
        // Safety: the pointer is either the embedded table or the finished
        // big table, both of which outlive every worker.
        let table = unsafe { &*sh.table.load(Ordering::Acquire) };
        let mut job = Job { table, key: hand.key, counter, seed_index };
        hand.batch_start = counter;
        let n = engine.run(&job, units, &mut hand);
        job.key.zeroize();
        counter = counter.wrapping_add(n);
        sh.attempts.fetch_add(n, Ordering::Relaxed);
        if hand.hit {
            // Never derive two published keys from the same random base.
            hand.key.zeroize();
            hand.key = fresh_key();
            hand.hit = false;
            counter = 0;
        }
        units = (units * 2).min(MAX_UNITS);
    }
    hand.key.zeroize();
}

pub fn human_count(x: f64) -> String {
    const UNITS: [&str; 5] = ["", " k", " M", " G", " T"];
    let mut v = x;
    let mut u = 0;
    while v >= 1000.0 && u < UNITS.len() - 1 {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{v:.0}")
    } else if v >= 100.0 {
        format!("{v:.0}{}", UNITS[u])
    } else if v >= 10.0 {
        format!("{v:.1}{}", UNITS[u])
    } else {
        format!("{v:.2}{}", UNITS[u])
    }
}

pub fn human_duration(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1e-3 {
        format!("{:.1} µs", s * 1e6)
    } else if s < 1.0 {
        format!("{:.2} ms", s * 1e3)
    } else if s < 120.0 {
        format!("{s:.2} s")
    } else if s < 7200.0 {
        format!("{:.1} min", s / 60.0)
    } else if s < 172_800.0 {
        format!("{:.1} h", s / 3600.0)
    } else if s < 63_072_000.0 {
        format!("{:.1} days", s / 86_400.0)
    } else {
        format!("{:.1} years", s / 31_536_000.0)
    }
}

/// Watches the clock, reports progress and looks after the GPU.
fn auxiliary(sh: &Shared<'_>) {
    let _ = sh.aux.set(thread::current());
    // The GPU takes longest to get going, so it is started first.
    let session = sh
        .gpu_cfg
        .lock()
        .unwrap()
        .take()
        .map(|cfg| crate::gpu::Session::start(cfg, thread::current(), sh.gpu_tried.clone()));
    let mut table_given = false;
    let opts = sh.opts;
    let seed_index = match opts.kind {
        KeyKind::Seed { index } => Some(index),
        KeyKind::Private => None,
    };
    let expected = opts.matcher.expected_attempts();
    let mut shown = false;
    let mut last_print = Instant::now();
    let mut last_attempts = 0u64;
    let mut rate = 0.0f64;
    loop {
        if sh.stop.load(Ordering::Acquire) {
            break;
        }
        if let Some(session) = &session {
            if !table_given {
                if let Some(t) = sh.gpu_table() {
                    session.give_table(t.layout, crate::gpu::convert_table(t));
                    table_given = true;
                    crate::trace::mark("gpu: table handed over");
                }
            }
            while let Ok(ev) = session.events.try_recv() {
                match ev {
                    crate::gpu::Event::Ready { name } => *sh.gpu_name.lock().unwrap() = Some(name),
                    crate::gpu::Event::Failed(e) => *sh.gpu_error.lock().unwrap() = Some(e),
                    crate::gpu::Event::Hits { key, start, tried_before, counters } => {
                        let mut hand = Hand {
                            sh,
                            key: *key,
                            seed_index,
                            batch_start: start,
                            hit: false,
                            gpu_before: Some(tried_before),
                        };
                        for c in counters {
                            if hand.hit || sh.stop.load(Ordering::Acquire) {
                                break;
                            }
                            hand.candidate(c, true);
                        }
                        hand.key.zeroize();
                    }
                }
            }
            if sh.stop.load(Ordering::Acquire) {
                break;
            }
        }
        let now = Instant::now();
        let elapsed = now.duration_since(opts.start);
        let mut wait = Duration::from_millis(250);
        if let Some(limit) = opts.timeout {
            if elapsed >= limit {
                sh.timed_out.store(true, Ordering::Release);
                sh.stop.store(true, Ordering::Release);
                break;
            }
            wait = wait.min(limit - elapsed);
        }
        if opts.progress && elapsed >= Duration::from_millis(400) {
            let dt = now.duration_since(last_print).as_secs_f64();
            if !shown || dt >= 0.5 {
                let attempts = sh.total();
                if shown && dt > 0.0 {
                    let inst = (attempts - last_attempts) as f64 / dt;
                    rate = if rate == 0.0 { inst } else { 0.7 * rate + 0.3 * inst };
                } else {
                    rate = attempts as f64 / elapsed.as_secs_f64().max(1e-9);
                }
                last_print = now;
                last_attempts = attempts;
                shown = true;
                let mut line = format!(
                    "\r{}  {} keys/s  {} tried",
                    human_duration(elapsed),
                    human_count(rate),
                    human_count(attempts as f64)
                );
                if let Some(e) = expected {
                    if e.is_finite() && rate > 0.0 {
                        let done = 1.0 - (-(attempts as f64) / e).exp();
                        let median = e * std::f64::consts::LN_2 / rate;
                        line.push_str(&format!(
                            "  {:.0}% likely found by now  (typical: {})",
                            done * 100.0,
                            human_duration(Duration::from_secs_f64(median.min(1e15)))
                        ));
                    }
                }
                line.push_str("    ");
                let mut err = std::io::stderr().lock();
                let _ = err.write_all(line.as_bytes());
                let _ = err.flush();
            }
        }
        thread::park_timeout(wait);
    }
    if shown {
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(format!("\r{:width$}\r", "", width = 110).as_bytes());
        let _ = err.flush();
    }
}

/// Run a search. `on_found` is called (serialised) for every match as soon
/// as it has been verified.
pub fn run(opts: &Options, on_found: &(dyn Fn(&Found) + Sync)) -> Summary {
    let threads = opts.threads.max(1);
    let expected = opts.matcher.expected_attempts().unwrap_or(f64::INFINITY) * opts.count as f64;
    let workers =
        if expected < SINGLE_THREAD_BELOW && opts.timeout.is_none() { 1 } else { threads };

    let (window, forced) = match opts.window {
        Some(0) => (None, false),
        Some(w) => (Some(w), true),
        None => (choose_window(expected), false),
    };
    let builder = window.filter(|&w| w != STATIC_WINDOW).map(Builder::new);

    // GPU: only for patterns its kernel can filter, and (unless forced)
    // only for searches that outlast its start-up.
    let mut gpu_cfg = None;
    let mut gpu_builder = None;
    let wanted = match opts.gpu.mode {
        GpuMode::Off => false,
        GpuMode::On => true,
        GpuMode::Auto => expected >= GPU_FROM,
    };
    if wanted && crate::gpu::driver_present() {
        let long = expected >= GPU_LONG_FROM;
        let g_threads =
            opts.gpu.threads.unwrap_or(if long { crate::gpu::LONG_THREADS } else { crate::gpu::DEFAULT_THREADS });
        let g_batch = opts.gpu.batch.unwrap_or(if long { crate::gpu::LONG_BATCH } else { crate::gpu::DEFAULT_BATCH });
        if let Some(filter) = crate::gpu::Filter::for_matcher(opts.matcher, g_threads as u64 * g_batch as u64) {
            let seed_index = match opts.kind {
                KeyKind::Seed { index } => Some(index),
                KeyKind::Private => None,
            };
            gpu_cfg = Some(crate::gpu::Config { filter, seed_index, threads: g_threads, batch: g_batch });
            let gw = opts.gpu.window.unwrap_or(crate::gpu::DEFAULT_WINDOW);
            if window != Some(gw) {
                gpu_builder = Some(Builder::new(gw));
            }
        }
    }
    let use_gpu = gpu_cfg.is_some();

    let sh = Shared {
        opts,
        on_found,
        stop: AtomicBool::new(false),
        timed_out: AtomicBool::new(false),
        attempts: AtomicU64::new(0),
        found: Mutex::new(0),
        mismatches: AtomicU64::new(0),
        table: AtomicPtr::new(static_table() as *const Table as *mut Table),
        builder,
        aux: OnceLock::new(),
        gpu_cfg: Mutex::new(gpu_cfg),
        gpu_builder,
        gpu_tried: Arc::new(AtomicU64::new(0)),
        gpu_name: Mutex::new(None),
        gpu_error: Mutex::new(None),
    };

    // With a single worker (or an explicitly requested table) the table is
    // built up front; otherwise the helpers build it while this thread
    // already searches with the embedded one.
    if sh.builder.is_some() && (forced || workers == 1) {
        let helpers = if forced { threads } else { 1 };
        thread::scope(|s| {
            for _ in 1..helpers {
                s.spawn(|| sh.help_build());
            }
            sh.help_build();
        });
    }

    let search_start = Instant::now();
    let need_aux = workers > 1 || use_gpu || opts.timeout.is_some() || opts.progress && expected >= 1e6;
    thread::scope(|s| {
        let sh = &sh;
        // The GPU takes longest to get going, so its thread goes first.
        if use_gpu {
            s.spawn(move || auxiliary(sh));
        }
        // This thread starts the others itself: a ninth thread doing it
        // would hold a core that one of the workers then waits for.
        for _ in 1..workers {
            s.spawn(move || worker(sh, true));
        }
        if need_aux && !use_gpu {
            s.spawn(move || auxiliary(sh));
        }
        worker(sh, false);
    });

    let table = unsafe { &*sh.table.load(Ordering::Acquire) };
    let found = *sh.found.lock().unwrap();
    let gpu_attempts = sh.gpu_tried.load(Ordering::Relaxed);
    let gpu = sh.gpu_name.lock().unwrap().take();
    let gpu_error = sh.gpu_error.lock().unwrap().take();
    Summary {
        gpu,
        gpu_attempts,
        gpu_error,
        attempts: sh.attempts.load(Ordering::Relaxed) + gpu_attempts,
        elapsed: opts.start.elapsed(),
        search_elapsed: search_start.elapsed(),
        found,
        timed_out: sh.timed_out.load(Ordering::Acquire),
        threads: workers,
        backend: opts.backend,
        window: table.layout.w,
        mismatches: sh.mismatches.load(Ordering::Relaxed),
    }
}
