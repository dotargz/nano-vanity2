//! GPU search on NVIDIA cards through the CUDA driver API.
//!
//! The driver library (`nvcuda.dll` / `libcuda.so`) ships with the graphics
//! driver and is loaded at run time, so the program also runs on machines
//! without it. The kernel (see `kernel.cu`) is embedded as PTX.
//!
//! The GPU only ever *nominates* candidates: every hit is re-derived on the
//! CPU by the reference implementation before it is shown, exactly like
//! hits from the CPU engines.

use crate::pattern::Matcher;
use crate::table::Table;
use crate::tablegen::{Layout, ENTRY_WORDS};
use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::Thread;
use std::time::Duration;
use zeroize::{Zeroize, Zeroizing};

static PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

const MAX_BATCH: u32 = 64;
const MAX_FIX: usize = 60;
const MAX_SEG: usize = 8;
const MAX_FLOAT: usize = 64;
const MAX_HITS: u32 = 4096;
const BLOCK: u32 = 128;
/// Words per table entry on the GPU: three 256-bit field elements.
pub const GPU_ENTRY_WORDS: usize = 12;

/// Default work shape: threads per launch and keys per thread. Larger
/// launches are a little more efficient, smaller ones report sooner.
pub const DEFAULT_THREADS: u32 = 1 << 17;
pub const DEFAULT_BATCH: u32 = 32;
pub const LONG_THREADS: u32 = 1 << 18;
pub const LONG_BATCH: u32 = 64;
/// Default table window on the GPU.
pub const DEFAULT_WINDOW: u32 = 16;

/// Mirrors `struct Params` in kernel.cu.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    key: [u64; 4],
    off: [u64; 4],
    m0: u64,
    v0: u64,
    mc: u64,
    vc: u64,
    index_be: u32,
    w: u32,
    n: u32,
    batch: u32,
    max_hits: u32,
    dump: u32,
    nfix: u32,
    nseg: u32,
    float_lo: u32,
    float_hi: u32,
    fix_pos: [u32; MAX_FIX],
    fix_mask: [u32; MAX_FIX],
    seg_len: [u32; MAX_SEG],
    fl_mask: [u32; MAX_FLOAT],
}

/// What the kernel tests on every key. Hits are a superset of the real
/// matches; the CPU settles the rest.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// Compute complete keys and checksums (otherwise the first 8 bytes).
    pub full: bool,
    pub m0: u64,
    pub v0: u64,
    pub mc: u64,
    pub vc: u64,
    /// (address position, allowed symbols)
    pub fix: Vec<(u8, u32)>,
    pub segments: Vec<Vec<u32>>,
    pub float_lo: u32,
    pub float_hi: u32,
}

impl Filter {
    /// The filter for a pattern, if the kernel can express it selectively
    /// enough that a launch of `keys` keys reports only a handful of hits.
    pub fn for_matcher(m: &Matcher, keys: u64) -> Option<Filter> {
        const MAX_HITS_PER_LAUNCH: f64 = 8.0;
        let quick_hits = m.quick_probability() * keys as f64;
        // The first 8 bytes are enough unless the pattern has conditions
        // elsewhere and the leading ones let too many keys through.
        let full = m.has_other && quick_hits > MAX_HITS_PER_LAUNCH;
        let hits = if full { m.probability? * keys as f64 } else { quick_hits };
        if hits > MAX_HITS_PER_LAUNCH {
            return None;
        }
        let (fix, segments, float_lo, float_hi) = m.gpu_parts(full);
        let floats: usize = segments.iter().map(|s| s.len()).sum();
        if fix.len() > MAX_FIX || segments.len() > MAX_SEG || floats > MAX_FLOAT {
            return None;
        }
        let pf = m.prefilter;
        Some(Filter {
            full,
            m0: pf.m0,
            v0: pf.v0,
            mc: pf.mc,
            vc: pf.vc,
            fix,
            segments,
            float_lo: float_lo as u32,
            float_hi: float_hi as u32,
        })
    }
}

/// The table in the kernel's format: per entry three canonical field
/// elements as 256-bit little-endian integers.
pub fn convert_table(t: &Table) -> Vec<u64> {
    let entries = t.layout.entries;
    let mut out = vec![0u64; entries * GPU_ENTRY_WORDS];
    for (i, o) in out.chunks_exact_mut(GPU_ENTRY_WORDS).enumerate() {
        let e = t.entry(i as u32);
        debug_assert_eq!(e.len(), ENTRY_WORDS);
        for k in 0..3 {
            let l = &e[k * 5..k * 5 + 5];
            o[k * 4] = l[0] | (l[1] << 51);
            o[k * 4 + 1] = (l[1] >> 13) | (l[2] << 38);
            o[k * 4 + 2] = (l[2] >> 26) | (l[3] << 25);
            o[k * 4 + 3] = (l[3] >> 39) | (l[4] << 12);
        }
    }
    out
}

/// Sum of 2^(w-1) * 2^(w*i) over all digits but the top one.
fn recode_offset(l: &Layout) -> [u64; 4] {
    let mut off = [0u64; 4];
    for i in 0..l.n - 1 {
        let bit = l.w as usize * i + l.w as usize - 1;
        off[bit >> 6] |= 1u64 << (bit & 63);
    }
    off
}

type Cu = i32;

struct Api {
    _lib: libloading::Library,
    init: unsafe extern "system" fn(u32) -> Cu,
    device_get: unsafe extern "system" fn(*mut i32, i32) -> Cu,
    device_name: unsafe extern "system" fn(*mut c_char, i32, i32) -> Cu,
    ctx_create: unsafe extern "system" fn(*mut *mut c_void, u32, i32) -> Cu,
    ctx_destroy: unsafe extern "system" fn(*mut c_void) -> Cu,
    module_load: unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> Cu,
    module_function: unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> Cu,
    mem_alloc: unsafe extern "system" fn(*mut u64, usize) -> Cu,
    mem_free: unsafe extern "system" fn(u64) -> Cu,
    htod: unsafe extern "system" fn(u64, *const c_void, usize) -> Cu,
    dtoh: unsafe extern "system" fn(*mut c_void, u64, usize) -> Cu,
    memset32: unsafe extern "system" fn(u64, u32, usize) -> Cu,
    launch: unsafe extern "system" fn(
        *mut c_void,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> Cu,
    synchronize: unsafe extern "system" fn() -> Cu,
}

fn check(code: Cu, what: &str) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    let name = match code {
        2 => "out of memory",
        3 => "not initialized",
        100 => "no CUDA device",
        101 => "invalid device",
        200 => "invalid kernel image",
        218 => "invalid PTX",
        700 => "illegal address",
        719 => "launch failed",
        999 => "unknown error",
        _ => "error",
    };
    Err(format!("CUDA {what}: {name} (code {code})"))
}

impl Api {
    fn load() -> Result<Api, String> {
        let names: &[&str] =
            if cfg!(windows) { &["nvcuda.dll"] } else { &["libcuda.so.1", "libcuda.so"] };
        let lib = names
            .iter()
            .find_map(|n| unsafe { libloading::Library::new(n).ok() })
            .ok_or_else(|| "no NVIDIA CUDA driver found".to_string())?;
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|_| format!("CUDA driver lacks {}", $name))?
            };
        }
        Ok(Api {
            init: sym!("cuInit"),
            device_get: sym!("cuDeviceGet"),
            device_name: sym!("cuDeviceGetName"),
            ctx_create: sym!("cuCtxCreate_v2"),
            ctx_destroy: sym!("cuCtxDestroy_v2"),
            module_load: sym!("cuModuleLoadData"),
            module_function: sym!("cuModuleGetFunction"),
            mem_alloc: sym!("cuMemAlloc_v2"),
            mem_free: sym!("cuMemFree_v2"),
            htod: sym!("cuMemcpyHtoD_v2"),
            dtoh: sym!("cuMemcpyDtoH_v2"),
            memset32: sym!("cuMemsetD32_v2"),
            launch: sym!("cuLaunchKernel"),
            synchronize: sym!("cuCtxSynchronize"),
            _lib: lib,
        })
    }
}

/// An open device with the kernel loaded. Must stay on the thread that
/// created it (the CUDA context is bound to that thread).
pub struct Gpu {
    api: Api,
    ctx: *mut c_void,
    module: *mut c_void,
    pub name: String,
    params: Params,
    d_params: u64,
    d_table: u64,
    d_out: u64,
    out_bytes: usize,
    func: *mut c_void,
    threads: u32,
}

impl Gpu {
    pub fn open() -> Result<Gpu, String> {
        let api = Api::load()?;
        crate::trace::mark("gpu: driver library loaded");
        unsafe {
            check((api.init)(0), "init")?;
            crate::trace::mark("gpu: cuInit");
            let mut dev = 0i32;
            check((api.device_get)(&mut dev, 0), "device")?;
            let mut name = [0 as c_char; 128];
            check((api.device_name)(name.as_mut_ptr(), 127, dev), "device name")?;
            let name = std::ffi::CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let name = name.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ").to_string();
            let mut ctx = std::ptr::null_mut();
            // 4 = CU_CTX_SCHED_BLOCKING_SYNC: sleep, do not spin, while waiting.
            check((api.ctx_create)(&mut ctx, 4, dev), "context")?;
            crate::trace::mark("gpu: context created");
            let mut gpu = Gpu {
                api,
                ctx,
                module: std::ptr::null_mut(),
                name,
                params: std::mem::zeroed(),
                d_params: 0,
                d_table: 0,
                d_out: 0,
                out_bytes: 0,
                func: std::ptr::null_mut(),
                threads: 0,
            };
            let mut image = Vec::with_capacity(PTX.len() + 1);
            image.extend_from_slice(PTX);
            image.push(0);
            check((gpu.api.module_load)(&mut gpu.module, image.as_ptr() as *const c_void), "kernel load")?;
            crate::trace::mark("gpu: kernel loaded");
            check((gpu.api.mem_alloc)(&mut gpu.d_params, std::mem::size_of::<Params>()), "alloc")?;
            Ok(gpu)
        }
    }

    /// Upload a table produced by `convert_table`.
    pub fn set_table(&mut self, layout: &Layout, words: &[u64]) -> Result<(), String> {
        assert_eq!(words.len(), layout.entries * GPU_ENTRY_WORDS);
        unsafe {
            if self.d_table != 0 {
                (self.api.mem_free)(self.d_table);
                self.d_table = 0;
            }
            check((self.api.mem_alloc)(&mut self.d_table, words.len() * 8), "table alloc")?;
            check((self.api.htod)(self.d_table, words.as_ptr() as *const c_void, words.len() * 8), "table upload")?;
        }
        self.params.w = layout.w;
        self.params.n = layout.n as u32;
        self.params.off = recode_offset(layout);
        Ok(())
    }

    /// Choose kernel variant, filter and work shape.
    pub fn configure(
        &mut self,
        filter: &Filter,
        seed_index: Option<u32>,
        threads: u32,
        batch: u32,
        dump: bool,
    ) -> Result<(), String> {
        let threads = (threads.max(BLOCK) / BLOCK) * BLOCK;
        let batch = batch.clamp(1, MAX_BATCH);
        let p = &mut self.params;
        p.m0 = filter.m0;
        p.v0 = filter.v0;
        p.mc = filter.mc;
        p.vc = filter.vc;
        p.index_be = seed_index.unwrap_or(0).swap_bytes();
        p.batch = batch;
        p.max_hits = MAX_HITS;
        p.dump = dump as u32;
        p.nfix = filter.fix.len() as u32;
        for (i, &(pos, mask)) in filter.fix.iter().enumerate() {
            p.fix_pos[i] = pos as u32;
            p.fix_mask[i] = mask;
        }
        p.nseg = filter.segments.len() as u32;
        let mut k = 0;
        for (i, seg) in filter.segments.iter().enumerate() {
            p.seg_len[i] = seg.len() as u32;
            for &m in seg {
                p.fl_mask[k] = m;
                k += 1;
            }
        }
        p.float_lo = filter.float_lo;
        p.float_hi = filter.float_hi;
        self.threads = threads;

        let name = match (seed_index.is_some(), filter.full) {
            (false, false) => "search_pq",
            (false, true) => "search_pf",
            (true, false) => "search_sq",
            (true, true) => "search_sf",
        };
        let cname = CString::new(name).unwrap();
        let keys = threads as usize * batch as usize;
        let out_bytes = if dump {
            keys * 8 * if filter.full { 5 } else { 1 }
        } else {
            (MAX_HITS as usize + 1) * 4
        };
        unsafe {
            check((self.api.module_function)(&mut self.func, self.module, cname.as_ptr()), "kernel lookup")?;
            if out_bytes != self.out_bytes {
                if self.d_out != 0 {
                    (self.api.mem_free)(self.d_out);
                    self.d_out = 0;
                }
                check((self.api.mem_alloc)(&mut self.d_out, out_bytes), "alloc")?;
                self.out_bytes = out_bytes;
            }
            check((self.api.memset32)(self.d_out, 0, out_bytes / 4), "clear")?;
        }
        self.upload_params()
    }

    /// Keys covered by one launch.
    pub fn keys_per_launch(&self) -> u64 {
        self.threads as u64 * self.params.batch as u64
    }

    fn upload_params(&mut self) -> Result<(), String> {
        unsafe {
            check(
                (self.api.htod)(
                    self.d_params,
                    &self.params as *const Params as *const c_void,
                    std::mem::size_of::<Params>(),
                ),
                "parameter upload",
            )
        }
    }

    pub fn set_key(&mut self, key: &[u64; 4]) -> Result<(), String> {
        self.params.key = *key;
        self.upload_params()
    }

    /// Start the kernel on candidates `counter ..` (asynchronous).
    pub fn launch(&mut self, counter: u64) -> Result<(), String> {
        let mut a0 = self.d_params;
        let mut a1 = self.d_table;
        let mut a2 = self.d_out;
        let mut a3 = counter;
        let mut args: [*mut c_void; 4] = [
            &mut a0 as *mut u64 as *mut c_void,
            &mut a1 as *mut u64 as *mut c_void,
            &mut a2 as *mut u64 as *mut c_void,
            &mut a3 as *mut u64 as *mut c_void,
        ];
        unsafe {
            check(
                (self.api.launch)(
                    self.func,
                    self.threads / BLOCK,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "launch",
            )
        }
    }

    pub fn wait(&mut self) -> Result<(), String> {
        unsafe { check((self.api.synchronize)(), "kernel") }
    }

    /// Candidate indexes (relative to the launch counter) reported by the
    /// finished kernel, sorted; clears the list on the device.
    pub fn take_hits(&mut self) -> Result<Vec<u32>, String> {
        let mut count = 0u32;
        unsafe {
            check((self.api.dtoh)(&mut count as *mut u32 as *mut c_void, self.d_out, 4), "read")?;
            if count == 0 {
                return Ok(Vec::new());
            }
            let n = count.min(MAX_HITS) as usize;
            let mut hits = vec![0u32; n];
            check((self.api.dtoh)(hits.as_mut_ptr() as *mut c_void, self.d_out + 4, n * 4), "read")?;
            check((self.api.memset32)(self.d_out, 0, 1), "clear")?;
            hits.sort_unstable();
            Ok(hits)
        }
    }

    /// Debugging: per key the first 8 public key bytes (quick) or the four
    /// key words and checksum (full), for one launch. Needs `dump` set.
    pub fn dump(&mut self, counter: u64) -> Result<Vec<u64>, String> {
        assert!(self.params.dump != 0);
        self.launch(counter)?;
        self.wait()?;
        let mut out = vec![0u64; self.out_bytes / 8];
        unsafe {
            check((self.api.dtoh)(out.as_mut_ptr() as *mut c_void, self.d_out, self.out_bytes), "read")?;
        }
        Ok(out)
    }

    /// Overwrite the secret material held on the device.
    pub fn wipe(&mut self) {
        self.params.key.zeroize();
        let _ = self.upload_params();
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.wipe();
        unsafe {
            for p in [self.d_params, self.d_table, self.d_out] {
                if p != 0 {
                    (self.api.mem_free)(p);
                }
            }
            (self.api.ctx_destroy)(self.ctx);
        }
    }
}

/// Is a CUDA driver installed? Only looks for the library file: loading
/// it takes milliseconds, which the caller's thread should not spend.
pub fn driver_present() -> bool {
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
        std::path::Path::new(&root).join("System32").join("nvcuda.dll").exists()
    } else {
        ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/lib/wsl/lib", "/usr/local/cuda/lib64"]
            .iter()
            .any(|d| std::path::Path::new(d).join("libcuda.so.1").exists())
    }
}

// ------------------------------------------------------------- sessions

pub struct Config {
    pub filter: Filter,
    pub seed_index: Option<u32>,
    pub threads: u32,
    pub batch: u32,
}

pub enum Event {
    /// The device is searching.
    Ready { name: String },
    /// Candidates `key[0] + c` for each `c` in `counters` passed the kernel's
    /// filter. `tried_before` is the session's key count before the launch
    /// that found them and `start` that launch's first counter.
    Hits { key: Zeroizing<[u64; 4]>, start: u64, tried_before: u64, counters: Vec<u64> },
    Failed(String),
}

/// A search running on a background thread.
pub struct Session {
    stop: Arc<AtomicBool>,
    tried: Arc<AtomicU64>,
    pub events: mpsc::Receiver<Event>,
    table: mpsc::Sender<(Layout, Vec<u64>)>,
}

impl Session {
    /// Start initialising the device. The search begins once `give_table`
    /// has been called. `notify` is unparked whenever an event is queued.
    pub fn start(cfg: Config, notify: Thread, tried: Arc<AtomicU64>) -> Session {
        let stop = Arc::new(AtomicBool::new(false));
        let (etx, erx) = mpsc::channel();
        let (ttx, trx) = mpsc::channel();
        let (stop2, tried2) = (stop.clone(), tried.clone());
        let spawned = std::thread::Builder::new().name("gpu".into()).spawn(move || {
            let etx2 = etx.clone();
            if let Err(e) = session_main(cfg, &stop2, &tried2, etx, trx, &notify) {
                let _ = etx2.send(Event::Failed(e));
                notify.unpark();
            }
        });
        if spawned.is_err() {
            stop.store(true, Ordering::Release);
        }
        Session { stop, tried, events: erx, table: ttx }
    }

    pub fn give_table(&self, layout: Layout, words: Vec<u64>) {
        let _ = self.table.send((layout, words));
    }

    /// Keys tried on the device so far.
    pub fn tried(&self) -> u64 {
        self.tried.load(Ordering::Relaxed)
    }

    /// Ask the thread to finish; it cleans up on its own.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The GPU thread only runs for microseconds between launches, but every
/// delay there idles the whole device; let it preempt the CPU workers.
fn raise_priority() {
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentThread() -> isize;
            fn SetThreadPriority(thread: isize, priority: i32) -> i32;
        }
        SetThreadPriority(GetCurrentThread(), 2); // THREAD_PRIORITY_HIGHEST
    }
}

fn session_main(
    cfg: Config,
    stop: &AtomicBool,
    tried: &AtomicU64,
    events: mpsc::Sender<Event>,
    table: mpsc::Receiver<(Layout, Vec<u64>)>,
    notify: &Thread,
) -> Result<(), String> {
    raise_priority();
    let mut gpu = Gpu::open()?;
    let (layout, words) = loop {
        if stop.load(Ordering::Acquire) {
            return Ok(());
        }
        match table.recv_timeout(Duration::from_millis(20)) {
            Ok(t) => break t,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(()),
        }
    };
    gpu.set_table(&layout, &words)?;
    drop(words);
    crate::trace::mark("gpu: table uploaded");
    gpu.configure(&cfg.filter, cfg.seed_index, cfg.threads, cfg.batch, false)?;
    let mut key = Zeroizing::new(crate::search::fresh_key());
    gpu.set_key(&key)?;
    // Short launches first, so that an early hit is reported early.
    let full_threads = gpu.threads;
    gpu.threads = (full_threads / 8).max(BLOCK * 32).min(full_threads);
    let _ = events.send(Event::Ready { name: gpu.name.clone() });
    notify.unpark();
    let mut counter = 0u64;
    let mut first = true;
    while !stop.load(Ordering::Acquire) {
        let keys = gpu.keys_per_launch();
        gpu.launch(counter)?;
        gpu.wait()?;
        gpu.threads = (gpu.threads * 2).min(full_threads);
        if first {
            crate::trace::mark("gpu: first launch done");
            first = false;
        }
        let hits = gpu.take_hits()?;
        let before = tried.fetch_add(keys, Ordering::Relaxed);
        if hits.is_empty() {
            counter = counter.wrapping_add(keys);
            continue;
        }
        // Never derive two published keys from one random base: hand the
        // base over and continue with a fresh one.
        let old = std::mem::replace(&mut key, Zeroizing::new(crate::search::fresh_key()));
        gpu.set_key(&key)?;
        let counters = hits.iter().map(|&h| counter.wrapping_add(h as u64)).collect();
        if events.send(Event::Hits { key: old, start: counter, tried_before: before, counters }).is_err() {
            break;
        }
        notify.unpark();
        counter = 0;
    }
    Ok(())
}

// ---------------------------------------------------------- diagnostics

/// Compare the kernel's output with the CPU for a few thousand keys in
/// every mode. Returns the device name.
pub fn self_check(table: &Table) -> Result<String, String> {
    use crate::engine::scalar;
    let mut gpu = Gpu::open()?;
    gpu.set_table(&table.layout, &convert_table(table))?;
    let key = [0x0123_4567_89ab_cdefu64, 0xfeed_face_cafe_beef, 0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00];
    let counter = 0xffff_ffff_ffff_ff00u64; // wraps during the run
    for seed_index in [None, Some(0u32), Some(0x0102_0304)] {
        for full in [false, true] {
            let filter = Filter { full, ..Filter::default() };
            gpu.configure(&filter, seed_index, 256, 20, true)?;
            gpu.set_key(&key)?;
            let out = gpu.dump(counter)?;
            let per = if full { 5 } else { 1 };
            for i in 0..gpu.keys_per_launch() as usize {
                let secret = [key[0].wrapping_add(counter.wrapping_add(i as u64)), key[1], key[2], key[3]];
                let pk = scalar::derive(crate::table::static_table(), &secret, seed_index);
                let got = &out[i * per..(i + 1) * per];
                let good = if full {
                    got[..4] == pk && got[4] == scalar::checksum_words(&pk)
                } else {
                    got[0] == pk[0]
                };
                if !good {
                    return Err(format!(
                        "GPU result differs from CPU (key {i}, seed index {seed_index:?}, full {full}, window {})",
                        table.layout.w
                    ));
                }
            }
        }
    }
    Ok(gpu.name.clone())
}

/// Raw kernel throughput in keys per second.
pub fn benchmark(
    table: &Table,
    seed: bool,
    full: bool,
    threads: u32,
    batch: u32,
    seconds: f64,
) -> Result<(String, f64, f64), String> {
    let mut gpu = Gpu::open()?;
    gpu.set_table(&table.layout, &convert_table(table))?;
    // A filter that (practically) never matches.
    let filter = Filter { full, m0: u64::MAX, v0: 0x5a5a_5a5a_5a5a_5a5a, ..Filter::default() };
    gpu.configure(&filter, seed.then_some(0), threads, batch, false)?;
    gpu.set_key(&crate::search::fresh_key())?;
    let keys = gpu.keys_per_launch();
    // Warm-up launch.
    gpu.launch(0)?;
    gpu.wait()?;
    let t = std::time::Instant::now();
    let mut done = 0u64;
    let mut counter = keys;
    while t.elapsed().as_secs_f64() < seconds {
        gpu.launch(counter)?;
        gpu.wait()?;
        counter += keys;
        done += keys;
    }
    let el = t.elapsed().as_secs_f64();
    gpu.take_hits()?;
    Ok((gpu.name.clone(), done as f64 / el, el / (done / keys) as f64))
}
