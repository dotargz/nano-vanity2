//! Lookup tables at run time: the table embedded in the binary (usable
//! immediately) and larger tables built on demand for long searches.

use crate::tablegen::{self, Bases, Layout, ENTRY_BYTES, ENTRY_WORDS};
use std::alloc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Window width of the embedded table. Must match `build.rs`.
pub const STATIC_WINDOW: u32 = 8;

pub struct Table {
    pub layout: Layout,
    /// 64-byte aligned, `layout.entries * 16` little-endian words.
    pub data: *const u64,
}

// The table is immutable once published.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    pub fn bytes(&self) -> usize {
        self.layout.bytes()
    }

    #[inline(always)]
    pub fn entry(&self, index: u32) -> &[u64; ENTRY_WORDS] {
        debug_assert!((index as usize) < self.layout.entries);
        unsafe { &*(self.data.add(index as usize * ENTRY_WORDS) as *const [u64; ENTRY_WORDS]) }
    }
}

#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

static STATIC_BYTES: &Aligned<[u8]> =
    &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/static_table.bin")));

#[cfg(not(target_endian = "little"))]
compile_error!("the embedded table is stored little endian");

/// The table embedded in the executable.
pub fn static_table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(|| {
        let layout = Layout::new(STATIC_WINDOW);
        assert_eq!(STATIC_BYTES.0.len(), layout.entries * ENTRY_BYTES);
        Table { layout, data: STATIC_BYTES.0.as_ptr() as *const u64 }
    })
}

/// Page-aligned heap buffer of u64 words.
pub struct AlignedBuf {
    ptr: *mut u64,
    words: usize,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(words: usize) -> AlignedBuf {
        let layout = alloc::Layout::from_size_align(words.max(1) * 8, 4096).unwrap();
        let ptr = unsafe { alloc::alloc_zeroed(layout) } as *mut u64;
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        AlignedBuf { ptr, words }
    }

    pub fn as_ptr(&self) -> *mut u64 {
        self.ptr
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = alloc::Layout::from_size_align(self.words.max(1) * 8, 4096).unwrap();
        unsafe { alloc::dealloc(self.ptr as *mut u8, layout) }
    }
}

/// A table under construction. Any number of threads may call `help`; the
/// table may be used once `is_ready` returns true.
pub struct Builder {
    table: Table,
    bases: Bases,
    _buf: AlignedBuf,
    next: AtomicUsize,
    done: AtomicUsize,
}

impl Builder {
    pub fn new(w: u32) -> Builder {
        let layout = Layout::new(w);
        let buf = AlignedBuf::new(layout.entries * ENTRY_WORDS);
        Builder {
            table: Table { layout, data: buf.as_ptr() },
            bases: tablegen::bases(&layout),
            _buf: buf,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        }
    }

    /// Build sub-tables until none are left to claim. Returns true if this
    /// call completed the table.
    pub fn help(&self) -> bool {
        let l = &self.table.layout;
        let mut finished = false;
        loop {
            // Largest sub-tables are all the same size; claim in order.
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= l.n {
                break;
            }
            let words = l.count(i) * ENTRY_WORDS;
            // Safety: sub-table regions are disjoint and each index is
            // claimed by exactly one thread.
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    (self.table.data as *mut u64).add(l.base[i] as usize * ENTRY_WORDS),
                    words,
                )
            };
            tablegen::gen_subtable(l, &self.bases, i, out);
            if self.done.fetch_add(1, Ordering::AcqRel) + 1 == l.n {
                finished = true;
            }
        }
        finished
    }

    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire) == self.table.layout.n
    }

    /// The table; only valid for use once `is_ready()`.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Build completely on the calling thread.
    pub fn build_now(w: u32) -> Builder {
        let b = Builder::new(w);
        b.help();
        b
    }
}
