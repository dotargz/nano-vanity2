//! Generation of the fixed-base lookup tables used by the search engines.
//!
//! A clamped ed25519 scalar always has the form `a = 2^254 + 8*r` with a
//! 251-bit `r`. We write `r` in signed base-2^w digits `d_0 .. d_{n-1}` and
//! compute
//!
//! ```text
//!   a*B = (2^254*B + d_{n-1} * 2^(w(n-1)) * 8B)  +  sum_{i<n-1} d_i * 2^(w*i) * 8B
//! ```
//!
//! so one table lookup per digit and `n-1` mixed additions give the public
//! key, with no doublings at all. The top digit is never negative and has
//! the constant folded into its sub-table; the other digits are signed, so
//! their sub-tables only store the multiples `0 ..= 2^(w-1)`.
//!
//! Entry layout: 16 little-endian u64 words (128 bytes, two cache lines):
//! three canonical radix-2^51 field elements (5 limbs each) plus padding.
//!   * ordinary sub-tables: `((y+x)/2, (y-x)/2, d*x*y)`  ("halved Niels")
//!   * top sub-table:        `(y+x, y-x, x*y)`            (ready-made start point)
//!
//! Also compiled into the build script; may only depend on `fe51`/`edwards`.

use crate::edwards::{Point, D2};
use crate::fe51::{batch_invert, Fe};

pub const MAX_DIGITS: usize = 64;
pub const ENTRY_WORDS: usize = 16;
pub const ENTRY_BYTES: usize = ENTRY_WORDS * 8;
/// Number of free bits in a clamped scalar (bits 3..=253).
pub const SCALAR_BITS: u32 = 251;
pub const MIN_WINDOW: u32 = 4;
pub const MAX_WINDOW: u32 = 18;

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// Window width in bits.
    pub w: u32,
    /// Number of digits.
    pub n: usize,
    /// Number of scalar bits covered by the top digit.
    pub top_bits: u32,
    /// First entry index of each digit's sub-table.
    pub base: [u32; MAX_DIGITS],
    /// Total number of entries.
    pub entries: usize,
}

impl Layout {
    pub fn new(w: u32) -> Layout {
        assert!((MIN_WINDOW..=MAX_WINDOW).contains(&w), "window out of range");
        let n = SCALAR_BITS.div_ceil(w) as usize;
        let top_bits = SCALAR_BITS - w * (n as u32 - 1);
        let mut l = Layout { w, n, top_bits, base: [0; MAX_DIGITS], entries: 0 };
        let mut acc = 0u32;
        for i in 0..n {
            l.base[i] = acc;
            acc += l.count(i) as u32;
        }
        l.entries = acc as usize;
        l
    }

    /// Number of entries in sub-table `i`.
    pub fn count(&self, i: usize) -> usize {
        if i == self.n - 1 {
            (1usize << self.top_bits) + 1
        } else {
            (1usize << (self.w - 1)) + 1
        }
    }

    pub fn bytes(&self) -> usize {
        self.entries * ENTRY_BYTES
    }
}

/// Per-digit base points and shared constants.
pub struct Bases {
    /// Affine `2^(w*i) * 8B` for each digit.
    pub p: Vec<(Fe, Fe)>,
    /// `2^254 * B`
    pub c: Point,
    pub inv2: Fe,
    pub d4: Fe,
}

pub fn bases(l: &Layout) -> Bases {
    let mut pts = Vec::with_capacity(l.n);
    let mut cur = Point::BASE.double().double().double(); // 8B
    let mut k = 0u32; // cur = 2^k * 8B
    for i in 0..l.n {
        let target = l.w * i as u32;
        while k < target {
            cur = cur.double();
            k += 1;
        }
        pts.push(cur);
    }
    while k < SCALAR_BITS {
        cur = cur.double();
        k += 1;
    }
    let c = cur; // 2^251 * 8B = 2^254 * B

    let mut zs: Vec<Fe> = pts.iter().map(|p| p.z).collect();
    batch_invert(&mut zs);
    let p = pts.iter().zip(zs.iter()).map(|(p, zi)| (p.x.mul(zi), p.y.mul(zi))).collect();

    let two = Fe([2, 0, 0, 0, 0]);
    Bases { p, c, inv2: two.invert(), d4: D2.add(&D2) }
}

fn write_entry(out: &mut [u64], e: &[Fe; 3]) {
    for (k, fe) in e.iter().enumerate() {
        out[k * 5..k * 5 + 5].copy_from_slice(&fe.canonical().0);
    }
    out[15] = 0;
}

/// Fill sub-table `i`. `out` must hold exactly `l.count(i) * ENTRY_WORDS` words.
pub fn gen_subtable(l: &Layout, b: &Bases, i: usize, out: &mut [u64]) {
    let top = i == l.n - 1;
    let count = l.count(i);
    assert_eq!(out.len(), count * ENTRY_WORDS);

    let (px, py) = b.p[i];
    // The digit's base point in halved Niels form, used to walk the chain.
    let s_ypx = py.add(&px).mul(&b.inv2);
    let s_ymx = py.sub(&px).mul(&b.inv2);
    let s_xyd = px.mul(&py).mul(&b.d4).mul(&b.inv2).mul(&b.inv2);

    let mut pts: Vec<Point> = Vec::with_capacity(count);
    if top {
        // j*P + 2^254*B for j = 0 ..= 2^top_bits
        let mut acc = b.c;
        for j in 0..count {
            pts.push(acc);
            if j + 1 < count {
                acc = acc.add_halved(&s_ypx, &s_ymx, &s_xyd);
            }
        }
    } else {
        // j*P for j = 0 ..= 2^(w-1)
        pts.push(Point::IDENTITY);
        let mut acc = Point::from_affine(&px, &py);
        for j in 1..count {
            pts.push(acc);
            if j + 1 < count {
                acc = acc.add_halved(&s_ypx, &s_ymx, &s_xyd);
            }
        }
    }

    let mut zinv: Vec<Fe> = pts.iter().map(|p| p.z).collect();
    batch_invert(&mut zinv);

    for j in 0..count {
        let e = if top {
            let x = pts[j].x.mul(&zinv[j]);
            let y = pts[j].y.mul(&zinv[j]);
            [y.add(&x), y.sub(&x), x.mul(&y)]
        } else {
            let zh = zinv[j].mul(&b.inv2);
            let xh = pts[j].x.mul(&zh);
            let yh = pts[j].y.mul(&zh);
            [yh.add(&xh), yh.sub(&xh), xh.mul(&yh).mul(&b.d4)]
        };
        write_entry(&mut out[j * ENTRY_WORDS..(j + 1) * ENTRY_WORDS], &e);
    }
}

/// Generate a complete table (single threaded).
pub fn generate(l: &Layout) -> Vec<u64> {
    let b = bases(l);
    let mut data = vec![0u64; l.entries * ENTRY_WORDS];
    for i in 0..l.n {
        let lo = l.base[i] as usize * ENTRY_WORDS;
        let hi = lo + l.count(i) * ENTRY_WORDS;
        gen_subtable(l, &b, i, &mut data[lo..hi]);
    }
    data
}
