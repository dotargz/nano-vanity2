//! AVX-512 IFMA engine: eight candidate keys per vector.
//!
//! Field elements are five 51-bit limbs in the 64-bit lanes of a zmm
//! register and are multiplied with the 52-bit multiply-add instructions.
//! Additions and subtractions are folded into the (unreduced) product
//! accumulators, so every multiplier input is the direct output of a single
//! carry pass and no stand-alone reductions are needed:
//!
//! * `Fe8`  "carried": limbs < 2^51 + 2^15 (or < 2^52 for table operands)
//! * `Lz`   "lazy":    limbs < 2^61, only valid as input to `carry`
//!
//! Per vector of eight keys: 8-way BLAKE2b, signed-digit recoding, then one
//! table lookup and one 7-multiplication mixed addition per digit. Only the
//! numerator and denominator of the affine y coordinate are kept; a whole
//! batch of vectors shares one field inversion (Montgomery's trick).

use super::{Handler, Job, MAX_UNITS};
use crate::blake2b::{IV, SIGMA};
use crate::fe51::MASK51;
use crate::tablegen::{Layout, MAX_DIGITS};
use core::arch::x86_64::*;
use zeroize::Zeroize;

type V = __m512i;

/// Tables larger than this are not expected to stay in the L2 cache, so the
/// entries for the next vector are prefetched while the current one runs.
const PREFETCH_MIN_BYTES: usize = 640 * 1024;

#[derive(Clone, Copy)]
#[repr(transparent)]
struct Fe8([V; 5]);

#[derive(Clone, Copy)]
struct Lz([V; 5]);

#[repr(C, align(64))]
struct Digits {
    /// Byte offset of each lane's table entry, per digit.
    off: [[u64; 8]; MAX_DIGITS],
    /// Lanes whose digit is negative, per digit.
    neg: [u8; MAX_DIGITS],
}

#[repr(C, align(64))]
pub struct Scratch {
    a: [Fe8; MAX_UNITS],
    b: [Fe8; MAX_UNITS],
    c: [Fe8; MAX_UNITS],
    p: [Fe8; MAX_UNITS],
    dig: [Digits; 2],
}

impl Scratch {
    pub fn new() -> Box<Scratch> {
        let layout = std::alloc::Layout::new::<Scratch>();
        // Safety: all-zero bytes are a valid Scratch.
        unsafe {
            let p = std::alloc::alloc_zeroed(layout) as *mut Scratch;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(p)
        }
    }

    pub fn wipe(&mut self) {
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(self as *mut Scratch as *mut u8, std::mem::size_of::<Scratch>())
        };
        bytes.zeroize();
    }
}

#[inline(always)]
unsafe fn set1(x: u64) -> V {
    _mm512_set1_epi64(x as i64)
}
#[inline(always)]
unsafe fn add(a: V, b: V) -> V {
    _mm512_add_epi64(a, b)
}
#[inline(always)]
unsafe fn sub(a: V, b: V) -> V {
    _mm512_sub_epi64(a, b)
}
#[inline(always)]
unsafe fn and(a: V, b: V) -> V {
    _mm512_and_si512(a, b)
}
#[inline(always)]
unsafe fn or(a: V, b: V) -> V {
    _mm512_or_si512(a, b)
}
#[inline(always)]
unsafe fn xor(a: V, b: V) -> V {
    _mm512_xor_si512(a, b)
}

#[inline(always)]
unsafe fn mul19(x: V) -> V {
    _mm512_mullo_epi64(x, set1(19))
}

/// Product of two carried elements, folded to five limbs but not carried.
///
/// A partial product a_i*b_j is split by the multiplier into its low 52 bits
/// (weight 2^(51(i+j))) and its high bits (weight 2^52 = 2 * 2^51, i.e. one
/// position up and counted twice). Positions 5..9 are computed first and
/// wrap around with a factor of 19 (2^255 = 19 mod p); every accumulator
/// chain then starts from an already needed value instead of zero.
#[inline(always)]
unsafe fn mul_lazy(a: &Fe8, b: &Fe8) -> Lz {
    let [a0, a1, a2, a3, a4] = a.0;
    let [b0, b1, b2, b3, b4] = b.0;
    let z = _mm512_setzero_si512();
    macro_rules! lo {
        ($acc:expr, $x:expr, $y:expr) => {
            _mm512_madd52lo_epu64($acc, $x, $y)
        };
    }
    macro_rules! hi {
        ($acc:expr, $x:expr, $y:expr) => {
            _mm512_madd52hi_epu64($acc, $x, $y)
        };
    }

    // High halves landing on positions 5..9.
    let h5 = hi!(hi!(hi!(hi!(hi!(z, a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    let h6 = hi!(hi!(hi!(hi!(z, a1, b4), a2, b3), a3, b2), a4, b1);
    let h7 = hi!(hi!(hi!(z, a2, b4), a3, b3), a4, b2);
    let h8 = hi!(hi!(z, a3, b4), a4, b3);
    let h9 = hi!(z, a4, b4);
    // Complete positions 5..9.
    let z5 = lo!(lo!(lo!(lo!(add(h5, h5), a1, b4), a2, b3), a3, b2), a4, b1);
    let z6 = lo!(lo!(lo!(add(h6, h6), a2, b4), a3, b3), a4, b2);
    let z7 = lo!(lo!(add(h7, h7), a3, b4), a4, b3);
    let z8 = lo!(add(h8, h8), a4, b4);
    let z9 = add(h9, h9);
    // High halves landing on positions 1..4.
    let h1 = hi!(z, a0, b0);
    let h2 = hi!(hi!(z, a0, b1), a1, b0);
    let h3 = hi!(hi!(hi!(z, a0, b2), a1, b1), a2, b0);
    let h4 = hi!(hi!(hi!(hi!(z, a0, b3), a1, b2), a2, b1), a3, b0);
    // Positions 0..4 on top of the wrapped positions.
    let r0 = lo!(mul19(z5), a0, b0);
    let r1 = lo!(lo!(add(mul19(z6), add(h1, h1)), a0, b1), a1, b0);
    let r2 = lo!(lo!(lo!(add(mul19(z7), add(h2, h2)), a0, b2), a1, b1), a2, b0);
    let r3 = lo!(lo!(lo!(lo!(add(mul19(z8), add(h3, h3)), a0, b3), a1, b2), a2, b1), a3, b0);
    let r4 = lo!(lo!(lo!(lo!(lo!(add(mul19(z9), add(h4, h4)), a0, b4), a1, b3), a2, b2), a3, b1), a4, b0);
    Lz([r0, r1, r2, r3, r4])
}

/// One parallel carry pass: lazy -> carried.
#[inline(always)]
unsafe fn carry(r: Lz) -> Fe8 {
    let m = set1(MASK51);
    let [r0, r1, r2, r3, r4] = r.0;
    let c0 = _mm512_srli_epi64::<51>(r0);
    let c1 = _mm512_srli_epi64::<51>(r1);
    let c2 = _mm512_srli_epi64::<51>(r2);
    let c3 = _mm512_srli_epi64::<51>(r3);
    let c4 = _mm512_srli_epi64::<51>(r4);
    Fe8([
        // c4 is tiny, so the multiply-add does the wrap-around in one step.
        _mm512_madd52lo_epu64(and(r0, m), c4, set1(19)),
        add(and(r1, m), c0),
        add(and(r2, m), c1),
        add(and(r3, m), c2),
        add(and(r4, m), c3),
    ])
}

#[inline(always)]
unsafe fn lz_add(a: &Lz, b: &Lz) -> Lz {
    Lz([add(a.0[0], b.0[0]), add(a.0[1], b.0[1]), add(a.0[2], b.0[2]), add(a.0[3], b.0[3]), add(a.0[4], b.0[4])])
}

/// a - b + 512p, which keeps every limb non-negative.
#[inline(always)]
unsafe fn lz_sub(a: &Lz, b: &Lz) -> Lz {
    let k0 = set1((1u64 << 60) - 19 * 512);
    let kn = set1((1u64 << 60) - 512);
    Lz([
        add(sub(a.0[0], b.0[0]), k0),
        add(sub(a.0[1], b.0[1]), kn),
        add(sub(a.0[2], b.0[2]), kn),
        add(sub(a.0[3], b.0[3]), kn),
        add(sub(a.0[4], b.0[4]), kn),
    ])
}

#[inline(always)]
unsafe fn fmul(a: &Fe8, b: &Fe8) -> Fe8 {
    carry(mul_lazy(a, b))
}

#[inline(always)]
unsafe fn pow2k(a: &Fe8, k: u32) -> Fe8 {
    let mut r = *a;
    for _ in 0..k {
        r = fmul(&r, &r);
    }
    r
}

/// Eight field inversions at once (z^(p-2)).
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn invert(z: &Fe8) -> Fe8 {
    let z2 = fmul(z, z);
    let z8 = pow2k(&z2, 2);
    let z9 = fmul(z, &z8);
    let z11 = fmul(&z2, &z9);
    let z22 = fmul(&z11, &z11);
    let z_5_0 = fmul(&z9, &z22);
    let z_10_0 = fmul(&pow2k(&z_5_0, 5), &z_5_0);
    let z_20_0 = fmul(&pow2k(&z_10_0, 10), &z_10_0);
    let z_40_0 = fmul(&pow2k(&z_20_0, 20), &z_20_0);
    let z_50_0 = fmul(&pow2k(&z_40_0, 10), &z_10_0);
    let z_100_0 = fmul(&pow2k(&z_50_0, 50), &z_50_0);
    let z_200_0 = fmul(&pow2k(&z_100_0, 100), &z_100_0);
    let z_250_0 = fmul(&pow2k(&z_200_0, 50), &z_50_0);
    fmul(&pow2k(&z_250_0, 5), &z11)
}

/// Canonical limbs (each < 2^51, value < p) of a carried element.
#[inline(always)]
unsafe fn freeze(a: &Fe8) -> [V; 5] {
    let m = set1(MASK51);
    let [mut l0, mut l1, mut l2, mut l3, mut l4] = a.0;
    let mut q = _mm512_srli_epi64::<51>(add(l0, set1(19)));
    q = _mm512_srli_epi64::<51>(add(l1, q));
    q = _mm512_srli_epi64::<51>(add(l2, q));
    q = _mm512_srli_epi64::<51>(add(l3, q));
    q = _mm512_srli_epi64::<51>(add(l4, q));
    l0 = add(l0, mul19(q));
    l1 = add(l1, _mm512_srli_epi64::<51>(l0));
    l0 = and(l0, m);
    l2 = add(l2, _mm512_srli_epi64::<51>(l1));
    l1 = and(l1, m);
    l3 = add(l3, _mm512_srli_epi64::<51>(l2));
    l2 = and(l2, m);
    l4 = add(l4, _mm512_srli_epi64::<51>(l3));
    l3 = and(l3, m);
    l4 = and(l4, m);
    [l0, l1, l2, l3, l4]
}

/// 8x8 transpose of 64-bit elements.
#[inline(always)]
unsafe fn transpose(r: [V; 8]) -> [V; 8] {
    let t0 = _mm512_unpacklo_epi64(r[0], r[1]);
    let t1 = _mm512_unpackhi_epi64(r[0], r[1]);
    let t2 = _mm512_unpacklo_epi64(r[2], r[3]);
    let t3 = _mm512_unpackhi_epi64(r[2], r[3]);
    let t4 = _mm512_unpacklo_epi64(r[4], r[5]);
    let t5 = _mm512_unpackhi_epi64(r[4], r[5]);
    let t6 = _mm512_unpacklo_epi64(r[6], r[7]);
    let t7 = _mm512_unpackhi_epi64(r[6], r[7]);

    let u0 = _mm512_shuffle_i64x2::<0x88>(t0, t2);
    let u1 = _mm512_shuffle_i64x2::<0xdd>(t0, t2);
    let u2 = _mm512_shuffle_i64x2::<0x88>(t4, t6);
    let u3 = _mm512_shuffle_i64x2::<0xdd>(t4, t6);
    let u4 = _mm512_shuffle_i64x2::<0x88>(t1, t3);
    let u5 = _mm512_shuffle_i64x2::<0xdd>(t1, t3);
    let u6 = _mm512_shuffle_i64x2::<0x88>(t5, t7);
    let u7 = _mm512_shuffle_i64x2::<0xdd>(t5, t7);

    [
        _mm512_shuffle_i64x2::<0x88>(u0, u2),
        _mm512_shuffle_i64x2::<0x88>(u4, u6),
        _mm512_shuffle_i64x2::<0x88>(u1, u3),
        _mm512_shuffle_i64x2::<0x88>(u5, u7),
        _mm512_shuffle_i64x2::<0xdd>(u0, u2),
        _mm512_shuffle_i64x2::<0xdd>(u4, u6),
        _mm512_shuffle_i64x2::<0xdd>(u1, u3),
        _mm512_shuffle_i64x2::<0xdd>(u5, u7),
    ]
}

/// Gather one table entry per lane into limb-sliced form.
#[inline(always)]
unsafe fn load(data: *const u8, off: &[u64; 8]) -> ([V; 5], [V; 5], [V; 5]) {
    let p0 = data.add(off[0] as usize) as *const V;
    let p1 = data.add(off[1] as usize) as *const V;
    let p2 = data.add(off[2] as usize) as *const V;
    let p3 = data.add(off[3] as usize) as *const V;
    let p4 = data.add(off[4] as usize) as *const V;
    let p5 = data.add(off[5] as usize) as *const V;
    let p6 = data.add(off[6] as usize) as *const V;
    let p7 = data.add(off[7] as usize) as *const V;
    let a = transpose([
        _mm512_load_si512(p0),
        _mm512_load_si512(p1),
        _mm512_load_si512(p2),
        _mm512_load_si512(p3),
        _mm512_load_si512(p4),
        _mm512_load_si512(p5),
        _mm512_load_si512(p6),
        _mm512_load_si512(p7),
    ]);
    let b = transpose([
        _mm512_load_si512(p0.add(1)),
        _mm512_load_si512(p1.add(1)),
        _mm512_load_si512(p2.add(1)),
        _mm512_load_si512(p3.add(1)),
        _mm512_load_si512(p4.add(1)),
        _mm512_load_si512(p5.add(1)),
        _mm512_load_si512(p6.add(1)),
        _mm512_load_si512(p7.add(1)),
    ]);
    ([a[0], a[1], a[2], a[3], a[4]], [a[5], a[6], a[7], b[0], b[1]], [b[2], b[3], b[4], b[5], b[6]])
}

/// Table entries with the digit signs applied: negating a point swaps
/// (y+x)/2 with (y-x)/2 and negates d*x*y.
#[inline(always)]
unsafe fn load_signed(data: *const u8, off: &[u64; 8], neg: u8) -> (Fe8, Fe8, Fe8) {
    let (ypx, ymx, xyd) = load(data, off);
    let p2_0 = set1((1u64 << 52) - 38);
    let p2_n = set1((1u64 << 52) - 2);
    let mut y = ypx;
    let mut m = ymx;
    let mut x = xyd;
    for k in 0..5 {
        y[k] = _mm512_mask_blend_epi64(neg, ypx[k], ymx[k]);
        m[k] = _mm512_mask_blend_epi64(neg, ymx[k], ypx[k]);
        x[k] = _mm512_mask_sub_epi64(xyd[k], neg, if k == 0 { p2_0 } else { p2_n }, xyd[k]);
    }
    (Fe8(y), Fe8(m), Fe8(x))
}

/// Eight single-block BLAKE2b hashes; returns the first four state words.
#[inline(always)]
unsafe fn blake2b8(m: &[V; 16], len: u64, outlen: u64) -> [V; 4] {
    let h0 = set1(IV[0] ^ 0x0101_0000 ^ outlen);
    let mut v: [V; 16] = [
        h0,
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        set1(IV[4]),
        set1(IV[5]),
        set1(IV[6]),
        set1(IV[7]),
        set1(IV[0]),
        set1(IV[1]),
        set1(IV[2]),
        set1(IV[3]),
        set1(IV[4] ^ len),
        set1(IV[5]),
        set1(!IV[6]),
        set1(IV[7]),
    ];
    macro_rules! g {
        ($a:literal, $b:literal, $c:literal, $d:literal, $x:expr, $y:expr) => {
            v[$a] = add(add(v[$a], v[$b]), $x);
            v[$d] = _mm512_ror_epi64::<32>(xor(v[$d], v[$a]));
            v[$c] = add(v[$c], v[$d]);
            v[$b] = _mm512_ror_epi64::<24>(xor(v[$b], v[$c]));
            v[$a] = add(add(v[$a], v[$b]), $y);
            v[$d] = _mm512_ror_epi64::<16>(xor(v[$d], v[$a]));
            v[$c] = add(v[$c], v[$d]);
            v[$b] = _mm512_ror_epi64::<63>(xor(v[$b], v[$c]));
        };
    }
    macro_rules! round {
        ($r:literal) => {
            g!(0, 4, 8, 12, m[SIGMA[$r][0]], m[SIGMA[$r][1]]);
            g!(1, 5, 9, 13, m[SIGMA[$r][2]], m[SIGMA[$r][3]]);
            g!(2, 6, 10, 14, m[SIGMA[$r][4]], m[SIGMA[$r][5]]);
            g!(3, 7, 11, 15, m[SIGMA[$r][6]], m[SIGMA[$r][7]]);
            g!(0, 5, 10, 15, m[SIGMA[$r][8]], m[SIGMA[$r][9]]);
            g!(1, 6, 11, 12, m[SIGMA[$r][10]], m[SIGMA[$r][11]]);
            g!(2, 7, 8, 13, m[SIGMA[$r][12]], m[SIGMA[$r][13]]);
            g!(3, 4, 9, 14, m[SIGMA[$r][14]], m[SIGMA[$r][15]]);
        };
    }
    round!(0);
    round!(1);
    round!(2);
    round!(3);
    round!(4);
    round!(5);
    round!(6);
    round!(7);
    round!(8);
    round!(9);
    round!(10);
    round!(11);
    [
        xor(xor(h0, v[0]), v[8]),
        xor(xor(set1(IV[1]), v[1]), v[9]),
        xor(xor(set1(IV[2]), v[2]), v[10]),
        xor(xor(set1(IV[3]), v[3]), v[11]),
    ]
}

/// Hash the eight secrets starting at `counter` and recode their scalars
/// into table offsets and sign masks.
#[inline(always)]
unsafe fn hash_recode(job: &Job, lay: &Layout, counter: u64, d: &mut Digits) {
    let z = _mm512_setzero_si512();
    let lanes = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    let m0 = add(set1(job.key[0].wrapping_add(counter)), lanes);
    let m1 = set1(job.key[1]);
    let m2 = set1(job.key[2]);
    let m3 = set1(job.key[3]);
    let s = match job.seed_index {
        Some(index) => {
            let m4 = set1(index.swap_bytes() as u64);
            let m = [m0, m1, m2, m3, m4, z, z, z, z, z, z, z, z, z, z, z];
            let p = blake2b8(&m, 36, 32);
            let m = [p[0], p[1], p[2], p[3], z, z, z, z, z, z, z, z, z, z, z, z];
            blake2b8(&m, 32, 64)
        }
        None => {
            let m = [m0, m1, m2, m3, z, z, z, z, z, z, z, z, z, z, z, z];
            blake2b8(&m, 32, 64)
        }
    };

    // r = bits 3..=253 of the scalar (clamping fixes all the others).
    let a = [
        or(_mm512_srli_epi64::<3>(s[0]), _mm512_slli_epi64::<61>(s[1])),
        or(_mm512_srli_epi64::<3>(s[1]), _mm512_slli_epi64::<61>(s[2])),
        or(_mm512_srli_epi64::<3>(s[2]), _mm512_slli_epi64::<61>(s[3])),
        and(_mm512_srli_epi64::<3>(s[3]), set1((1u64 << 59) - 1)),
    ];
    let w = lay.w as usize;
    let n = lay.n;
    let maskw = set1((1u64 << w) - 1);
    let half = set1(1u64 << (w - 1));
    let two_w = set1(1u64 << w);
    let mut cy = z;
    for i in 0..n {
        let bit = w * i;
        let q = bit >> 6;
        let r = bit & 63;
        let mut win = _mm512_srl_epi64(a[q], _mm_cvtsi64_si128(r as i64));
        if r + w > 64 && q < 3 {
            win = or(win, _mm512_sll_epi64(a[q + 1], _mm_cvtsi64_si128((64 - r) as i64)));
        }
        let mut dd = add(and(win, maskw), cy);
        if i < n - 1 {
            let neg = _mm512_cmpgt_epu64_mask(dd, half);
            dd = _mm512_mask_sub_epi64(dd, neg, two_w, dd);
            cy = _mm512_maskz_set1_epi64(neg, 1);
            d.neg[i] = neg;
        }
        let off = _mm512_slli_epi64::<7>(add(dd, set1(lay.base[i] as u64)));
        _mm512_storeu_si512(d.off[i].as_mut_ptr() as *mut V, off);
    }
}

/// Request one digit's table entries (two cache lines each) for all lanes.
#[inline(always)]
unsafe fn prefetch_digit(data: *const u8, off: &[u64; 8]) {
    for l in 0..8 {
        let p = data.add(off[l] as usize) as *const i8;
        _mm_prefetch::<_MM_HINT_T0>(p);
        _mm_prefetch::<_MM_HINT_T0>(p.add(64));
    }
}

/// Process `nvec` vectors (8 candidates each) starting at `job.counter`.
///
/// # Safety
/// The CPU must support AVX-512 F, DQ and IFMA.
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn run<H: Handler>(job: &Job, nvec: usize, sc: &mut Scratch, h: &mut H) {
    debug_assert!((1..=MAX_UNITS).contains(&nvec));
    let tbl = job.table;
    let lay = &tbl.layout;
    let n = lay.n;
    let data = tbl.data as *const u8;
    let full = h.full();
    let pf = h.prefilter();
    let prefetch = tbl.bytes() > PREFETCH_MIN_BYTES;
    let zero = _mm512_setzero_si512();
    let one = Fe8([set1(1), zero, zero, zero, zero]);
    let Scratch { a: sa, b: sb, c: sx, p: sp, dig } = sc;

    hash_recode(job, lay, job.counter, &mut dig[0]);
    for v in 0..nvec {
        // Hash the next vector first: its table entries are then requested
        // digit by digit while this vector is being computed, a steady
        // trickle that arrives a whole vector ahead of its use.
        let ahead = prefetch && v + 1 < nvec;
        if v + 1 < nvec {
            hash_recode(job, lay, job.counter.wrapping_add(8 * (v as u64 + 1)), &mut dig[(v + 1) & 1]);
        }
        let d = &dig[v & 1];
        let nd = &dig[(v + 1) & 1];
        if ahead {
            prefetch_digit(data, &nd.off[n - 1]);
        }

        // Start from the top digit's entry: an affine point (Z = 1) stored
        // as (Y+X, Y-X, T).
        let (s0, d0, t0) = load(data, &d.off[n - 1]);
        let mut ps = Fe8(s0);
        let mut pd = Fe8(d0);
        let mut pt = Fe8(t0);
        // Z only ever enters sums, so it is never carried.
        let mut pz = Lz(one.0);

        let mut i = n - 2;
        while i > 0 {
            if ahead {
                prefetch_digit(data, &nd.off[i]);
            }
            let (ypx, ymx, xyd) = load_signed(data, &d.off[i], d.neg[i]);
            let ta = mul_lazy(&pd, &ymx);
            let tb = mul_lazy(&ps, &ypx);
            let tc = mul_lazy(&pt, &xyd);
            let fe = carry(lz_sub(&tb, &ta));
            let fh = carry(lz_add(&tb, &ta));
            let ff = carry(lz_sub(&pz, &tc));
            let fg = carry(lz_add(&pz, &tc));
            let x3 = mul_lazy(&fe, &ff);
            let y3 = mul_lazy(&fg, &fh);
            ps = carry(lz_add(&y3, &x3));
            pd = carry(lz_sub(&y3, &x3));
            pz = mul_lazy(&ff, &fg);
            pt = carry(mul_lazy(&fe, &fh));
            i -= 1;
        }

        // Last digit: only the pieces of the result that are needed.
        if ahead {
            prefetch_digit(data, &nd.off[0]);
        }
        let (ypx, ymx, xyd) = load_signed(data, &d.off[0], d.neg[0]);
        let ta = mul_lazy(&pd, &ymx);
        let tb = mul_lazy(&ps, &ypx);
        let tc = mul_lazy(&pt, &xyd);
        let fh = carry(lz_add(&tb, &ta));
        let ff = carry(lz_sub(&pz, &tc));
        if full {
            let fe = carry(lz_sub(&tb, &ta));
            let fg = carry(lz_add(&pz, &tc));
            sa[v] = fmul(&ff, &fg); // Z
            sb[v] = fmul(&fg, &fh); // Y
            sx[v] = fmul(&fe, &ff); // X
        } else {
            sa[v] = ff; // y = H / F
            sb[v] = fh;
        }
    }

    // One inversion for the whole batch.
    sp[0] = sa[0];
    for i in 1..nvec {
        sp[i] = fmul(&sp[i - 1], &sa[i]);
    }
    let mut inv = invert(&sp[nvec - 1]);

    let m0 = set1(pf.m0);
    let v0 = set1(pf.v0);
    let mut i = nvec;
    while i > 0 {
        i -= 1;
        let di = if i > 0 {
            let t = fmul(&inv, &sp[i - 1]);
            inv = fmul(&inv, &sa[i]);
            t
        } else {
            inv
        };
        let y = freeze(&fmul(&sb[i], &di));
        let w0 = or(y[0], _mm512_slli_epi64::<51>(y[1]));
        let mut k = _mm512_cmpeq_epi64_mask(and(w0, m0), v0);
        let base = job.counter.wrapping_add(8 * i as u64);
        if !full {
            if k != 0 {
                let mut out = [0u64; 8];
                _mm512_storeu_si512(out.as_mut_ptr() as *mut V, w0);
                while k != 0 {
                    let l = k.trailing_zeros() as usize;
                    k &= k - 1;
                    h.on_quick(base.wrapping_add(l as u64), out[l]);
                }
            }
            continue;
        }
        if k == 0 {
            continue;
        }
        let x = freeze(&fmul(&sx[i], &di));
        let w1 = or(_mm512_srli_epi64::<13>(y[1]), _mm512_slli_epi64::<38>(y[2]));
        let w2 = or(_mm512_srli_epi64::<26>(y[2]), _mm512_slli_epi64::<25>(y[3]));
        let w3 = or(
            or(_mm512_srli_epi64::<39>(y[3]), _mm512_slli_epi64::<12>(y[4])),
            _mm512_slli_epi64::<63>(x[0]),
        );
        let z = zero;
        let msg = [w0, w1, w2, w3, z, z, z, z, z, z, z, z, z, z, z, z];
        let cks = and(blake2b8(&msg, 32, 5)[0], set1(0xff_ffff_ffff));
        k &= _mm512_cmpeq_epi64_mask(and(cks, set1(pf.mc)), set1(pf.vc));
        if k != 0 {
            let mut o0 = [0u64; 8];
            let mut o1 = [0u64; 8];
            let mut o2 = [0u64; 8];
            let mut o3 = [0u64; 8];
            let mut oc = [0u64; 8];
            _mm512_storeu_si512(o0.as_mut_ptr() as *mut V, w0);
            _mm512_storeu_si512(o1.as_mut_ptr() as *mut V, w1);
            _mm512_storeu_si512(o2.as_mut_ptr() as *mut V, w2);
            _mm512_storeu_si512(o3.as_mut_ptr() as *mut V, w3);
            _mm512_storeu_si512(oc.as_mut_ptr() as *mut V, cks);
            while k != 0 {
                let l = k.trailing_zeros() as usize;
                k &= k - 1;
                h.on_full(base.wrapping_add(l as u64), [o0[l], o1[l], o2[l], o3[l]], oc[l]);
            }
        }
    }
}

/// Timing of the isolated primitives, for tuning (`examples/uarch.rs`).
#[doc(hidden)]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
pub unsafe fn microbench(cycle_ns: f64) {
    use std::hint::black_box;
    use std::time::Instant;
    const N: usize = 4_000_000;
    let seed = |k: u64| -> Fe8 {
        let k = black_box(k);
        Fe8([
            set1((k.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & MASK51),
            set1((k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) & MASK51),
            set1((k.wrapping_mul(0x1656_67B1_9E37_79F9)) & MASK51),
            set1((k.wrapping_mul(0x27D4_EB2F_1656_67C5)) & MASK51),
            set1((k.wrapping_mul(0x85EB_CA77_C2B2_AE63)) & MASK51),
        ])
    };
    let report = |name: &str, dt: f64, ops: usize| {
        println!("{name:44} {:7.1} cycles/op", dt * 1e9 / ops as f64 / cycle_ns);
    };

    // Latency: one dependent chain of multiplications.
    let b = seed(2);
    let mut x = seed(1);
    let t = Instant::now();
    for _ in 0..N {
        x = fmul(&x, &b);
    }
    report("fmul, 1 dependent chain", t.elapsed().as_secs_f64(), N);
    black_box(x.0);

    // Throughput: several independent chains.
    let mut x = [seed(1), seed(3), seed(4), seed(5)];
    let t = Instant::now();
    for _ in 0..N / 2 {
        x[0] = fmul(&x[0], &b);
        x[1] = fmul(&x[1], &b);
    }
    report("fmul, 2 independent chains", t.elapsed().as_secs_f64(), N);
    let t = Instant::now();
    for _ in 0..N / 4 {
        x[0] = fmul(&x[0], &b);
        x[1] = fmul(&x[1], &b);
        x[2] = fmul(&x[2], &b);
        x[3] = fmul(&x[3], &b);
    }
    report("fmul, 4 independent chains", t.elapsed().as_secs_f64(), N);
    black_box([x[0].0, x[1].0, x[2].0, x[3].0]);

    // The mixed addition without table lookups.
    let (ypx, ymx, xyd) = (seed(6), seed(7), seed(8));
    let step = |ps: &mut Fe8, pd: &mut Fe8, pz: &mut Lz, pt: &mut Fe8| {
        let ta = mul_lazy(pd, &ymx);
        let tb = mul_lazy(ps, &ypx);
        let tc = mul_lazy(pt, &xyd);
        let fe = carry(lz_sub(&tb, &ta));
        let fh = carry(lz_add(&tb, &ta));
        let ff = carry(lz_sub(pz, &tc));
        let fg = carry(lz_add(pz, &tc));
        let x3 = mul_lazy(&fe, &ff);
        let y3 = mul_lazy(&fg, &fh);
        *ps = carry(lz_add(&y3, &x3));
        *pd = carry(lz_sub(&y3, &x3));
        *pz = mul_lazy(&ff, &fg);
        *pt = carry(mul_lazy(&fe, &fh));
    };
    let (mut ps, mut pd, mut pz, mut pt) = (seed(9), seed(10), Lz(seed(11).0), seed(12));
    let t = Instant::now();
    for _ in 0..N / 4 {
        step(&mut ps, &mut pd, &mut pz, &mut pt);
    }
    report("point add, 1 chain", t.elapsed().as_secs_f64(), N / 4);
    let (mut qs, mut qd, mut qz, mut qt) = (seed(13), seed(14), Lz(seed(15).0), seed(16));
    let t = Instant::now();
    for _ in 0..N / 8 {
        step(&mut ps, &mut pd, &mut pz, &mut pt);
        step(&mut qs, &mut qd, &mut qz, &mut qt);
    }
    report("point add, 2 chains back to back", t.elapsed().as_secs_f64(), N / 4);
    black_box([ps.0, pd.0, pz.0, pt.0, qs.0, qd.0, qz.0, qt.0]);

    // Table lookup (transposing gather) alone, from the embedded table.
    let tbl = crate::table::static_table();
    let data = tbl.data as *const u8;
    let mut off = [0u64; 8];
    let mut acc = _mm512_setzero_si512();
    let mut r = 0x1234_5678_9abc_def1u64;
    let t = Instant::now();
    for _ in 0..N / 4 {
        for o in off.iter_mut() {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            *o = (r % 4000) * 128;
        }
        let (y, m, x) = load_signed(data, &off, r as u8);
        acc = add(acc, add(y.0[0], add(m.0[4], x.0[2])));
    }
    report("table lookup x8 + sign (incl. rng)", t.elapsed().as_secs_f64(), N / 4);
    black_box(acc);

    // 8-way BLAKE2b.
    let z = _mm512_setzero_si512();
    let mut m0 = seed(17).0[0];
    let t = Instant::now();
    for _ in 0..N / 8 {
        let m = [m0, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z];
        m0 = blake2b8(&m, 32, 64)[0];
    }
    report("blake2b x8 (dependent)", t.elapsed().as_secs_f64(), N / 8);
    black_box(m0);

    let inv_in = seed(18);
    let t = Instant::now();
    let mut y = inv_in;
    for _ in 0..N / 400 {
        y = invert(&y);
    }
    report("field inversion x8", t.elapsed().as_secs_f64(), N / 400);
    black_box(y.0);
}
