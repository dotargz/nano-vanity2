//! Portable engine: one key at a time with 64-bit arithmetic, but using the
//! same tricks as the vector engine (table-only fixed-base multiplication,
//! no doublings, y-only output, batched inversion).

use super::{Handler, Job};
use crate::blake2b;
use crate::fe51::Fe;
use crate::table::Table;
use crate::tablegen::MAX_DIGITS;

const BATCH: usize = 128;

/// The (unclamped) 256-bit ed25519 scalar of a secret, as little-endian words.
#[inline(always)]
pub fn scalar_words(secret: &[u64; 4], seed_index: Option<u32>) -> [u64; 4] {
    let private = match seed_index {
        Some(index) => {
            // seed || big-endian index
            let m4 = index.swap_bytes() as u64;
            let h = blake2b::hash_words([secret[0], secret[1], secret[2], secret[3], m4], 36, 32);
            [h[0], h[1], h[2], h[3]]
        }
        None => *secret,
    };
    let h = blake2b::hash_words([private[0], private[1], private[2], private[3], 0], 32, 64);
    [h[0], h[1], h[2], h[3]]
}

#[inline(always)]
fn select(mask: u64, a: &[u64], b: &[u64]) -> Fe {
    // mask = all ones selects b
    let mut r = [0u64; 5];
    for k in 0..5 {
        r[k] = (a[k] & !mask) | (b[k] & mask);
    }
    Fe(r)
}

/// Fixed-base multiplication by the clamped scalar `s` using `tbl`.
/// Returns (E, F, G, H) with x = E/G and y = H/F.
#[inline(always)]
pub fn ec(tbl: &Table, s: &[u64; 4]) -> (Fe, Fe, Fe, Fe) {
    let l = &tbl.layout;
    let w = l.w as usize;
    let n = l.n;
    // r = bits 3..=253 of the scalar
    let a = [
        (s[0] >> 3) | (s[1] << 61),
        (s[1] >> 3) | (s[2] << 61),
        (s[2] >> 3) | (s[3] << 61),
        (s[3] >> 3) & ((1u64 << 59) - 1),
    ];
    let maskw = (1u64 << w) - 1;
    let half = 1u64 << (w - 1);

    // Signed digit recoding.
    let mut idx = [0u32; MAX_DIGITS];
    let mut neg = [0u64; MAX_DIGITS];
    let mut carry = 0u64;
    for i in 0..n {
        let bit = w * i;
        let q = bit >> 6;
        let r = bit & 63;
        let mut win = a[q] >> r;
        if r + w > 64 && q < 3 {
            win |= a[q + 1] << (64 - r);
        }
        let mut d = (win & maskw) + carry;
        if i < n - 1 {
            let m = ((d > half) as u64).wrapping_neg();
            d = (d & !m) | (((1u64 << w) - d) & m);
            carry = m & 1;
            neg[i] = m;
        }
        idx[i] = l.base[i] + d as u32;
    }

    // Start point straight from the top sub-table (affine, so Z = 1).
    let e = tbl.entry(idx[n - 1]);
    let mut ps = Fe([e[0], e[1], e[2], e[3], e[4]]); // Y + X
    let mut pd = Fe([e[5], e[6], e[7], e[8], e[9]]); // Y - X
    let mut pt = Fe([e[10], e[11], e[12], e[13], e[14]]); // T
    let mut pz = Fe::ONE;

    const P2_0: u64 = (1 << 52) - 38;
    const P2_N: u64 = (1 << 52) - 2;

    let mut i = n - 2;
    loop {
        let e = tbl.entry(idx[i]);
        let m = neg[i];
        let ypx = select(m, &e[0..5], &e[5..10]);
        let ymx = select(m, &e[5..10], &e[0..5]);
        let nx = [P2_0 - e[10], P2_N - e[11], P2_N - e[12], P2_N - e[13], P2_N - e[14]];
        let xyd = select(m, &e[10..15], &nx);

        let ta = pd.mul(&ymx);
        let tb = ps.mul(&ypx);
        let tc = pt.mul(&xyd);
        let fe = tb.sub(&ta);
        let fh = tb.add(&ta);
        let ff = pz.sub(&tc);
        let fg = pz.add(&tc);
        if i == 0 {
            return (fe, ff, fg, fh);
        }
        let x3 = fe.mul(&ff);
        let y3 = fg.mul(&fh);
        ps = y3.add(&x3);
        pd = y3.sub(&x3);
        pz = ff.mul(&fg);
        pt = fe.mul(&fh);
        i -= 1;
    }
}

/// Public key words from affine coordinates.
#[inline(always)]
fn pack(x: &Fe, y: &Fe) -> [u64; 4] {
    let mut w = y.to_words();
    w[3] |= (x.canonical().0[0] & 1) << 63;
    w
}

/// Public key (little-endian words) of one secret, using the lookup table.
pub fn derive(tbl: &Table, secret: &[u64; 4], seed_index: Option<u32>) -> [u64; 4] {
    let s = scalar_words(secret, seed_index);
    let (e, f, g, h) = ec(tbl, &s);
    let zi = f.mul(&g).invert();
    let x = e.mul(&f).mul(&zi);
    let y = h.mul(&g).mul(&zi);
    pack(&x, &y)
}

#[inline(always)]
pub fn checksum_words(pk: &[u64; 4]) -> u64 {
    blake2b::hash_words([pk[0], pk[1], pk[2], pk[3], 0], 32, 5)[0] & 0xff_ffff_ffff
}

pub fn run<H: Handler>(job: &Job, count: usize, h: &mut H) {
    let full = h.full();
    let pf = h.prefilter();
    let tbl = job.table;
    let mut num = [Fe::ZERO; BATCH];
    let mut den = [Fe::ZERO; BATCH];
    let mut xnum = [Fe::ZERO; BATCH];
    let mut prefix = [Fe::ZERO; BATCH];

    let mut done = 0usize;
    while done < count {
        let k = BATCH.min(count - done);
        let base = job.counter.wrapping_add(done as u64);
        for i in 0..k {
            let secret =
                [job.key[0].wrapping_add(base.wrapping_add(i as u64)), job.key[1], job.key[2], job.key[3]];
            let s = scalar_words(&secret, job.seed_index);
            let (e, f, g, hh) = ec(tbl, &s);
            if full {
                den[i] = f.mul(&g);
                num[i] = g.mul(&hh);
                xnum[i] = e.mul(&f);
            } else {
                den[i] = f;
                num[i] = hh;
            }
        }
        // Montgomery batch inversion of the denominators.
        let mut acc = Fe::ONE;
        for i in 0..k {
            acc = acc.mul(&den[i]);
            prefix[i] = acc;
        }
        let mut inv = acc.invert();
        for i in (0..k).rev() {
            let di = if i > 0 { inv.mul(&prefix[i - 1]) } else { inv };
            inv = inv.mul(&den[i]);
            let y = num[i].mul(&di);
            let counter = base.wrapping_add(i as u64);
            if full {
                let x = xnum[i].mul(&di);
                let pk = pack(&x, &y);
                if pk[0] & pf.m0 != pf.v0 {
                    continue;
                }
                let cks = checksum_words(&pk);
                if cks & pf.mc == pf.vc {
                    h.on_full(counter, pk, cks);
                }
            } else {
                let yc = y.canonical().0;
                let w0 = yc[0] | (yc[1] << 51);
                if w0 & pf.m0 == pf.v0 {
                    h.on_quick(counter, w0);
                }
            }
        }
        done += k;
    }
}
