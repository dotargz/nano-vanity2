//! Arithmetic in GF(2^255 - 19), radix 2^51, portable 64-bit implementation.
//!
//! This is the reference / fallback field implementation. It is also compiled
//! into the build script (which generates the embedded lookup table), so it
//! must not depend on anything outside this file.

pub const MASK51: u64 = (1u64 << 51) - 1;

/// A field element as five 51-bit limbs (little endian). Limbs may carry a
/// few excess bits between operations; `canonical` gives the unique form.
#[derive(Clone, Copy, Debug)]
pub struct Fe(pub [u64; 5]);

#[inline(always)]
fn m(a: u64, b: u64) -> u128 {
    (a as u128) * (b as u128)
}

impl Fe {
    pub const ZERO: Fe = Fe([0; 5]);
    pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    /// Load 255 bits little endian (the top bit of the last byte is ignored).
    pub fn from_bytes(b: &[u8; 32]) -> Fe {
        let w = |i: usize| -> u64 {
            let mut t = [0u8; 8];
            t.copy_from_slice(&b[i..i + 8]);
            u64::from_le_bytes(t)
        };
        Fe([
            w(0) & MASK51,
            (w(6) >> 3) & MASK51,
            (w(12) >> 6) & MASK51,
            (w(19) >> 1) & MASK51,
            (w(24) >> 12) & MASK51,
        ])
    }

    /// One parallel carry pass: limbs end up below 2^51 + 2^13 or so.
    #[inline(always)]
    pub fn weak_reduce(&self) -> Fe {
        let l = &self.0;
        let c0 = l[0] >> 51;
        let c1 = l[1] >> 51;
        let c2 = l[2] >> 51;
        let c3 = l[3] >> 51;
        let c4 = l[4] >> 51;
        Fe([
            (l[0] & MASK51) + c4 * 19,
            (l[1] & MASK51) + c0,
            (l[2] & MASK51) + c1,
            (l[3] & MASK51) + c2,
            (l[4] & MASK51) + c3,
        ])
    }

    /// The unique representative: every limb < 2^51 and the value < p.
    pub fn canonical(&self) -> Fe {
        let mut l = self.weak_reduce().0;
        // q = 1 iff value >= p
        let mut q = (l[0] + 19) >> 51;
        q = (l[1] + q) >> 51;
        q = (l[2] + q) >> 51;
        q = (l[3] + q) >> 51;
        q = (l[4] + q) >> 51;
        l[0] += 19 * q;
        l[1] += l[0] >> 51;
        l[0] &= MASK51;
        l[2] += l[1] >> 51;
        l[1] &= MASK51;
        l[3] += l[2] >> 51;
        l[2] &= MASK51;
        l[4] += l[3] >> 51;
        l[3] &= MASK51;
        l[4] &= MASK51;
        Fe(l)
    }

    /// Canonical value as four little-endian 64-bit words.
    pub fn to_words(&self) -> [u64; 4] {
        let l = self.canonical().0;
        [
            l[0] | (l[1] << 51),
            (l[1] >> 13) | (l[2] << 38),
            (l[2] >> 26) | (l[3] << 25),
            (l[3] >> 39) | (l[4] << 12),
        ]
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let w = self.to_words();
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&w[i].to_le_bytes());
        }
        out
    }

    /// Limb-wise addition without carrying (inputs must be weakly reduced).
    #[inline(always)]
    pub fn add(&self, o: &Fe) -> Fe {
        let (a, b) = (&self.0, &o.0);
        Fe([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4]])
    }

    /// self - o, weakly reduced. `o` may have limbs up to 2^54.
    #[inline(always)]
    pub fn sub(&self, o: &Fe) -> Fe {
        // 16p, limb-wise, so the subtraction never underflows.
        const P16_0: u64 = 36028797018963664; // 16 * (2^51 - 19)
        const P16_N: u64 = 36028797018963952; // 16 * (2^51 - 1)
        let (a, b) = (&self.0, &o.0);
        Fe([
            (a[0] + P16_0) - b[0],
            (a[1] + P16_N) - b[1],
            (a[2] + P16_N) - b[2],
            (a[3] + P16_N) - b[3],
            (a[4] + P16_N) - b[4],
        ])
        .weak_reduce()
    }

    #[inline(always)]
    pub fn neg(&self) -> Fe {
        Fe::ZERO.sub(self)
    }

    /// Product, weakly reduced. Inputs may have limbs up to 2^54.
    #[inline(always)]
    pub fn mul(&self, o: &Fe) -> Fe {
        let a = &self.0;
        let b = &o.0;
        let b1_19 = b[1] * 19;
        let b2_19 = b[2] * 19;
        let b3_19 = b[3] * 19;
        let b4_19 = b[4] * 19;

        let c0 = m(a[0], b[0]) + m(a[4], b1_19) + m(a[3], b2_19) + m(a[2], b3_19) + m(a[1], b4_19);
        let mut c1 = m(a[1], b[0]) + m(a[0], b[1]) + m(a[4], b2_19) + m(a[3], b3_19) + m(a[2], b4_19);
        let mut c2 = m(a[2], b[0]) + m(a[1], b[1]) + m(a[0], b[2]) + m(a[4], b3_19) + m(a[3], b4_19);
        let mut c3 = m(a[3], b[0]) + m(a[2], b[1]) + m(a[1], b[2]) + m(a[0], b[3]) + m(a[4], b4_19);
        let mut c4 = m(a[4], b[0]) + m(a[3], b[1]) + m(a[2], b[2]) + m(a[1], b[3]) + m(a[0], b[4]);

        Self::carry_wide(c0, &mut c1, &mut c2, &mut c3, &mut c4)
    }

    #[inline(always)]
    fn carry_wide(c0: u128, c1: &mut u128, c2: &mut u128, c3: &mut u128, c4: &mut u128) -> Fe {
        let mut out = [0u64; 5];
        *c1 += (c0 >> 51) as u64 as u128;
        out[0] = (c0 as u64) & MASK51;
        *c2 += (*c1 >> 51) as u64 as u128;
        out[1] = (*c1 as u64) & MASK51;
        *c3 += (*c2 >> 51) as u64 as u128;
        out[2] = (*c2 as u64) & MASK51;
        *c4 += (*c3 >> 51) as u64 as u128;
        out[3] = (*c3 as u64) & MASK51;
        let carry = (*c4 >> 51) as u64;
        out[4] = (*c4 as u64) & MASK51;
        out[0] += carry * 19;
        out[1] += out[0] >> 51;
        out[0] &= MASK51;
        Fe(out)
    }

    #[inline(always)]
    pub fn square(&self) -> Fe {
        let a = &self.0;
        let a3_19 = 19 * a[3];
        let a4_19 = 19 * a[4];
        let c0 = m(a[0], a[0]) + 2 * (m(a[1], a4_19) + m(a[2], a3_19));
        let mut c1 = m(a[3], a3_19) + 2 * (m(a[0], a[1]) + m(a[2], a4_19));
        let mut c2 = m(a[1], a[1]) + 2 * (m(a[0], a[2]) + m(a[4], a3_19));
        let mut c3 = m(a[4], a4_19) + 2 * (m(a[0], a[3]) + m(a[1], a[2]));
        let mut c4 = m(a[2], a[2]) + 2 * (m(a[0], a[4]) + m(a[1], a[3]));
        Self::carry_wide(c0, &mut c1, &mut c2, &mut c3, &mut c4)
    }

    /// self^(2^k)
    pub fn pow2k(&self, k: u32) -> Fe {
        let mut r = *self;
        for _ in 0..k {
            r = r.square();
        }
        r
    }

    /// Returns (self^(2^250 - 1), self^11).
    fn pow22501(&self) -> (Fe, Fe) {
        let z = self;
        let z2 = z.square();
        let z8 = z2.pow2k(2);
        let z9 = z.mul(&z8);
        let z11 = z2.mul(&z9);
        let z22 = z11.square();
        let z_5_0 = z9.mul(&z22);
        let z_10_0 = z_5_0.pow2k(5).mul(&z_5_0);
        let z_20_0 = z_10_0.pow2k(10).mul(&z_10_0);
        let z_40_0 = z_20_0.pow2k(20).mul(&z_20_0);
        let z_50_0 = z_40_0.pow2k(10).mul(&z_10_0);
        let z_100_0 = z_50_0.pow2k(50).mul(&z_50_0);
        let z_200_0 = z_100_0.pow2k(100).mul(&z_100_0);
        let z_250_0 = z_200_0.pow2k(50).mul(&z_50_0);
        (z_250_0, z11)
    }

    /// Multiplicative inverse (self^(p-2)); zero maps to zero.
    pub fn invert(&self) -> Fe {
        let (t, z11) = self.pow22501();
        t.pow2k(5).mul(&z11)
    }

    /// self^((p-5)/8)
    pub fn pow_p58(&self) -> Fe {
        let (t, _) = self.pow22501();
        t.pow2k(2).mul(self)
    }

    pub fn is_negative(&self) -> bool {
        self.canonical().0[0] & 1 == 1
    }

    pub fn is_zero(&self) -> bool {
        self.canonical().0 == [0; 5]
    }

    pub fn equals(&self, o: &Fe) -> bool {
        self.canonical().0 == o.canonical().0
    }
}

/// Invert every element of `v` in place using one field inversion
/// (Montgomery's trick). No element may be zero.
pub fn batch_invert(v: &mut [Fe]) {
    if v.is_empty() {
        return;
    }
    let mut prefix = Vec::with_capacity(v.len());
    let mut run = Fe::ONE;
    for x in v.iter() {
        run = run.mul(x);
        prefix.push(run);
    }
    let mut inv = run.invert();
    for j in (0..v.len()).rev() {
        let r = if j > 0 { inv.mul(&prefix[j - 1]) } else { inv };
        inv = inv.mul(&v[j]);
        v[j] = r;
    }
}
