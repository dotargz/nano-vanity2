//! Reference twisted-Edwards arithmetic on edwards25519 (extended coordinates).
//!
//! Simple and independent of the lookup tables used by the fast search
//! engines: every key the program prints is re-derived through this code.
//! Also compiled into the build script, so it may only depend on `fe51`.

use crate::fe51::Fe;

/// d = -121665/121666
pub const D: Fe = Fe([
    929955233495203,
    466365720129213,
    1662059464998953,
    2033849074728123,
    1442794654840575,
]);

/// 2d
pub const D2: Fe = Fe([
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
]);

/// sqrt(-1)
pub const SQRT_M1: Fe = Fe([
    1718705420411056,
    234908883556509,
    2233514472574048,
    2117202627021982,
    765476049583133,
]);

/// A point in extended coordinates: x = X/Z, y = Y/Z, T = XY/Z.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

impl Point {
    pub const IDENTITY: Point = Point { x: Fe::ZERO, y: Fe::ONE, z: Fe::ONE, t: Fe::ZERO };

    /// The standard base point (y = 4/5, x even).
    pub const BASE: Point = Point {
        x: Fe([
            1738742601995546,
            1146398526822698,
            2070867633025821,
            562264141797630,
            587772402128613,
        ]),
        y: Fe([
            1801439850948184,
            1351079888211148,
            450359962737049,
            900719925474099,
            1801439850948198,
        ]),
        z: Fe::ONE,
        t: Fe([
            1841354044333475,
            16398895984059,
            755974180946558,
            900171276175154,
            1821297809914039,
        ]),
    };

    pub fn from_affine(x: &Fe, y: &Fe) -> Point {
        Point { x: *x, y: *y, z: Fe::ONE, t: x.mul(y) }
    }

    /// Complete unified addition (add-2008-hwcd-3).
    pub fn add(&self, q: &Point) -> Point {
        let a = self.y.sub(&self.x).mul(&q.y.sub(&q.x));
        let b = self.y.add(&self.x).mul(&q.y.add(&q.x));
        let c = self.t.mul(&D2).mul(&q.t);
        let zz = self.z.mul(&q.z);
        let d = zz.add(&zz);
        let e = b.sub(&a);
        let f = d.sub(&c);
        let g = d.add(&c);
        let h = b.add(&a);
        Point { x: e.mul(&f), y: g.mul(&h), z: f.mul(&g), t: e.mul(&h) }
    }

    /// Doubling (dbl-2008-hwcd).
    pub fn double(&self) -> Point {
        let a = self.x.square();
        let b = self.y.square();
        let zz = self.z.square();
        let c = zz.add(&zz);
        let d = a.neg();
        let e = self.x.add(&self.y).square().sub(&a).sub(&b);
        let g = d.add(&b);
        let f = g.sub(&c);
        let h = d.sub(&b);
        Point { x: e.mul(&f), y: g.mul(&h), z: f.mul(&g), t: e.mul(&h) }
    }

    /// Mixed addition with an affine point given in "halved Niels" form
    /// `((y+x)/2, (y-x)/2, d*x*y)`. Costs 7 multiplications. The result is
    /// the true sum with all four coordinates scaled by 1/4, which is the
    /// same projective point.
    #[inline(always)]
    pub fn add_halved(&self, ypx: &Fe, ymx: &Fe, xyd: &Fe) -> Point {
        let a = self.y.sub(&self.x).mul(ymx);
        let b = self.y.add(&self.x).mul(ypx);
        let c = self.t.mul(xyd);
        let e = b.sub(&a);
        let f = self.z.sub(&c);
        let g = self.z.add(&c);
        let h = b.add(&a);
        Point { x: e.mul(&f), y: g.mul(&h), z: f.mul(&g), t: e.mul(&h) }
    }

    /// Plain double-and-add over a 256-bit little-endian scalar.
    pub fn mul(&self, scalar: &[u8; 32]) -> Point {
        let mut acc = Point::IDENTITY;
        for i in (0..256).rev() {
            acc = acc.double();
            if (scalar[i >> 3] >> (i & 7)) & 1 == 1 {
                acc = acc.add(self);
            }
        }
        acc
    }

    pub fn affine(&self) -> (Fe, Fe) {
        let zi = self.z.invert();
        (self.x.mul(&zi), self.y.mul(&zi))
    }

    /// Standard 32-byte encoding: y little endian, sign of x in the top bit.
    pub fn compress(&self) -> [u8; 32] {
        let (x, y) = self.affine();
        let mut out = y.to_bytes();
        out[31] |= (x.is_negative() as u8) << 7;
        out
    }

    pub fn decompress(bytes: &[u8; 32]) -> Option<Point> {
        let y = Fe::from_bytes(bytes);
        // Reject non-canonical y.
        let mut masked = *bytes;
        masked[31] &= 0x7f;
        if y.to_bytes() != masked {
            return None;
        }
        let sign = bytes[31] >> 7 == 1;
        let yy = y.square();
        let u = yy.sub(&Fe::ONE);
        let v = yy.mul(&D).add(&Fe::ONE);
        // x = sqrt(u/v)
        let v3 = v.square().mul(&v);
        let v7 = v3.square().mul(&v);
        let mut x = u.mul(&v3).mul(&u.mul(&v7).pow_p58());
        let check = v.mul(&x.square());
        if !check.equals(&u) {
            if check.equals(&u.neg()) {
                x = x.mul(&SQRT_M1);
            } else {
                return None;
            }
        }
        if x.is_zero() && sign {
            return None;
        }
        if x.is_negative() != sign {
            x = x.neg();
        }
        Some(Point::from_affine(&x, &y))
    }

    pub fn equals(&self, o: &Point) -> bool {
        // x1/z1 == x2/z2 and y1/z1 == y2/z2
        self.x.mul(&o.z).equals(&o.x.mul(&self.z)) && self.y.mul(&o.z).equals(&o.y.mul(&self.z))
    }
}
