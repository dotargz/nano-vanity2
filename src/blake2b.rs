//! BLAKE2b for single-block, unkeyed messages (at most 128 bytes).
//!
//! That is all Nano key derivation needs: 32-byte private keys, 36-byte
//! `seed || index` strings and 32-byte public keys for the address checksum.

pub const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

pub const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Hash one final block. `m` holds the message as little-endian words
/// (zero padded), `len` its length in bytes, `outlen` the digest length.
/// The digest is the first `outlen` bytes of the returned words.
#[inline(always)]
pub fn compress(m: &[u64; 16], len: u64, outlen: u64) -> [u64; 8] {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ outlen;
    let mut v = [
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
        IV[0], IV[1], IV[2], IV[3], IV[4] ^ len, IV[5], !IV[6], IV[7],
    ];
    macro_rules! g {
        ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {
            v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add($x);
            v[$d] = (v[$d] ^ v[$a]).rotate_right(32);
            v[$c] = v[$c].wrapping_add(v[$d]);
            v[$b] = (v[$b] ^ v[$c]).rotate_right(24);
            v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add($y);
            v[$d] = (v[$d] ^ v[$a]).rotate_right(16);
            v[$c] = v[$c].wrapping_add(v[$d]);
            v[$b] = (v[$b] ^ v[$c]).rotate_right(63);
        };
    }
    for s in SIGMA.iter() {
        g!(0, 4, 8, 12, m[s[0]], m[s[1]]);
        g!(1, 5, 9, 13, m[s[2]], m[s[3]]);
        g!(2, 6, 10, 14, m[s[4]], m[s[5]]);
        g!(3, 7, 11, 15, m[s[6]], m[s[7]]);
        g!(0, 5, 10, 15, m[s[8]], m[s[9]]);
        g!(1, 6, 11, 12, m[s[10]], m[s[11]]);
        g!(2, 7, 8, 13, m[s[12]], m[s[13]]);
        g!(3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
    h
}

/// Hash a message whose only non-zero words are the first five.
#[inline(always)]
pub fn hash_words(m: [u64; 5], len: u64, outlen: u64) -> [u64; 8] {
    let mm = [m[0], m[1], m[2], m[3], m[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    compress(&mm, len, outlen)
}

/// BLAKE2b of `msg` (at most 128 bytes) with an `outlen`-byte digest. The
/// digest occupies the first `outlen` bytes of the result.
pub fn hash(outlen: usize, msg: &[u8]) -> [u8; 64] {
    assert!(msg.len() <= 128 && (1..=64).contains(&outlen));
    let mut block = [0u8; 128];
    block[..msg.len()].copy_from_slice(msg);
    let mut m = [0u64; 16];
    for i in 0..16 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&block[i * 8..i * 8 + 8]);
        m[i] = u64::from_le_bytes(t);
    }
    let h = compress(&m, msg.len() as u64, outlen as u64);
    let mut out = [0u8; 64];
    for i in 0..8 {
        out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    for b in out[outlen..].iter_mut() {
        *b = 0;
    }
    // Best effort: do not leave message copies behind.
    for b in block.iter_mut() {
        *b = 0;
    }
    for w in m.iter_mut() {
        *w = 0;
    }
    out
}
