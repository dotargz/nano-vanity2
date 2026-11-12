//! Reference Nano key derivation. Straightforward, table-free code that is
//! used to verify every result before it is shown to the user.

use crate::blake2b;
use crate::edwards::Point;
use zeroize::Zeroize;

/// Private key number `index` of a Nano seed: `blake2b_256(seed || index_be)`.
pub fn seed_to_private(seed: &[u8; 32], index: u32) -> [u8; 32] {
    let mut msg = [0u8; 36];
    msg[..32].copy_from_slice(seed);
    msg[32..].copy_from_slice(&index.to_be_bytes());
    let mut h = blake2b::hash(32, &msg);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h[..32]);
    msg.zeroize();
    h.zeroize();
    out
}

/// Nano public key of a private key: ed25519 with BLAKE2b-512 as the hash.
pub fn private_to_public(private: &[u8; 32]) -> [u8; 32] {
    let mut h = blake2b::hash(64, private);
    let mut scalar = [0u8; 32];
    scalar.copy_from_slice(&h[..32]);
    scalar[0] &= 248;
    scalar[31] &= 127;
    scalar[31] |= 64;
    let public = Point::BASE.mul(&scalar).compress();
    h.zeroize();
    scalar.zeroize();
    public
}

pub fn hex_upper(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}

pub fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let s = s.trim();
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    let b = s.as_bytes();
    for i in 0..32 {
        let hi = (b[2 * i] as char).to_digit(16)?;
        let lo = (b[2 * i + 1] as char).to_digit(16)?;
        out[i] = (hi * 16 + lo) as u8;
    }
    Some(out)
}
