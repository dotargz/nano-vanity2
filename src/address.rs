//! Nano account address encoding, decoding and validation.
//!
//! `nano_` + 52 base32 characters (4 zero bits + 256-bit public key)
//!         +  8 base32 characters (40-bit BLAKE2b checksum, byte reversed).

use crate::blake2b;
use std::fmt;

pub const ALPHABET: &[u8; 32] = b"13456789abcdefghijkmnopqrstuwxyz";
/// Characters of an address after the `nano_` prefix.
pub const BODY_LEN: usize = 60;
/// Characters that encode the public key.
pub const KEY_LEN: usize = 52;

/// Index of an address character in the alphabet.
pub fn symbol(c: u8) -> Option<u8> {
    ALPHABET.iter().position(|&a| a == c).map(|p| p as u8)
}

/// The 40-bit checksum as an integer whose top 5 bits are the first
/// checksum character.
pub fn checksum(public: &[u8; 32]) -> u64 {
    let mut m = [0u64; 5];
    for i in 0..4 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&public[i * 8..i * 8 + 8]);
        m[i] = u64::from_le_bytes(t);
    }
    blake2b::hash_words(m, 32, 5)[0] & 0xff_ffff_ffff
}

pub fn encode(public: &[u8; 32]) -> String {
    let mut s = String::with_capacity(5 + BODY_LEN);
    s.push_str("nano_");
    // 4 zero bits, then the key, most significant bit first.
    let mut acc: u32 = 0;
    let mut bits = 4;
    for &b in public.iter() {
        acc = (acc << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            s.push(ALPHABET[((acc >> (bits - 5)) & 31) as usize] as char);
            bits -= 5;
        }
    }
    let c = checksum(public);
    for i in 0..8 {
        s.push(ALPHABET[((c >> (35 - 5 * i)) & 31) as usize] as char);
    }
    s
}

#[derive(Debug, PartialEq, Eq)]
pub enum AddressError {
    Prefix,
    Length(usize),
    Character(char),
    FirstCharacter(char),
    Checksum,
}

impl fmt::Display for AddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressError::Prefix => write!(f, "address must start with nano_ or xrb_"),
            AddressError::Length(n) => {
                write!(f, "expected {BODY_LEN} characters after the prefix, found {n}")
            }
            AddressError::Character(c) => {
                write!(f, "character {c:?} is not in the Nano alphabet")
            }
            AddressError::FirstCharacter(c) => {
                write!(f, "first character after the prefix must be 1 or 3, found {c:?}")
            }
            AddressError::Checksum => write!(f, "checksum does not match (mistyped address?)"),
        }
    }
}

/// Validate an address and return its public key.
pub fn decode(addr: &str) -> Result<[u8; 32], AddressError> {
    let body = addr
        .strip_prefix("nano_")
        .or_else(|| addr.strip_prefix("xrb_"))
        .ok_or(AddressError::Prefix)?;
    let n = body.chars().count();
    if n != BODY_LEN {
        return Err(AddressError::Length(n));
    }
    let mut syms = [0u8; BODY_LEN];
    for (i, c) in body.chars().enumerate() {
        let s = if c.is_ascii() { symbol(c as u8) } else { None };
        syms[i] = s.ok_or(AddressError::Character(c))?;
    }
    if syms[0] > 1 {
        return Err(AddressError::FirstCharacter(body.chars().next().unwrap()));
    }
    let mut public = [0u8; 32];
    let mut acc: u32 = syms[0] as u32;
    let mut bits = 1;
    let mut k = 0;
    for &s in &syms[1..KEY_LEN] {
        acc = (acc << 5) | s as u32;
        bits += 5;
        while bits >= 8 {
            public[k] = (acc >> (bits - 8)) as u8;
            k += 1;
            bits -= 8;
        }
    }
    let mut c: u64 = 0;
    for &s in &syms[KEY_LEN..] {
        c = (c << 5) | s as u64;
    }
    if c != checksum(&public) {
        return Err(AddressError::Checksum);
    }
    Ok(public)
}
