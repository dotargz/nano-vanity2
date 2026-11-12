//! Known-answer tests and cross-checks of the fast engines against the
//! reference implementation. Used by `nano-vanity2 selftest` and the tests.

use crate::engine::{Backend, Engine, Handler, Job, Prefilter, UNIT};
use crate::table::Table;
use crate::{address, blake2b, keys};

/// (seed, index, private key, public key, address) from the Nano docs.
pub const SEED_VECTOR: (&str, u32, &str, &str, &str) = (
    "0000000000000000000000000000000000000000000000000000000000000000",
    0,
    "9F0E444C69F77A49BD0BE89DB92C38FE713E0963165CCA12FAF5712D7657120F",
    "C008B814A7D269A1FA3C6528B19201A24D797912DB9996FF02A1FF356E45552B",
    "nano_3i1aq1cchnmbn9x5rsbap8b15akfh7wj7pwskuzi7ahz8oq6cobd99d4r3b7",
);

/// (private key, public key, address) from the Nano docs.
pub const KEY_VECTOR: (&str, &str, &str) = (
    "781186FB9EF17DB6E3D1056550D9FAE5D5BBADA6A6BC370E4CBB938B1DC71DA3",
    "3068BB1CA04525BB0E416C485FE6A67FD52540227D267CC8B6E8DA958A7FA039",
    "nano_1e5aqegc1jb7qe964u4adzmcezyo6o146zb8hm6dft8tkp79za3sxwjym5rx",
);

/// Reference derivation checked against published vectors.
pub fn known_answers() -> Result<(), String> {
    // BLAKE2b-512("abc"), RFC 7693.
    let h = blake2b::hash(64, b"abc");
    let want = "BA80A53F981C4D0D6A2797B69F12F6E94C212F14685AC4B74B12BB6FDBFFA2D1\
                7D87C5392AAB792DC252D5DE4533CC9518D38AA8DBF1925AB92386EDD4009923";
    if keys::hex_upper(&h) != want {
        return Err("BLAKE2b known answer mismatch".into());
    }

    let (seed, index, private, public, addr) = SEED_VECTOR;
    let seed = keys::parse_hex32(seed).unwrap();
    let got_private = keys::seed_to_private(&seed, index);
    if keys::hex_upper(&got_private) != private {
        return Err("seed -> private key mismatch".into());
    }
    let got_public = keys::private_to_public(&got_private);
    if keys::hex_upper(&got_public) != public {
        return Err("private -> public key mismatch (seed vector)".into());
    }
    if address::encode(&got_public) != addr {
        return Err("address encoding mismatch (seed vector)".into());
    }
    if address::decode(addr) != Ok(got_public) {
        return Err("address decoding mismatch (seed vector)".into());
    }

    let (private, public, addr) = KEY_VECTOR;
    let got_public = keys::private_to_public(&keys::parse_hex32(private).unwrap());
    if keys::hex_upper(&got_public) != public {
        return Err("private -> public key mismatch (key vector)".into());
    }
    if address::encode(&got_public) != addr {
        return Err("address encoding mismatch (key vector)".into());
    }
    Ok(())
}

struct Collect {
    full: bool,
    quick: Vec<(u64, u64)>,
    fulls: Vec<(u64, [u64; 4], u64)>,
}

impl Handler for Collect {
    fn full(&self) -> bool {
        self.full
    }
    fn prefilter(&self) -> Prefilter {
        Prefilter::default()
    }
    fn on_quick(&mut self, counter: u64, w0: u64) {
        self.quick.push((counter, w0));
    }
    fn on_full(&mut self, counter: u64, pk: [u64; 4], cks: u64) {
        self.fulls.push((counter, pk, cks));
    }
}

fn reference(key: &[u64; 4], counter: u64, seed_index: Option<u32>) -> ([u64; 4], u64) {
    let words = [key[0].wrapping_add(counter), key[1], key[2], key[3]];
    let mut secret = [0u8; 32];
    for i in 0..4 {
        secret[i * 8..i * 8 + 8].copy_from_slice(&words[i].to_le_bytes());
    }
    let private = match seed_index {
        Some(i) => keys::seed_to_private(&secret, i),
        None => secret,
    };
    let public = keys::private_to_public(&private);
    let mut pk = [0u64; 4];
    for i in 0..4 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&public[i * 8..i * 8 + 8]);
        pk[i] = u64::from_le_bytes(t);
    }
    (pk, address::checksum(&public))
}

/// Run `units` work units through `backend` in both engine modes and compare
/// every public key and checksum with the reference implementation.
pub fn engine_check(
    backend: Backend,
    table: &Table,
    seed_index: Option<u32>,
    key: [u64; 4],
    counter: u64,
    units: usize,
) -> Result<usize, String> {
    let mut engine = Engine::new(backend);
    let job = Job { table, key, counter, seed_index };
    let total = units * UNIT;
    let tag = format!("{} w={} {}", backend.name(), table.layout.w, if seed_index.is_some() { "seed" } else { "key" });

    let mut c = Collect { full: true, quick: Vec::new(), fulls: Vec::new() };
    engine.run(&job, units, &mut c);
    if c.fulls.len() != total {
        return Err(format!("{tag}: full mode reported {} of {total} candidates", c.fulls.len()));
    }
    c.fulls.sort_by_key(|e| e.0.wrapping_sub(counter));
    for (i, (ctr, pk, cks)) in c.fulls.iter().enumerate() {
        if *ctr != counter.wrapping_add(i as u64) {
            return Err(format!("{tag}: full mode candidate numbering is wrong"));
        }
        let (rpk, rcks) = reference(&key, *ctr, seed_index);
        if *pk != rpk {
            return Err(format!("{tag}: wrong public key for candidate {i}"));
        }
        if *cks != rcks {
            return Err(format!("{tag}: wrong checksum for candidate {i}"));
        }
    }

    let mut c = Collect { full: false, quick: Vec::new(), fulls: Vec::new() };
    engine.run(&job, units, &mut c);
    if c.quick.len() != total {
        return Err(format!("{tag}: quick mode reported {} of {total} candidates", c.quick.len()));
    }
    c.quick.sort_by_key(|e| e.0.wrapping_sub(counter));
    for (i, (ctr, w0)) in c.quick.iter().enumerate() {
        if *ctr != counter.wrapping_add(i as u64) {
            return Err(format!("{tag}: quick mode candidate numbering is wrong"));
        }
        let (rpk, _) = reference(&key, *ctr, seed_index);
        if *w0 != rpk[0] {
            return Err(format!("{tag}: wrong key prefix for candidate {i}"));
        }
    }
    Ok(total * 2)
}
