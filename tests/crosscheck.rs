//! Cross-checks against independent implementations and published vectors.

use blake2::digest::{Update, VariableOutput};
use curve25519_dalek::edwards::EdwardsPoint;
use nano_vanity2::engine::Backend;
use nano_vanity2::pattern::{GlobOptions, Matcher};
use nano_vanity2::table::{static_table, Builder};
use nano_vanity2::{address, blake2b, edwards, fe51::Fe, keys, selftest};

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random32(state: &mut u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    for c in out.chunks_mut(8) {
        c.copy_from_slice(&rng(state).to_le_bytes());
    }
    out
}

fn blake2_ref(outlen: usize, msg: &[u8]) -> Vec<u8> {
    let mut h = blake2::Blake2bVar::new(outlen).unwrap();
    h.update(msg);
    let mut out = vec![0u8; outlen];
    h.finalize_variable(&mut out).unwrap();
    out
}

/// Public key of a private key via curve25519-dalek and the blake2 crate.
fn dalek_public(private: &[u8; 32]) -> [u8; 32] {
    let h = blake2_ref(64, private);
    let mut scalar = [0u8; 32];
    scalar.copy_from_slice(&h[..32]);
    EdwardsPoint::mul_base_clamped(scalar).compress().to_bytes()
}

#[test]
fn published_vectors() {
    selftest::known_answers().unwrap();
}

#[test]
fn curve_constants() {
    // d = -121665/121666
    let d = Fe([121665, 0, 0, 0, 0]).neg().mul(&Fe([121666, 0, 0, 0, 0]).invert());
    assert!(d.equals(&edwards::D));
    assert!(d.add(&d).equals(&edwards::D2));
    assert!(edwards::SQRT_M1.square().equals(&Fe::ONE.neg()));
    let mut b = [0x66u8; 32];
    b[0] = 0x58;
    assert_eq!(edwards::Point::BASE.compress(), b);
    assert!(edwards::Point::decompress(&b).unwrap().equals(&edwards::Point::BASE));
    assert!(edwards::Point::BASE.x.mul(&edwards::Point::BASE.y).equals(&edwards::Point::BASE.t));
}

#[test]
fn blake2b_matches_reference_crate() {
    let mut s = 0x1234_5678_9abc_def1u64;
    for len in [0usize, 1, 5, 32, 36, 64, 127, 128] {
        for outlen in [5usize, 32, 64] {
            let msg: Vec<u8> = (0..len).map(|_| rng(&mut s) as u8).collect();
            assert_eq!(&blake2b::hash(outlen, &msg)[..outlen], &blake2_ref(outlen, &msg)[..]);
        }
    }
}

#[test]
fn public_keys_match_dalek() {
    let mut s = 0xfeed_beef_cafe_f00du64;
    for _ in 0..300 {
        let private = random32(&mut s);
        assert_eq!(keys::private_to_public(&private), dalek_public(&private));
    }
}

#[test]
fn address_round_trip_and_rejection() {
    let mut s = 77u64;
    for _ in 0..200 {
        let pk = random32(&mut s);
        let a = address::encode(&pk);
        assert_eq!(a.len(), 65);
        assert_eq!(address::decode(&a), Ok(pk));
        assert_eq!(address::decode(&a.replacen("nano_", "xrb_", 1)), Ok(pk));
        // Change one character: must be rejected.
        let mut bytes = a.clone().into_bytes();
        let i = 5 + (rng(&mut s) % 60) as usize;
        bytes[i] = if bytes[i] == b'a' { b'b' } else { b'a' };
        assert!(address::decode(std::str::from_utf8(&bytes).unwrap()).is_err());
    }
    assert!(address::decode("nano_abc").is_err());
    assert!(address::decode("ban_1111111111111111111111111111111111111111111111111111hifc8npp").is_err());
    assert!(address::decode("nano_1111111111111111111111111111111111111111111111111111hifc8npp").is_ok());
}

#[test]
fn engines_match_reference() {
    let mut s = 0xabcdu64;
    for backend in [Backend::Scalar, Backend::Ifma] {
        if !backend.available() {
            continue;
        }
        for w in [8u32, 4, 9, 13, 18] {
            let built;
            let table = if w == 8 {
                static_table()
            } else {
                built = Builder::build_now(w);
                built.table()
            };
            for seed_index in [None, Some(0), Some(0xdead_beef)] {
                let key = [rng(&mut s), rng(&mut s), rng(&mut s), rng(&mut s)];
                selftest::engine_check(backend, table, seed_index, key, rng(&mut s), 12).unwrap();
            }
        }
    }
}

fn matches(pattern: &str, opt: GlobOptions, addr: &str) -> bool {
    let m = Matcher::glob(pattern, &opt).unwrap();
    m.check_public(&address::decode(addr).unwrap())
}

#[test]
fn patterns() {
    let a = "nano_3i1aq1cchnmbn9x5rsbap8b15akfh7wj7pwskuzi7ahz8oq6cobd99d4r3b7";
    let d = GlobOptions::default();
    assert!(matches("i1aq", d, a));
    assert!(matches("I1AQ", d, a));
    assert!(!matches("i1ar", d, a));
    assert!(matches("nano_3i1", d, a));
    assert!(!matches("nano_1i1", d, a));
    assert!(matches("i?aq", d, a));
    assert!(matches("i[13]aq", d, a));
    assert!(!matches("i[!1]aq", d, a));
    assert!(matches("*r3b7", d, a));
    assert!(matches("r3b7", GlobOptions { suffix: true, ..d }, a));
    assert!(!matches("*r3b8", d, a));
    assert!(matches("i1*b7", d, a));
    assert!(matches("*skuzi*", d, a));
    assert!(matches("skuzi", GlobOptions { anywhere: true, ..d }, a));
    assert!(matches("i1*cchn*x5rs*b7", d, a));
    assert!(!matches("i1*x5rs*cchn*b7", d, a));
    assert!(matches("*99d4r3b7", d, a));
    // Lookalikes: l -> 1/i, 9 -> q ...
    let l = GlobOptions { lookalike: true, ..d };
    assert!(matches("ilaq", l, a));
    assert!(matches("11a9", l, a));
    assert!(Matcher::glob("ilaq", &d).is_err());
    assert!(Matcher::glob("nano_2abc", &d).is_err());
    assert!(Matcher::glob("a[", &d).is_err());

    let re = Matcher::regex("^nano_[13]i1aq.*b7$").unwrap();
    assert!(re.check_public(&address::decode(a).unwrap()));
    assert!(!re.need_full, "anchored regex should use the quick filter");
    let re = Matcher::regex("cchn.*skuz").unwrap();
    assert!(re.check_public(&address::decode(a).unwrap()));
    assert!(re.need_full);
    let re = Matcher::regex("^nano_1").unwrap();
    assert!(!re.check_public(&address::decode(a).unwrap()));
}

#[test]
fn search_finds_valid_keys() {
    use nano_vanity2::search::{self, KeyKind, Options};
    use std::sync::Mutex;
    let plain = GlobOptions::default();
    let look = GlobOptions { lookalike: true, ..plain };
    for (pattern, opt, kind) in [
        ("ab", plain, KeyKind::Seed { index: 0 }),
        ("abc", plain, KeyKind::Private),
        ("*xy", plain, KeyKind::Seed { index: 3 }),
        ("a*7", plain, KeyKind::Private),
        ("te5t", look, KeyKind::Seed { index: 0 }),
    ] {
        for backend in [Backend::Scalar, Backend::detect()] {
            let m = Matcher::glob(pattern, &opt).unwrap();
            let got = Mutex::new(Vec::new());
            let opts = Options {
                matcher: &m,
                kind,
                threads: 2,
                count: 2,
                timeout: None,
                backend,
                window: None,
                start: std::time::Instant::now(),
                progress: false,
                gpu: Default::default(),
            };
            let summary = search::run(&opts, &|f| {
                got.lock().unwrap().push((*f.secret, f.public, f.address.clone()));
            });
            assert_eq!(summary.mismatches, 0);
            let got = got.into_inner().unwrap();
            assert_eq!(got.len(), 2);
            assert_ne!(got[0].0, got[1].0);
            for (secret, public, addr) in got {
                let private = match kind {
                    KeyKind::Seed { index } => keys::seed_to_private(&secret, index),
                    KeyKind::Private => secret,
                };
                let want = dalek_public(&private);
                assert_eq!(public, want);
                assert_eq!(address::decode(&addr), Ok(want));
                assert!(m.check_public(&want), "{addr} does not match {pattern}");
            }
        }
    }
}

#[test]
fn gpu_kernel_matches_cpu() {
    use nano_vanity2::gpu;
    if !gpu::driver_present() {
        eprintln!("no CUDA driver: GPU test skipped");
        return;
    }
    for w in [9u32, 16] {
        let built = Builder::build_now(w);
        match gpu::self_check(built.table()) {
            Ok(_) => {}
            // A driver without a usable device is not a failure of this code.
            Err(e) if e.contains("no CUDA device") || e.contains("init") => {
                eprintln!("GPU test skipped: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        }
    }
}

#[test]
fn gpu_search_finds_valid_keys() {
    use nano_vanity2::search::{self, GpuMode, GpuOptions, KeyKind, Options};
    use std::sync::Mutex;
    if !nano_vanity2::gpu::driver_present() {
        return;
    }
    let plain = GlobOptions::default();
    for (pattern, opt, kind) in [
        ("abcd", plain, KeyKind::Seed { index: 2 }),
        ("ab*yz", plain, KeyKind::Private),
        ("*wxyz", plain, KeyKind::Seed { index: 0 }),
        ("wxy9", GlobOptions { anywhere: true, ..plain }, KeyKind::Private),
    ] {
        let m = Matcher::glob(pattern, &opt).unwrap();
        let got = Mutex::new(Vec::new());
        let opts = Options {
            matcher: &m,
            kind,
            // One slow CPU thread, so that the GPU finds (nearly) everything.
            threads: 1,
            count: 3,
            timeout: None,
            backend: Backend::Scalar,
            window: Some(0),
            start: std::time::Instant::now(),
            progress: false,
            gpu: GpuOptions { mode: GpuMode::On, threads: Some(8192), batch: Some(16), window: Some(12) },
        };
        let summary = search::run(&opts, &|f| {
            got.lock().unwrap().push((*f.secret, f.public, f.address.clone()));
        });
        assert_eq!(summary.mismatches, 0);
        if summary.gpu.is_none() {
            eprintln!("GPU not usable ({:?}); search test ran on the CPU", summary.gpu_error);
        }
        let got = got.into_inner().unwrap();
        assert_eq!(got.len(), 3);
        for (secret, public, addr) in got {
            let private = match kind {
                KeyKind::Seed { index } => keys::seed_to_private(&secret, index),
                KeyKind::Private => secret,
            };
            let want = dalek_public(&private);
            assert_eq!(public, want);
            assert_eq!(address::decode(&addr), Ok(want));
            assert!(m.check_public(&want), "{addr} does not match {pattern}");
        }
    }
}
