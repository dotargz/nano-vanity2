# nano-vanity2

A very fast vanity address generator for [Nano](https://nano.org). It finds a
wallet seed (or raw private key) whose address matches a pattern.

```
> nano-vanity2 nano
Address:     nano_3nanokr1fwsjnqkq5biuf7bsdcuioifh5gin3h4o4m76xzakributd79fns1
Seed:        <64 hex characters>
Public key:  D114ACB006F331A5E571A61B695395AB70AC1AF1BA140BC5514CA4EFD12C413B
Found in 15.52 ms: 393 k keys tried at 25.6 M keys/s (avx512-ifma, 8 threads)
```

## Build

```
cargo build --release
```

The binary is `target/release/nano-vanity2`. Run `nano-vanity2 --selftest`
once on a new machine: it checks the build against Nano's documented test
vectors and cross-checks every engine, including the GPU kernel, against the
reference implementation.

## Usage

```
nano-vanity2 [OPTIONS] <PATTERN>
nano-vanity2 [OPTIONS] --regex <REGEX>
nano-vanity2 --check <ADDRESS>...
```

| Pattern      | Matches                                                        |
|--------------|----------------------------------------------------------------|
| `abc`        | `nano_1abc…` or `nano_3abc…` (the first character is always 1 or 3) |
| `nano_3abc`  | pins the first character too                                   |
| `a?c`        | `?` is any one character                                       |
| `a[bc]d`     | one character from a set; `[a-f]` and `[!ab]` also work         |
| `*abc`       | suffix (`-s abc` is the same)                                  |
| `abc*xyz`    | prefix and suffix                                              |
| `*abc*`      | anywhere (`-a abc` is the same)                                |

Nano's alphabet is `13456789abcdefghijkmnopqrstuwxyz` — there is no `0`, `2`,
`l` or `v`.

* `-l, --lookalike` — every character also matches its lookalikes, and you may
  type `0 2 l v`: `o=0  1=i=l  z=2  e=3  a=4  s=5  b=6=8  t=7  g=q=9  u=v`.
  Each affected character makes the search 2–3x faster.
* `-r, --regex` — matched against the whole address. Expressions anchored
  like `^nano_1(cafe|beef)` are as fast as prefixes; unanchored ones are slower.
* `-k, --private-key` — output a raw private key instead of a seed (~5% faster).
* `-i, --index N` — which account of the seed gets the address (default 0).
* `-n, --count N`, `-t, --threads N`, `--timeout SECONDS`, `--json`, `-q`.
* `-o, --output FILE` — append results to a file and keep the secret off the
  terminal (created with mode 0600 on Unix).
* `--gpu` / `--no-gpu` — by default an NVIDIA GPU joins searches expected to
  need more than ~12 million keys (five characters and up); shorter ones are
  over before it has started. `--gpu` forces it on, `--no-gpu` off.
* `--check ADDRESS...` — validate prefix, length, alphabet and checksum.
* `--bench [SECONDS]`, `--selftest`.

## Speed

Measured on a Ryzen 7 9800X3D (8 cores) with an RTX 5070, Windows 11:

| | keys/s |
|---|---|
| nano-vanity2, CPU + GPU | 325–340 M |
| nano-vanity2, RTX 5070 alone | 270–295 M |
| nano-vanity2, AVX-512 IFMA, 8 threads | 46–50 M |
| nano-vanity2, AVX-512 IFMA, 1 thread | 6.3–7.4 M |
| nano-vanity2, portable scalar backend, 8 threads | 5.1 M |
| conventional pipeline (curve25519-dalek per key), 8 threads | 0.67 M |

| Search | Median time | Runs |
|---|---|---|
| 2 characters | 0.6 ms | 100 |
| 3 characters | 1.5 ms | 200 |
| 4 characters | 23 ms | 60 |
| 5 characters | 190 ms (CPU only: 640 ms) | 80 |
| 6 characters | 2.5 s (90% within 6.1 s) | 40 |

Times are measured from the first line of `main` to a verified result. A
search is a lottery: the median needs 0.69x the average number of keys and
one run in ten needs more than 2.3x.

Up to four characters the time is start-up: threads take 0.2–0.3 ms to
come up and the lookup table is cold. Five characters are dominated by the
NVIDIA driver, which needs 90–180 ms to initialise before the first GPU
key; from six characters on the full rate applies.

The "conventional pipeline" row is `cargo run --release --example baseline`:
the per-key hash + fixed-base multiplication + compression that existing CPU
generators perform. It is a stand-in, not a run of any particular tool.

### Why it is fast

* **No doublings, one inversion per batch.** A clamped ed25519 scalar is
  `2^254 + 8r`; `r` is recoded into signed base-2^w digits and the public key
  is one table lookup plus one 7-multiplication mixed addition per digit.
  Only the affine `y` numerator/denominator is kept, and a whole batch shares
  one field inversion.
* **Eight keys per instruction.** With AVX-512 IFMA, field elements are
  5×51-bit limbs in zmm lanes, multiplied with `vpmadd52`. Additions are folded
  into the unreduced product accumulators so no stand-alone reductions are
  needed. BLAKE2b runs 8-way as well.
* **Tables sized to the cache.** A 513 KB table is embedded in the binary so
  short searches start instantly. Longer searches build a larger table (up to
  what fits in the L3 cache) in the background and switch to it; lookups for
  the next vector are prefetched while the current one is computed.
* **Bit-level matching.** Patterns compile to a mask/compare on the first 64
  key bits that runs on all eight lanes at once; the checksum is only hashed
  when the pattern needs it.
* **Low start-up cost.** No argument-parsing framework, threads are started
  while the first batch is already running.

* **GPU.** The same algorithm as a CUDA kernel (`src/gpu/kernel.cu`):
  256-bit field arithmetic on 32-bit limbs with multiply-add carry chains,
  32 or 64 keys per GPU thread sharing one inversion, a 50 MB table. The
  kernel tests the pattern itself (prefix, suffix, sets, lookalikes,
  floating segments) and reports only candidates. It starts in the
  background while the CPU is already searching.

CPUs without AVX-512 IFMA (most Intel desktop chips, AMD before Zen 4) use
the scalar backend; there is no AVX2 backend. The GPU backend needs an
NVIDIA card and driver (nothing else: the kernel is embedded as PTX and
`nvcuda.dll` / `libcuda.so` is loaded at run time). Building from changed
kernel source needs the CUDA toolkit; otherwise the checked-in
`src/gpu/kernel.ptx` is used. Unanchored regular expressions run on the CPU
only.

## Security

* Secrets come from the operating system's CSPRNG (256 bits per thread).
  Candidates are `base + counter`; only matches are ever shown, and a thread
  draws a fresh base after every match, so no two results are related.
* Every result is re-derived with a separate, table-free reference
  implementation (plain double-and-add, scalar BLAKE2b), its address is
  decoded and checksum-verified, and the pattern is re-checked before
  anything is printed. A disagreement is discarded and reported.
* The GPU only nominates candidates; each one goes through the same CPU
  verification. Its random base lives in GPU memory while it searches, is
  replaced after every hit and overwritten when the search ends normally;
  if the process exits first, the driver frees that memory without this
  program wiping it.
* Key material is zeroized after use. Nothing touches the network.
* `-o FILE` keeps the secret out of the terminal and its scrollback.
* The search itself is not constant-time (it indexes tables by secret
  digits); it is meant for a machine you control.

A vanity prefix does not make an address safe to recognise by eye — always
compare the whole address. Lookalike mode in particular produces addresses
that only resemble the pattern.

## Tests

```
cargo test --release
```

Covers Nano's documented vectors, BLAKE2b against the `blake2` crate, key
derivation against `curve25519-dalek`, both engines against the reference for
several table sizes, pattern semantics, the GPU kernel's output key by key
against the CPU (skipped without an NVIDIA GPU), and end-to-end searches on
CPU and GPU whose results are re-derived with `curve25519-dalek`.
