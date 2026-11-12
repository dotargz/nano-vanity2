use nano_vanity2::edwards::Point;
use nano_vanity2::engine::Backend;
use nano_vanity2::pattern::{GlobOptions, Matcher};
use nano_vanity2::search::{
    self, human_count, human_duration, Found, GpuMode, GpuOptions, KeyKind, Options, Summary,
};
use nano_vanity2::table::{static_table, Builder, STATIC_WINDOW};
use nano_vanity2::{address, keys, selftest};
use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use zeroize::Zeroizing;

const AFTER_HELP: &str = "\
PATTERNS (case insensitive):
  abc          addresses starting with nano_1abc or nano_3abc
  nano_3abc    pin the first character too (it can only be 1 or 3)
  a?c          ? matches any one character
  a[bc]d       [...] matches one character from a set; [a-f] and [!ab] work too
  *abc         * matches any run: this is a suffix (the last characters)
  abc*xyz      prefix and suffix
  *abc*        anywhere in the address
  Nano's alphabet is 13456789abcdefghijkmnopqrstuwxyz (no 0, 2, l or v).

LOOKALIKES (-l):
  Every character also matches its lookalikes, which makes a search several
  times faster per affected character and lets you type 0, 2, l and v:
    o=0   1=i=l   z=2   e=3   a=4   s=5   b=6=8   t=7   g=q=9   u=v

REGEX (-r):
  Matched against the whole address, e.g.  -r '^nano_[13](cafe|beef)'
  Expressions anchored with ^nano_... are much faster than unanchored ones.

EXAMPLES:
  nano-vanity2 nano                 seed for an address like nano_1nano...
  nano-vanity2 -l test              matches nano_1test, nano_37e57, ...
  nano-vanity2 -k -n 5 'me*me'      five private keys, prefix and suffix
  nano-vanity2 -o keys.txt abcde    keep the secret out of the terminal
  nano-vanity2 --check nano_3i1aq1cchnmbn9x5rsbap8b15akfh7wj7pwskuzi7ahz8oq6cobd99d4r3b7";

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackendArg {
    Auto,
    Ifma,
    Scalar,
}

const USAGE: &str = "\
Extremely fast vanity address generator for Nano.

Finds a wallet seed (or private key) whose address matches a pattern.
Everything happens offline; every result is re-derived and verified with an
independent implementation before it is printed.

Usage: nano-vanity2 [OPTIONS] <PATTERN>
       nano-vanity2 [OPTIONS] --regex <REGEX>
       nano-vanity2 --check <ADDRESS>...

Options:
  -r, --regex <REGEX>     Search with a regular expression instead of a pattern
  -s, --suffix            Match the pattern at the end of the address
  -a, --anywhere          Match the pattern anywhere in the address
  -l, --lookalike         Let characters also match their lookalikes
  -k, --private-key       Generate a raw private key instead of a wallet seed
                          (slightly faster)
  -i, --index <N>         Account index within the seed that gets the vanity
                          address [default: 0]
  -n, --count <N>         Number of matches to find [default: 1]
  -t, --threads <N>       Worker threads [default: all logical CPUs]
  -o, --output <FILE>     Append results to FILE, keep secrets off the terminal
      --json              Print results as JSON, one object per line
  -q, --quiet             No progress or statistics on stderr
      --timeout <SECONDS> Give up after this many seconds
      --backend <NAME>    auto, ifma (AVX-512 IFMA) or scalar [default: auto]
      --gpu               Always use the GPU (default: for long searches only)
      --no-gpu            Never use the GPU
      --check <ADDRESS>...  Validate addresses (prefix, alphabet, checksum)
      --bench [SECONDS]   Measure key generation speed [default: 3]
      --selftest          Verify this build on this CPU against known answers
  -h, --help              Print help
  -V, --version           Print version
";

#[derive(Default)]
struct Cli {
    pattern: Option<String>,
    regex: Option<String>,
    suffix: bool,
    anywhere: bool,
    lookalike: bool,
    private_key: bool,
    index: u32,
    count: u64,
    threads: Option<usize>,
    output: Option<PathBuf>,
    json: bool,
    quiet: bool,
    timeout: Option<f64>,
    backend: Option<BackendArg>,
    window: Option<u32>,
    check: Vec<String>,
    bench: Option<f64>,
    selftest: bool,
    trace: bool,
    gpu: GpuMode,
    gpu_check: bool,
    gpu_bench: bool,
    gpu_threads: Option<u32>,
    gpu_batch: Option<u32>,
    gpu_window: Option<u32>,
}

type ArgIter = std::iter::Peekable<std::vec::IntoIter<String>>;

fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
    v.parse().map_err(|_| format!("invalid value {v:?} for --{name}"))
}

fn value(name: &str, inline: &Option<String>, it: &mut ArgIter) -> Result<String, String> {
    match inline {
        Some(v) => Ok(v.clone()),
        None => it.next().ok_or_else(|| format!("--{name} needs a value")),
    }
}

/// Hand-rolled argument parsing: a general-purpose parser costs more
/// start-up time than a whole two-character search.
fn parse_args(args: Vec<String>) -> Result<Cli, String> {
    let mut cli = Cli { count: 1, ..Cli::default() };
    let mut it: ArgIter = args.into_iter().peekable();
    let mut positional_only = false;
    while let Some(arg) = it.next() {
        if positional_only || !arg.starts_with('-') || arg == "-" {
            if cli.pattern.is_some() {
                return Err(format!("unexpected extra argument {arg:?} (one pattern per run)"));
            }
            cli.pattern = Some(arg);
            continue;
        }
        if arg == "--" {
            positional_only = true;
            continue;
        }
        // Normalise to (name, optional inline value); expand short clusters.
        let mut items: Vec<(String, Option<String>)> = Vec::new();
        if let Some(long) = arg.strip_prefix("--") {
            match long.split_once('=') {
                Some((n, v)) => items.push((n.to_string(), Some(v.to_string()))),
                None => items.push((long.to_string(), None)),
            }
        } else {
            let chars: Vec<char> = arg[1..].chars().collect();
            let mut k = 0;
            while k < chars.len() {
                let name = match chars[k] {
                    'r' => "regex",
                    's' => "suffix",
                    'a' => "anywhere",
                    'l' => "lookalike",
                    'k' => "private-key",
                    'i' => "index",
                    'n' => "count",
                    't' => "threads",
                    'o' => "output",
                    'q' => "quiet",
                    'h' => "help",
                    'V' => "version",
                    c => return Err(format!("unknown option -{c} (see --help)")),
                };
                k += 1;
                let takes = matches!(name, "regex" | "index" | "count" | "threads" | "output");
                if takes && k < chars.len() {
                    let v: String = chars[k..].iter().collect();
                    items.push((name.to_string(), Some(v.trim_start_matches('=').to_string())));
                    k = chars.len();
                } else {
                    items.push((name.to_string(), None));
                }
            }
        }
        for (name, inline) in items {
            let n = name.as_str();
            match n {
                "regex" => cli.regex = Some(value(n, &inline, &mut it)?),
                "suffix" => cli.suffix = true,
                "anywhere" => cli.anywhere = true,
                "lookalike" => cli.lookalike = true,
                "private-key" => cli.private_key = true,
                "index" => cli.index = num(n, &value(n, &inline, &mut it)?)?,
                "count" => cli.count = num(n, &value(n, &inline, &mut it)?)?,
                "threads" => cli.threads = Some(num(n, &value(n, &inline, &mut it)?)?),
                "output" => cli.output = Some(PathBuf::from(value(n, &inline, &mut it)?)),
                "json" => cli.json = true,
                "quiet" => cli.quiet = true,
                "timeout" => cli.timeout = Some(num(n, &value(n, &inline, &mut it)?)?),
                "window" => cli.window = Some(num(n, &value(n, &inline, &mut it)?)?),
                "trace" => cli.trace = true,
                "gpu" => cli.gpu = GpuMode::On,
                "no-gpu" => cli.gpu = GpuMode::Off,
                "gpu-check" => cli.gpu_check = true,
                "gpu-bench" => cli.gpu_bench = true,
                "gpu-threads" => cli.gpu_threads = Some(num(n, &value(n, &inline, &mut it)?)?),
                "gpu-batch" => cli.gpu_batch = Some(num(n, &value(n, &inline, &mut it)?)?),
                "gpu-window" => cli.gpu_window = Some(num(n, &value(n, &inline, &mut it)?)?),
                "selftest" => cli.selftest = true,
                "backend" => {
                    cli.backend = Some(match value(n, &inline, &mut it)?.as_str() {
                        "auto" => BackendArg::Auto,
                        "ifma" => BackendArg::Ifma,
                        "scalar" => BackendArg::Scalar,
                        v => return Err(format!("unknown backend {v:?} (auto, ifma, scalar)")),
                    })
                }
                "check" => {
                    if let Some(v) = inline {
                        cli.check.push(v);
                    }
                    while let Some(a) = it.peek() {
                        if a.starts_with('-') {
                            break;
                        }
                        cli.check.push(it.next().unwrap());
                    }
                    if cli.check.is_empty() {
                        return Err("--check needs at least one address".into());
                    }
                }
                "bench" => {
                    let secs = match inline {
                        Some(v) => num(n, &v)?,
                        None => match it.peek().and_then(|a| a.parse::<f64>().ok()) {
                            Some(v) => {
                                it.next();
                                v
                            }
                            None => 3.0,
                        },
                    };
                    cli.bench = Some(secs);
                }
                "help" => {
                    print!("{USAGE}\n{AFTER_HELP}\n");
                    std::process::exit(0);
                }
                "version" => {
                    println!("nano-vanity2 {}", env!("CARGO_PKG_VERSION"));
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option --{other} (see --help)")),
            }
        }
    }
    if cli.threads == Some(0) {
        return Err("--threads must be at least 1".into());
    }
    if let Some(w) = cli.window {
        use nano_vanity2::tablegen::{MAX_WINDOW, MIN_WINDOW};
        if w != 0 && !(MIN_WINDOW..=MAX_WINDOW).contains(&w) {
            return Err(format!("--window must be 0 or {MIN_WINDOW}..={MAX_WINDOW}"));
        }
    }
    Ok(cli)
}

fn main() {
    let start = Instant::now();
    let cli = match parse_args(std::env::args().skip(1).collect()) {
        Ok(c) => c,
        Err(e) => std::process::exit(fail(e)),
    };
    if cli.trace {
        nano_vanity2::trace::enable();
    }
    nano_vanity2::trace::mark("arguments parsed");
    let code = if cli.gpu_check || cli.gpu_bench {
        cmd_gpu(&cli)
    } else if cli.selftest {
        cmd_selftest()
    } else if !cli.check.is_empty() {
        cmd_check(&cli.check)
    } else if let Some(seconds) = cli.bench {
        cmd_bench(&cli, seconds)
    } else {
        cmd_search(&cli, start)
    };
    std::process::exit(code);
}

fn fail(msg: impl std::fmt::Display) -> i32 {
    eprintln!("error: {msg}");
    2
}

fn pick_backend(arg: Option<BackendArg>) -> Result<Backend, String> {
    match arg.unwrap_or(BackendArg::Auto) {
        BackendArg::Auto => Ok(Backend::detect()),
        BackendArg::Scalar => Ok(Backend::Scalar),
        BackendArg::Ifma => {
            if Backend::Ifma.available() {
                Ok(Backend::Ifma)
            } else {
                Err("this CPU does not support AVX-512 IFMA".into())
            }
        }
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn open_secret_file(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

fn json_line(f: &Found, with_secret: bool) -> Zeroizing<String> {
    let mut s = Zeroizing::new(String::with_capacity(400));
    s.push_str(&format!("{{\"address\":\"{}\",\"public_key\":\"{}\"", f.address, keys::hex_upper(&f.public)));
    if with_secret {
        let hex = Zeroizing::new(keys::hex_upper(&f.secret[..]));
        match f.kind {
            KeyKind::Seed { index } => {
                s.push_str(&format!(",\"seed\":\"{}\",\"index\":{}", hex.as_str(), index))
            }
            KeyKind::Private => s.push_str(&format!(",\"private_key\":\"{}\"", hex.as_str())),
        }
    }
    s.push_str(&format!(",\"attempts\":{},\"seconds\":{:.9}}}", f.attempts, f.elapsed.as_secs_f64()));
    s
}

fn text_block(f: &Found, with_secret: bool) -> Zeroizing<String> {
    let mut s = Zeroizing::new(String::with_capacity(400));
    s.push_str(&format!("Address:     {}\n", f.address));
    if with_secret {
        let hex = Zeroizing::new(keys::hex_upper(&f.secret[..]));
        match f.kind {
            KeyKind::Seed { index } => {
                s.push_str(&format!("Seed:        {}\n", hex.as_str()));
                if index != 0 {
                    s.push_str(&format!("Index:       {index}\n"));
                }
            }
            KeyKind::Private => s.push_str(&format!("Private key: {}\n", hex.as_str())),
        }
    }
    s.push_str(&format!("Public key:  {}\n", keys::hex_upper(&f.public)));
    s
}

fn print_summary(cli: &Cli, matcher: &Matcher, s: &Summary) {
    if cli.quiet {
        return;
    }
    let rate = s.attempts as f64 / s.search_elapsed.as_secs_f64().max(1e-9);
    let gpu = match &s.gpu {
        Some(name) if s.gpu_attempts > 0 => format!(" + {name}"),
        _ => String::new(),
    };
    let how = format!(
        "{} keys tried at {} keys/s ({}, {} thread{}{gpu})",
        human_count(s.attempts as f64),
        human_count(rate),
        s.backend.name(),
        s.threads,
        if s.threads == 1 { "" } else { "s" }
    );
    if s.found == 0 {
        eprintln!("No match for {} after {}: {how}", matcher.display, human_duration(s.elapsed));
    } else if s.found == 1 {
        eprintln!("Found in {}: {how}", human_duration(s.elapsed));
    } else {
        eprintln!("Found {} matches in {}: {how}", s.found, human_duration(s.elapsed));
    }
    if let (Some(e), GpuMode::On) = (&s.gpu_error, cli.gpu) {
        eprintln!("note: the GPU was not used: {e}");
    }
    if s.mismatches > 0 {
        eprintln!(
            "warning: {} candidate(s) failed independent verification and were discarded. \
             Results shown are verified, but please run --selftest and report this.",
            s.mismatches
        );
    }
}

fn cmd_search(cli: &Cli, start: Instant) -> i32 {
    let matcher = match (&cli.pattern, &cli.regex) {
        (Some(p), None) => Matcher::glob(
            p,
            &GlobOptions { suffix: cli.suffix, anywhere: cli.anywhere, lookalike: cli.lookalike },
        ),
        (None, Some(r)) => {
            if cli.suffix || cli.anywhere || cli.lookalike {
                return fail("--suffix, --anywhere and --lookalike do not apply to --regex");
            }
            Matcher::regex(r)
        }
        _ => return fail("give a pattern (or --regex); try --help for the syntax and examples"),
    };
    let matcher = match matcher {
        Ok(m) => m,
        Err(e) => return fail(e),
    };
    let backend = match pick_backend(cli.backend) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };
    if cli.count == 0 {
        return fail("--count must be at least 1");
    }
    let timeout = match cli.timeout {
        Some(t) if t.is_finite() && t >= 0.0 => Some(Duration::from_secs_f64(t)),
        Some(_) => return fail("--timeout must be a non-negative number of seconds"),
        None => None,
    };
    let file = match &cli.output {
        Some(p) => match open_secret_file(p) {
            Ok(f) => Some(Mutex::new(f)),
            Err(e) => return fail(format!("cannot open {}: {e}", p.display())),
        },
        None => None,
    };

    let opts = Options {
        matcher: &matcher,
        kind: if cli.private_key { KeyKind::Private } else { KeyKind::Seed { index: cli.index } },
        threads: cli.threads.unwrap_or_else(default_threads),
        count: cli.count,
        timeout,
        backend,
        window: cli.window,
        start,
        progress: !cli.quiet && std::io::stderr().is_terminal(),
        gpu: GpuOptions { mode: cli.gpu, threads: cli.gpu_threads, batch: cli.gpu_batch, window: cli.gpu_window },
    };

    let first = std::sync::atomic::AtomicBool::new(true);
    let on_found = |f: &Found| {
        let to_file = file.is_some();
        if let Some(file) = &file {
            let mut file = file.lock().unwrap();
            let rec = if cli.json {
                let mut l = json_line(f, true);
                l.push('\n');
                l
            } else {
                let mut b = text_block(f, true);
                b.push('\n');
                b
            };
            if let Err(e) = file.write_all(rec.as_bytes()).and_then(|_| file.flush()) {
                // Do not lose a found key because the file went away.
                eprintln!("error: writing to the output file failed ({e}); printing the result instead");
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(rec.as_bytes());
                return;
            }
        }
        let mut out = std::io::stdout().lock();
        if cli.json {
            let _ = writeln!(out, "{}", json_line(f, !to_file).as_str());
        } else {
            if !first.swap(false, std::sync::atomic::Ordering::Relaxed) {
                let _ = writeln!(out);
            }
            let _ = out.write_all(text_block(f, !to_file).as_bytes());
            if let Some(p) = &cli.output {
                let what = if cli.private_key { "Private key" } else { "Seed" };
                let _ = writeln!(out, "{what} saved to {}", p.display());
            }
        }
        let _ = out.flush();
    };

    nano_vanity2::trace::mark("pattern compiled, starting search");
    let summary = search::run(&opts, &on_found);
    nano_vanity2::trace::mark("search returned");
    nano_vanity2::trace::dump(start);
    print_summary(cli, &matcher, &summary);
    if summary.found >= cli.count {
        0
    } else {
        1
    }
}

fn cmd_check(addresses: &[String]) -> i32 {
    let mut bad = 0;
    for a in addresses {
        let a = a.trim();
        match address::decode(a) {
            Ok(public) => {
                let curve = if Point::decompress(&public).is_some() {
                    ""
                } else {
                    "  (not a curve point: nobody can hold its key)"
                };
                println!("valid    {a}\n         public key {}{curve}", keys::hex_upper(&public));
            }
            Err(e) => {
                bad += 1;
                println!("INVALID  {a}\n         {e}");
            }
        }
    }
    if bad == 0 {
        0
    } else {
        1
    }
}

fn cmd_bench(cli: &Cli, seconds: f64) -> i32 {
    if !(seconds.is_finite() && seconds > 0.0) {
        return fail("--bench needs a positive number of seconds");
    }
    let backend = match pick_backend(cli.backend) {
        Ok(b) => b,
        Err(e) => return fail(e),
    };
    let threads = cli.threads.unwrap_or_else(default_threads);
    let kind = if cli.private_key { KeyKind::Private } else { KeyKind::Seed { index: cli.index } };
    println!(
        "{} backend, {} thread{}, generating {}",
        backend.name(),
        threads,
        if threads == 1 { "" } else { "s" },
        if cli.private_key { "private keys" } else { "seeds" }
    );
    // Patterns that will not be found: twelve leading z's, eight trailing z's.
    for (label, pattern) in [("prefix search", "zzzzzzzzzzzz"), ("suffix search", "*zzzzzzzz")] {
        let matcher = Matcher::glob(pattern, &GlobOptions::default()).unwrap();
        let opts = Options {
            matcher: &matcher,
            kind,
            threads,
            count: 1,
            timeout: Some(Duration::from_secs_f64(seconds)),
            backend,
            window: cli.window.or(search::choose_window(f64::INFINITY)),
            start: Instant::now(),
            progress: false,
            gpu: GpuOptions { mode: GpuMode::Off, ..GpuOptions::default() },
        };
        let s = search::run(&opts, &|_| {});
        let rate = s.attempts as f64 / s.search_elapsed.as_secs_f64().max(1e-9);
        println!(
            "{label:14} {:>9} keys/s  ({} per thread, table window {})",
            human_count(rate),
            human_count(rate / s.threads as f64),
            s.window
        );
    }
    0
}

fn cmd_selftest() -> i32 {
    let mut failures = 0;
    let mut report = |name: &str, r: Result<String, String>| match r {
        Ok(detail) => println!("ok    {name}{detail}"),
        Err(e) => {
            failures += 1;
            println!("FAIL  {name}: {e}");
        }
    };
    report("known answers (Nano documentation vectors)", selftest::known_answers().map(|_| String::new()));

    let mut key = [0u8; 32];
    if getrandom::fill(&mut key).is_err() {
        report("operating system random numbers", Err("unavailable".into()));
        return 1;
    }
    let mut words = [0u64; 4];
    for i in 0..4 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&key[i * 8..i * 8 + 8]);
        words[i] = u64::from_le_bytes(t);
    }
    for backend in [Backend::Scalar, Backend::Ifma] {
        if !backend.available() {
            println!("skip  {} engine (not supported by this CPU)", backend.name());
            continue;
        }
        for w in [STATIC_WINDOW, 5, 11, search::BIG_WINDOW, 16] {
            let built;
            let table = if w == STATIC_WINDOW {
                static_table()
            } else {
                built = Builder::build_now(w);
                built.table()
            };
            for seed_index in [None, Some(0), Some(7)] {
                let name = format!(
                    "{} engine, table window {w}, {}",
                    backend.name(),
                    match seed_index {
                        None => "private keys".to_string(),
                        Some(i) => format!("seeds (index {i})"),
                    }
                );
                // Start just below a counter wrap to exercise it as well.
                let r = selftest::engine_check(backend, table, seed_index, words, u64::MAX - 100, 40)
                    .map(|n| format!(" ({n} keys verified)"));
                report(&name, r);
            }
        }
    }
    if !nano_vanity2::gpu::driver_present() {
        println!("skip  GPU (no NVIDIA CUDA driver)");
    } else {
        for w in [search::BIG_WINDOW, nano_vanity2::gpu::DEFAULT_WINDOW] {
            let built = Builder::build_now(w);
            let r = nano_vanity2::gpu::self_check(built.table())
                .map(|name| format!(" on {name} (all modes, 15360 keys compared with the CPU)"));
            report(&format!("GPU kernel, table window {w}"), r);
        }
    }
    if failures == 0 {
        println!("all tests passed");
        0
    } else {
        println!("{failures} test(s) FAILED - do not use this build");
        1
    }
}

/// Hidden developer commands: --gpu-check and --gpu-bench.
fn cmd_gpu(cli: &Cli) -> i32 {
    use nano_vanity2::gpu;
    let w = cli.gpu_window.unwrap_or(gpu::DEFAULT_WINDOW);
    let built = Builder::new(w);
    std::thread::scope(|s| {
        for _ in 0..default_threads() {
            s.spawn(|| built.help());
        }
    });
    if cli.gpu_check {
        match gpu::self_check(built.table()) {
            Ok(name) => println!("ok    {name}, table window {w}"),
            Err(e) => return fail(e),
        }
    }
    if cli.gpu_bench {
        let threads = cli.gpu_threads.unwrap_or(gpu::DEFAULT_THREADS);
        let batch = cli.gpu_batch.unwrap_or(gpu::DEFAULT_BATCH);
        for (label, full) in [("prefix", false), ("suffix", true)] {
            match gpu::benchmark(built.table(), !cli.private_key, full, threads, batch, cli.bench.unwrap_or(2.0)) {
                Ok((name, rate, per)) => println!(
                    "{name} {label}: {} keys/s (window {w}, {threads} threads x {batch}, {:.1} ms per launch)",
                    human_count(rate),
                    per * 1e3
                ),
                Err(e) => return fail(e),
            }
        }
    }
    0
}
