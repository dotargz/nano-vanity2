//! Pattern compilation and matching.
//!
//! Glob syntax (case insensitive):
//!   * a Nano alphabet character matches itself
//!   * `?` matches any one character
//!   * `[abc]`, `[a-f]`, `[!abc]` match one character from (or not from) a set
//!   * `*` matches any run of characters
//!
//! A pattern without `*` is a prefix, placed right after the leading `1`/`3`
//! that every address has; write `nano_1...` / `nano_3...` to pin that
//! character too. `*abc` is a suffix, `abc*xyz` a prefix plus a suffix and
//! `*abc*` matches anywhere.
//!
//! Matching works on the address *bits* rather than on text: an address body
//! is 60 base32 characters = 300 bits (4 zero bits, the 256-bit public key,
//! the 40-bit checksum), kept as a left-aligned bit string in five words.

use crate::address::{self, symbol, ALPHABET, BODY_LEN, KEY_LEN};
use crate::engine::Prefilter;
use std::fmt;

const ALL: u32 = u32::MAX;
/// Positions whose characters lie entirely within the first 8 key bytes.
const QUICK_LAST: usize = 12;
/// A quick filter at least this selective makes the slower follow-up checks
/// rare enough that the engine can skip computing full keys for everybody.
const QUICK_ENOUGH: f64 = 1.0 / 4096.0;

#[derive(Debug)]
pub enum PatternError {
    Empty,
    InvalidChar { c: char, hint: Option<&'static str> },
    UnclosedSet,
    EmptySet,
    TooLong { len: usize, max: usize },
    FirstChar,
    Conflict(&'static str),
    Regex(String),
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::Empty => write!(f, "the pattern is empty"),
            PatternError::InvalidChar { c, hint: Some(h) } => write!(
                f,
                "{c:?} never appears in a Nano address (alphabet: {}); with --lookalike it would match {}",
                std::str::from_utf8(ALPHABET).unwrap(),
                h.chars().map(|c| format!("'{c}'")).collect::<Vec<_>>().join(" or ")
            ),
            PatternError::InvalidChar { c, hint: None } => write!(
                f,
                "{c:?} never appears in a Nano address (alphabet: {})",
                std::str::from_utf8(ALPHABET).unwrap()
            ),
            PatternError::UnclosedSet => write!(f, "'[' without a closing ']'"),
            PatternError::EmptySet => write!(f, "a [...] set matches no Nano address character"),
            PatternError::TooLong { len, max } => {
                write!(f, "the pattern needs {len} characters but only {max} are available")
            }
            PatternError::FirstChar => {
                write!(f, "the first character after nano_ can only be 1 or 3")
            }
            PatternError::Conflict(s) => write!(f, "{s}"),
            PatternError::Regex(e) => write!(f, "invalid regular expression: {e}"),
        }
    }
}

impl std::error::Error for PatternError {}

#[derive(Clone, Copy, Default, Debug)]
pub struct GlobOptions {
    /// Treat a pattern without `*` as a suffix.
    pub suffix: bool,
    /// Treat a pattern without `*` as matching anywhere.
    pub anywhere: bool,
    /// Let every character also match its lookalikes.
    pub lookalike: bool,
}

/// Characters that look like `c`, as address characters.
pub fn lookalikes(c: char) -> Option<&'static str> {
    Some(match c {
        '0' | 'o' => "o",
        '1' | 'i' | 'l' => "1i",
        '2' | 'z' => "z",
        '3' | 'e' => "3e",
        '4' | 'a' => "4a",
        '5' | 's' => "5s",
        '6' => "6b",
        'b' => "b68",
        '8' => "8b",
        '7' | 't' => "7t",
        '9' => "9gq",
        'g' => "g9",
        'q' => "q9",
        'u' | 'v' => "u",
        _ => return None,
    })
}

fn mask_of(chars: &str) -> u32 {
    chars.bytes().fold(0, |m, c| m | 1 << symbol(c).unwrap())
}

fn char_set(c: char, lookalike: bool) -> Result<u32, PatternError> {
    if lookalike {
        if let Some(s) = lookalikes(c) {
            return Ok(mask_of(s));
        }
    }
    if c.is_ascii() {
        if let Some(s) = symbol(c as u8) {
            return Ok(1 << s);
        }
    }
    Err(PatternError::InvalidChar { c, hint: lookalikes(c) })
}

enum Tok {
    Set(u32),
    Star,
}

fn tokenize(s: &str, lookalike: bool) -> Result<Vec<Tok>, PatternError> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        match c {
            '*' => {
                if !matches!(toks.last(), Some(Tok::Star)) {
                    toks.push(Tok::Star);
                }
            }
            '?' => toks.push(Tok::Set(ALL)),
            '[' => {
                let mut negate = false;
                if i < chars.len() && (chars[i] == '!' || chars[i] == '^') {
                    negate = true;
                    i += 1;
                }
                let mut set = 0u32;
                let mut closed = false;
                while i < chars.len() {
                    let a = chars[i];
                    i += 1;
                    if a == ']' {
                        closed = true;
                        break;
                    }
                    if i + 1 < chars.len() && chars[i] == '-' && chars[i + 1] != ']' {
                        // Range in ASCII order, restricted to the alphabet.
                        let b = chars[i + 1];
                        i += 2;
                        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                        let mut any = false;
                        for &sym in ALPHABET.iter() {
                            if (lo..=hi).contains(&(sym as char)) {
                                set |= 1 << symbol(sym).unwrap();
                                any = true;
                            }
                        }
                        if !any {
                            return Err(PatternError::EmptySet);
                        }
                    } else {
                        set |= char_set(a, lookalike)?;
                    }
                }
                if !closed {
                    return Err(PatternError::UnclosedSet);
                }
                if negate {
                    set = !set;
                }
                if set == 0 {
                    return Err(PatternError::EmptySet);
                }
                toks.push(Tok::Set(set));
            }
            _ => toks.push(Tok::Set(char_set(c, lookalike)?)),
        }
    }
    Ok(toks)
}

/// The 300-bit address body, left aligned in 320 bits, from a public key
/// given as little-endian words and its checksum.
#[inline(always)]
pub fn bits_from_words(pk: &[u64; 4], cks: u64) -> [u64; 5] {
    [pk[0].swap_bytes(), pk[1].swap_bytes(), pk[2].swap_bytes(), pk[3].swap_bytes(), cks << 24]
}

pub fn bits_from_public(public: &[u8; 32]) -> [u64; 5] {
    let mut w = [0u64; 4];
    for i in 0..4 {
        let mut t = [0u8; 8];
        t.copy_from_slice(&public[i * 8..i * 8 + 8]);
        w[i] = u64::from_le_bytes(t);
    }
    bits_from_words(&w, address::checksum(public))
}

/// Alphabet index of address character `p` (0-based, after `nano_`).
#[inline(always)]
pub fn sym_at(u: &[u64; 5], p: usize) -> u32 {
    if p == 0 {
        // Four padding bits plus the first key bit.
        return (u[0] >> 63) as u32;
    }
    let b = 5 * p - 4;
    let w = b >> 6;
    let r = b & 63;
    if r <= 59 {
        ((u[w] >> (59 - r)) & 31) as u32
    } else {
        (((u[w] << (r - 59)) | (u[w + 1] >> (123 - r))) & 31) as u32
    }
}

pub fn symbols(u: &[u64; 5]) -> [u8; BODY_LEN] {
    let mut s = [0u8; BODY_LEN];
    for (p, o) in s.iter_mut().enumerate() {
        *o = sym_at(u, p) as u8;
    }
    s
}

/// A compiled pattern.
pub struct Matcher {
    /// (position, allowed symbols), most selective first.
    fixed: Vec<(u8, u32)>,
    /// The subset of `fixed` within the first 8 key bytes.
    quick: Vec<(u8, u32)>,
    /// Segments that may float, in order, within `float_lo..float_hi`.
    floating: Vec<Vec<u32>>,
    float_lo: usize,
    float_hi: usize,
    regex: Option<regex::Regex>,
    /// Bit filter the engine applies to every candidate.
    pub prefilter: Prefilter,
    /// The engine must compute the complete key and checksum of every
    /// candidate (the quick filter alone is not selective enough).
    pub need_full: bool,
    /// There are conditions beyond what the quick positions can decide.
    pub has_other: bool,
    /// Chance that a random address matches, if it can be estimated.
    pub probability: Option<f64>,
    /// True when `probability` is only an upper bound (regular expressions).
    pub probability_is_bound: bool,
    /// Human-readable form.
    pub display: String,
}

fn set_probability(pos: usize, mask: u32) -> f64 {
    if pos == 0 {
        (mask & 3).count_ones() as f64 / 2.0
    } else {
        mask.count_ones() as f64 / 32.0
    }
}

fn show_set(mask: u32) -> String {
    if mask == ALL {
        return "?".into();
    }
    let chars: String =
        (0..32).filter(|s| mask >> s & 1 == 1).map(|s| ALPHABET[s as usize] as char).collect();
    if chars.len() == 1 {
        chars
    } else {
        format!("[{chars}]")
    }
}

impl Matcher {
    fn finish(
        mut fixed: Vec<(u8, u32)>,
        floating: Vec<Vec<u32>>,
        float_lo: usize,
        float_hi: usize,
        regex: Option<regex::Regex>,
        display: String,
    ) -> Result<Matcher, PatternError> {
        // Normalise: drop unconstrained positions, restrict position 0.
        let mut clean = Vec::new();
        for (pos, mut mask) in fixed.drain(..) {
            if pos == 0 {
                mask &= 3;
                if mask == 0 {
                    return Err(PatternError::FirstChar);
                }
                if mask == 3 {
                    continue;
                }
            } else if mask == ALL {
                continue;
            }
            clean.push((pos, mask));
        }
        clean.sort_by_key(|&(_, mask)| mask.count_ones());
        let fixed = clean;
        let quick: Vec<(u8, u32)> =
            fixed.iter().copied().filter(|&(p, _)| p as usize <= QUICK_LAST).collect();

        // Prefilter: for every constrained position, the bits on which all
        // allowed symbols agree.
        let mut m_be = 0u64;
        let mut v_be = 0u64;
        let mut mc = 0u64;
        let mut vc = 0u64;
        for &(pos, mask) in &fixed {
            let pos = pos as usize;
            let mut and_all = 31u32;
            let mut or_all = 0u32;
            for s in 0..32u32 {
                if mask >> s & 1 == 1 {
                    and_all &= s;
                    or_all |= s;
                }
            }
            let agree = !(and_all ^ or_all) & 31;
            let value = and_all & agree;
            if pos == 0 {
                m_be |= 1 << 63;
                v_be |= ((mask >> 1 & 1) as u64) << 63;
            } else if pos <= QUICK_LAST {
                let shift = 63 - 5 * pos;
                m_be |= (agree as u64) << shift;
                v_be |= (value as u64) << shift;
            } else if pos >= KEY_LEN {
                let shift = 35 - 5 * (pos - KEY_LEN);
                mc |= (agree as u64) << shift;
                vc |= (value as u64) << shift;
            }
        }
        let prefilter = Prefilter { m0: m_be.swap_bytes(), v0: v_be.swap_bytes(), mc, vc };

        let quick_p: f64 = quick.iter().map(|&(p, m)| set_probability(p as usize, m)).product();
        let has_other =
            regex.is_some() || !floating.is_empty() || fixed.iter().any(|&(p, _)| p as usize > QUICK_LAST);
        let need_full = has_other && quick_p > QUICK_ENOUGH;

        let mut p: f64 = fixed.iter().map(|&(pos, m)| set_probability(pos as usize, m)).product();
        for seg in &floating {
            let seg_p: f64 = seg.iter().map(|m| m.count_ones() as f64 / 32.0).product();
            let places = (float_hi - float_lo + 1).saturating_sub(seg.len()).max(1);
            p *= (seg_p * places as f64).min(1.0);
        }
        let (probability, probability_is_bound) = match (&regex, fixed.is_empty()) {
            (Some(_), true) => (None, false),
            (Some(_), false) => (Some(p), true),
            (None, _) => (Some(p), false),
        };

        Ok(Matcher {
            fixed,
            quick,
            floating,
            float_lo,
            float_hi,
            regex,
            prefilter,
            need_full,
            has_other,
            probability,
            probability_is_bound,
            display,
        })
    }

    /// Compile a glob pattern (see the module documentation).
    pub fn glob(pattern: &str, opt: &GlobOptions) -> Result<Matcher, PatternError> {
        let lower = pattern.trim().to_lowercase();
        let (body, pinned) = match lower.strip_prefix("nano_").or_else(|| lower.strip_prefix("xrb_")) {
            Some(b) => (b, true),
            None => (lower.as_str(), false),
        };
        let toks = tokenize(body, opt.lookalike)?;
        if toks.is_empty() {
            return Err(PatternError::Empty);
        }
        let has_star = toks.iter().any(|t| matches!(t, Tok::Star));
        if opt.suffix && opt.anywhere {
            return Err(PatternError::Conflict("--suffix and --anywhere cannot be combined"));
        }
        if has_star && (opt.suffix || opt.anywhere) {
            return Err(PatternError::Conflict(
                "--suffix/--anywhere only apply to patterns without '*'",
            ));
        }
        let (leading, trailing) = if has_star {
            (matches!(toks.first(), Some(Tok::Star)), matches!(toks.last(), Some(Tok::Star)))
        } else if opt.suffix {
            (true, false)
        } else if opt.anywhere {
            (true, true)
        } else {
            (false, true)
        };
        if pinned && leading {
            return Err(PatternError::Conflict(
                "a pattern starting with nano_ cannot also be a suffix or float",
            ));
        }

        let mut segs: Vec<Vec<u32>> = Vec::new();
        let mut cur = Vec::new();
        for t in toks {
            match t {
                Tok::Set(m) => cur.push(m),
                Tok::Star => {
                    if !cur.is_empty() {
                        segs.push(std::mem::take(&mut cur));
                    }
                }
            }
        }
        if !cur.is_empty() {
            segs.push(cur);
        }

        let prefix = if !leading && !segs.is_empty() { segs.remove(0) } else { Vec::new() };
        let suffix = if !trailing && !segs.is_empty() { segs.pop().unwrap() } else { Vec::new() };
        let floating = segs;

        let start = if pinned { 0 } else { 1 };
        let need = prefix.len() + suffix.len() + floating.iter().map(|s| s.len()).sum::<usize>();
        if need > BODY_LEN - start {
            return Err(PatternError::TooLong { len: need, max: BODY_LEN - start });
        }

        let mut fixed = Vec::new();
        for (k, &m) in prefix.iter().enumerate() {
            fixed.push(((start + k) as u8, m));
        }
        let suffix_at = BODY_LEN - suffix.len();
        for (k, &m) in suffix.iter().enumerate() {
            fixed.push(((suffix_at + k) as u8, m));
        }

        let mut display = String::from("nano_");
        if !pinned {
            display.push_str("[13]");
        }
        for &m in &prefix {
            display.push_str(&show_set(m));
        }
        if !floating.is_empty() || !suffix.is_empty() || prefix.len() + start < BODY_LEN {
            display.push('…');
        }
        for seg in &floating {
            for &m in seg {
                display.push_str(&show_set(m));
            }
            display.push('…');
        }
        for &m in &suffix {
            display.push_str(&show_set(m));
        }

        Matcher::finish(fixed, floating, start + prefix.len(), suffix_at, None, display)
    }

    /// Compile a regular expression, matched against the complete address
    /// (`nano_` included).
    pub fn regex(pattern: &str) -> Result<Matcher, PatternError> {
        let re = regex::Regex::new(pattern).map_err(|e| PatternError::Regex(e.to_string()))?;
        let fixed = regex_prefix_sets(pattern)
            .map(|sets| sets.into_iter().enumerate().map(|(p, m)| (p as u8, m)).collect())
            .unwrap_or_default();
        Matcher::finish(fixed, Vec::new(), 0, BODY_LEN, Some(re), format!("/{pattern}/"))
    }

    /// Exact test of every condition that lies within the first 8 key bytes.
    /// `be0` is those bytes as a big-endian word.
    #[inline]
    pub fn check_quick(&self, be0: u64) -> bool {
        for &(pos, mask) in &self.quick {
            let sym = if pos == 0 { be0 >> 63 } else { (be0 >> (63 - 5 * pos as u32)) & 31 };
            if mask >> sym & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Complete test against the address bits.
    pub fn check_bits(&self, u: &[u64; 5]) -> bool {
        for &(pos, mask) in &self.fixed {
            if mask >> sym_at(u, pos as usize) & 1 == 0 {
                return false;
            }
        }
        if self.floating.is_empty() && self.regex.is_none() {
            return true;
        }
        let syms = symbols(u);
        let mut at = self.float_lo;
        for seg in &self.floating {
            let mut found = false;
            while at + seg.len() <= self.float_hi {
                if seg.iter().zip(&syms[at..]).all(|(&m, &s)| m >> s & 1 == 1) {
                    found = true;
                    break;
                }
                at += 1;
            }
            if !found {
                return false;
            }
            at += seg.len();
        }
        if let Some(re) = &self.regex {
            let mut text = [0u8; 5 + BODY_LEN];
            text[..5].copy_from_slice(b"nano_");
            for (o, &s) in text[5..].iter_mut().zip(syms.iter()) {
                *o = ALPHABET[s as usize];
            }
            // The buffer only holds ASCII.
            return re.is_match(std::str::from_utf8(&text).unwrap());
        }
        true
    }

    /// Complete test against a public key.
    pub fn check_public(&self, public: &[u8; 32]) -> bool {
        self.check_bits(&bits_from_public(public))
    }

    /// The position-wise conditions a GPU kernel can test: the fixed
    /// positions it sees in this mode, the floating segments and their range.
    pub fn gpu_parts(&self, full: bool) -> (Vec<(u8, u32)>, Vec<Vec<u32>>, usize, usize) {
        if full {
            (self.fixed.clone(), self.floating.clone(), self.float_lo, self.float_hi)
        } else {
            (self.quick.clone(), Vec::new(), 0, 0)
        }
    }

    /// Chance that a random key passes the conditions on the first 8 bytes.
    pub fn quick_probability(&self) -> f64 {
        self.quick.iter().map(|&(p, m)| set_probability(p as usize, m)).product()
    }

    /// Expected number of keys to try per match.
    pub fn expected_attempts(&self) -> Option<f64> {
        self.probability.map(|p| if p > 0.0 { 1.0 / p } else { f64::INFINITY })
    }
}

/// For a regex anchored at the start, the sets of characters its matches can
/// have at each leading address position (a necessary condition only).
fn regex_prefix_sets(pattern: &str) -> Option<Vec<u32>> {
    use regex_syntax::hir::literal::{ExtractKind, Extractor};
    use regex_syntax::hir::Look;

    let hir = regex_syntax::Parser::new().parse(pattern).ok()?;
    if !hir.properties().look_set_prefix().contains(Look::Start) {
        return None;
    }
    let seq = Extractor::new().kind(ExtractKind::Prefix).extract(&hir);
    let lits = seq.literals()?;
    let mut sets: Option<Vec<u32>> = None;
    for lit in lits {
        let Some(body) = lit.as_bytes().strip_prefix(b"nano_") else {
            if b"nano_".starts_with(lit.as_bytes()) {
                // Too short to say anything about the address body.
                return None;
            }
            // A literal that cannot start an address never matches.
            continue;
        };
        let mut these = Vec::new();
        let mut possible = true;
        for &c in body.iter().take(BODY_LEN) {
            match symbol(c) {
                Some(s) => these.push(1u32 << s),
                None => {
                    possible = false;
                    break;
                }
            }
        }
        if !possible {
            continue;
        }
        sets = Some(match sets {
            None => these,
            Some(mut acc) => {
                acc.truncate(these.len());
                for (a, t) in acc.iter_mut().zip(these.iter()) {
                    *a |= t;
                }
                acc
            }
        });
    }
    sets.filter(|s| !s.is_empty())
}
