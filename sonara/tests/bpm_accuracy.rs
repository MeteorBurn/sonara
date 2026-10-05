//! BPM accuracy regression harness (Layer 1): synthetic signals and real-music
//! envelope fixtures.
//!
//! This suite guards *correctness* of tempo detection against deterministic
//! ground-truth signals — complementing the speed benches in `benches/`, which
//! only guard throughput.
//!
//! Motivation: beat trackers commonly make two classes of error that are
//! invisible to speed benchmarks:
//!   1. **Octave errors** — reporting ~0.5x or ~2x the true tempo. These are
//!      especially common on patterns with strong offbeat energy (e.g. hats
//!      between kicks), which can make the detector lock onto a pulse at twice
//!      the true beat rate.
//!   2. **Near-miss drift** — reporting a tempo 1-3 BPM off the truth.
//!
//! We synthesize signals with *exactly* known tempo (click trains, kick
//! patterns, and syncopated kick+hat patterns) across a spread of BPMs that
//! covers the historically problematic zones, run the analysis pipeline's
//! tempo detector, and compute:
//!   - accuracy @ +/-0.5 BPM
//!   - accuracy @ +/-2%
//!   - octave-error rate  (MUST be 0 for the non-known-failing subset)
//!   - median / p95 absolute BPM error
//!
//! NOTE FOR THE DETECTOR TEAM: the pass/fail thresholds and the
//! `KNOWN_FAILING` list are grouped together at the top of this file. This
//! suite does **not** modify any detection logic (`src/beat.rs` etc.) — it only
//! measures. After the detector is improved, re-run this suite, tighten
//! `MAX_MEDIAN_ABS_ERR` / trim `KNOWN_FAILING`, and un-`#[ignore]` the per-case
//! tests below.

use ndarray::Array1;
use std::f32::consts::PI;

type Float = f32;

// ============================================================
// TUNABLE THRESHOLDS + KNOWN-FAILING LIST  (edit here only)
// ============================================================

/// Sample rate used for all synthetic signals.
const SR: u32 = 22050;

/// Duration (seconds) of each synthetic signal. Longer = better tempo
/// resolution; 12s comfortably resolves the tempo grid.
const DUR_SEC: Float = 12.0;

/// A detected tempo counts as "correct" if within this fraction of truth.
const PCT_TOL: Float = 0.02; // +/-2%

/// A detected tempo counts as an "octave error" if it is NOT correct but is
/// within this fraction of a 0.5x / 2x / (1/3)x / 3x multiple of truth.
const OCTAVE_TOL: Float = 0.04;

/// Aggregate guard: median absolute BPM error over the non-known-failing
/// subset must stay under this. Current `main` achieves a guarded-subset median
/// of ~0.66 BPM (p95 ~2.0). This threshold leaves headroom for detector noise
/// while still catching a broad regression — ratchet it *down* as the detector
/// improves.
const MAX_MEDIAN_ABS_ERR: Float = 1.5;

/// Aggregate guard: octave-error rate over the non-known-failing subset must be
/// exactly this (0). Octave errors are the headline failure mode; we never
/// tolerate them in the guarded subset.
const MAX_OCTAVE_ERRORS: usize = 0;

/// Duration (seconds) of the beat-period precision signals: long enough for
/// the beat-pair consensus at every tested tempo (70 beats at 70 BPM).
const PRECISION_DUR_SEC: Float = 60.0;

/// Beat-period precision guard: the reported BPM, folded into the octave of
/// the true tempo, must be within this many BPM of it.
const MAX_PRECISION_ABS_ERR: Float = 0.03;

/// Cases that current `main` gets wrong (octave error or large drift). Listed
/// as `(bpm, pattern)` so they are excluded from the aggregate hard-asserts but
/// still measured and printed in the report. Each also has a `#[ignore]`d
/// per-case test below documenting the observed failure. TRIM THIS as the
/// detector improves.
///
/// Measured on `main` after the tempo-candidate selection improvements (2026-07).
/// The port fixed the 126/140 BPM halving (all patterns now within +/-0.5 BPM);
/// the remaining known failures are:
///   - 192: all patterns -> ~95.9 (octave). The metrical-lift tiers only rescue
///     raw detections below 95 BPM, so 192->96 sits just outside them. The
///     supported fix is a `bpm_min`/`bpm_max` project range whose floor
///     excludes ~96 (e.g. 100-200 doubles 96 to 192; note bpm_max must be >= 2x bpm_min); an
///     unconditional lift at ~96 would regress real ~96 BPM material.
///   - 60/63/70 kick+hats -> exactly 2x (octave). Cost of the metrical lift:
///     a strong offbeat 8th-note hat layer at 2x the pulse is treated as
///     evidence for the doubled tempo. On real-world DJ material, halving
///     errors dominate and this trade wins; genuinely
///     slow material with dense offbeat hats needs `bpm_max` or beat-grid
///     regularity scoring (planned) to disambiguate.
/// TRIM this list (and un-ignore the per-case tests) as the detector improves.
const KNOWN_FAILING: &[(Float, Pattern)] = &[
    (60.0, Pattern::KickOffbeatHats),
    (63.0, Pattern::KickOffbeatHats),
    (70.0, Pattern::KickOffbeatHats),
    (192.0, Pattern::Click),
    (192.0, Pattern::Kick),
    (192.0, Pattern::KickOffbeatHats),
];

// ============================================================
// Signal synthesis
// ============================================================

#[derive(Clone, Copy, PartialEq, Debug)]
enum Pattern {
    /// Sine-burst click on every beat (straight 4/4 pulse).
    Click,
    /// Decaying low sine "kick" on every beat (straight 4/4).
    Kick,
    /// Kicks on the beat + high decaying-noise "hats" on the offbeat 8ths.
    /// This is the classic half/double-tempo trap.
    KickOffbeatHats,
}

impl Pattern {
    fn label(self) -> &'static str {
        match self {
            Pattern::Click => "click",
            Pattern::Kick => "kick",
            Pattern::KickOffbeatHats => "kick+hats",
        }
    }
}

/// Deterministic value-noise burst (no rng crate; hash-based PRNG).
fn noise_at(seed: u64, i: usize) -> Float {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    (seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15)).hash(&mut h);
    (h.finish() as Float / u64::MAX as Float) * 2.0 - 1.0
}

/// Beat positions (in samples) for a given tempo across the signal.
fn beat_positions(bpm: Float, n: usize) -> Vec<usize> {
    let interval = 60.0 / bpm * SR as Float;
    let mut pos = Vec::new();
    let mut t = 0.0;
    while (t as usize) < n {
        pos.push(t as usize);
        t += interval;
    }
    pos
}

/// Add a short 1 kHz sine click at `start`.
fn add_click(y: &mut Array1<Float>, start: usize) {
    let n = y.len();
    for i in 0..80.min(n.saturating_sub(start)) {
        y[start + i] += (2.0 * PI * 1000.0 * i as Float / SR as Float).sin();
    }
}

/// Add a decaying low-frequency "kick" (~55 Hz) at `start`.
fn add_kick(y: &mut Array1<Float>, start: usize) {
    let n = y.len();
    let len = (SR as Float * 0.12) as usize; // 120 ms body
    for i in 0..len.min(n.saturating_sub(start)) {
        let t = i as Float / SR as Float;
        let env = (-t * 35.0).exp();
        y[start + i] += env * (2.0 * PI * 55.0 * t).sin();
    }
}

/// Add a decaying high-frequency "hat" (filtered noise burst) at `start`.
fn add_hat(y: &mut Array1<Float>, start: usize, seed: u64) {
    let n = y.len();
    let len = (SR as Float * 0.04) as usize; // 40 ms tick
    for i in 0..len.min(n.saturating_sub(start)) {
        let t = i as Float / SR as Float;
        let env = (-t * 120.0).exp();
        // high-passed-ish noise: differentiate to emphasize highs
        let hp = noise_at(seed, start + i) - noise_at(seed, start + i.saturating_sub(1));
        y[start + i] += 0.5 * env * hp;
    }
}

/// Click train of `dur_sec` seconds at exactly `bpm`: click `k` starts at the
/// sample nearest `k * 60 / bpm` seconds, computed in f64 so a fractional
/// period does not drift (unlike `beat_positions`, which accumulates in f32).
fn exact_click_train(bpm: Float, dur_sec: Float) -> Array1<Float> {
    let n = (SR as Float * dur_sec) as usize;
    let interval = 60.0 / bpm as f64 * SR as f64;
    let mut y = Array1::<Float>::zeros(n);
    let mut clicks = 0usize;
    let mut start = 0;
    while start < n {
        add_click(&mut y, start);
        clicks += 1;
        start = (clicks as f64 * interval).round() as usize;
    }
    y
}

/// Synthesize a deterministic signal at exactly `bpm` for the given pattern.
fn synth(bpm: Float, pattern: Pattern) -> Array1<Float> {
    let n = (SR as Float * DUR_SEC) as usize;
    let mut y = Array1::<Float>::zeros(n);
    let beats = beat_positions(bpm, n);
    match pattern {
        Pattern::Click => {
            for &b in &beats {
                add_click(&mut y, b);
            }
        }
        Pattern::Kick => {
            for &b in &beats {
                add_kick(&mut y, b);
            }
        }
        Pattern::KickOffbeatHats => {
            let interval = 60.0 / bpm * SR as Float;
            for &b in &beats {
                add_kick(&mut y, b);
                // hat on the offbeat 8th (halfway to next beat)
                let off = b + (interval * 0.5) as usize;
                if off < n {
                    add_hat(&mut y, off, 0xA5A5);
                }
            }
        }
    }
    y
}

// ============================================================
// Detection + metrics
// ============================================================

/// The BPMs under test — chosen to cover historically problematic zones,
/// including fractional (128.3) and the half/double boundaries.
const TEST_BPMS: &[Float] = &[
    60.0, 63.0, 70.0, 79.0, 85.0, 92.0, 100.0, 118.0, 126.0, 128.3, 140.0, 150.0, 160.0, 174.0,
    192.0,
];

const PATTERNS: &[Pattern] = &[Pattern::Click, Pattern::Kick, Pattern::KickOffbeatHats];

/// Tempos of the beat-period precision suite.
const PRECISION_BPMS: &[Float] = &[70.0, 90.0, 128.0, 140.0, 166.0, 174.0, 190.0];

/// BPM ranges under which the octave-free tempo must stay the same.
const PRECISION_RANGES: &[(Float, Float)] =
    &[(79.0, 192.0), (70.0, 180.0), (60.0, 120.0), (90.0, 180.0)];

/// `detected` moved into the octave of `reference` by a power of two.
fn fold_to_octave(detected: Float, reference: Float) -> Float {
    let octaves = (detected / reference).log2().round();
    detected * (-octaves).exp2()
}

fn detect_bpm(bpm: Float, pattern: Pattern) -> Float {
    let y = synth(bpm, pattern);
    let r = sonara::analyze::analyze_signal(y.view(), SR, &sonara::analyze::compact()).unwrap();
    r.bpm
}

#[derive(Clone, Copy)]
struct CaseResult {
    ref_bpm: Float,
    pattern: Pattern,
    detected: Float,
    abs_err: Float,
    correct_pct: bool, // within +/-2%
    within_half_bpm: bool,
    octave_error: bool,
}

fn is_octave(detected: Float, reference: Float) -> bool {
    for &mult in &[0.5f32, 2.0, 1.0 / 3.0, 3.0] {
        let target = reference * mult;
        if (detected - target).abs() <= OCTAVE_TOL * target {
            return true;
        }
    }
    false
}

fn evaluate(reference: Float, pattern: Pattern) -> CaseResult {
    let detected = detect_bpm(reference, pattern);
    let abs_err = (detected - reference).abs();
    let correct_pct = abs_err <= PCT_TOL * reference;
    let within_half_bpm = abs_err <= 0.5;
    let octave_error = !correct_pct && is_octave(detected, reference);
    CaseResult {
        ref_bpm: reference,
        pattern,
        detected,
        abs_err,
        correct_pct,
        within_half_bpm,
        octave_error,
    }
}

fn is_known_failing(bpm: Float, pattern: Pattern) -> bool {
    KNOWN_FAILING
        .iter()
        .any(|&(b, p)| (b - bpm).abs() < 1e-3 && p == pattern)
}

fn median(mut v: Vec<Float>) -> Float {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}

fn percentile(mut v: Vec<Float>, p: Float) -> Float {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (v.len() - 1) as Float).round() as usize;
    v[idx.min(v.len() - 1)]
}

/// Run the full grid, print the metrics report, and return every case result.
fn run_suite() -> Vec<CaseResult> {
    let mut results = Vec::new();
    for &pattern in PATTERNS {
        for &bpm in TEST_BPMS {
            results.push(evaluate(bpm, pattern));
        }
    }

    // Per-case table
    println!("\n=== Synthetic BPM accuracy — per case ===");
    println!(
        "{:>7}  {:<10} {:>9} {:>9} {:>6} {:>7} {:>8}",
        "ref", "pattern", "detected", "abs_err", "<=2%", "octave", "known"
    );
    for c in &results {
        println!(
            "{:>7.1}  {:<10} {:>9.2} {:>9.2} {:>6} {:>7} {:>8}",
            c.ref_bpm,
            c.pattern.label(),
            c.detected,
            c.abs_err,
            if c.correct_pct { "yes" } else { "NO" },
            if c.octave_error { "OCTAVE" } else { "-" },
            if is_known_failing(c.ref_bpm, c.pattern) {
                "known"
            } else {
                "-"
            },
        );
    }

    print_metrics("ALL cases", &results);
    let guarded: Vec<CaseResult> = results
        .iter()
        .copied()
        .filter(|c| !is_known_failing(c.ref_bpm, c.pattern))
        .collect();
    print_metrics("GUARDED subset (excl. known-failing)", &guarded);

    results
}

fn print_metrics(label: &str, cases: &[CaseResult]) {
    if cases.is_empty() {
        println!("\n--- Metrics [{}]: no cases ---", label);
        return;
    }
    let n = cases.len() as Float;
    let acc_half = cases.iter().filter(|c| c.within_half_bpm).count() as Float / n;
    let acc_pct = cases.iter().filter(|c| c.correct_pct).count() as Float / n;
    let octaves = cases.iter().filter(|c| c.octave_error).count();
    let errs: Vec<Float> = cases.iter().map(|c| c.abs_err).collect();
    println!("\n--- Metrics [{}]  (n={}) ---", label, cases.len());
    println!("  accuracy @ +/-0.5 BPM : {:.1}%", acc_half * 100.0);
    println!("  accuracy @ +/-2%      : {:.1}%", acc_pct * 100.0);
    println!(
        "  octave-error rate     : {:.1}%  ({} cases)",
        octaves as Float / n * 100.0,
        octaves
    );
    println!("  median abs error      : {:.2} BPM", median(errs.clone()));
    println!(
        "  p95 abs error         : {:.2} BPM",
        percentile(errs, 95.0)
    );
}

// ============================================================
// The guarding test
// ============================================================

/// Primary regression guard. Prints the full metrics report, then asserts on
/// the GUARDED subset (all cases except `KNOWN_FAILING`):
///   - octave-error rate == 0
///   - median absolute BPM error <= `MAX_MEDIAN_ABS_ERR`
#[test]
fn bpm_accuracy_report() {
    let results = run_suite();

    let guarded: Vec<CaseResult> = results
        .iter()
        .copied()
        .filter(|c| !is_known_failing(c.ref_bpm, c.pattern))
        .collect();

    let octaves = guarded.iter().filter(|c| c.octave_error).count();
    let med = median(guarded.iter().map(|c| c.abs_err).collect());

    assert!(
        octaves <= MAX_OCTAVE_ERRORS,
        "GUARDED octave-error count {} exceeds allowed {} — see report above. \
         A new octave error appeared; if the detector legitimately regressed, fix it; \
         if this case is genuinely hard, add it to KNOWN_FAILING with justification.",
        octaves,
        MAX_OCTAVE_ERRORS
    );
    assert!(
        med <= MAX_MEDIAN_ABS_ERR,
        "GUARDED median abs error {:.2} BPM exceeds threshold {:.2} — see report above.",
        med,
        MAX_MEDIAN_ABS_ERR
    );
}

// ============================================================
// Known-failing per-case tests (documenting the octave-error zone)
//
// These assert the *correct* behaviour and are `#[ignore]`d because current
// `main` fails them (see KNOWN_FAILING). They are not loosened — they encode
// what a fixed detector should do. After the detector is improved, run
// `cargo test -p sonara --test bpm_accuracy -- --ignored` and remove the
// `#[ignore]` (and the matching KNOWN_FAILING entries) for the ones that pass.
// ============================================================

/// Asserts a tempo is detected within +/-2% and is NOT an octave error, for
/// all three patterns.
fn assert_correct_all_patterns(reference: Float) {
    for &pattern in PATTERNS {
        let c = evaluate(reference, pattern);
        assert!(
            c.correct_pct && !c.octave_error,
            "{:.1} BPM [{}]: detected {:.2} (abs_err {:.2}, octave={})",
            reference,
            pattern.label(),
            c.detected,
            c.abs_err,
            c.octave_error
        );
    }
}

/// Fixed by the metrical-multiple candidate selection: 126 BPM was halved
/// (click/kick+hats -> ~63, kick drifted to 123.05) before the metrical lift.
#[test]
fn fixed_126_bpm() {
    assert_correct_all_patterns(126.0);
}

/// Fixed by the metrical-multiple candidate selection: 140 BPM was halved to
/// ~69.84 across all patterns before the metrical lift.
#[test]
fn fixed_140_bpm() {
    assert_correct_all_patterns(140.0);
}

#[test]
#[ignore = "detector halves 192 BPM to ~95.9 across all patterns; the metrical-lift tiers stop below 95 BPM — use bpm_min/bpm_max, or fix via beat-grid regularity scoring"]
fn known_failing_192_bpm() {
    assert_correct_all_patterns(192.0);
}

#[test]
#[ignore = "metrical lift doubles slow (60/63/70 BPM) kick+hats patterns to exactly 2x; needs bpm_max or beat-grid regularity scoring to disambiguate"]
fn known_failing_slow_offbeat_hats() {
    for &bpm in &[60.0, 63.0, 70.0] {
        let c = evaluate(bpm, Pattern::KickOffbeatHats);
        assert!(
            c.correct_pct && !c.octave_error,
            "{bpm:.1} BPM [kick+hats]: detected {:.2} (abs_err {:.2}, octave={})",
            c.detected,
            c.abs_err,
            c.octave_error
        );
    }
}

/// A `bpm_min`/`bpm_max` project range whose floor excludes the halved value
/// must rescue the remaining 192 BPM octave failure: the raw ~96 detection
/// doubles into a 100-200 (fast-genre) range. Note a broad 79-192 range
/// does NOT rescue this case — 96 lies inside it and is correctly left alone.
#[test]
fn bpm_range_rescues_192_bpm() {
    let config = sonara::analyze::AnalysisConfig {
        bpm_min: Some(100.0),
        bpm_max: Some(200.0),
        ..sonara::analyze::compact()
    };
    for &pattern in PATTERNS {
        let y = synth(192.0, pattern);
        let r = sonara::analyze::analyze_signal(y.view(), SR, &config).unwrap();
        let err = (r.bpm - 192.0).abs();
        assert!(
            err <= 192.0 * 0.02,
            "192.0 BPM [{}] with range 79-192: detected {:.2} (abs_err {err:.2})",
            pattern.label(),
            r.bpm
        );
    }
}

// ============================================================
// Beat-period precision
//
// The reported BPM is the period of the tracked beats. Tempo-class and octave
// selection are not part of this guard: fast clicks may still be tracked at
// half tempo (a known 0.3.x behaviour), so the error is measured octave-free,
// with the result folded into the octave of the true tempo.
//
// The beat tracker runs at the range-aligned ACF tempo, so ranges that yield
// the same beats report bit-identical octave-free BPM. A range that moves the
// tracked level (e.g. 60-120 tracks every other click of 128 BPM) measures the
// period on other beats; there the octave-free values must agree within the
// precision guard.
// ============================================================

// ============================================================
// Real-music fixtures
//
// `tests/fixtures/rhythm/` holds the analysis pass's broadband and beat-tracker
// onset envelopes of real tracks (u16-quantized; no audio), written by
// `dev-docs/bench/scripts/bpm-grid/make_fixtures.py`, and `manifest.tsv` with
// each track's validator values (MIK, Rekordbox, Beat This!) and what the tempo
// must be: an exact integer, or a fraction within a tolerance of the label.
// ============================================================

struct RhythmFixture {
    sr: u32,
    hop: usize,
    onset: Array1<Float>,
    beat: Array1<Float>,
}

fn read_rhythm_fixture(path: &std::path::Path) -> RhythmFixture {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&bytes[..8], b"SNRFIX01", "{}: not a rhythm fixture", path.display());
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    let (sr, hop, count, frames) = (word(8), word(12) as usize, word(16) as usize, word(20) as usize);
    let mut at = 24;
    let mut heads = Vec::with_capacity(count);
    for _ in 0..count {
        let len = bytes[at] as usize;
        let name = String::from_utf8(bytes[at + 1..at + 1 + len].to_vec()).unwrap();
        let scale = f32::from_le_bytes(bytes[at + 1 + len..at + 5 + len].try_into().unwrap());
        heads.push((name, scale));
        at += 5 + len;
    }
    let envelope = |name: &str| -> Array1<Float> {
        let index = heads.iter().position(|(n, _)| n == name).unwrap();
        let start = at + index * 2 * frames;
        bytes[start..start + 2 * frames]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as Float * heads[index].1)
            .collect()
    };
    let (onset, beat) = (envelope("onset"), envelope("beat"));
    at += count * 2 * frames;
    assert_eq!(at, bytes.len(), "{}: trailing bytes", path.display());
    RhythmFixture { sr, hop, onset, beat }
}

/// Tempo of a fixture as the analysis pass computes it (no BPM range).
fn fixture_tempo(fixture: &RhythmFixture) -> sonara::beat::TempoEstimate {
    sonara::beat::beat_track_detailed_with_dp_envelope(
        None,
        Some(fixture.onset.view()),
        fixture.sr,
        fixture.hop,
        120.0,
        100.0,
        true,
        None,
        None,
        Some(fixture.beat.view()),
    )
    .unwrap()
    .0
}

/// Steady tracks at an integer tempo report exactly that integer; tracks at a
/// fractional tempo, by the validators, report the fraction instead of the
/// nearest integer.
#[test]
fn real_music_integer_and_fractional_tempos() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/rhythm");
    let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    println!("\n=== Real-music tempo fixtures ===");
    for line in manifest.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let (name, kind) = (cols[0], cols[1]);
        let label: Float = cols[2].parse().unwrap();
        let tolerance: Float = cols[3].parse().unwrap();
        let tempo = fixture_tempo(&read_rhythm_fixture(&dir.join(format!("{name}.env16")))).tempo;
        let folded = fold_to_octave(tempo, label);
        let is_integer = (folded - folded.round()).abs() < 1e-4;
        let ok = match kind {
            "integer" => is_integer && folded.round() == label.round(),
            "fractional" => !is_integer && (folded - label).abs() <= tolerance,
            "not-integer" => !is_integer,
            other => panic!("{name}: unknown kind {other}"),
        };
        println!("{name:>16} {kind:>11} label {label:>9.4} tempo {tempo:>10.4} {}", if ok { "ok" } else { "FAIL" });
        if !ok {
            failures.push(format!("{name} ({kind}, label {label}): tempo {tempo}"));
        }
        checked += 1;
    }
    assert!(checked > 0, "no fixtures in {}", dir.display());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn bpm_precision_from_beat_period() {
    println!("\n=== Beat-period precision — octave-free ===");
    println!(
        "{:>7}  {:>9} {:>6} {:>11} {:>11} {:>11} {:>9}",
        "ref", "range", "beats", "bpm", "bpm_raw", "folded", "abs_err"
    );
    let mut failures = Vec::new();
    let mut errors = Vec::new();
    for &truth in PRECISION_BPMS {
        let y = exact_click_train(truth, PRECISION_DUR_SEC);
        // (tracked beats, octave-free BPM) per range
        let mut by_range: Vec<(Vec<usize>, Float)> = Vec::new();
        for &(bpm_min, bpm_max) in PRECISION_RANGES {
            let config = sonara::analyze::AnalysisConfig {
                bpm_min: Some(bpm_min),
                bpm_max: Some(bpm_max),
                ..sonara::analyze::compact()
            };
            let r = sonara::analyze::analyze_signal(y.view(), SR, &config).unwrap();
            let folded = fold_to_octave(r.bpm, truth);
            let abs_err = (folded - truth).abs();
            println!(
                "{:>7.1}  {:>4}-{:<4} {:>6} {:>11.5} {:>11.5} {:>11.5} {:>9.5}",
                truth,
                bpm_min,
                bpm_max,
                r.beats.len(),
                r.bpm,
                r.bpm_raw,
                folded,
                abs_err
            );
            if abs_err > MAX_PRECISION_ABS_ERR {
                failures.push(format!(
                    "{truth:.1} BPM, range {bpm_min}-{bpm_max}: bpm {:.5} folds to {folded:.5} \
                     (abs_err {abs_err:.5} > {MAX_PRECISION_ABS_ERR})",
                    r.bpm
                ));
            }
            if fold_to_octave(r.bpm_raw, truth).to_bits() != folded.to_bits() {
                failures.push(format!(
                    "{truth:.1} BPM, range {bpm_min}-{bpm_max}: bpm {} is not an octave of \
                     bpm_raw {}",
                    r.bpm, r.bpm_raw
                ));
            }
            errors.push(abs_err);
            by_range.push((r.beats, folded));
        }
        let folded: Vec<Float> = by_range.iter().map(|(_, folded)| *folded).collect();
        for (i, (beats, value)) in by_range.iter().enumerate() {
            if by_range[i + 1..].iter().any(|(other_beats, other)| {
                other_beats == beats && other.to_bits() != value.to_bits()
            }) {
                failures.push(format!(
                    "{truth:.1} BPM: identical beats, different octave-free BPM across \
                     ranges {PRECISION_RANGES:?}: {folded:?}"
                ));
            }
        }
        let spread = folded.iter().copied().fold(Float::NEG_INFINITY, Float::max)
            - folded.iter().copied().fold(Float::INFINITY, Float::min);
        println!("{truth:>7.1}  octave-free spread across ranges {spread:.5}");
        if spread > MAX_PRECISION_ABS_ERR {
            failures.push(format!(
                "{truth:.1} BPM: octave-free BPM spreads {spread:.5} across ranges \
                 {PRECISION_RANGES:?}: {folded:?}"
            ));
        }
    }
    println!(
        "  abs error median {:.5} BPM, max {:.5} BPM (n={})",
        median(errors.clone()),
        errors.iter().copied().fold(0.0, Float::max),
        errors.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
