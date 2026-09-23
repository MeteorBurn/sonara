//! Rhythmic regularity — how percussive events distribute *inside* the metric
//! grid.
//!
//! This is deliberately **not** [`crate::beatgrid::grid_stability`]. That score
//! answers "do the beats sit on a constant-tempo lattice?" — a question about
//! temporal drift of the grid itself. This module takes the grid as given and
//! asks the orthogonal question: *given* the metric grid, where do the drum
//! hits land within it? A track can have a rock-solid grid (`grid_stability`
//! ≈ 1) and a heavily syncopated pattern on top of it, so a stable breakbeat
//! must not read as regular merely because its grid is stable.
//!
//! High values describe a straight pattern whose accents sit on the pulse
//! lattice (house/techno); low values describe accents displaced onto weak
//! metrical positions, strong beats left uncovered, or pulses competing with
//! the metre (garage, 2-step, breakbeat, drum & bass). The measure never
//! distinguishes those irregular idioms from one another — it is a rhythm
//! descriptor, not a genre classifier, and carries no model, no training data
//! and no bundled artifact.
//!
//! # Algorithm
//!
//! Everything is computed from curves the fused pipeline has already produced
//! ([`crate::onset::onset_strength_bands`], the tracked beats, and the
//! [`crate::beatgrid`] downbeat phase) — no second decode and no second FFT.
//!
//! 1. **Tatum grid.** Each inter-beat interval is subdivided into
//!    [`TATUMS_PER_BEAT`] equal cells by linear interpolation between
//!    consecutive tracked beats, giving `beats_per_bar * TATUMS_PER_BEAT`
//!    metrical positions per bar (16 in 4/4). Interpolating *between the
//!    actual beats* is what decouples this from tempo drift, and makes the
//!    whole measure beat-relative — hence invariant to the sample rate and hop
//!    length that define the frame indices it is handed.
//! 2. **Accent sampling.** For every band, each tatum cell takes the peak
//!    onset-strength value over the frames it spans.
//! 3. **Windows.** Bars are grouped into non-overlapping [`WINDOW_BARS`]-bar
//!    windows and every window is measured independently, so a fill, a
//!    breakdown or a drumless intro cannot decide the track.
//! 4. **Cyclic metrical profile.** Within a window, cells are folded modulo
//!    the bar (phase aligned to the detected downbeats) and averaged over its
//!    bars.
//! 5. **DC removal.** Only contrast above a band's own floor carries metrical
//!    information — a constant pedestal is pure DC and describes no rhythm —
//!    so the profile is reduced to `P̃[j] = P[j] - min_j P[j]`. Its L1 mass is
//!    the band's evidence weight, so a flat band (steady sixteenth hats, or no
//!    percussion at all) contributes nothing without any hand-set weighting.
//! 6. **Reliability gate.** A window is measured only when its profile is real
//!    structure rather than sampling noise — see [`window_reliability`].
//! 7. **Two bounded measures** per window, pooled over all bands by evidence
//!    mass and combined with equal weight:
//!    - [`band_metrical_alignment`] — accents on strong versus weak metrical
//!      positions (Longuet-Higgins & Lee 1984);
//!    - [`band_metrical_periodicity`] — the share of the figure that recurs
//!      inside the bar; its complement is the energy of pulses competing with
//!      the metre.
//!
//!    [`beat_concentration`] and [`beat_coverage`] — the two "is the kick on
//!    the quarters" measures — are computed and reported alongside them but
//!    are **not** part of the score. Against manual labels they separate the
//!    classes at or near chance; see [`beat_concentration`] for the
//!    measurements and the two structural reasons.
//! 8. **Robust aggregation.** The track value is the evidence-weighted median
//!    over valid windows, so a short passage cannot move it.
//!
//! Every measure is normalized against an explicit reference pair, so the
//! combination introduces no fitted coefficient, and the result is a
//! deterministic function of the inputs.

use ndarray::ArrayView2;

use crate::types::Float;

/// Label for a straight pattern aligned to the pulse lattice.
pub const LABEL_REGULAR: &str = "regular";
/// Label for a syncopated, polyrhythmic or breakbeat-style pattern.
pub const LABEL_IRREGULAR: &str = "irregular";

/// Metrical resolution: tatum cells per beat (sixteenth notes in 4/4).
///
/// The sixteenth is the grid the idioms in scope are actually written on, and
/// it is what the pipeline's 23.2 ms hop can sample: at 175 BPM — the fast end
/// of the range — a sixteenth still spans ~3.7 frames.
///
/// It is *not* comfortably above the representation's time resolution. The
/// 2048-sample STFT window spans 93 ms while a sixteenth at 128 BPM is 117 ms,
/// so a transient's log-mel flux does leak into the neighbouring cell; on a
/// 192-track real-music sample that pins `beat_concentration` at zero for
/// about 30% of tracks. Dropping to an eighth grid removes the leakage but
/// measures less: it was tried and scored *worse* end to end
/// (class-separation AUC 0.737 against 0.772 at the sixteenth), because the
/// syncopation that distinguishes these idioms lives on the sixteenth. The
/// smear is therefore accepted, and shows up as the compressed range of the
/// two phase-sensitive components rather than as lost discrimination.
pub const TATUMS_PER_BEAT: usize = 4;

/// Bars per measurement window.
///
/// Four bars is the shortest span that is both a musical phrase unit and long
/// enough to separate a repeating figure from sampling noise (the reliability
/// estimate needs several bars per slot).
pub const WINDOW_BARS: usize = 4;

/// Decision boundary between the two labels: the midpoint of the normalized
/// scale, not a fitted threshold.
const LABEL_BOUNDARY: Float = 0.5;

/// Rhythmic-regularity analysis of one track.
///
/// `regularity`, `label` and `candidates` are `None` together when the track
/// carried no measurable rhythmic evidence (silence, ambient, drumless).
/// `confidence` is always present, so a caller can always distinguish "not
/// regular" from "could not tell".
#[derive(Debug, Clone, PartialEq)]
pub struct RhythmicRegularity {
    /// Regularity in `[0, 1]`: `1.0` a straight pattern on the pulse lattice,
    /// `0.0` a syncopated / polyrhythmic / breakbeat pattern. `None` when the
    /// measure abstained.
    pub regularity: Option<Float>,
    /// [`LABEL_REGULAR`] when `regularity >= 0.5`, else [`LABEL_IRREGULAR`];
    /// `None` when the measure abstained.
    pub label: Option<&'static str>,
    /// Quality of the rhythmic evidence in `[0, 1]` — **not** a class
    /// probability, and available even when the measure abstained. See
    /// [`evidence_confidence`].
    pub confidence: Float,
    /// Both labels with their DSP scores, ranked by score descending. The two
    /// scores are complementary by construction (`regularity` and
    /// `1 - regularity`); they are measurements, not model probabilities.
    pub candidates: Option<Vec<(&'static str, Float)>>,
    /// Accent mass on the quarters rather than between them (diagnostic).
    pub beat_concentration: Option<Float>,
    /// Share of the strong beats actually struck (diagnostic).
    pub beat_coverage: Option<Float>,
    /// Alignment with the metrical-weight hierarchy (diagnostic).
    pub metrical_alignment: Option<Float>,
    /// Share of the figure recurring inside the bar (diagnostic).
    pub metrical_periodicity: Option<Float>,
    /// Energy of pulses competing with the metre — the complement of
    /// `metrical_periodicity` (diagnostic).
    pub pulse_competition: Option<Float>,
    /// Agreement of the accent pattern across windows, in `[0, 1]`. Feeds
    /// `confidence` only: a *stable* broken pattern is still broken.
    pub pattern_stability: Float,
    /// Windows the bar grid yielded.
    pub windows_total: usize,
    /// Windows that passed the reliability gate and were measured.
    pub windows_valid: usize,
    /// Meter the bar profile was folded at.
    pub beats_per_bar: usize,
}

impl RhythmicRegularity {
    /// An abstaining result carrying only what evidence was available.
    fn abstained(confidence: Float, beats_per_bar: usize, windows_total: usize) -> Self {
        Self {
            regularity: None,
            label: None,
            confidence,
            candidates: None,
            beat_concentration: None,
            beat_coverage: None,
            metrical_alignment: None,
            metrical_periodicity: None,
            pulse_competition: None,
            pattern_stability: 0.0,
            windows_total,
            windows_valid: 0,
            beats_per_bar,
        }
    }
}

/// Metrical strides of a bar: one stride per level of the metrical hierarchy,
/// coarsest first.
///
/// For the usual 4/4 sixteenth grid (`beats_per_bar = 4`,
/// `tatums_per_beat = 4`) this is `[16, 8, 4, 2, 1]` — bar, half-bar, beat,
/// eighth, sixteenth — reproducing the classical Longuet-Higgins & Lee metrical
/// tree exactly. Above the beat, one level per grouping of beats (each divisor
/// of the meter); at and below the beat, binary subdivision.
fn metrical_strides(beats_per_bar: usize, tatums_per_beat: usize) -> Vec<usize> {
    let tatums = tatums_per_beat.max(1);
    let beats = beats_per_bar.max(1);
    let mut strides = Vec::new();
    for grouping in (2..=beats).rev() {
        if beats % grouping == 0 {
            strides.push(tatums * grouping);
        }
    }
    let mut stride = tatums;
    loop {
        strides.push(stride);
        if stride == 1 {
            break;
        }
        if stride % 2 != 0 {
            // Non-power-of-two subdivision: drop straight to the tatum level
            // rather than inventing a fractional one.
            strides.push(1);
            break;
        }
        stride /= 2;
    }
    strides
}

/// Normalized metrical weight of every slot in a bar, in `[0, 1]`.
///
/// A slot's raw weight is the number of metrical levels it begins (see
/// [`metrical_strides`]); the downbeat begins all of them, an odd sixteenth
/// only the tatum level. Weights are then affinely mapped so the weakest slot
/// is `0` and the downbeat `1`.
fn metrical_weights(beats_per_bar: usize, tatums_per_beat: usize) -> Vec<f64> {
    let slots = beats_per_bar.max(1) * tatums_per_beat.max(1);
    let strides = metrical_strides(beats_per_bar, tatums_per_beat);
    let raw: Vec<usize> = (0..slots)
        .map(|slot| strides.iter().filter(|&&d| slot % d == 0).count())
        .collect();
    let highest = raw.iter().copied().max().unwrap_or(1);
    let lowest = raw.iter().copied().min().unwrap_or(1);
    let span = highest.saturating_sub(lowest) as f64;
    if span <= 0.0 {
        return vec![0.0; slots];
    }
    raw.iter()
        .map(|&weight| (weight - lowest) as f64 / span)
        .collect()
}

/// Coarsest metrical period strictly shorter than the bar, in slots.
///
/// [`metrical_strides`] lists the bar first — a period every bar profile
/// trivially has, and therefore no evidence of anything — so the first genuine
/// recurrence a figure can show is the second entry: the half-bar in 4/4, the
/// beat in 3/4. `None` for a meter with no level below the bar.
fn sub_bar_period(beats_per_bar: usize, tatums_per_beat: usize) -> Option<usize> {
    metrical_strides(beats_per_bar, tatums_per_beat)
        .get(1)
        .copied()
}

/// Peak band energy over the frames a tatum cell `[lo, hi)` spans.
///
/// Frame bounds are rounded and the cell always consumes at least one frame, so
/// every metrical slot is sampled even when the tatum is shorter than a hop.
///
/// Cells are *centred* on their metrical position rather than starting at it —
/// see the sampling loop in [`analyze`]. A tracked beat can sit a frame or two
/// away from the true onset peak (the same effect
/// [`crate::beatgrid`] handles with its accent window), and a forward-looking
/// cell would credit an early attack to the preceding slot, collapsing the
/// on-beat share of a perfectly straight kick.
fn cell_peak(bands: ArrayView2<Float>, band: usize, lo: f64, hi: f64, n_frames: usize) -> f64 {
    let first = (lo.round().max(0.0) as usize).min(n_frames - 1);
    let end = (hi.round().max(0.0) as usize).clamp(first + 1, n_frames);
    (first..end)
        .map(|frame| bands[(band, frame)] as f64)
        .fold(0.0_f64, f64::max)
}

/// Accent mass on the beats rather than between them, in `[0, 1]`.
///
/// This is the pair "concentration of rhythmic energy on the quarters" and
/// "energy between the quarters" expressed as one quantity, since they are
/// complements of the same split. The share landing on the `beats_per_bar`
/// beat slots is read against the share a meter-blind profile would put there
/// by slot count alone, `1 / tatums_per_beat`.
///
/// A kick on every beat and a backbeat clap on 2 and 4 both score `1.0`; hats
/// placed only between the beats score `0.0`.
///
/// **Reported as a diagnostic, deliberately excluded from the score** — as is
/// [`beat_coverage`]. Both ask "is the kick on the quarters", and on real
/// audio neither answers this target.
///
/// Measured against 130 manually labelled tracks (100 straight
/// four-on-the-floor, 30 broken-beat tech house at the same tempo), the
/// class-separation AUC of this measure is 0.469 — at chance — and of
/// `beat_coverage` 0.603, against 0.829 for [`band_metrical_alignment`] and
/// 0.882 for [`band_metrical_periodicity`].
///
/// Two reasons, both structural rather than incidental:
///
/// - the time resolution noted on [`TATUMS_PER_BEAT`]: the 93 ms analysis
///   window cannot resolve *which* sixteenth a low-band attack occupies, so
///   the on-beat share collapses toward zero for straight and broken material
///   alike (means 0.145 and 0.134);
/// - broken-beat house keeps a kick near most quarters and displaces it
///   *within* the beat, so "on the quarters" cannot see the very
///   displacement that defines the class.
///
/// Averaging measures that are near-constant across classes only added an
/// offset: balanced accuracy at the unchanged `0.5` boundary rose from 0.657
/// (all four, equal weight) to 0.830 once the score used alignment and
/// periodicity alone, whose equal-error threshold is 0.480 — i.e. the
/// principled midpoint is already the right operating point.
fn beat_concentration(profile: &[f64], beats_per_bar: usize, tatums_per_beat: usize) -> f64 {
    let mass: f64 = profile.iter().sum();
    if mass <= 0.0 || tatums_per_beat == 0 {
        return 0.0;
    }
    let on_beat: f64 = (0..beats_per_bar)
        .filter_map(|beat| profile.get(beat * tatums_per_beat))
        .sum();
    let null = 1.0 / tatums_per_beat as f64;
    if null >= 1.0 {
        return 0.0;
    }
    ((on_beat / mass - null) / (1.0 - null)).clamp(0.0, 1.0)
}

/// Share of the strong beats actually struck, in `[0, 1]`.
///
/// The accent mass sitting on the beat slots is treated as a distribution over
/// those beats, and its perplexity `exp(H)` is the effective number of beats
/// carrying the pattern. Normalized between one beat (the bar-length figure
/// that drops the other three) and all of them.
///
/// This is the "coverage of the four strong beats" measure, and its complement
/// is "dropped strong beats" — a kick only on beat one scores `0.0`, a kick on
/// every beat `1.0`.
fn beat_coverage(profile: &[f64], beats_per_bar: usize, tatums_per_beat: usize) -> f64 {
    if beats_per_bar < 2 {
        return 0.0;
    }
    let on_beat: Vec<f64> = (0..beats_per_bar)
        .map(|beat| profile.get(beat * tatums_per_beat).copied().unwrap_or(0.0))
        .collect();
    let total: f64 = on_beat.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let entropy: f64 = on_beat
        .iter()
        .map(|&value| value / total)
        .filter(|&share| share > 0.0)
        .map(|share| -share * share.ln())
        .sum();
    let effective = entropy.exp();
    let span = beats_per_bar as f64 - 1.0;
    if span <= 0.0 {
        return 0.0;
    }
    ((effective - 1.0) / span).clamp(0.0, 1.0)
}

/// Metrical alignment of one band's DC-removed profile, in `[0, 1]`.
///
/// The measure is the inner product `⟨P̂, ŵ⟩` of the L1-normalized profile with
/// the metrical weights — the share of accent mass, weighted by how strong a
/// metrical position it occupies. It is read against two explicit references:
///
/// - `uniform` = `mean(ŵ)`, the expected value when accent mass is spread
///   across slots without regard to meter — the no-information null;
/// - `lattice` = `mean(ŵ)` over the beat slots, the value of an even
///   four-on-the-floor accent pattern — the reference *regular* figure.
///
/// The score is the position between them, clamped: an even beat lattice
/// scores exactly `1.0`, meter-blind mass scores `0.0`, and mass on the weak
/// sixteenths scores below the null and clamps to `0.0`.
fn band_metrical_alignment(
    profile: &[f64],
    mass: f64,
    weights: &[f64],
    reference: (f64, f64),
) -> f64 {
    let (uniform, lattice) = reference;
    let span = lattice - uniform;
    if mass <= 0.0 || span <= 0.0 {
        return 0.0;
    }
    let weighted: f64 = profile
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value * weight)
        .sum::<f64>()
        / mass;
    ((weighted - uniform) / span).clamp(0.0, 1.0)
}

/// Sub-bar metrical periodicity of one band's DC-removed profile, in `[0, 1]`.
///
/// Signals of period `p` inside a cycle of `L` slots form a subspace whose
/// length-`L` DFT is supported only on bins that are multiples of `L / p`, and
/// those subspaces nest as `p` grows. So the fraction of AC energy (bin `0` is
/// the mean, which describes no rhythm, and is excluded) on the multiples of
/// `L / sub_bar_period` is exactly the share of the figure that *recurs within
/// the bar* rather than existing only as a bar-length shape. The complement is
/// energy at pulses that compete with the metre — cross-rhythm, polyrhythm, or
/// a figure that only closes over the whole bar.
///
/// A kick on every beat (period 4 in a 16-slot bar) and a backbeat clap on 2
/// and 4 (period 8) both score `1.0`; a chopped break spreads energy onto the
/// remaining bins and scores near the null.
///
/// The null is the share a meter-blind profile would land there by bin count
/// alone, `(sub_bar_period - 1) / (L - 1)`. Unlike [`band_metrical_alignment`]
/// this is a magnitude spectrum and therefore blind to a wholesale phase shift
/// of the pattern — which is why both measures are needed.
fn band_metrical_periodicity(profile: &[f64], sub_bar_period: usize) -> f64 {
    let slots = profile.len();
    if slots < 2 || sub_bar_period == 0 || sub_bar_period >= slots || slots % sub_bar_period != 0 {
        return 0.0;
    }
    let step = slots / sub_bar_period;
    let mut ac_energy = 0.0_f64;
    let mut recurrent_energy = 0.0_f64;
    for bin in 1..slots {
        let mut real = 0.0_f64;
        let mut imaginary = 0.0_f64;
        for (slot, &value) in profile.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * ((slot * bin) % slots) as f64 / slots as f64;
            real += value * angle.cos();
            imaginary += value * angle.sin();
        }
        let energy = real * real + imaginary * imaginary;
        ac_energy += energy;
        if bin % step == 0 {
            recurrent_energy += energy;
        }
    }
    if ac_energy <= 0.0 {
        return 0.0;
    }
    let null = (sub_bar_period - 1) as f64 / (slots - 1) as f64;
    if null >= 1.0 {
        return 0.0;
    }
    ((recurrent_energy / ac_energy - null) / (1.0 - null)).clamp(0.0, 1.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64
}

/// Per-slot values of one band across a window's bars.
fn slot_series(bars: &[Vec<f64>], slot: usize) -> Vec<f64> {
    bars.iter().map(|bar| bar[slot]).collect()
}

/// How much of a window's metrical profile is real structure rather than
/// sampling noise, in `[0, 1]`.
///
/// The profile is a per-slot mean over the window's bars, so its between-slot
/// variance contains both genuine metrical contrast and the noise of averaging
/// finitely many bars. Estimating the latter from the within-slot variance
/// across bars gives the standard variance decomposition
///
/// ```text
/// noise  = mean_j Var_bars(A[bar][j]) / n_bars
/// signal = Var_j(profile) - noise
/// ```
///
/// and the reliability is `signal / (signal + noise)`. Drums drive it to ~1; a
/// drumless or ambient passage, whose slots differ only by chance, sits at the
/// noise floor.
///
/// A window is measured only when `signal > noise`, i.e. reliability `> 0.5` —
/// the point where metrical structure explains more of the profile's shape
/// than sampling noise does. That is a statement about the decomposition, not
/// a tuned cutoff.
fn window_reliability(bars: &[Vec<f64>]) -> f64 {
    let n_bars = bars.len();
    if n_bars < 2 {
        return 0.0;
    }
    let slots = bars[0].len();
    if slots < 2 {
        return 0.0;
    }
    let profile: Vec<f64> = (0..slots).map(|slot| mean(&slot_series(bars, slot))).collect();
    let between = variance(&profile);
    let within: f64 = mean(
        &(0..slots)
            .map(|slot| variance(&slot_series(bars, slot)))
            .collect::<Vec<_>>(),
    );
    let noise = within / n_bars as f64;
    let signal = between - noise;
    if signal <= 0.0 {
        return 0.0;
    }
    (signal / (signal + noise)).clamp(0.0, 1.0)
}

/// Geometric mean of bounded evidence factors; `0.0` if any factor is `0.0`.
///
/// Evidence quality is conjunctive — an untrustworthy tempo cannot be made up
/// for by a long track — so the factors multiply rather than average. The
/// geometric mean keeps the result on the same `[0, 1]` scale regardless of how
/// many factors are available (all-equal factors return that value), which is
/// what lets the optional meter factor join without rescaling the others.
fn geometric_mean(factors: &[f64]) -> f64 {
    if factors.is_empty() {
        return 0.0;
    }
    if factors.iter().any(|&factor| factor <= 0.0) {
        return 0.0;
    }
    let log_mean = factors.iter().map(|&factor| factor.ln()).sum::<f64>() / factors.len() as f64;
    log_mean.exp().clamp(0.0, 1.0)
}

/// Weighted median of `(value, weight)` pairs — the value at which cumulative
/// weight first reaches half the total.
///
/// Robust aggregation is what keeps a fill or a breakdown from deciding the
/// track: unlike a mean, the median ignores a minority of deviant windows
/// however extreme they are.
fn weighted_median(samples: &mut [(f64, f64)]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = samples.iter().map(|&(_, weight)| weight).sum();
    if total <= 0.0 {
        return samples[samples.len() / 2].0;
    }
    let mut cumulative = 0.0;
    for &(value, weight) in samples.iter() {
        cumulative += weight;
        if cumulative * 2.0 >= total {
            return value;
        }
    }
    samples[samples.len() - 1].0
}

/// Cosine similarity of two non-negative vectors, in `[0, 1]`.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&y| y * y).sum::<f64>().sqrt();
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(0.0, 1.0)
}

/// Quality of the rhythmic evidence behind a regularity score, in `[0, 1]`.
///
/// Five factors are always present, plus the meter's own confidence when the
/// caller knows it:
///
/// - **tempo** — `bpm_confidence`: the tatum grid is built from tracked beats,
///   so an unanchored tempo makes every downstream slot meaningless;
/// - **grid** — `grid_stability`: how well those beats form a lattice at all;
/// - **windows** — `1 - 1/sqrt(n_valid)`: the track value is an aggregate over
///   windows, whose relative standard error falls as `1/sqrt(n)`. A single
///   window carries no aggregation and scores `0`;
/// - **reliability** — how far the measured profiles rose above the sampling
///   noise floor (see [`window_reliability`]);
/// - **stability** — how consistently the same accent pattern recurred across
///   windows. This belongs to *confidence*, never to the score: a stable
///   broken pattern is still broken.
fn evidence_confidence(
    bpm_confidence: Float,
    grid_stability: Float,
    windows_valid: usize,
    reliability: f64,
    pattern_stability: f64,
    time_signature_confidence: Option<Float>,
) -> Float {
    if windows_valid == 0 {
        return 0.0;
    }
    let mut factors = vec![
        bpm_confidence.clamp(0.0, 1.0) as f64,
        grid_stability.clamp(0.0, 1.0) as f64,
        1.0 - 1.0 / (windows_valid as f64).sqrt(),
        reliability.clamp(0.0, 1.0),
        pattern_stability.clamp(0.0, 1.0),
    ];
    if let Some(meter) = time_signature_confidence {
        factors.push(meter.clamp(0.0, 1.0) as f64);
    }
    geometric_mean(&factors) as Float
}

/// One window's measurement.
struct Window {
    regularity: f64,
    beat_concentration: f64,
    beat_coverage: f64,
    metrical_alignment: f64,
    metrical_periodicity: f64,
    reliability: f64,
    weight: f64,
    shape: Vec<f64>,
}

/// Measure rhythmic regularity from already-computed rhythm curves.
///
/// `bands` is the `(n_bands, n_frames)` multiband onset strength, lowest band
/// first; `beats` and `downbeats` are frame indices in the same frame space
/// (`downbeats` only supplies the bar phase and may be empty, in which case
/// the first beat is assumed to start a bar). `beats_per_bar` is the meter
/// numerator.
///
/// Always returns a result. `regularity`/`label`/`candidates` are `None` when
/// no window carried measurable rhythmic structure — silence, ambient, or a
/// drumless passage — while `confidence` always reports how much evidence
/// there was. That is an absence of evidence, not a score of zero.
pub fn analyze(
    bands: ArrayView2<Float>,
    beats: &[usize],
    downbeats: &[usize],
    beats_per_bar: usize,
    bpm_confidence: Float,
    grid_stability: Float,
    time_signature_confidence: Option<Float>,
) -> RhythmicRegularity {
    let tatums = TATUMS_PER_BEAT;
    let meter = beats_per_bar.max(1);
    let slots = meter * tatums;
    let n_bands = bands.nrows();
    let n_frames = bands.ncols();
    if n_bands == 0 || n_frames == 0 || beats.len() < 2 {
        return RhythmicRegularity::abstained(0.0, meter, 0);
    }

    // Bar phase: which tracked beat opens a bar. `downbeats` is a subset of
    // `beats` by construction (see `crate::beatgrid::detect_downbeats`).
    let phase = downbeats
        .first()
        .and_then(|first| beats.iter().position(|beat| beat == first))
        .unwrap_or(0);
    // The last beat opens no interval, so it cannot host tatum cells.
    let usable_beats = beats.len() - 1;
    if phase >= usable_beats {
        return RhythmicRegularity::abstained(0.0, meter, 0);
    }
    let bars_total = (usable_beats - phase) / meter;
    if bars_total == 0 {
        return RhythmicRegularity::abstained(0.0, meter, 0);
    }

    // --- accent matrix: [band][bar][slot] ---
    let mut accents = vec![vec![vec![0.0_f64; slots]; bars_total]; n_bands];
    for bar in 0..bars_total {
        for beat_in_bar in 0..meter {
            let index = phase + bar * meter + beat_in_bar;
            let start = beats[index] as f64;
            let span = beats[index + 1] as f64 - start;
            let half_cell = span / (2.0 * tatums as f64);
            for tatum in 0..tatums {
                let centre = start + span * tatum as f64 / tatums as f64;
                let lo = centre - half_cell;
                let hi = centre + half_cell;
                let slot = beat_in_bar * tatums + tatum;
                for band in 0..n_bands {
                    accents[band][bar][slot] = cell_peak(bands, band, lo, hi, n_frames);
                }
            }
        }
    }

    let weights = metrical_weights(meter, tatums);
    let uniform = weights.iter().sum::<f64>() / weights.len() as f64;
    let lattice = (0..meter).map(|beat| weights[beat * tatums]).sum::<f64>() / meter as f64;
    let reference = (uniform, lattice);
    let recurrence_period = sub_bar_period(meter, tatums).unwrap_or(0);

    let windows_total = bars_total.div_ceil(WINDOW_BARS);
    let mut measured: Vec<Window> = Vec::with_capacity(windows_total);

    for start_bar in (0..bars_total).step_by(WINDOW_BARS) {
        let end_bar = (start_bar + WINDOW_BARS).min(bars_total);
        // The reliability decomposition needs at least two bars per slot.
        if end_bar - start_bar < 2 {
            continue;
        }

        let mut total_mass = 0.0_f64;
        let mut alignment_acc = 0.0_f64;
        let mut periodicity_acc = 0.0_f64;
        let mut reliability_acc = 0.0_f64;
        let mut kick_contrast: Option<Vec<f64>> = None;
        let mut pooled = vec![0.0_f64; slots];

        for band in 0..n_bands {
            let bars = &accents[band][start_bar..end_bar];
            let profile: Vec<f64> = (0..slots).map(|slot| mean(&slot_series(bars, slot))).collect();
            let floor = profile.iter().copied().fold(f64::INFINITY, f64::min);
            let contrast: Vec<f64> = profile.iter().map(|&v| (v - floor).max(0.0)).collect();
            let mass: f64 = contrast.iter().sum();
            if mass <= 0.0 {
                continue;
            }
            // The kick pattern is read from the lowest band, the accent
            // placement from every band — "low-band kick, mid/high accents".
            if band == 0 {
                kick_contrast = Some(contrast.clone());
            }
            for (slot, &value) in contrast.iter().enumerate() {
                pooled[slot] += value;
            }
            total_mass += mass;
            alignment_acc += mass * band_metrical_alignment(&contrast, mass, &weights, reference);
            periodicity_acc += mass * band_metrical_periodicity(&contrast, recurrence_period);
            reliability_acc += mass * window_reliability(bars);
        }

        if total_mass <= 0.0 {
            continue;
        }
        let reliability = reliability_acc / total_mass;
        // Measure only where metrical structure outweighs sampling noise.
        if reliability <= 0.5 {
            continue;
        }

        let kick = kick_contrast.unwrap_or_else(|| pooled.clone());
        let concentration = beat_concentration(&kick, meter, tatums);
        let coverage = beat_coverage(&kick, meter, tatums);
        let alignment = alignment_acc / total_mass;
        let periodicity = periodicity_acc / total_mass;
        // Equal weight over the two measures that separate the classes on
        // manual labels. `concentration` and `coverage` are reported but
        // excluded — see [`beat_concentration`]. No fitted coefficient and no
        // moved boundary enter here.
        let regularity = 0.5 * (alignment + periodicity);

        measured.push(Window {
            regularity: regularity.clamp(0.0, 1.0),
            beat_concentration: concentration,
            beat_coverage: coverage,
            metrical_alignment: alignment,
            metrical_periodicity: periodicity,
            reliability,
            weight: total_mass * reliability,
            shape: pooled,
        });
    }

    if measured.is_empty() {
        return RhythmicRegularity::abstained(0.0, meter, windows_total);
    }

    let mut consensus = vec![0.0_f64; slots];
    for window in &measured {
        for (slot, &value) in window.shape.iter().enumerate() {
            consensus[slot] += window.weight * value;
        }
    }
    let pattern_stability = if measured.len() < 2 {
        // A single window agrees with itself; that is not evidence of
        // consistency, so it earns no credit.
        0.0
    } else {
        mean(
            &measured
                .iter()
                .map(|window| cosine(&window.shape, &consensus))
                .collect::<Vec<_>>(),
        )
    };
    let reliability = mean(&measured.iter().map(|w| w.reliability).collect::<Vec<_>>());

    let aggregate = |pick: fn(&Window) -> f64| -> f64 {
        let mut samples: Vec<(f64, f64)> = measured
            .iter()
            .map(|window| (pick(window), window.weight))
            .collect();
        weighted_median(&mut samples)
    };

    let regularity = (aggregate(|w| w.regularity) as Float).clamp(0.0, 1.0);
    let periodicity = aggregate(|w| w.metrical_periodicity) as Float;
    let irregular = 1.0 - regularity;
    let candidates = if regularity >= irregular {
        vec![(LABEL_REGULAR, regularity), (LABEL_IRREGULAR, irregular)]
    } else {
        vec![(LABEL_IRREGULAR, irregular), (LABEL_REGULAR, regularity)]
    };

    RhythmicRegularity {
        regularity: Some(regularity),
        label: Some(if regularity >= LABEL_BOUNDARY {
            LABEL_REGULAR
        } else {
            LABEL_IRREGULAR
        }),
        confidence: evidence_confidence(
            bpm_confidence,
            grid_stability,
            measured.len(),
            reliability,
            pattern_stability,
            time_signature_confidence,
        ),
        candidates: Some(candidates),
        beat_concentration: Some(aggregate(|w| w.beat_concentration) as Float),
        beat_coverage: Some(aggregate(|w| w.beat_coverage) as Float),
        metrical_alignment: Some(aggregate(|w| w.metrical_alignment) as Float),
        metrical_periodicity: Some(periodicity),
        pulse_competition: Some((1.0 - periodicity).clamp(0.0, 1.0)),
        pattern_stability: pattern_stability as Float,
        windows_total,
        windows_valid: measured.len(),
        beats_per_bar: meter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const FRAMES_PER_BEAT: usize = 24;

    /// Build `(n_bands, n_frames)` band envelopes plus the matching beat list,
    /// letting the pattern vary by bar.
    ///
    /// `pattern(band, bar)` returns the 16 slot amplitudes for that band in
    /// that bar. The beat grid is exactly constant, so nothing here varies
    /// tempo — that is `grid_stability`'s domain, not this module's.
    fn synth_with(
        n_bands: usize,
        n_bars: usize,
        mut pattern: impl FnMut(usize, usize) -> [Float; 16],
    ) -> (Array2<Float>, Vec<usize>) {
        // Fixtures are always written as sixteenths per bar, independent of
        // the module's own metrical resolution.
        let frames_per_tatum = FRAMES_PER_BEAT / 4;
        let n_beats = n_bars * 4;
        let n_frames = (n_beats + 1) * FRAMES_PER_BEAT;
        let mut bands = Array2::<Float>::zeros((n_bands, n_frames));
        for bar in 0..n_bars {
            for band in 0..n_bands {
                let slots = pattern(band, bar);
                for (slot, &amplitude) in slots.iter().enumerate() {
                    let frame = bar * 4 * FRAMES_PER_BEAT + slot * frames_per_tatum;
                    bands[(band, frame)] = amplitude;
                }
            }
        }
        let beats: Vec<usize> = (0..=n_beats).map(|b| b * FRAMES_PER_BEAT).collect();
        (bands, beats)
    }

    fn synth(patterns: &[[Float; 16]], n_bars: usize) -> (Array2<Float>, Vec<usize>) {
        synth_with(patterns.len(), n_bars, |band, _bar| patterns[band])
    }

    fn downbeats_of(beats: &[usize]) -> Vec<usize> {
        beats.iter().copied().step_by(4).collect()
    }

    fn measure(bands: &Array2<Float>, beats: &[usize]) -> RhythmicRegularity {
        let downbeats = downbeats_of(beats);
        analyze(bands.view(), beats, &downbeats, 4, 0.9, 0.95, None)
    }

    fn run(patterns: &[[Float; 16]], n_bars: usize) -> RhythmicRegularity {
        let (bands, beats) = synth(patterns, n_bars);
        measure(&bands, &beats)
    }

    fn score(result: &RhythmicRegularity) -> Float {
        result
            .regularity
            .expect("a measured window must yield a score")
    }

    // --- canonical patterns -------------------------------------------------

    /// Four-on-the-floor kick, clap on 2 and 4, offbeat open hat, even
    /// sixteenth closed hats — the canonical house/techno arrangement.
    fn house() -> [[Float; 16]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0],
            [0.4; 16],
        ]
    }

    /// Chopped breakbeat: kick on 1 and the "and" of 3, snare on 2 and 4 with
    /// ghost notes — a bar-length figure, not a beat-length one.
    fn breakbeat() -> [[Float; 16]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.9, 0.0, 0.4, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0, 0.5],
            [0.2, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.4, 0.2, 0.0, 0.0, 0.5, 0.0, 0.4, 0.0, 0.3],
        ]
    }

    /// 2-step: kick on 1 and the weak sixteenth before 3, snare on 2 and 4,
    /// strong beats 2 and 3 left uncovered in the low band.
    fn two_step() -> [[Float; 16]; 2] {
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
        ]
    }

    /// Three-against-four: a pulse every 3 sixteenths over a 16-slot bar, so it
    /// never aligns with the metre and never closes inside the bar.
    fn polyrhythm() -> [[Float; 16]; 1] {
        let mut slots = [0.0; 16];
        let mut position = 0;
        while position < 16 {
            slots[position] = 1.0;
            position += 3;
        }
        [slots]
    }

    // --- metrical scaffolding ----------------------------------------------

    #[test]
    fn metrical_weights_reproduce_the_classical_4_4_tree() {
        assert_eq!(metrical_strides(4, 4), vec![16, 8, 4, 2, 1]);
        let w = metrical_weights(4, 4);
        let expected = [
            1.0, 0.0, 0.25, 0.0, 0.5, 0.0, 0.25, 0.0, 0.75, 0.0, 0.25, 0.0, 0.5, 0.0, 0.25, 0.0,
        ];
        for (slot, (&got, &want)) in w.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "slot {slot}: {got} != {want}");
        }
    }

    #[test]
    fn metrical_weights_handle_a_three_four_meter() {
        assert_eq!(metrical_strides(3, 4), vec![12, 4, 2, 1]);
        let w = metrical_weights(3, 4);
        assert_eq!(w.len(), 12);
        assert!((w[0] - 1.0).abs() < 1e-12, "downbeat must be 1.0");
        assert!((w[4] - w[8]).abs() < 1e-12);
        assert!(w[4] > w[2] && w[2] > w[1], "beat > eighth > sixteenth");
        assert!((w[1] - 0.0).abs() < 1e-12, "weakest slot must be 0.0");
    }

    // --- the four component measures ---------------------------------------

    #[test]
    fn beat_concentration_separates_on_beat_from_between_beat_mass() {
        let on_beat = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let backbeat = [
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let offbeat = [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        assert!((beat_concentration(&on_beat, 4, 4) - 1.0).abs() < 1e-12);
        assert!(
            (beat_concentration(&backbeat, 4, 4) - 1.0).abs() < 1e-12,
            "a backbeat is still entirely on the quarters"
        );
        assert!((beat_concentration(&offbeat, 4, 4) - 0.0).abs() < 1e-12);
        assert!((beat_concentration(&[1.0 / 16.0; 16], 4, 4) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn beat_coverage_counts_struck_strong_beats() {
        let all_four = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let only_one = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let two_of_four = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert!((beat_coverage(&all_four, 4, 4) - 1.0).abs() < 1e-12);
        assert!(
            (beat_coverage(&only_one, 4, 4) - 0.0).abs() < 1e-12,
            "three dropped beats must read as no coverage"
        );
        let half = beat_coverage(&two_of_four, 4, 4);
        assert!(half > 0.3 && half < 0.4, "two of four covered: {half}");
    }

    #[test]
    fn the_kick_placement_measures_are_reported_but_kept_out_of_the_score() {
        let r = run(&house(), 32);
        let score = score(&r);
        let expected = 0.5 * (r.metrical_alignment.unwrap() + r.metrical_periodicity.unwrap());
        assert!(
            (score - expected).abs() < 1e-5,
            "score must be the mean of alignment/periodicity: {score} vs {expected}"
        );
        assert!(
            r.beat_concentration.is_some() && r.beat_coverage.is_some(),
            "both are still reported as diagnostics"
        );
    }

    #[test]
    fn the_components_are_not_redundant() {
        // Every kick displaced onto the weak sixteenth after the beat: still
        // recurrent within the bar, but nothing lands on a quarter.
        let displaced = [[
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ]];
        let r = run(&displaced, 16);
        assert!(
            r.metrical_periodicity.unwrap() > 0.9,
            "a displaced but recurrent lattice stays periodic: {:?}",
            r.metrical_periodicity
        );
        assert!(
            r.metrical_alignment.unwrap() < 0.05,
            "mass entirely on weak sixteenths is not aligned: {:?}",
            r.metrical_alignment
        );
        assert!(
            r.beat_concentration.unwrap() < 0.05,
            "nothing is on a quarter: {:?}",
            r.beat_concentration
        );
    }

    #[test]
    fn a_bar_length_figure_does_not_recur() {
        let single = [[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]];
        let r = run(&single, 16);
        assert!(
            r.metrical_periodicity.unwrap() < 0.1,
            "a once-per-bar accent must not read as recurrent: {:?}",
            r.metrical_periodicity
        );
        assert!(
            r.pulse_competition.unwrap() > 0.9,
            "its complement is competing-pulse energy: {:?}",
            r.pulse_competition
        );
        assert!(
            r.beat_coverage.unwrap() < 0.05,
            "three strong beats are dropped: {:?}",
            r.beat_coverage
        );
    }

    // --- the required discriminations --------------------------------------

    #[test]
    fn four_on_the_floor_is_regular() {
        let r = run(&house(), 32);
        assert_eq!(r.label, Some(LABEL_REGULAR));
        assert!(
            score(&r) > 0.8,
            "house pattern scored {} — expected a straight reading",
            score(&r)
        );
        assert!(r.beat_concentration.unwrap() > 0.95);
        assert!(r.beat_coverage.unwrap() > 0.95);
    }

    #[test]
    fn straight_four_on_the_floor_with_offbeat_hats_stays_regular() {
        // Criterion: offbeat hats must not drag a straight kick into irregular.
        let r = run(&house(), 32);
        assert_eq!(r.label, Some(LABEL_REGULAR));
        assert!(score(&r) > 0.7, "offbeat hats dragged it to {}", score(&r));
    }

    #[test]
    fn chopped_breakbeat_is_irregular() {
        let r = run(&breakbeat(), 32);
        assert_eq!(r.label, Some(LABEL_IRREGULAR));
        assert!(score(&r) < 0.5, "breakbeat scored {}", score(&r));
    }

    #[test]
    fn two_step_is_irregular() {
        let r = run(&two_step(), 32);
        assert_eq!(r.label, Some(LABEL_IRREGULAR), "score {}", score(&r));
        assert!(
            r.beat_coverage.unwrap() < 0.6,
            "2-step leaves strong beats uncovered: {:?}",
            r.beat_coverage
        );
    }

    #[test]
    fn a_competing_pulse_is_irregular() {
        let r = run(&polyrhythm(), 32);
        assert_eq!(r.label, Some(LABEL_IRREGULAR), "score {}", score(&r));
        assert!(
            r.pulse_competition.unwrap() > 0.5,
            "three-against-four must register as competing: {:?}",
            r.pulse_competition
        );
    }

    #[test]
    fn straight_pattern_outscores_the_same_kit_syncopated() {
        let straight = score(&run(&house(), 32));
        let chopped = score(&run(&breakbeat(), 32));
        assert!(
            straight > chopped + 0.3,
            "separation too small: {straight} vs {chopped}"
        );
    }

    #[test]
    fn a_stable_breakbeat_does_not_become_regular() {
        // Criterion: high grid stability must never rescue a broken pattern.
        let (bands, beats) = synth(&breakbeat(), 32);
        let downbeats = downbeats_of(&beats);
        let rigid = analyze(bands.view(), &beats, &downbeats, 4, 1.0, 1.0, None);
        assert_eq!(rigid.label, Some(LABEL_IRREGULAR));
        assert!(
            rigid.pattern_stability > 0.9,
            "the looped break is highly stable: {}",
            rigid.pattern_stability
        );
        assert!(
            score(&rigid) < 0.5,
            "stability must feed confidence, not the score: {}",
            score(&rigid)
        );
    }

    // --- abstention ---------------------------------------------------------

    #[test]
    fn silence_abstains_but_still_reports_confidence() {
        let (bands, beats) = synth(&[[0.0; 16]], 32);
        let r = measure(&bands, &beats);
        assert!(r.regularity.is_none(), "silence must not produce a score");
        assert!(r.label.is_none(), "silence must not produce a label");
        assert!(r.candidates.is_none());
        assert!(
            r.confidence < 0.05,
            "confidence must be reported and low: {}",
            r.confidence
        );
        assert_eq!(r.windows_valid, 0);
    }

    #[test]
    fn a_steady_stream_abstains() {
        // An unbroken constant is pure DC: no metrical information at all.
        let (bands, beats) = synth(&[[0.5; 16]], 32);
        let r = measure(&bands, &beats);
        assert!(r.regularity.is_none() && r.label.is_none());
        assert_eq!(r.windows_valid, 0);
    }

    #[test]
    fn a_drumless_noise_bed_is_not_confidently_irregular() {
        // Slot values differ only by chance; the reliability gate must reject
        // the windows rather than calling the noise a broken rhythm.
        let mut state = 0x9E37_79B9_u32;
        let (bands, beats) = synth_with(2, 32, |_band, _bar| {
            let mut slots = [0.0_f32; 16];
            for slot in slots.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *slot = 0.30 + 0.02 * ((state >> 16) as f32 / 65_536.0);
            }
            slots
        });
        let r = measure(&bands, &beats);
        assert!(
            r.regularity.is_none() || r.confidence < 0.5,
            "a drumless bed must abstain or be low-confidence, got {:?} at {}",
            r.regularity,
            r.confidence
        );
    }

    #[test]
    fn too_few_beats_abstains() {
        let (bands, beats) = synth(&house(), 32);
        let downbeats = downbeats_of(&beats);
        let one = analyze(bands.view(), &beats[..1], &downbeats, 4, 0.9, 0.95, None);
        assert!(one.regularity.is_none() && one.confidence == 0.0);
        // Three beats cannot close a 4/4 bar.
        let three = analyze(bands.view(), &beats[..3], &[], 4, 0.9, 0.95, None);
        assert!(three.regularity.is_none());
    }

    #[test]
    fn sparse_drums_still_measure_but_score_lower_than_a_full_kit() {
        // One kick per bar plus a backbeat: real but thin evidence.
        let sparse = [
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
            ],
        ];
        let thin = run(&sparse, 32);
        let full = run(&house(), 32);
        assert!(thin.regularity.is_some(), "sparse drums are still drums");
        assert!(
            score(&thin) < score(&full),
            "sparse coverage must not outscore a full straight kit: {} vs {}",
            score(&thin),
            score(&full)
        );
    }

    // --- robust aggregation -------------------------------------------------

    #[test]
    fn a_short_breakdown_does_not_decide_the_track() {
        // 28 bars of house with a 4-bar drumless breakdown in the middle.
        let house_pattern = house();
        let (bands, beats) = synth_with(4, 32, |band, bar| {
            if (12..16).contains(&bar) {
                [0.0; 16]
            } else {
                house_pattern[band]
            }
        });
        let r = measure(&bands, &beats);
        assert_eq!(r.label, Some(LABEL_REGULAR));
        assert!(
            score(&r) > 0.8,
            "a 4-bar breakdown must not move the track: {}",
            score(&r)
        );
        assert!(
            r.windows_valid < r.windows_total,
            "the silent window must be excluded: {} of {}",
            r.windows_valid,
            r.windows_total
        );
    }

    #[test]
    fn a_short_fill_does_not_decide_the_track() {
        let house_pattern = house();
        let break_pattern = breakbeat();
        let (bands, beats) = synth_with(4, 32, |band, bar| {
            if (16..20).contains(&bar) {
                break_pattern[band]
            } else {
                house_pattern[band]
            }
        });
        let r = measure(&bands, &beats);
        assert_eq!(r.label, Some(LABEL_REGULAR));
        assert!(
            score(&r) > 0.8,
            "a 4-bar fill must not flip the track: {}",
            score(&r)
        );
    }

    #[test]
    fn a_majority_broken_track_reads_irregular() {
        // The converse: the median must follow the majority, not the outlier.
        let house_pattern = house();
        let break_pattern = breakbeat();
        let (bands, beats) = synth_with(4, 32, |band, bar| {
            if bar < 8 {
                house_pattern[band]
            } else {
                break_pattern[band]
            }
        });
        let r = measure(&bands, &beats);
        assert_eq!(r.label, Some(LABEL_IRREGULAR), "score {}", score(&r));
    }

    // --- invariances --------------------------------------------------------

    #[test]
    fn score_is_independent_of_tempo_drift() {
        // Same pattern, but each beat interval one frame longer than the last:
        // a drifting grid that `grid_stability` would penalise and this measure
        // must not, because the tatum grid interpolates between real beats.
        let patterns = house();
        let mut beats = vec![0usize];
        let mut interval = FRAMES_PER_BEAT;
        for _ in 0..128 {
            let next = beats.last().unwrap() + interval;
            beats.push(next);
            interval += 1;
        }
        let n_frames = beats.last().unwrap() + interval;
        let mut bands = Array2::<Float>::zeros((patterns.len(), n_frames));
        for (beat_index, window) in beats.windows(2).enumerate() {
            let span = (window[1] - window[0]) as f64;
            for tatum in 0..TATUMS_PER_BEAT {
                let frame =
                    window[0] + (span * tatum as f64 / TATUMS_PER_BEAT as f64).round() as usize;
                let slot = (beat_index % 4) * TATUMS_PER_BEAT + tatum;
                for (band, pattern) in patterns.iter().enumerate() {
                    bands[(band, frame)] = pattern[slot];
                }
            }
        }
        let downbeats: Vec<usize> = beats.iter().copied().step_by(4).collect();
        let drifting = analyze(bands.view(), &beats, &downbeats, 4, 0.9, 0.5, None);
        assert_eq!(drifting.label, Some(LABEL_REGULAR));
        assert!(
            score(&drifting) > 0.8,
            "tempo drift must not make a straight pattern read as syncopated: {}",
            score(&drifting)
        );
    }

    #[test]
    fn score_is_invariant_to_a_gain_change() {
        // Every measure normalizes by its own mass, so scaling the accents
        // must not move the score at all.
        let (bands, beats) = synth(&house(), 32);
        let quiet = bands.mapv(|v| v * 0.1);
        let loud = bands.mapv(|v| v * 7.5);
        let base = measure(&bands, &beats);
        assert_eq!(score(&measure(&quiet, &beats)), score(&base));
        assert_eq!(score(&measure(&loud, &beats)), score(&base));
    }

    #[test]
    fn downbeat_phase_is_honoured() {
        let (bands, beats) = synth(&house(), 32);
        let shifted: Vec<usize> = beats[2..].to_vec();
        let aligned_downbeats: Vec<usize> = shifted.iter().copied().skip(2).step_by(4).collect();
        let aligned = analyze(bands.view(), &shifted, &aligned_downbeats, 4, 0.9, 0.95, None);
        let misaligned = analyze(bands.view(), &shifted, &[], 4, 0.9, 0.95, None);
        assert!(
            aligned.metrical_alignment.unwrap() >= misaligned.metrical_alignment.unwrap(),
            "honouring the downbeat phase must not hurt alignment: {:?} vs {:?}",
            aligned.metrical_alignment,
            misaligned.metrical_alignment
        );
        assert_eq!(aligned.label, Some(LABEL_REGULAR));
    }

    // --- contract -----------------------------------------------------------

    #[test]
    fn candidates_are_ranked_and_complementary() {
        for patterns in [&house()[..], &breakbeat()[..]] {
            let r = run(patterns, 32);
            let candidates = r.candidates.clone().expect("candidates");
            assert_eq!(candidates.len(), 2);
            assert!(
                candidates[0].1 >= candidates[1].1,
                "candidates must be ranked: {candidates:?}"
            );
            assert_eq!(Some(candidates[0].0), r.label);
            let total: Float = candidates.iter().map(|c| c.1).sum();
            assert!((total - 1.0).abs() < 1e-5, "complementary scores sum to 1");
            let regular = candidates
                .iter()
                .find(|c| c.0 == LABEL_REGULAR)
                .expect("regular candidate present");
            assert!((regular.1 - score(&r)).abs() < 1e-6);
        }
    }

    #[test]
    fn score_and_confidence_stay_in_range() {
        let r = run(&house(), 32);
        assert!((0.0..=1.0).contains(&score(&r)));
        assert!((0.0..=1.0).contains(&r.confidence));
        for component in [
            r.beat_concentration,
            r.beat_coverage,
            r.metrical_alignment,
            r.metrical_periodicity,
            r.pulse_competition,
        ] {
            let value = component.expect("component");
            assert!(
                (0.0..=1.0).contains(&value),
                "component {value} out of range"
            );
        }
    }

    #[test]
    fn confidence_grows_with_observed_windows() {
        let short = run(&house(), 8).confidence;
        let long = run(&house(), 64).confidence;
        assert!(
            long > short,
            "more windows must raise evidence quality: {long} vs {short}"
        );
    }

    #[test]
    fn confidence_falls_with_untrustworthy_inputs() {
        let (bands, beats) = synth(&house(), 32);
        let downbeats = downbeats_of(&beats);
        let trusted = analyze(bands.view(), &beats, &downbeats, 4, 0.9, 0.95, None);
        let shaky = analyze(bands.view(), &beats, &downbeats, 4, 0.2, 0.95, None);
        assert!(shaky.confidence < trusted.confidence);
        // The score itself is a property of the pattern, not of our trust in it.
        assert_eq!(shaky.regularity, trusted.regularity);

        let metered = analyze(bands.view(), &beats, &downbeats, 4, 0.9, 0.95, Some(0.4));
        assert!(
            metered.confidence < trusted.confidence,
            "a weakly detected meter must lower evidence quality"
        );
    }

    #[test]
    fn repeated_analysis_is_bit_identical() {
        let (bands, beats) = synth(&breakbeat(), 32);
        let downbeats = downbeats_of(&beats);
        let first = analyze(bands.view(), &beats, &downbeats, 4, 0.7, 0.8, Some(0.6));
        let second = analyze(bands.view(), &beats, &downbeats, 4, 0.7, 0.8, Some(0.6));
        assert_eq!(first, second);
    }

    #[test]
    fn weighted_median_ignores_a_light_outlier() {
        let mut samples = vec![(0.9, 10.0), (0.9, 10.0), (0.1, 1.0), (0.9, 10.0)];
        assert_eq!(weighted_median(&mut samples), 0.9);
    }
}
