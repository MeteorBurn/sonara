//! Beat tracking.
//!
//! Beat tracking via dynamic programming (Ellis 2007 algorithm).
//! Includes beat_track, plp, tempo_curve, and tempo_variability.

use ndarray::{Array1, ArrayView1};

use crate::error::{Result, SonaraError};
use crate::onset;
use crate::types::Float;

/// Number of strongest tempo candidates surfaced in a [`TempoEstimate`].
const MAX_TEMPO_CANDIDATES: usize = 5;

/// Fewest tracked beats whose period can replace the ACF tempo estimate.
const MIN_BEAT_PERIOD_BEATS: usize = 17;

/// Pair separations, in beats, of the hierarchical beat-period consensus.
const BEAT_PAIR_SEPARATIONS: [usize; 3] = [32, 64, 128];

/// Pairs whose span lies within this many frames of the current span agree.
const BEAT_PAIR_TOLERANCE_FRAMES: Float = 1.0;

/// Mean-refinement passes per consensus stage.
const BEAT_PAIR_PASSES: usize = 3;

/// Largest relative deviation of the beat period from the ACF tempo; beyond it
/// the ACF estimate is kept.
const MAX_BEAT_TEMPO_DEVIATION: Float = 0.05;

/// Integer tempo of steady tracks. Tracks made in a DAW usually have an
/// integer tempo; a constant grid over the whole track tells whether the
/// tracked beats keep one tempo and whether the nearest integer fits them.
/// The beats are split into this many consecutive segments, and each
/// segment's grid phase is compared with the phase of the whole track.
const STEADY_GRID_SEGMENTS: usize = 4;

/// Fewest beats per segment; shorter tracks are never rounded.
const STEADY_GRID_MIN_SEGMENT_BEATS: usize = 16;

/// Largest phase offset (seconds) of a segment from the whole-track grid for
/// the tempo to count as constant, and for the integer grid to fit.
const STEADY_GRID_MAX_OFFSET_SEC: f64 = 0.010;

/// The grid phase is taken modulo a quarter beat (fourth harmonic of the beat
/// period), so a tracker that sits half or a quarter of a beat off during one
/// section of a syncopated pattern is not read as tempo drift.
const STEADY_GRID_HARMONIC: f64 = 4.0;

/// Largest distance (BPM) of the measured tempo from an integer to round it.
const INTEGER_TEMPO_TOLERANCE: Float = 0.03;

/// Fixed frame (BPM) in which the tempo level is chosen, whatever the caller's
/// range: the caller's `bpm_min`/`bpm_max` only fold the reported tempo, so the
/// level does not depend on the range.
const LEVEL_FRAME_BPM: (Float, Float) = (79.0, 192.0);

/// Further ACF peaks retracked as alternatives to the selected tempo level.
const MAX_ALTERNATIVE_LEVELS: usize = 4;

/// Weight of the mean local score at the beats in a level's evidence (the
/// beats' pair agreement plus this times their mean local score).
const LEVEL_LOCAL_SCORE_WEIGHT: Float = 0.75;

/// Evidence margin an alternative level needs over the selected one to be
/// reported instead.
const LEVEL_SWITCH_MARGIN: Float = 0.25;

/// Octave guard: an alternative within this ratio of an octave multiple of the
/// selected level (but not at its octave) is skipped.
const LEVEL_OCTAVE_GUARD_RATIO: Float = 1.03;

/// Result of tempo estimation, including diagnostic tempo candidates.
///
/// - `tempo`: final BPM, after optional `bpm_min`/`bpm_max` range alignment
///   and clamping to `[30, 320]`. It is the period of the tracked beats,
///   measured by a hierarchical consensus of beat pairs, at the tempo level
///   chosen among the ACF peaks (the metrical selection unless another peak's
///   beats fit the onsets clearly better). The level's ACF tempo (fractional
///   ACF-peak refinement) sets the beat tracker's period and is the fallback:
///   it is reported when fewer than 17 beats are tracked or the beat period
///   deviates from it by more than 5%.
/// - `tempo_raw`: the same tempo *before* optional BPM-range alignment, at the
///   octave of the chosen ACF lag.
/// - `candidates`: the strongest ACF tempo candidates as `(bpm, score)` pairs,
///   sorted by score descending.
#[derive(Debug, Clone)]
pub struct TempoEstimate {
    /// Final tempo in BPM: the tracked-beat period, or the ACF estimate as the
    /// fallback (post range-alignment and clamping).
    pub tempo: Float,
    /// The same tempo in BPM before optional BPM-range alignment.
    pub tempo_raw: Float,
    /// Strongest `(bpm, score)` candidates, sorted by score descending.
    pub candidates: Vec<(Float, Float)>,
}

impl TempoEstimate {
    /// Fallback estimate carrying a single tempo and no ACF candidates.
    fn fallback(tempo: Float) -> Self {
        Self {
            tempo,
            tempo_raw: tempo,
            candidates: Vec::new(),
        }
    }
}

/// Track beats in an audio signal.
///
/// Returns `(tempo, beat_frames)`:
/// - `tempo`: Estimated tempo in BPM
/// - `beat_frames`: Frame indices of detected beats
pub fn beat_track(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
) -> Result<(Float, Vec<usize>)> {
    beat_track_with_bpm_range(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        None,
        None,
    )
}

/// Track beats in an audio signal with optional octave-folding BPM range.
///
/// If both `bpm_min` and `bpm_max` are provided, the estimated tempo is doubled
/// or halved by octaves until it falls inside the requested range. This mirrors
/// DJ-library workflows that constrain BPM display to a preferred range.
pub fn beat_track_with_bpm_range(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<(Float, Vec<usize>)> {
    let (estimate, beats) = beat_track_detailed(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        bpm_min,
        bpm_max,
    )?;
    Ok((estimate.tempo, beats))
}

/// Track beats and return the full [`TempoEstimate`] alongside the beat frames.
///
/// Like [`beat_track_with_bpm_range`], but also surfaces the pre-range-alignment
/// tempo and the strongest ACF tempo candidates for reporting.
///
/// The autocorrelation (ACF) estimate sets the integer beat period of the
/// dynamic-programming tracker. The tempo level is chosen in a fixed 79–192
/// BPM frame: the metrically selected ACF lag and up to four further ACF
/// peaks are each tracked, and another level replaces the selected one only
/// when its beats agree better with each other and with the onsets (pair
/// agreement plus 0.75 × mean local score, a margin of 0.25), never an octave
/// relative. The reported tempo is then the period of the chosen level's
/// beats: a hierarchical consensus of beat pairs 32, 64 and 128 beats apart
/// (±1 frame), at the octave of its ACF lag, folded into the caller's range.
/// The ACF estimate stays as the fallback when fewer than 17 beats are tracked
/// or the beat period deviates from it by more than 5%. A steady track whose
/// beats stay on one constant grid that also fits the nearest integer tempo
/// reports that integer. The candidates do not depend on the level choice.
///
/// The beats are tracked on the same broadband onset envelope; see
/// [`beat_track_detailed_with_dp_envelope`] for a separate tracking envelope.
#[allow(clippy::too_many_arguments)]
pub fn beat_track_detailed(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<(TempoEstimate, Vec<usize>)> {
    beat_track_detailed_with_dp_envelope(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        bpm_min,
        bpm_max,
        None,
    )
}

/// Like [`beat_track_detailed`], with a separate envelope for the beat tracker.
///
/// The tempo, its ACF candidates and the tracker's integer beat period come
/// from the broadband onset envelope (`onset_envelope`, or computed from `y`).
/// The dynamic-programming tracker (local score, DP and trimming) runs on
/// `dp_envelope`, and the beat-period consensus then measures the tempo on
/// those beats. `dp_envelope = None` tracks on the broadband envelope, which is
/// [`beat_track_detailed`]. Both envelopes share one frame grid: a
/// `dp_envelope` of another length is an error.
#[allow(clippy::too_many_arguments)]
pub fn beat_track_detailed_with_dp_envelope(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
    dp_envelope: Option<ArrayView1<Float>>,
) -> Result<(TempoEstimate, Vec<usize>)> {
    let (estimate, beats, _) = beat_track_with_range_free_beats(
        y,
        onset_envelope,
        sr,
        hop_length,
        start_bpm,
        tightness,
        trim,
        bpm_min,
        bpm_max,
        dp_envelope,
        false,
    )?;
    Ok((estimate, beats))
}

/// [`beat_track_detailed_with_dp_envelope`], plus, when `range_free_beats` is
/// set, the beats a call without a BPM range would return (bit-identical to
/// one). `tempo_raw` is that call's tempo. The aggression model reads both, so
/// its score does not move with the caller's display range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn beat_track_with_range_free_beats(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    tightness: Float,
    trim: bool,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
    dp_envelope: Option<ArrayView1<Float>>,
    range_free_beats: bool,
) -> Result<(TempoEstimate, Vec<usize>, Option<Vec<usize>>)> {
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    // Get onset envelope
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "either y or onset_envelope must be provided".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };
    if let Some(dp) = dp_envelope {
        if dp.len() != oenv.len() {
            return Err(SonaraError::InvalidParameter {
                param: "dp_envelope",
                reason: format!(
                    "length {} differs from the onset envelope length {}",
                    dp.len(),
                    oenv.len()
                ),
            });
        }
    }

    if oenv.len() < 4 {
        return Ok((
            TempoEstimate::fallback(start_bpm),
            vec![],
            range_free_beats.then(Vec::new),
        ));
    }

    // Guard against flat / degenerate onset envelopes (silence, DC): with no
    // dynamic range there is no meaningful autocorrelation peak to track.
    let (min_onset, max_onset) = oenv.iter().copied().fold(
        (Float::INFINITY, Float::NEG_INFINITY),
        |(min_v, max_v), v| (min_v.min(v), max_v.max(v)),
    );
    if !min_onset.is_finite()
        || !max_onset.is_finite()
        || max_onset <= 1e-10
        || (max_onset - min_onset) <= 1e-10
    {
        return Ok((
            TempoEstimate::fallback(start_bpm),
            vec![],
            range_free_beats.then(Vec::new),
        ));
    }

    // Normalize the tracking envelope: the DP envelope, else the broadband one
    let track_env = dp_envelope.unwrap_or_else(|| oenv.view());
    let mean = track_env.iter().sum::<Float>() / track_env.len() as Float;
    let std = (track_env.iter().map(|&v| (v - mean).powi(2)).sum::<Float>()
        / track_env.len() as Float)
        .sqrt();
    let track_norm = if std > 0.0 {
        track_env.mapv(|v| (v - mean) / (std + 1e-10))
    } else {
        track_env.to_owned()
    };

    // DP beats at an integer period, with the local score they were tracked on
    let track = |frames_per_beat: usize| {
        // Compute local score via Gaussian-windowed autocorrelation
        let local_score = beat_local_score(track_norm.view(), frames_per_beat);
        // Run DP beat tracker
        let beats = beat_track_dp(local_score.view(), frames_per_beat, tightness);
        // Trim beats with low onset strength
        let beats = if trim {
            trim_beats(local_score.view(), &beats)
        } else {
            beats
        };
        (beats, local_score)
    };
    let frames_per_beat_of = |tempo: Float| (60.0 * frame_rate / tempo).round() as usize;

    let Some(acf) = acf_candidates(&oenv, sr, hop_length, start_bpm)? else {
        // No autocorrelation peak to choose a level from: track at start_bpm.
        let mut estimate = TempoEstimate::fallback(start_bpm);
        let frames_per_beat = frames_per_beat_of(start_bpm);
        if frames_per_beat == 0 {
            return Ok((estimate, vec![], range_free_beats.then(Vec::new)));
        }
        let (beats, _) = track(frames_per_beat);
        if let Some((tempo, tempo_raw)) =
            beat_period_tempo(&beats, start_bpm, 1.0, frame_rate, bpm_min, bpm_max)?
        {
            estimate.tempo = tempo;
            estimate.tempo_raw = tempo_raw;
        }
        // Tracked at start_bpm whatever the range: the range-free beats are these.
        let range_free = range_free_beats.then(|| beats.clone());
        return Ok((estimate, beats, range_free));
    };
    let candidates = acf.ranked();

    // Tempo levels in the fixed frame: the selected ACF lag first, then up to
    // MAX_ALTERNATIVE_LEVELS further ACF peaks, one level per beat period.
    let mut levels: Vec<TempoLevel> = Vec::with_capacity(MAX_ALTERNATIVE_LEVELS + 1);
    let mut seen_periods = Vec::with_capacity(MAX_ALTERNATIVE_LEVELS + 1);
    for lag in std::iter::once(acf.selected_lag()).chain(acf.peak_lags()) {
        if levels.len() > MAX_ALTERNATIVE_LEVELS {
            break;
        }
        let refined = refine_tempo_from_acf_peak(acf.acf.view(), lag, frame_rate);
        let folded =
            align_tempo_to_bpm_range(refined, Some(LEVEL_FRAME_BPM.0), Some(LEVEL_FRAME_BPM.1))?
                .clamp(30.0, 320.0);
        let frames_per_beat = frames_per_beat_of(folded);
        if seen_periods.contains(&frames_per_beat) {
            continue;
        }
        seen_periods.push(frames_per_beat);
        if let Some(selected) = levels.first() {
            // Octave guard: an octave relative of the selected level never competes.
            let octaves = (folded / selected.folded).log2();
            if octaves.round() != 0.0
                && (octaves - octaves.round()).abs() < LEVEL_OCTAVE_GUARD_RATIO.log2()
            {
                continue;
            }
        }
        if frames_per_beat == 0 {
            if levels.is_empty() {
                // Even the selected level has no integer period to track.
                let estimate = TempoEstimate {
                    tempo: align_tempo_to_bpm_range(refined, bpm_min, bpm_max)?.clamp(30.0, 320.0),
                    tempo_raw: refined.clamp(30.0, 320.0),
                    candidates,
                };
                return Ok((estimate, vec![], range_free_beats.then(Vec::new)));
            }
            continue;
        }
        let (beats, local_score) = track(frames_per_beat);
        let consensus = beat_period_consensus(&beats);
        let agreement = consensus.map_or(0.0, |(_, agreement)| agreement);
        // The evidence belongs to this level only when its beats keep its period:
        // a tracker pulled onto another level's pulse measures that level instead.
        let measured = consensus.is_some_and(|(period, _)| {
            (60.0 * frame_rate / period / folded - 1.0).abs() <= MAX_BEAT_TEMPO_DEVIATION
        });
        let mean_local =
            beats.iter().map(|&b| local_score[b] as f64).sum::<f64>() / beats.len() as f64;
        let mean_local = if mean_local.is_finite() {
            mean_local as Float
        } else {
            Float::NEG_INFINITY
        };
        levels.push(TempoLevel {
            refined,
            folded,
            frames_per_beat,
            measured,
            evidence: agreement + LEVEL_LOCAL_SCORE_WEIGHT * mean_local,
            beats,
        });
    }

    // Keep the selected level unless another one has clearly better evidence.
    // Only a level whose own beats are measured and keep its period can win:
    // with too few beats the evidence is just the local score, and a tracker
    // pulled onto another pulse scores that pulse, not its level.
    let mut chosen = 0;
    for (index, level) in levels.iter().enumerate().skip(1) {
        if level.measured && level.evidence > levels[chosen].evidence {
            chosen = index;
        }
    }
    if chosen != 0 && levels[chosen].evidence - levels[0].evidence < LEVEL_SWITCH_MARGIN {
        chosen = 0;
    }
    let level = levels.swap_remove(chosen);

    // Report the period of the tracked beats at the chosen level, folded into
    // the caller's range. The ACF estimate stays when the beats are too few or
    // their period strays too far from it.
    let raw = level.refined.clamp(30.0, 320.0);
    let acf_tempo = align_tempo_to_bpm_range(level.refined, bpm_min, bpm_max)?.clamp(30.0, 320.0);
    let (tempo, tempo_raw) = beat_period_tempo(
        &level.beats,
        level.folded,
        raw / level.folded,
        frame_rate,
        bpm_min,
        bpm_max,
    )?
    .unwrap_or((acf_tempo, raw));

    // The beats of the chosen level as the caller's range folds it: retracked
    // only when that changes the integer beat period.
    let beats_at = |frames_per_beat: usize| {
        if frames_per_beat == level.frames_per_beat {
            level.beats.clone()
        } else if frames_per_beat == 0 {
            vec![]
        } else {
            track(frames_per_beat).0
        }
    };
    let frames_per_beat = frames_per_beat_of(acf_tempo);
    let beats = beats_at(frames_per_beat);
    // The same level's beats as a call without a range tracks them (`raw` is
    // `acf_tempo` unfolded); the level choice itself never depends on the range.
    let range_free_beats = range_free_beats.then(|| {
        let raw_frames_per_beat = frames_per_beat_of(raw);
        if raw_frames_per_beat == frames_per_beat {
            beats.clone()
        } else {
            beats_at(raw_frames_per_beat)
        }
    });

    Ok((
        TempoEstimate {
            tempo,
            tempo_raw,
            candidates,
        },
        beats,
        range_free_beats,
    ))
}

/// One tempo level: an ACF lag, its tempo folded into [`LEVEL_FRAME_BPM`] and
/// the beats the DP tracks at that period.
struct TempoLevel {
    /// Fractional-lag tempo of the ACF peak (BPM).
    refined: Float,
    /// `refined` folded into the level frame and clamped to `[30, 320]`.
    folded: Float,
    /// Integer beat period of `folded`, in frames.
    frames_per_beat: usize,
    /// Whether the beats are enough for the beat-pair consensus and keep this
    /// level's period (within [`MAX_BEAT_TEMPO_DEVIATION`]).
    measured: bool,
    /// Pair agreement of the beats plus the weighted mean local score at them.
    evidence: Float,
    beats: Vec<usize>,
}

/// Tempo of `beats` from their period, when the beat-period consensus applies:
/// `Some((tempo, tempo_raw))`, with `tempo_raw = beat tempo × octave` and
/// `tempo` its fold into the caller's range.
///
/// `level_tempo` is the ACF tempo the beats were tracked at; a beat period
/// more than [`MAX_BEAT_TEMPO_DEVIATION`] away from it, or too few beats,
/// gives `None`. A steady track whose beats fit the nearest integer tempo
/// reports that integer.
fn beat_period_tempo(
    beats: &[usize],
    level_tempo: Float,
    octave: Float,
    frame_rate: Float,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<Option<(Float, Float)>> {
    let Some((period, _agreement)) = beat_period_consensus(beats) else {
        return Ok(None);
    };
    let beat_tempo = 60.0 * frame_rate / period;
    if (beat_tempo / level_tempo - 1.0).abs() > MAX_BEAT_TEMPO_DEVIATION {
        return Ok(None);
    }
    let beat_tempo = integer_grid_tempo(beats, beat_tempo, frame_rate).unwrap_or(beat_tempo);
    let tempo_raw = beat_tempo * octave;
    let tempo = align_tempo_to_bpm_range(tempo_raw, bpm_min, bpm_max)?.clamp(30.0, 320.0);
    Ok(Some((tempo, tempo_raw)))
}

/// Period of the tracked beats from a hierarchical consensus of beat pairs.
///
/// A pair `sep` beats apart spans `sep` beat periods. The first stage seeds the
/// span of pairs `min(32, n/2)` apart with their median (the upper one for an
/// even count), then refines it with mean passes over the pairs within ±1
/// frame of the current span. Stages with pairs 64 and 128 beats apart (capped
/// at `n/2`) start from the previous period times their separation instead of
/// a fresh median, so a tracker slip by half a beat cannot capture them.
///
/// `beats` are strictly increasing frame indices, as the DP backtrack yields.
/// Returns `(period_frames, agreement)`, where `agreement` is the share of the
/// first-stage pairs within the tolerance, or `None` below
/// [`MIN_BEAT_PERIOD_BEATS`] beats.
fn beat_period_consensus(beats: &[usize]) -> Option<(Float, Float)> {
    let n = beats.len();
    if n < MIN_BEAT_PERIOD_BEATS {
        return None;
    }
    let pair_spans = |sep: usize| -> Vec<Float> {
        (0..n - sep)
            .map(|i| (beats[i + sep] - beats[i]) as Float)
            .collect()
    };

    let sep = BEAT_PAIR_SEPARATIONS[0].min(n / 2);
    let spans = pair_spans(sep);
    let mut sorted = spans.clone();
    sorted.sort_by(Float::total_cmp);
    let (span, agreeing) = refine_pair_span(&spans, sorted[sorted.len() / 2])?;
    let mut period = span / sep as Float;
    let agreement = agreeing as Float / spans.len() as Float;

    for &stage_sep in &BEAT_PAIR_SEPARATIONS[1..] {
        let sep = stage_sep.min(n / 2);
        if let Some((span, _)) = refine_pair_span(&pair_spans(sep), period * sep as Float) {
            period = span / sep as Float;
        }
    }
    Some((period, agreement))
}

/// Refine a pair span by repeated means over the spans within
/// [`BEAT_PAIR_TOLERANCE_FRAMES`] of the current value.
///
/// Returns the refined span and the agreeing-pair count of the last pass, or
/// `None` when no span lies within the tolerance of `span` on the first pass.
fn refine_pair_span(spans: &[Float], mut span: Float) -> Option<(Float, usize)> {
    let mut agreeing = None;
    for _ in 0..BEAT_PAIR_PASSES {
        let (mut sum, mut count) = (0.0f64, 0usize);
        for &x in spans {
            if (x - span).abs() <= BEAT_PAIR_TOLERANCE_FRAMES {
                sum += x as f64;
                count += 1;
            }
        }
        if count == 0 {
            break;
        }
        span = (sum / count as f64) as Float;
        agreeing = Some(count);
    }
    agreeing.map(|count| (span, count))
}

/// The integer tempo nearest `tempo` (BPM, at the level of `beats`) when a
/// constant grid at that integer stays on the tracked beats over the whole
/// track; `None` otherwise.
///
/// The grid fits when no beat segment drifts more than
/// [`STEADY_GRID_MAX_OFFSET_SEC`] off it. A grid that fits the whole track
/// means the tempo is both constant and that integer: a drifting tempo or a
/// fractional one slides off a constant integer grid (0.02 BPM at 128 BPM moves
/// it by about 47 ms in five minutes). The measured tempo itself is not tested,
/// because its own small error would make its grid drift on a steady track.
fn integer_grid_tempo(beats: &[usize], tempo: Float, frame_rate: Float) -> Option<Float> {
    let integer = tempo.round();
    if integer <= 0.0 || (tempo - integer).abs() > INTEGER_TEMPO_TOLERANCE {
        return None;
    }
    max_grid_offset_sec(beats, 60.0 * frame_rate as f64 / integer as f64, frame_rate as f64)
        .is_some_and(|offset| offset <= STEADY_GRID_MAX_OFFSET_SEC)
        .then_some(integer)
}

/// Largest phase offset (seconds) of [`STEADY_GRID_SEGMENTS`] consecutive beat
/// segments from a constant grid of `period_frames`, the grid phase fitted to
/// all beats by a circular mean. Phases are taken at
/// [`STEADY_GRID_HARMONIC`] times the beat rate. `None` with fewer than
/// [`STEADY_GRID_MIN_SEGMENT_BEATS`] beats per segment.
fn max_grid_offset_sec(beats: &[usize], period_frames: f64, frame_rate: f64) -> Option<f64> {
    let segments = STEADY_GRID_SEGMENTS;
    if beats.len() < segments * STEADY_GRID_MIN_SEGMENT_BEATS
        || !period_frames.is_finite()
        || period_frames <= 0.0
    {
        return None;
    }
    let omega = STEADY_GRID_HARMONIC * std::f64::consts::TAU / period_frames;
    let phase = |part: &[usize]| {
        let (sin, cos) = part.iter().fold((0.0f64, 0.0f64), |(s, c), &b| {
            let (bs, bc) = (omega * b as f64).sin_cos();
            (s + bs, c + bc)
        });
        sin.atan2(cos)
    };
    let global = phase(beats);
    // Consecutive segments, the first `len % segments` one beat longer.
    let (base, extra) = (beats.len() / segments, beats.len() % segments);
    let mut start = 0;
    let mut worst = 0.0f64;
    for segment in 0..segments {
        let end = start + base + usize::from(segment < extra);
        let offset = (phase(&beats[start..end]) - global + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        worst = worst.max(offset.abs() / omega / frame_rate);
        start = end;
    }
    Some(worst)
}

/// Autocorrelation tempo candidates of an onset envelope.
struct AcfCandidates {
    /// Autocorrelation of the envelope, by lag in frames.
    acf: Array1<Float>,
    /// `(lag, bpm, score)` for every lag in the 30–300 BPM window, lag ascending.
    candidates: Vec<(usize, Float, Float)>,
    /// Smallest lag of the window.
    min_lag: usize,
}

impl AcfCandidates {
    /// Lag of the preferred candidate (metrical selection), or the smallest lag.
    fn selected_lag(&self) -> usize {
        select_preferred_tempo_candidate(&self.candidates).map_or(self.min_lag, |c| c.0)
    }

    /// The strongest candidates (by score) as `(bpm, score)` pairs.
    fn ranked(&self) -> Vec<(Float, Float)> {
        let mut ranked: Vec<(Float, Float)> = self
            .candidates
            .iter()
            .filter(|(_, bpm, score)| bpm.is_finite() && score.is_finite())
            .map(|&(_, bpm, score)| (bpm, score))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(MAX_TEMPO_CANDIDATES);
        ranked
    }

    /// Lags of the local score maxima, strongest first.
    fn peak_lags(&self) -> Vec<usize> {
        let c = &self.candidates;
        let score = |i: Option<usize>| {
            i.and_then(|i| c.get(i))
                .map_or(Float::NEG_INFINITY, |c| c.2)
        };
        let mut peaks: Vec<(usize, Float)> = (0..c.len())
            .filter(|&i| {
                c[i].2 >= score(i.checked_sub(1))
                    && c[i].2 >= score(Some(i + 1))
                    && c[i].2.is_finite()
            })
            .map(|i| (c[i].0, c[i].2))
            .collect();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peaks.into_iter().map(|(lag, _)| lag).collect()
    }
}

/// Autocorrelation tempo candidates of `oenv`, weighted by a log-normal prior
/// centred at `start_bpm`; `None` when the envelope is too short for a lag
/// window.
fn acf_candidates(
    oenv: &Array1<Float>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
) -> Result<Option<AcfCandidates>> {
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    // Autocorrelate onset envelope
    let max_lag = (4.0 * frame_rate).min(oenv.len() as Float) as usize; // up to 4 seconds
    let acf = crate::core::audio::autocorrelate(oenv.view(), Some(max_lag))?;

    if acf.is_empty() {
        return Ok(None);
    }

    // Find peaks in BPM range [30, 300]
    let min_lag = (60.0 * frame_rate / 300.0).ceil() as usize;
    let max_lag = (60.0 * frame_rate / 30.0).floor() as usize;
    let max_lag = max_lag.min(acf.len() - 1);

    if min_lag >= max_lag {
        return Ok(None);
    }

    // Weight by log-normal prior centered at start_bpm, collecting every
    // candidate so downstream metrical-multiple lifting can inspect them.
    let mut candidates = Vec::with_capacity(max_lag - min_lag + 1);

    for lag in min_lag..=max_lag {
        let bpm = 60.0 * frame_rate / lag as Float;
        let log_prior = -0.5 * ((bpm.log2() - start_bpm.log2()) / 1.0).powi(2);
        let score = acf[lag] * (1.0 + log_prior.exp());
        candidates.push((lag, bpm, score));
    }

    Ok(Some(AcfCandidates {
        acf,
        candidates,
        min_lag,
    }))
}

/// Refine a tempo from an integer ACF lag using parabolic interpolation of the
/// autocorrelation peak. This removes the 1–3 BPM quantization drift caused by
/// snapping tempo to integer lags.
fn refine_tempo_from_acf_peak(acf: ArrayView1<Float>, lag: usize, frame_rate: Float) -> Float {
    let integer_tempo = if lag > 0 {
        60.0 * frame_rate / lag as Float
    } else {
        return 0.0;
    };

    if lag + 1 >= acf.len() || !frame_rate.is_finite() || frame_rate <= 0.0 {
        return integer_tempo;
    }

    let left = acf[lag - 1];
    let center = acf[lag];
    let right = acf[lag + 1];
    if !left.is_finite()
        || !center.is_finite()
        || !right.is_finite()
        || center < left
        || center < right
    {
        return integer_tempo;
    }

    let denominator = left - 2.0 * center + right;
    if denominator.abs() <= 1e-12 {
        return integer_tempo;
    }

    let offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
    let refined_lag = lag as Float + offset;
    if refined_lag <= 0.0 || !refined_lag.is_finite() {
        integer_tempo
    } else {
        60.0 * frame_rate / refined_lag
    }
}

/// Choose a preferred tempo candidate, lifting a supported metrical multiple
/// (2x / 1.5x) when the raw best BPM is suspiciously low. Electronic music
/// frequently produces a strong half-tempo ACF peak; when a supported
/// double/dotted multiple exists in the higher, danceable range we prefer it.
fn select_preferred_tempo_candidate(
    candidates: &[(usize, Float, Float)],
) -> Option<(usize, Float, Float)> {
    let best = candidates
        .iter()
        .copied()
        .filter(|(_, bpm, score)| bpm.is_finite() && score.is_finite())
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))?;

    if best.1 < 75.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 115.0, 150.0, 1.80, 2.20, 0.50)
        {
            return Some(candidate);
        }
    } else if best.1 < 90.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 120.0, 145.0, 1.42, 1.62, 0.75)
        {
            return Some(candidate);
        }
    } else if best.1 < 95.0 && best.2 >= 4.0 {
        if let Some(candidate) =
            best_supported_metrical_candidate(candidates, best, 120.0, 145.0, 1.42, 1.62, 0.85)
        {
            return Some(candidate);
        }
    }

    Some(best)
}

/// Find the strongest candidate that is a supported metrical multiple of `best`
/// (within the given BPM window, multiple range, and score-ratio floor).
fn best_supported_metrical_candidate(
    candidates: &[(usize, Float, Float)],
    best: (usize, Float, Float),
    min_bpm: Float,
    max_bpm: Float,
    min_multiple: Float,
    max_multiple: Float,
    min_score_ratio: Float,
) -> Option<(usize, Float, Float)> {
    candidates
        .iter()
        .copied()
        .filter(|candidate| {
            let multiple = candidate.1 / best.1;
            candidate.1 >= min_bpm
                && candidate.1 <= max_bpm
                && multiple >= min_multiple
                && multiple <= max_multiple
                && candidate.2 >= best.2 * min_score_ratio
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
}

/// Relative tolerance at the range edges: a tempo within 0.1% outside a bound
/// stays at that bound's octave (179.994 and 180.001 both stay near 180 in a
/// 70–180 range) instead of jumping a whole octave on measurement noise.
const RANGE_EDGE_TOLERANCE: Float = 0.001;

/// Deterministically double/halve a tempo into an optional user-supplied BPM
/// range. Both bounds must be supplied together and span about one octave or
/// more: `bpm_max + 1 >= 2 * bpm_min`, so Rekordbox's one-octave ranges
/// `[a, 2a - 1]` (e.g. 68–135) are accepted. The edges are inclusive within
/// [`RANGE_EDGE_TOLERANCE`]. A tempo halved from above `bpm_max` can land in
/// the small gap below `bpm_min` of a one-octave range; it stays there, inside
/// `(bpm_max / 2, bpm_max]`, the window Rekordbox itself shows.
fn align_tempo_to_bpm_range(
    mut tempo: Float,
    bpm_min: Option<Float>,
    bpm_max: Option<Float>,
) -> Result<Float> {
    let (min_bpm, max_bpm) = match (bpm_min, bpm_max) {
        (None, None) => return Ok(tempo),
        (Some(min_bpm), Some(max_bpm)) => (min_bpm, max_bpm),
        _ => {
            return Err(SonaraError::InvalidParameter {
                param: "bpm_range",
                reason: "bpm_min and bpm_max must be provided together".into(),
            });
        }
    };

    if !min_bpm.is_finite() || !max_bpm.is_finite() || min_bpm <= 0.0 || max_bpm <= min_bpm {
        return Err(SonaraError::InvalidParameter {
            param: "bpm_range",
            reason: "expected finite values with 0 < bpm_min < bpm_max".into(),
        });
    }
    if max_bpm + 1.0 < min_bpm * 2.0 {
        return Err(SonaraError::InvalidParameter {
            param: "bpm_range",
            reason: "bpm_max + 1 must be at least double bpm_min for octave folding".into(),
        });
    }
    if !tempo.is_finite() || tempo <= 0.0 {
        return Ok(tempo);
    }

    while tempo < min_bpm * (1.0 - RANGE_EDGE_TOLERANCE) {
        tempo *= 2.0;
    }
    while tempo > max_bpm * (1.0 + RANGE_EDGE_TOLERANCE) {
        tempo /= 2.0;
    }
    Ok(tempo)
}

/// Compute local beat score via Gaussian convolution.
fn beat_local_score(oenv: ArrayView1<Float>, frames_per_beat: usize) -> Array1<Float> {
    let n = oenv.len();
    let fpb = frames_per_beat as Float;

    // Build Gaussian window: exp(-0.5 * (lag * 32 / fpb)^2)
    let half_win = frames_per_beat;
    let win_len = 2 * half_win + 1;
    let window: Vec<Float> = (0..win_len)
        .map(|i| {
            let lag = (i as Float - half_win as Float) * 32.0 / fpb;
            (-0.5 * lag * lag).exp()
        })
        .collect();
    let win_sum: Float = window.iter().sum();

    // Convolve (same mode)
    let mut score = Array1::<Float>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..win_len {
            let idx = i as i64 + j as i64 - half_win as i64;
            if idx >= 0 && (idx as usize) < n {
                sum += oenv[idx as usize] * window[j];
            }
        }
        score[i] = sum / win_sum;
    }

    score
}

/// Dynamic programming beat tracker.
///
/// Scoring: cumscore[i] = localscore[i] + max_j(cumscore[j] - tightness * (log(i-j) - log(fpb))^2)
/// where j ranges over [i - 2*fpb, i - fpb/2]
fn beat_track_dp(
    local_score: ArrayView1<Float>,
    frames_per_beat: usize,
    tightness: Float,
) -> Vec<usize> {
    let n = local_score.len();
    if n == 0 {
        return vec![];
    }

    let fpb = frames_per_beat;
    let log_fpb = (fpb as Float).ln();

    // Pre-compute ln() lookup table to avoid transcendental calls in the inner loop.
    // Intervals range from fpb/2 to 2*fpb, so we need ln(1) through ln(2*fpb).
    let max_interval = 2 * fpb + 1;
    let ln_table: Vec<Float> = (0..=max_interval)
        .map(|i| if i == 0 { 0.0 } else { (i as Float).ln() })
        .collect();

    let mut cumscore = vec![0.0_f32; n];
    let mut backlink = vec![0usize; n];

    // Forward pass — every frame gets a backlink to its best predecessor.
    // No score threshold here; trimming happens post-hoc (Ellis 2007).
    for i in 0..n {
        // Search backwards for best predecessor
        let search_start = i.saturating_sub(2 * fpb);
        let search_end = i.saturating_sub(fpb / 2);

        let mut best_score = Float::NEG_INFINITY;
        let mut best_j = 0usize;

        for j in search_start..search_end.min(i) {
            let interval = i - j;
            let ln_interval = if interval <= max_interval {
                ln_table[interval]
            } else {
                (interval as Float).ln()
            };
            let penalty = tightness * (ln_interval - log_fpb).powi(2);
            let score = cumscore[j] - penalty;
            if score > best_score {
                best_score = score;
                best_j = j;
            }
        }

        if best_score > Float::NEG_INFINITY {
            cumscore[i] = local_score[i] + best_score;
            backlink[i] = best_j;
        } else {
            cumscore[i] = local_score[i];
            backlink[i] = i; // self-link (no predecessor found yet)
        }
    }

    // Find last beat: highest cumscore in the tail region
    let tail_start = n.saturating_sub(2 * fpb);
    let mut last_beat = tail_start;
    let mut best = Float::NEG_INFINITY;
    for i in tail_start..n {
        if cumscore[i] > best {
            best = cumscore[i];
            last_beat = i;
        }
    }

    // Backtrack
    let mut beats = vec![last_beat];
    let mut current = last_beat;
    while backlink[current] != current && backlink[current] < current {
        current = backlink[current];
        beats.push(current);
    }

    beats.reverse();
    beats
}

/// Trim beats with low onset strength at the edges.
fn trim_beats(local_score: ArrayView1<Float>, beats: &[usize]) -> Vec<usize> {
    if beats.is_empty() {
        return vec![];
    }

    let threshold = 0.01 * local_score.iter().copied().fold(0.0_f32, Float::max);

    let start = beats
        .iter()
        .position(|&b| b < local_score.len() && local_score[b] >= threshold)
        .unwrap_or(0);

    let end = beats
        .iter()
        .rposition(|&b| b < local_score.len() && local_score[b] >= threshold)
        .unwrap_or(beats.len() - 1);

    beats[start..=end].to_vec()
}

/// Predominant Local Pulse.
///
/// Estimates a pulse curve from the onset envelope using Fourier tempogram.
pub fn plp(
    y: ArrayView1<Float>,
    sr: u32,
    hop_length: usize,
    tempo_min: Float,
    tempo_max: Float,
) -> Result<Array1<Float>> {
    let oenv = onset::onset_strength(y, sr, hop_length)?;
    let n = oenv.len();

    if n < 4 {
        return Ok(Array1::zeros(n));
    }

    // Simple PLP: autocorrelate onset envelope in windows
    let win_length = 384.min(n);
    let acf = crate::core::audio::autocorrelate(oenv.view(), Some(win_length))?;

    // Weight by tempo range
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;
    let min_lag = (60.0 * frame_rate / tempo_max).ceil() as usize;
    let max_lag = (60.0 * frame_rate / tempo_min).floor() as usize;

    let mut pulse = Array1::<Float>::zeros(n);

    // Find peak in tempo range
    let mut best_lag = min_lag;
    let mut best_val = 0.0;
    for lag in min_lag..max_lag.min(acf.len()) {
        if acf[lag] > best_val {
            best_val = acf[lag];
            best_lag = lag;
        }
    }

    // Generate pulse at detected tempo
    if best_lag > 0 {
        let period = best_lag;
        for i in (0..n).step_by(period) {
            pulse[i] = 1.0;
        }
    }

    Ok(pulse)
}

/// Compute a per-beat tempo curve from beat frame positions.
///
/// Returns a vector of BPM values, one per inter-beat interval
/// (length = `beat_frames.len() - 1`).
///
/// - `smooth`: optional median filter window size for smoothing.
///   Use an odd number (e.g., 5) to reduce jitter.
pub fn tempo_curve(
    beat_frames: &[usize],
    sr: u32,
    hop_length: usize,
    smooth: Option<usize>,
) -> Result<Vec<Float>> {
    if beat_frames.len() < 2 {
        return Ok(vec![]);
    }

    let sr_f = sr as Float;
    let hop_f = hop_length as Float;

    // Convert frame intervals to BPM
    let mut bpms: Vec<Float> = beat_frames
        .windows(2)
        .map(|w| {
            let dt = (w[1] as Float - w[0] as Float) * hop_f / sr_f;
            if dt > 0.0 {
                60.0 / dt
            } else {
                0.0
            }
        })
        .collect();

    // Optional median filter smoothing
    if let Some(k) = smooth {
        if k >= 3 && bpms.len() >= k {
            let half = k / 2;
            let orig = bpms.clone();
            for i in half..orig.len().saturating_sub(half) {
                let mut window: Vec<Float> = orig[i - half..i + half + 1].to_vec();
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                bpms[i] = window[window.len() / 2];
            }
        }
    }

    Ok(bpms)
}

/// Compute the coefficient of variation (std/mean) of the tempo curve.
///
/// A low value (< 0.05) indicates steady tempo; a high value (> 0.1)
/// indicates significant tempo variation.
pub fn tempo_variability(tempo_curve: &[Float]) -> Float {
    if tempo_curve.is_empty() {
        return 0.0;
    }
    let n = tempo_curve.len() as Float;
    let mean = tempo_curve.iter().sum::<Float>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = tempo_curve
        .iter()
        .map(|&b| (b - mean).powi(2))
        .sum::<Float>()
        / n;
    variance.sqrt() / mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn click_train(sr: u32, dur: Float, bpm: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        let interval = (60.0 / bpm * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        y
    }

    /// Click train at exactly `bpm`: click `k` starts at the sample nearest
    /// `k * 60 / bpm` seconds, computed in f64 so a fractional period does not
    /// drift (unlike `click_train`, which truncates the interval).
    fn exact_click_train(sr: u32, dur: Float, bpm: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        let interval = 60.0 / bpm as f64 * sr as f64;
        let mut y = Array1::<Float>::zeros(n);
        let mut clicks = 0usize;
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            clicks += 1;
            pos = (clicks as f64 * interval).round() as usize;
        }
        y
    }

    #[test]
    fn test_beat_track_clicks() {
        let y = click_train(22050, 4.0, 120.0);
        let (tempo, beats) =
            beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            tempo > 80.0 && tempo < 180.0,
            "tempo {tempo} should be near 120"
        );
        assert!(beats.len() >= 3, "expected >=3 beats, got {}", beats.len());
    }

    #[test]
    fn test_beat_track_tempo() {
        let y = click_train(22050, 4.0, 120.0);
        let (tempo, _) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            (tempo - 120.0).abs() < 30.0,
            "tempo {tempo} should be ~120 BPM"
        );
    }

    #[test]
    fn test_tempo_candidate_selection_lifts_supported_half_tempo() {
        let selected = select_preferred_tempo_candidate(&[
            (41, 63.024010, 11.464),
            (31, 83.354332, 9.647),
            (21, 123.046875, 7.414),
            (20, 129.199219, 7.407),
        ])
        .unwrap();
        assert_eq!(selected.0, 21);
    }

    #[test]
    fn test_tempo_candidate_selection_lifts_supported_three_halves_tempo() {
        let selected = select_preferred_tempo_candidate(&[
            (29, 89.102913, 9.025),
            (19, 135.999176, 8.613),
            (39, 66.256012, 7.820),
            (20, 129.199219, 6.865),
        ])
        .unwrap();
        assert_eq!(selected.0, 19);
    }

    #[test]
    fn test_tempo_candidate_selection_keeps_weakly_supported_true_90_bpm() {
        let selected = select_preferred_tempo_candidate(&[
            (29, 89.102913, 2.050),
            (20, 129.199219, 1.338),
            (19, 135.999176, 0.939),
            (39, 66.256012, 0.747),
        ])
        .unwrap();
        assert_eq!(selected.0, 29);
    }

    #[test]
    fn test_tempo_candidate_selection_keeps_low_confidence_92_bpm() {
        let selected = select_preferred_tempo_candidate(&[
            (28, 92.285156, 2.170),
            (37, 69.837418, 2.145),
            (18, 143.554688, 2.144),
            (19, 135.999176, 2.082),
        ])
        .unwrap();
        assert_eq!(selected.0, 28);
    }

    #[test]
    fn test_tempo_refinement_uses_fractional_acf_peak() {
        let mut acf = Array1::<Float>::zeros(24);
        acf[19] = 7.0;
        acf[20] = 10.0;
        acf[21] = 10.0;

        let frame_rate = 22050.0 / 512.0;
        let refined = refine_tempo_from_acf_peak(acf.view(), 20, frame_rate);

        let expected = 60.0 * frame_rate / 20.5;
        assert!(
            (refined - expected).abs() < 1e-5,
            "expected fractional-lag tempo {expected}, got {refined}"
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_doubles_low_values() {
        assert_eq!(
            align_tempo_to_bpm_range(63.02401, Some(79.0), Some(192.0)).unwrap(),
            126.04802
        );
        assert_eq!(
            align_tempo_to_bpm_range(66.25601, Some(79.0), Some(192.0)).unwrap(),
            132.51202
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_halves_high_values() {
        assert!(
            (align_tempo_to_bpm_range(250.0, Some(79.0), Some(192.0)).unwrap() - 125.0).abs()
                < 1e-6
        );
        assert!(
            (align_tempo_to_bpm_range(401.0, Some(79.0), Some(192.0)).unwrap() - 100.25).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_align_tempo_to_bpm_range_requires_complete_valid_range() {
        assert!(align_tempo_to_bpm_range(120.0, None, None).is_ok());
        assert!(align_tempo_to_bpm_range(120.0, Some(79.0), None).is_err());
        assert!(align_tempo_to_bpm_range(120.0, None, Some(192.0)).is_err());
        assert!(align_tempo_to_bpm_range(120.0, Some(192.0), Some(79.0)).is_err());
        assert!(align_tempo_to_bpm_range(120.0, Some(100.0), Some(150.0)).is_err());
    }

    #[test]
    fn test_align_tempo_to_bpm_range_accepts_one_octave_ranges() {
        // Rekordbox's one-octave presets [a, 2a - 1].
        for a in [48.0, 58.0, 68.0, 78.0, 88.0, 98.0, 108.0, 118.0, 128.0] {
            let tempo = align_tempo_to_bpm_range(150.0, Some(a), Some(2.0 * a - 1.0)).unwrap();
            assert!(tempo > (2.0 * a - 1.0) / 2.0 && tempo <= 2.0 * a - 1.0, "{a}: {tempo}");
        }
        // Halved from above the top it may land just below a, inside (max/2, max].
        let tempo = align_tempo_to_bpm_range(135.5, Some(68.0), Some(135.0)).unwrap();
        assert!((tempo - 67.75).abs() < 1e-4, "{tempo}");
    }

    #[test]
    fn test_align_tempo_to_bpm_range_edges_are_inclusive_within_tolerance() {
        let near = |t: Float| align_tempo_to_bpm_range(t, Some(70.0), Some(180.0)).unwrap();
        assert!((near(179.994) - 179.994).abs() < 1e-3);
        assert!((near(180.001) - 180.001).abs() < 1e-3);
        assert!((near(69.95) - 69.95).abs() < 1e-3);
        // Beyond the tolerance the octave still folds.
        assert!((near(181.0) - 90.5).abs() < 1e-3);
        assert!((near(69.0) - 138.0).abs() < 1e-3);
    }

    #[test]
    fn test_beat_track_flat_onset_envelope_falls_back_without_beats() {
        let env = Array1::<Float>::zeros(128);
        let (tempo, beats) =
            beat_track(None, Some(env.view()), 22050, 512, 120.0, 100.0, true).unwrap();
        assert!(
            (tempo - 120.0).abs() < 1e-6,
            "flat onset envelope should fall back to start BPM, got {tempo}"
        );
        assert!(
            beats.is_empty(),
            "flat onset envelope should not produce beats, got {}",
            beats.len()
        );

        // Too few tracked beats for the beat-period consensus: the ACF
        // estimate is reported unchanged.
        let y = click_train(22050, 4.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let (estimate, beats) = beat_track_detailed(
            None,
            Some(oenv.view()),
            22050,
            512,
            120.0,
            100.0,
            true,
            None,
            None,
        )
        .unwrap();
        assert!(
            !beats.is_empty() && beats.len() < MIN_BEAT_PERIOD_BEATS,
            "expected a few beats, got {}",
            beats.len()
        );
        let acf = acf_candidates(&oenv, 22050, 512, 120.0).unwrap().unwrap();
        let acf_tempo =
            refine_tempo_from_acf_peak(acf.acf.view(), acf.selected_lag(), 22050.0 / 512.0)
                .clamp(30.0, 320.0);
        assert_eq!(estimate.tempo.to_bits(), acf_tempo.to_bits());
        assert_eq!(estimate.tempo_raw.to_bits(), acf_tempo.to_bits());
    }

    #[test]
    fn test_beat_track_reports_tracked_beat_period() {
        let y = exact_click_train(22050, 60.0, 128.0);
        let (estimate, beats) = beat_track_detailed(
            Some(y.view()),
            None,
            22050,
            512,
            120.0,
            100.0,
            true,
            None,
            None,
        )
        .unwrap();
        assert!(
            beats.len() >= MIN_BEAT_PERIOD_BEATS,
            "expected the consensus to apply, got {} beats",
            beats.len()
        );
        assert!(
            (estimate.tempo - 128.0).abs() <= 0.005,
            "128 BPM clicks should report the beat period, got {}",
            estimate.tempo
        );
        assert_eq!(estimate.tempo.to_bits(), estimate.tempo_raw.to_bits());
    }

    #[test]
    fn test_dp_envelope_must_match_the_onset_envelope_length() {
        let onset_envelope = Array1::<Float>::zeros(128);
        let dp_envelope = Array1::<Float>::zeros(127);
        let result = beat_track_detailed_with_dp_envelope(
            None,
            Some(onset_envelope.view()),
            22050,
            512,
            120.0,
            100.0,
            true,
            None,
            None,
            Some(dp_envelope.view()),
        );
        assert!(matches!(
            result,
            Err(SonaraError::InvalidParameter {
                param: "dp_envelope",
                ..
            })
        ));
    }

    #[test]
    fn test_beat_track_silence() {
        let y = Array1::<Float>::zeros(44100);
        let (_, beats) = beat_track(Some(y.view()), None, 22050, 512, 120.0, 100.0, true).unwrap();
        // DP tracker may find some beats in silence; the important thing is it doesn't crash
        // and produces fewer beats than a click train would
        assert!(beats.len() < 50, "silence produced {} beats", beats.len());
    }

    #[test]
    fn test_plp_basic() {
        let y = click_train(22050, 2.0, 120.0);
        let pulse = plp(y.view(), 22050, 512, 30.0, 300.0).unwrap();
        assert!(pulse.len() > 0);
    }

    #[test]
    fn test_tempo_curve_steady() {
        // Steady 120 BPM → each beat every ~43 frames at sr=22050, hop=512
        let frames_per_beat = (60.0_f32 / 120.0 * 22050.0 / 512.0).round() as usize;
        let beats: Vec<usize> = (0..10).map(|i| i * frames_per_beat).collect();
        let curve = tempo_curve(&beats, 22050, 512, None).unwrap();
        assert_eq!(curve.len(), 9);
        for &bpm in &curve {
            assert!((bpm - 120.0).abs() < 5.0, "expected ~120 BPM, got {bpm}");
        }
        let var = tempo_variability(&curve);
        assert!(
            var < 0.01,
            "steady tempo should have low variability, got {var}"
        );
    }

    #[test]
    fn test_tempo_curve_empty() {
        let curve = tempo_curve(&[], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
        let curve = tempo_curve(&[10], 22050, 512, None).unwrap();
        assert!(curve.is_empty());
    }
}
