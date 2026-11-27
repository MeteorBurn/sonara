//! Fused audio analysis pipeline.
//!
//! Computes all common audio features in a single optimized pass,
//! eliminating redundant STFT computation.
//!
//! ## Modes
//!
//! - **Compact** (default): core signal features — tempo, beats, onsets, RMS,
//!   centroid, ZCR, dynamic range. ~0.6ms per 10s track.
//! - **Playlist**: everything needed for playlist generation — adds spectral
//!   features, MFCCs, chroma, plus perceptual features (energy, danceability,
//!   key, valence, acousticness). ~3ms per 10s track.
//! - **Full**: every registered feature — Playlist plus `tempo_curve`,
//!   `time_signature` and all opt-in families (see `FEATURE_REGISTRY`). The
//!   file-only ones (`ebur128`, `native_dynamics`, `amplitude_envelope`,
//!   `stereo_field`, `tags`) are measured by `analyze_file` / `analyze_batch`.
//!
//! ## Algorithms
//!
//! All features use their most accurate algorithms by default:
//! - **Chroma**: proper chroma filterbank (sparse projection)
//! - **Spectral contrast**: log-spaced frequency bands with partial sort
//! - **HPCP / chords**: spectral peak detection with harmonic weighting (Gomez 2006)
//! - **Dissonance**: Sethares (1998) Plomp-Levelt model
//! - All features are fused into a single FFT pass to minimize cache pressure.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::path::Path;

use ndarray::{s, Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::core::{audio, convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{Result, SonaraError};
use crate::filters;
use crate::perceptual;
use crate::types::*;
use crate::util::utils;

#[cfg(feature = "aggression")]
#[path = "aggression_dsp.rs"]
mod aggression_dsp;

/// Minimum number of frames to justify rayon thread overhead.
const PARALLEL_THRESHOLD: usize = 32;

// ---- Frame statistics (MIR_FEATURES group 1.2) ----
// Aggregates over every main-pass frame, the same frames the `_mean` fields use.
// Accumulated in f64 so long tracks do not lose precision in f32 sums.

/// Population standard deviation (ddof = 0); 0 for an empty series.
fn series_std(x: &[Float]) -> Float {
    if x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var.sqrt() as Float
}

/// Mean absolute difference between consecutive frames; 0 for under two frames.
fn series_delta_mean(x: &[Float]) -> Float {
    if x.len() < 2 {
        return 0.0;
    }
    let sum: f64 = x
        .windows(2)
        .map(|w| (w[1] as f64 - w[0] as f64).abs())
        .sum();
    (sum / (x.len() - 1) as f64) as Float
}

/// Ascending copy for [`sorted_quantile`].
fn sorted_series(x: &[Float]) -> Vec<Float> {
    let mut v = x.to_vec();
    v.sort_by(Float::total_cmp);
    v
}

/// Quantile `q` in `[0, 1]` of an ascending series, interpolating linearly
/// between order statistics (NumPy's default method); 0 for an empty series.
fn sorted_quantile(sorted: &[Float], q: f64) -> Float {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let lo_v = sorted[lo] as f64;
    (lo_v + (sorted[hi] as f64 - lo_v) * (pos - lo as f64)) as Float
}

/// Interquartile range (p75 - p25) of an ascending series.
fn sorted_iqr(sorted: &[Float]) -> Float {
    sorted_quantile(sorted, 0.75) - sorted_quantile(sorted, 0.25)
}

/// Zero-crossing rate of every main-pass frame: crossings among the frame's
/// `frame_length` samples of the centered (zero-padded) signal, divided by
/// `frame_length`. Same sign rule as [`validated_zero_crossing_rate`].
fn frame_zero_crossing_rates(
    padded: &[Float],
    frame_length: usize,
    hop_length: usize,
    n_frames: usize,
) -> Vec<Float> {
    // Counted per frame (each sample read frame_length / hop_length times):
    // no signal-length buffer next to the already padded signal.
    (0..n_frames)
        .map(|t| {
            let start = t * hop_length;
            let crossings = padded[start..start + frame_length]
                .windows(2)
                .filter(|p| (p[0] > 0.0) != (p[1] > 0.0))
                .count();
            crossings as Float / frame_length as Float
        })
        .collect()
}

// ---- Spectral shape (MIR_FEATURES group 3.1), per-frame math in f64 ----

/// Octave bands of `spectral_slope_db_per_octave`: `(log2 centre Hz, first
/// bin, end bin)` for centres 62.5·2^b Hz (b = 0..7), taking bins with
/// `centre/√2 <= f_k < min(centre·√2, sr/2)`, f_k = k·sr/n_fft. Empty bands
/// are left out.
fn spectral_shape_octave_bands(sr: u32, n_fft: usize, n_bins: usize) -> Vec<(f64, usize, usize)> {
    let bin_hz = sr as f64 / n_fft as f64;
    let nyquist = sr as f64 / 2.0;
    (0..8)
        .filter_map(|b| {
            let centre = 62.5 * 2.0_f64.powi(b);
            let lo = centre / std::f64::consts::SQRT_2;
            let hi = (centre * std::f64::consts::SQRT_2).min(nyquist);
            let first = (0..n_bins).find(|&k| k as f64 * bin_hz >= lo)?;
            let end = (first..n_bins)
                .find(|&k| k as f64 * bin_hz >= hi)
                .unwrap_or(n_bins);
            (end > first).then_some((centre.log2(), first, end))
        })
        .collect()
}

/// Spectral skewness and excess kurtosis about the centroid, weighted by
/// magnitude and standardized by the bandwidth `bw`. `None` unless
/// `cent_den` (Σ|X_k|) and `bw` are positive.
fn spectral_shape_moments(
    mag: &[Float],
    freqs: &[Float],
    centroid: Float,
    cent_den: Float,
    bw: Float,
) -> Option<(f64, f64)> {
    if !(cent_den > 0.0 && bw > 0.0) {
        return None;
    }
    let (mut m3, mut m4) = (0.0_f64, 0.0_f64);
    for (&m, &f) in mag.iter().zip(freqs) {
        let d = f as f64 - centroid as f64;
        let d2 = d * d;
        m3 += m as f64 * d2 * d;
        m4 += m as f64 * d2 * d2;
    }
    let (den, bw) = (cent_den as f64, bw as f64);
    Some((m3 / den / bw.powi(3), m4 / den / bw.powi(4) - 3.0))
}

/// Least-squares slope (dB per octave) of the octave-band power levels
/// against log2 of the band centres. `None` for a silent frame or fewer
/// than two bands.
fn spectral_shape_slope(power: &[Float], bands: &[(f64, usize, usize)]) -> Option<f64> {
    let total: f64 = power.iter().map(|&p| p as f64).sum();
    if !(total > 0.0) || bands.len() < 2 {
        return None;
    }
    let points: Vec<(f64, f64)> = bands
        .iter()
        .map(|&(x, first, end)| {
            let sum: f64 = power[first..end].iter().map(|&p| p as f64).sum();
            (x, 10.0 * (sum / (end - first) as f64).max(1e-10).log10())
        })
        .collect();
    let n = points.len() as f64;
    let x_mean = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y_mean = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &(x, y) in &points {
        sxy += (x - x_mean) * (y - y_mean);
        sxx += (x - x_mean) * (x - x_mean);
    }
    Some(sxy / sxx)
}

/// Spectral decrease with the DC bin as reference:
/// Σ_{k≥1} (|X_k| − |X_0|)/k / Σ_{k≥1} |X_k|. `None` when the denominator is 0.
fn spectral_shape_decrease(mag: &[Float]) -> Option<f64> {
    let m0 = *mag.first()? as f64;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (k, &m) in mag.iter().enumerate().skip(1) {
        num += (m as f64 - m0) / k as f64;
        den += m as f64;
    }
    (den > 0.0).then(|| num / den)
}

// ---- Spectral texture (MIR_FEATURES group 3.2), per-frame math in f64 ----

/// Frames computed (in parallel) and scattered per batch in the fused pass;
/// bounds the per-frame results, incl. `spectral_texture`'s normalized
/// spectra, held at once.
const FRAME_CHUNK: usize = 512;

/// One frame of `spectral_texture` from its magnitudes and their sum
/// `sum` = Σ|X_k| (the fused pass's `cent_den`): the normalized spectrum
/// p_k = |X_k| / sum kept for the next frame's flux, the crest
/// max|X_k| / mean|X_k| and the entropy −Σ_{p_k>0} p_k ln p_k / ln(n_bins),
/// all in one pass. `None` for a silent frame (sum = 0).
struct TextureFrame {
    p: Vec<Float>,
    crest: f64,
    entropy: f64,
}

fn spectral_texture_frame(mag: &[Float], sum: Float) -> Option<TextureFrame> {
    if !(sum > 0.0) {
        return None;
    }
    let inv = 1.0 / sum;
    let (mut max, mut h) = (0.0_f32, 0.0_f64);
    let p: Vec<Float> = mag
        .iter()
        .map(|&m| {
            max = max.max(m);
            let v = m * inv;
            if v > 0.0 {
                h -= (v * v.ln()) as f64;
            }
            v
        })
        .collect();
    let n = mag.len() as f64;
    Some(TextureFrame {
        p,
        crest: max as f64 * n / sum as f64,
        entropy: h / n.ln(),
    })
}

/// Half-wave rectified L2 distance of two normalized spectra:
/// sqrt(Σ max(0, cur_k − prev_k)²).
fn spectral_texture_flux(prev: &[Float], cur: &[Float]) -> f64 {
    prev.iter()
        .zip(cur)
        .map(|(&a, &b)| ((b - a).max(0.0) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Jensen irregularity Σ(a_{i+1} − a_i)² / Σa_i² over peaks sorted by
/// frequency. `None` with fewer than 2 peaks or zero energy.
fn spectral_texture_irregularity(freqs: &[Float], mags: &[Float]) -> Option<f64> {
    let mut peaks: Vec<(Float, f64)> = freqs
        .iter()
        .zip(mags)
        .map(|(&f, &a)| (f, a as f64))
        .collect();
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
    let energy: f64 = peaks.iter().map(|p| p.1 * p.1).sum();
    if peaks.len() < 2 || !(energy > 0.0) {
        return None;
    }
    let diff: f64 = peaks.windows(2).map(|w| (w[1].1 - w[0].1).powi(2)).sum();
    Some(diff / energy)
}

// ---- Tuning (deviation from A4 = 440 Hz), per-frame math in f64 ----

/// Mean resultant length below which `tuning_offset_cents` abstains.
const TUNING_MIN_CONFIDENCE: Float = 0.08;
/// With few voting frames the bar is `TUNING_SHORT_BAR / √frames` instead:
/// noise reads about 0.4/√frames, so under ~3 s it would clear the fixed bar
/// (20 % of 1 s clips, none of 300 at this factor). Above 352 frames (~8 s)
/// the fixed bar is the higher one.
const TUNING_SHORT_BAR: f64 = 1.5;
/// Strongest spectral peaks per frame that vote on the tuning.
const TUNING_PEAKS: usize = 10;
/// Peaks at or below this fraction of the frame's strongest bin (−80 dB) are
/// FFT round-off, not partials: a signal with nothing in the band must abstain.
const TUNING_REL_FLOOR: Float = 1e-4;

/// One frame's tuning vote `[re, im, wsum]` (Dressler & Streich, ISMIR 2007):
/// the 10 strongest strict local maxima of `mag` with bin centre in
/// 150–5000 Hz (on a tie the lower bin; peaks at or below −80 dB of the
/// frame's strongest bin are skipped), each refined by a parabola on ln|X|
/// and added as the unit vector at angle 2π·frac(12·log2(f/440)) with weight
/// sqrt|X_i|. Deliberately separate from the linear-magnitude peak
/// interpolation HPCP, dissonance and irregularity share.
fn tuning_frame(mag: &[Float], sr: u32, n_fft: usize) -> [f64; 3] {
    const TINY: f64 = 1e-30;
    let bin_hz = sr as f64 / n_fft as f64;
    let floor = mag.iter().copied().fold(0.0, Float::max) * TUNING_REL_FLOOR;
    let mut peaks: Vec<usize> = (1..mag.len().saturating_sub(1))
        .filter(|&i| {
            (150.0..=5000.0).contains(&(i as f64 * bin_hz))
                && mag[i] > floor
                && mag[i] > mag[i - 1]
                && mag[i] > mag[i + 1]
        })
        .collect();
    // Strongest first, the lower bin on a tie: a total order, so selecting the
    // top 10 before sorting them gives the same peaks in the same order.
    let strongest = |a: &usize, b: &usize| mag[*b].total_cmp(&mag[*a]).then(a.cmp(b));
    if peaks.len() > TUNING_PEAKS {
        peaks.select_nth_unstable_by(TUNING_PEAKS - 1, strongest);
        peaks.truncate(TUNING_PEAKS);
    }
    peaks.sort_unstable_by(strongest);
    let mut sums = [0.0_f64; 3];
    for i in peaks {
        let la = (mag[i - 1] as f64).max(TINY).ln();
        let lb = (mag[i] as f64).ln();
        let lg = (mag[i + 1] as f64).max(TINY).ln();
        let den = la - 2.0 * lb + lg;
        // Negative at a strict maximum; only a peak below TINY (both
        // neighbours clamped) fails that, and it keeps its bin centre.
        let p = if den < 0.0 {
            0.5 * (la - lg) / den
        } else {
            0.0
        };
        let semitones = 12.0 * ((i as f64 + p) * bin_hz / 440.0).log2();
        let theta = std::f64::consts::TAU * (semitones - semitones.floor());
        let w = (mag[i] as f64).sqrt();
        sums[0] += w * theta.cos();
        sums[1] += w * theta.sin();
        sums[2] += w;
    }
    sums
}

// ---- HPSS percussive ratio (median-filter separation), f32 cells, f64 sums ----

// The default kernel of `librosa.decompose.hpss`, 31 frames by 31 bins at
// librosa's default resolution (22050 Hz, n_fft 2048, hop 512), in seconds and
// hertz so that it spans the same time and frequency at every sample rate.
// 31 * 512 / 22050 s (0.72 s) by 31 * 22050 / 2048 Hz (334 Hz).
const HPSS_KERNEL: u64 = 31;
const HPSS_KERNEL_SR: u64 = 22050;
const HPSS_KERNEL_N_FFT: u64 = 2048;
const HPSS_KERNEL_HOP: u64 = 512;

/// The kernel in `(frames, bins)` of a pass at `sr`, converted as the FMP
/// reference code does (`libfmp.c8.hps`: rounded up, then made odd, so a
/// window has one middle element and is centred on its cell). In integers:
/// 22050 Hz gives exactly 31 by 31.
fn hpss_kernel(sr: u32, n_fft: usize, hop: usize) -> (usize, usize) {
    let (sr, n_fft, hop) = (sr as u64, n_fft as u64, hop as u64);
    let odd = |num: u64, den: u64| (num.div_ceil(den) | 1) as usize;
    (
        odd(HPSS_KERNEL * HPSS_KERNEL_HOP * sr, HPSS_KERNEL_SR * hop),
        odd(HPSS_KERNEL * HPSS_KERNEL_SR * n_fft, HPSS_KERNEL_N_FFT * sr),
    )
}

/// The in-range index that position `i` of an axis of length `n` mirrors to,
/// the edge sample repeated (`scipy.ndimage` `reflect`: d c b a | a b c d |
/// d c b a), folding as often as a short axis needs.
fn hpss_reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if (0..n).contains(&i) {
        return i as usize;
    }
    let m = i.rem_euclid(2 * n);
    (if m < n { m } else { 2 * n - 1 - m }) as usize
}

/// Replaces one `old` by `new` in the sorted window `win`, keeping it sorted.
fn hpss_window_replace(win: &mut [Float], old: Float, new: Float) {
    let from = win.partition_point(|w| w.total_cmp(&old).is_lt());
    debug_assert!(win[from].to_bits() == old.to_bits());
    let to = win.partition_point(|w| w.total_cmp(&new).is_lt());
    if to > from {
        win.copy_within(from + 1..to, from);
        win[to - 1] = new;
    } else {
        win.copy_within(to..from, to + 1);
        win[to] = new;
    }
}

/// Median of `x` over the `k` values centred on each index (`k` odd), the axis
/// mirrored at both ends: one sorted window, one value replaced per step.
fn hpss_median(x: &[Float], k: usize) -> Vec<Float> {
    let half = (k / 2) as isize;
    let at = |i: isize| x[hpss_reflect(i, x.len())];
    let mut win: Vec<Float> = (-half..=half).map(at).collect();
    win.sort_unstable_by(Float::total_cmp);
    (0..x.len() as isize)
        .map(|j| {
            let median = win[k / 2];
            hpss_window_replace(&mut win, at(j - half), at(j + half + 1));
            median
        })
        .collect()
}

/// Streaming state of `hpss_percussive_ratio`. The harmonic estimate of a
/// frame is a median over `kernel` frames centred on it, so a
/// frame waits here until its look-ahead has arrived or the track has ended;
/// memory stays at the chunk plus one kernel of columns.
struct HpssStream {
    /// Frames in the harmonic (time) window, odd.
    kernel: usize,
    n_frames: usize,
    /// `(|X|, its median over frequency)` of frames `first..first + cols.len()`:
    /// the waiting frames and the earlier ones their windows still read.
    cols: VecDeque<(Vec<Float>, Vec<Float>)>,
    first: usize,
    /// First frame not yet in the sums.
    next: usize,
    /// Per bin, the sorted time window of the last frame added to the sums.
    windows: Vec<Float>,
    /// Medians over time of the frames being added, bin-major; reused.
    harmonic: Vec<Float>,
    sum_h: f64,
    sum_p: f64,
}

impl HpssStream {
    fn new(n_bins: usize, n_frames: usize, kernel: usize) -> Self {
        Self {
            kernel,
            n_frames,
            cols: VecDeque::new(),
            first: 0,
            next: 0,
            windows: vec![0.0; n_bins * kernel],
            harmonic: Vec::new(),
            sum_h: 0.0,
            sum_p: 0.0,
        }
    }

    /// Takes the next frames' columns, in frame order, and adds every frame
    /// whose window is complete to the sums. Medians are exact and each
    /// frame's sums are added in frame order, so neither the chunking nor
    /// `parallel` changes a bit of the result.
    fn push(&mut self, chunk: impl Iterator<Item = (Vec<Float>, Vec<Float>)>, parallel: bool) {
        self.cols.extend(chunk);
        let (k, half) = (self.kernel, self.kernel / 2);
        let reach = half as isize;
        let arrived = self.first + self.cols.len();
        // At the end of the track the window mirrors at the right edge instead.
        let ready = if arrived == self.n_frames {
            arrived
        } else {
            arrived.saturating_sub(half)
        };
        let (next, count) = (self.next, ready.saturating_sub(self.next));
        if count == 0 {
            return;
        }
        let (cols, first, n_frames) = (&self.cols, self.first, self.n_frames);
        // |X| of bin `b` in the frame that time position `i` mirrors to.
        let at = |i: isize, b: usize| cols[hpss_reflect(i, n_frames) - first].0[b];

        // Harmonic estimate: per bin, slide the sorted window over the frames.
        self.harmonic.resize(self.windows.len() / k * count, 0.0);
        let slide = |(b, (win, out)): (usize, (&mut [Float], &mut [Float]))| {
            for (j, median) in out.iter_mut().enumerate() {
                let t = (next + j) as isize;
                if t == 0 {
                    for (i, w) in win.iter_mut().enumerate() {
                        *w = at(i as isize - reach, b);
                    }
                    win.sort_unstable_by(Float::total_cmp);
                } else {
                    hpss_window_replace(win, at(t - 1 - reach, b), at(t + reach, b));
                }
                *median = win[half];
            }
        };
        if parallel {
            self.windows
                .par_chunks_mut(k)
                .zip(self.harmonic.par_chunks_mut(count))
                .enumerate()
                .for_each(slide);
        } else {
            self.windows
                .chunks_mut(k)
                .zip(self.harmonic.chunks_mut(count))
                .enumerate()
                .for_each(slide);
        }

        // Soft mask (power 2, margin 1) and one frame's Σh², Σp² over its bins.
        let harmonic = &self.harmonic;
        let frame_sums = |j: usize| {
            let (mag, percussive) = &cols[next + j - first];
            let (mut sum_h, mut sum_p) = (0.0_f64, 0.0_f64);
            for (b, (&s, &pm)) in mag.iter().zip(percussive).enumerate() {
                let hm = harmonic[b * count + j];
                let (wh, wp) = (hm * hm, pm * pm);
                let tot = wh + wp;
                let (h, p) = if tot > 0.0 {
                    (s * wh / tot, s * wp / tot)
                } else {
                    (0.5 * s, 0.5 * s)
                };
                sum_h += h as f64 * h as f64;
                sum_p += p as f64 * p as f64;
            }
            (sum_h, sum_p)
        };
        let sums: Vec<(f64, f64)> = if parallel {
            (0..count).into_par_iter().map(frame_sums).collect()
        } else {
            (0..count).map(frame_sums).collect()
        };
        for (h, p) in sums {
            self.sum_h += h;
            self.sum_p += p;
        }

        self.next = ready;
        // The next slide still reads frame `next - 1 - half`; mirrored
        // positions never reach further back.
        while self.first + half + 1 < self.next {
            self.cols.pop_front();
            self.first += 1;
        }
    }

    /// Σp² / (Σh² + Σp²); 0 when the track holds no energy.
    fn ratio(&self) -> Float {
        let total = self.sum_h + self.sum_p;
        if total > 0.0 {
            (self.sum_p / total) as Float
        } else {
            0.0
        }
    }
}

// ---- Rhythm levels (onset autocorrelation at fractions of the beat period), math in f64 ----

// Autocorrelation frame and its step, seconds (Peeters 2011).
const RHYTHM_LEVELS_WINDOW_SEC: f64 = 8.0;
const RHYTHM_LEVELS_HOP_SEC: f64 = 0.5;
// Lags in beat periods: Peeters 2011 grid (factors 2 and 3/2 of `librosa.feature.tempogram_ratio`).
const RHYTHM_LEVELS_DOUBLE_LAG: f64 = 1.0 / 2.0;
const RHYTHM_LEVELS_ONE_AND_HALF_LAG: f64 = 2.0 / 3.0;

/// `(double_ratio, one_and_half_ratio, confidence)` of the onset envelope
/// `oenv` (one value per hop) at the tempo `bpm`: the normalized
/// autocorrelation of the envelope at 1/2 and at 2/3 of the beat period, each
/// relative to the one at the beat period, and that last one as the
/// confidence. All three are means over frames of 8 s every 0.5 s. The ratios
/// are `None` and the confidence 0 without a tempo, a full frame, a frame
/// with any variation, or a beat period shorter than the frame; the ratios
/// are also `None` when the envelope does not repeat at the beat period.
fn rhythm_levels(
    oenv: &[Float],
    bpm: Float,
    sr: u32,
    hop_length: usize,
) -> (Option<Float>, Option<Float>, Float) {
    let fr = sr as f64 / hop_length as f64;
    let window = (RHYTHM_LEVELS_WINDOW_SEC * fr).round() as usize;
    // At least one frame per step: a sample rate below the hop must not stall.
    let hop = ((RHYTHM_LEVELS_HOP_SEC * fr).round() as usize).max(1);
    let period = fr * 60.0 / bpm as f64;
    // The longest lag read is floor(period) + 1; the other two levels are shorter.
    if !bpm.is_finite()
        || bpm <= 0.0
        || oenv.len() < window
        || period.floor() + 1.0 >= window as f64
    {
        return (None, None, 0.0);
    }
    let lags = [
        period,
        period * RHYTHM_LEVELS_DOUBLE_LAG,
        period * RHYTHM_LEVELS_ONE_AND_HALF_LAG,
    ];
    let mut sums = [0.0_f64; 3];
    let mut frames = 0_usize;
    let mut x = vec![0.0_f64; window];
    for frame in oenv.windows(window).step_by(hop) {
        // The frame's mean is removed first (Tzanetakis & Cook 2002).
        let mean = frame.iter().map(|&v| v as f64).sum::<f64>() / window as f64;
        for (d, &v) in x.iter_mut().zip(frame) {
            *d = v as f64 - mean;
        }
        // Biased autocorrelation at an integer lag, rectangular window.
        let acf = |lag: usize| -> f64 { x.iter().zip(&x[lag..]).map(|(a, b)| a * b).sum() };
        let acf0 = acf(0);
        if acf0 <= 0.0 {
            continue;
        }
        frames += 1;
        for (sum, &lag) in sums.iter_mut().zip(&lags) {
            let i = lag.floor() as usize;
            let g = lag - lag.floor();
            // Linear interpolation between integer lags; negative
            // autocorrelation is set to zero (Tolonen & Karjalainen 2000).
            *sum += (((1.0 - g) * acf(i) + g * acf(i + 1)) / acf0).max(0.0);
        }
    }
    let confidence = if frames > 0 {
        sums[0] / frames as f64
    } else {
        0.0
    };
    let ratio = |sum: f64| (sums[0] > 0.0).then(|| (sum / sums[0]) as Float);
    (ratio(sums[1]), ratio(sums[2]), confidence as Float)
}

// ---- Stereo field (Side share and L/R correlation on the file's channels), math in f64 ----

// Width: Tardieu et al., DAFx 2011, eq. (3), sigma(L-R)/sigma(L+R), in the
// bounded form E_S / (E_M + E_S).
// Edge of the low band, Hz: project choice, no published norm. So is its
// filter, a Butterworth low-pass of order 4.
const STEREO_LOW_CUTOFF_HZ: f64 = 150.0;
// Correlation frame, seconds, no overlap, mean over frames: Ziemer 2023 (arXiv:2302.01090).
const STEREO_FRAME_SEC: f64 = 2048.0 / 22050.0;
// Silence threshold, a variance of -70 dB, and "no variance reads 0":
// Essentia FalseStereoDetector / RTW 1260C.
const STEREO_SILENCE_POWER: f64 = 1e-7;
// Low percentile of the frame correlations: project choice, no published norm.
const STEREO_CORRELATION_LOW_QUANTILE: f64 = 0.05;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct StereoField {
    side_ratio: Option<Float>,
    side_ratio_low: Option<Float>,
    correlation_mean: Option<Float>,
    correlation_p05: Option<Float>,
}

/// Running sums over a pair of channels.
#[derive(Clone, Copy, Default)]
struct PairSums {
    n: f64,
    l: f64,
    r: f64,
    ll: f64,
    rr: f64,
    lr: f64,
}

impl PairSums {
    #[inline]
    fn add(&mut self, l: f64, r: f64) {
        self.n += 1.0;
        self.l += l;
        self.r += r;
        self.ll += l * l;
        self.rr += r * r;
        self.lr += l * r;
    }

    /// `(1 - 2 cov(L, R) / (var L + var R)) / 2`, the Side share of the energy
    /// with each channel's mean removed; `None` without samples or when the
    /// mean variance is below the silence threshold.
    fn side_ratio(&self) -> Option<Float> {
        let (mean_l, mean_r) = (self.l / self.n, self.r / self.n);
        let var = (self.ll / self.n - mean_l * mean_l) + (self.rr / self.n - mean_r * mean_r);
        let cov = self.lr / self.n - mean_l * mean_r;
        (var / 2.0 >= STEREO_SILENCE_POWER)
            .then(|| (0.5 * (1.0 - 2.0 * cov / var)).clamp(0.0, 1.0) as Float)
    }

    /// Pearson correlation of the frame; 0 when a channel has no variance,
    /// `None` when the variance of both channels is below the silence
    /// threshold (a constant offset is not signal).
    fn correlation(&self) -> Option<Float> {
        // `n^2` times the variances.
        let var_l = self.n * self.ll - self.l * self.l;
        let var_r = self.n * self.rr - self.r * self.r;
        let floor = STEREO_SILENCE_POWER * self.n * self.n;
        if var_l < floor && var_r < floor {
            return None;
        }
        // A constant channel keeps a rounding residue of about 1e-12 of its
        // power here, so "no variance" is tested against the power, not zero.
        let flat = |var: f64, power: f64| var <= 1e-9 * self.n * power;
        Some(if flat(var_l, self.ll) || flat(var_r, self.rr) {
            0.0
        } else {
            ((self.n * self.lr - self.l * self.r) / (var_l * var_r).sqrt()).clamp(-1.0, 1.0)
                as Float
        })
    }
}

/// Stereo field of a file's channels (rows) at its sample rate `sr`: the Side
/// share of the energy, whole and below 150 Hz, and the mean and 5th
/// percentile of the per-frame correlation of the unfiltered channels. One
/// channel reads as `L = R`; everything is `None` for no or over two channels.
fn stereo_field(channels: ArrayView2<Float>, sr: u32) -> StereoField {
    let (left, right) = match channels.nrows() {
        1 => (channels.row(0), channels.row(0)),
        2 => (channels.row(0), channels.row(1)),
        _ => return StereoField::default(),
    };
    let fs = sr as f64;
    // Each channel's mean comes off before the low-pass: left in, a constant
    // offset would reach the filter as a step and read as bass.
    let mean =
        |x: ndarray::ArrayView1<Float>| x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
    let (mean_l, mean_r) = (mean(left), mean(right));
    // Low-pass biquads `[b0, b1, b2, a1, a2]`: bilinear transform with
    // prewarping, as `scipy.signal.butter(4, 150, fs=fs)`.
    let low_pass = (STEREO_LOW_CUTOFF_HZ < fs / 2.0).then(|| {
        use std::f64::consts::PI;
        let k = (PI * STEREO_LOW_CUTOFF_HZ / fs).tan();
        // The order-4 poles sit at pi/8 and 3 pi/8 off the real axis.
        [PI / 8.0, 3.0 * PI / 8.0].map(|angle| {
            let q = 1.0 / (2.0 * angle.cos());
            let a0 = 1.0 + k / q + k * k;
            [
                k * k / a0,
                2.0 * k * k / a0,
                k * k / a0,
                2.0 * (k * k - 1.0) / a0,
                (1.0 - k / q + k * k) / a0,
            ]
        })
    });
    // Transposed direct form II.
    let step = |c: &[f64; 5], x: f64, st: &mut [f64; 2]| {
        let y = c[0] * x + st[0];
        st[0] = c[1] * x - c[3] * y + st[1];
        st[1] = c[2] * x - c[4] * y;
        y
    };
    let frame_len = (STEREO_FRAME_SEC * fs).round();
    let (mut whole, mut low, mut frame) = (
        PairSums::default(),
        PairSums::default(),
        PairSums::default(),
    );
    // Filter state per section, left then right.
    let mut state = [[[0.0_f64; 2]; 2]; 2];
    let mut correlations: Vec<Float> = Vec::new();
    for (&l, &r) in left.iter().zip(right.iter()) {
        let (l, r) = (l as f64, r as f64);
        whole.add(l, r);
        frame.add(l, r);
        // A trailing partial frame is never closed, so it is dropped.
        if frame.n == frame_len {
            correlations.extend(frame.correlation());
            frame = PairSums::default();
        }
        if let Some(sections) = &low_pass {
            let (mut low_l, mut low_r) = (l - mean_l, r - mean_r);
            for (c, st) in sections.iter().zip(state.iter_mut()) {
                low_l = step(c, low_l, &mut st[0]);
                low_r = step(c, low_r, &mut st[1]);
            }
            low.add(low_l, low_r);
        }
    }
    let any = !correlations.is_empty();
    StereoField {
        side_ratio: whole.side_ratio(),
        // Without the filter `low` holds no samples and reads `None`.
        side_ratio_low: low.side_ratio(),
        correlation_mean: any.then(|| {
            let sum: f64 = correlations.iter().map(|&c| c as f64).sum();
            (sum / correlations.len() as f64) as Float
        }),
        correlation_p05: any.then(|| {
            sorted_quantile(
                &sorted_series(&correlations),
                STEREO_CORRELATION_LOW_QUANTILE,
            )
        }),
    }
}

#[inline]
pub(crate) fn validated_zero_crossing_rate(y: ndarray::ArrayView1<Float>) -> Result<Float> {
    if y.is_empty() {
        return Err(SonaraError::InvalidAudio(
            "signal must contain at least one sample".into(),
        ));
    }
    let mut non_finite = false;
    let mut crossings = 0_usize;
    if let Some(samples) = y.as_slice() {
        for pair in samples.windows(2) {
            non_finite |= !pair[0].is_finite();
            crossings += usize::from((pair[0] > 0.0) != (pair[1] > 0.0));
        }
        non_finite |= !samples[samples.len() - 1].is_finite();
    } else {
        let mut samples = y.iter().copied();
        let mut previous = samples
            .next()
            .expect("public signal validation rejects empty input");
        non_finite |= !previous.is_finite();
        for sample in samples {
            non_finite |= !sample.is_finite();
            crossings += usize::from((previous > 0.0) != (sample > 0.0));
            previous = sample;
        }
    }
    if non_finite {
        let index = y
            .iter()
            .position(|sample| !sample.is_finite())
            .expect("non-finite reduction must identify a sample");
        return Err(SonaraError::InvalidAudio(format!(
            "signal sample at index {index} is not finite"
        )));
    }
    Ok(crossings as Float / y.len() as Float)
}

// ============================================================
// Analysis mode & feature selection
// ============================================================

/// Analysis depth — controls which features are computed.
///
/// Use `AnalysisMode::Compact` for fast scanning, `Playlist` for music discovery
/// and playlist generation, or `Full` for comprehensive analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    /// Core signal features only: tempo, beats, onsets, RMS, centroid, ZCR,
    /// dynamic range. (~0.6ms per 10s track)
    Compact,
    /// All features for playlist generation: adds spectral bandwidth/rolloff/
    /// flatness/contrast, MFCCs, chroma, plus perceptual features (energy,
    /// danceability, key, valence, acousticness). (~3ms per 10s track)
    Playlist,
    /// Every feature: Playlist plus the expensive rhythm analysis
    /// (tempo_curve, time_signature via metrogram) and all opt-in features.
    /// The file-only ones (ebur128, native_dynamics, amplitude_envelope,
    /// stereo_field, tags) need `analyze_file` / `analyze_batch`.
    Full,
}

impl AnalysisMode {
    /// Parse mode from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "compact" => Some(Self::Compact),
            "playlist" => Some(Self::Playlist),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    /// Canonical lowercase name (the inverse of [`AnalysisMode::from_str`]).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Playlist => "playlist",
            Self::Full => "full",
        }
    }
}

impl Default for AnalysisMode {
    fn default() -> Self {
        Self::Compact
    }
}

/// How a feature's value could be reproduced without re-running a full
/// analysis — the coarse dependency classification behind
/// [`feature_dependencies`].
///
/// This classifies *evidence requirements*, not pass routing: the registry's
/// routing flags (extended pass, opt-in, full-only) are an orthogonal concern
/// and unaffected by this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyClass {
    /// Requires the decoded audio signal itself (sample-domain computation:
    /// zero crossings, true-peak oversampling, fingerprinting, container
    /// metadata). Never recomputable from a cached [`TrackAnalysis`].
    Audio,
    /// Derived from per-frame intermediate curves (STFT frames, per-frame
    /// RMS/chroma/onset envelope, …) that analysis aggregates and then
    /// **drops** — the persisted record keeps only summaries. NOT decode-free:
    /// reproducing the value means re-decoding and re-running the frame pass.
    FrameCurves,
    /// Decode-free: recomputable purely from the cached [`TrackAnalysis`]
    /// fields listed in [`FeatureDependency::required_evidence`].
    Scalars,
    /// The similarity embedding: assembled purely from other [`TrackAnalysis`]
    /// fields (see `required_evidence`), decode-free when those are present.
    Embedding,
}

// Shorthand aliases keeping the registry table readable.
use DependencyClass::Audio as A;
use DependencyClass::Embedding as E;
use DependencyClass::FrameCurves as C;
use DependencyClass::Scalars as S;

/// [`TrackAnalysis`] fields read by [`crate::similarity::embed`] when
/// assembling the similarity vector (absent optional fields fall back to
/// documented neutral values, but meaningful similarity needs all of them).
const EMBEDDING_EVIDENCE: &[&str] = &[
    "mfcc_mean",
    "chroma_mean",
    "spectral_contrast_mean",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    "bpm",
    "onset_density",
    "danceability",
    "beats",
    "loudness_lufs",
    "dynamic_range_db",
    "dissonance",
    "chord_change_rate",
    "key",
    "energy",
    "valence",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FeatureSpec {
    name: &'static str,
    needs_extended: bool,
    opt_in_only: bool,
    full_only: bool,
    /// Evidence classification for decode-free recomputation — see
    /// [`DependencyClass`]. Orthogonal to the routing booleans above.
    class: DependencyClass,
    /// [`TrackAnalysis`] field names a decode-free recompute reads. Non-empty
    /// exactly for the `Scalars`/`Embedding` classes.
    required_evidence: &'static [&'static str],
}

const fn feature(
    name: &'static str,
    needs_extended: bool,
    opt_in_only: bool,
    full_only: bool,
    class: DependencyClass,
    required_evidence: &'static [&'static str],
) -> FeatureSpec {
    FeatureSpec {
        name,
        needs_extended,
        opt_in_only,
        full_only,
        class,
        required_evidence,
    }
}

/// Canonical public feature registry. Validation, mode routing, provenance,
/// and Python configuration all resolve names through this table.
const FEATURE_REGISTRY: &[FeatureSpec] = &[
    feature("bpm", false, false, false, C, &[]),
    feature("beats", false, false, false, C, &[]),
    feature("onsets", false, false, false, C, &[]),
    feature("rms", false, false, false, C, &[]),
    feature("dynamic_range", false, false, false, C, &[]),
    feature("centroid", false, false, false, C, &[]),
    feature("zcr", false, false, false, A, &[]),
    feature(
        "onset_density",
        false,
        false,
        false,
        S,
        &["onset_frames", "duration_sec"],
    ),
    feature("bandwidth", true, false, false, C, &[]),
    feature("rolloff", true, false, false, C, &[]),
    feature("flatness", true, false, false, C, &[]),
    feature("contrast", true, false, false, C, &[]),
    feature("mfcc", true, false, false, C, &[]),
    feature("chroma", true, false, false, C, &[]),
    feature("spectral_shape", true, false, false, C, &[]),
    feature("spectral_texture", true, false, false, C, &[]),
    // Reads the log-mel matrix every pass builds, so it needs no extended pass.
    feature("mel_bands", false, false, false, C, &[]),
    // Reads the magnitude column every pass computes, so no extended pass.
    feature("tuning", false, false, false, C, &[]),
    feature("chords", true, false, false, C, &[]),
    feature("dissonance", true, false, false, C, &[]),
    feature(
        "energy",
        true,
        false,
        false,
        S,
        &[
            "rms_mean",
            "spectral_centroid_mean",
            "onset_density",
            "spectral_bandwidth_mean",
        ],
    ),
    feature(
        "danceability",
        true,
        false,
        false,
        S,
        &["bpm", "beats", "onset_density"],
    ),
    feature("key", true, false, false, S, &["chroma_mean"]),
    feature(
        "valence",
        true,
        false,
        false,
        S,
        &["chroma_mean", "bpm", "spectral_centroid_mean"],
    ),
    feature(
        "acousticness",
        true,
        false,
        false,
        S,
        &[
            "spectral_flatness_mean",
            "spectral_rolloff_mean",
            "spectral_centroid_mean",
            "onset_density",
        ],
    ),
    // Reads the beats' sub-frame positions, which a record does not keep.
    feature("tempo_curve", true, false, true, C, &[]),
    feature("time_signature", true, false, true, C, &[]),
    feature("onset_bands", false, true, false, C, &[]),
    feature("beatgrid", false, true, false, C, &[]),
    feature("rhythmic_regularity", false, true, false, C, &[]),
    // Reads the magnitude column every pass computes, so no extended pass.
    feature("hpss", false, true, false, C, &[]),
    // Reads the onset envelope and the tempo every pass computes.
    feature("rhythm_levels", false, false, false, C, &[]),
    feature("structure", true, true, false, C, &[]),
    feature("embedding", true, true, false, E, EMBEDDING_EVIDENCE),
    #[cfg(feature = "aggression")]
    feature("aggression", true, true, false, A, &[]),
    feature("fingerprint", false, true, false, A, &[]),
    feature("loudness", false, true, false, A, &[]),
    // File-only (analyze_file / analyze_batch): measured on the file's own
    // channels and sample rate, see `crate::native_dynamics`.
    feature("ebur128", false, true, false, A, &[]),
    feature("native_dynamics", false, true, false, A, &[]),
    feature("amplitude_envelope", false, true, false, A, &[]),
    feature("stereo_field", false, true, false, A, &[]),
    feature("silence", false, true, false, C, &[]),
    feature("key_candidates", true, true, false, S, &["chroma_mean"]),
    feature(
        "vocalness",
        true,
        true,
        false,
        S,
        &[
            "spectral_contrast_mean",
            "spectral_flatness_mean",
            "rms_mean",
        ],
    ),
    feature(
        "mood",
        true,
        true,
        false,
        S,
        &[
            "chroma_mean",
            "bpm",
            "rms_mean",
            "spectral_centroid_mean",
            "onset_density",
            "spectral_bandwidth_mean",
            "beats",
            "dissonance",
            "dynamic_range_db",
        ],
    ),
    feature(
        "instrumentalness",
        true,
        true,
        false,
        S,
        &[
            "spectral_contrast_mean",
            "spectral_flatness_mean",
            "rms_mean",
        ],
    ),
    feature("tags", false, true, false, A, &[]),
];

fn feature_spec(name: &str) -> Option<&'static FeatureSpec> {
    FEATURE_REGISTRY
        .iter()
        .find(|feature| feature.name.eq_ignore_ascii_case(name))
}

/// Resolve a public feature name to its canonical lowercase spelling.
pub fn canonical_feature_name(name: &str) -> Option<&'static str> {
    feature_spec(name).map(|feature| feature.name)
}

/// Iterate over every supported public feature name in canonical order.
pub fn analysis_feature_names() -> impl Iterator<Item = &'static str> {
    FEATURE_REGISTRY.iter().map(|feature| feature.name)
}

/// One row of the declared feature-dependency map — see
/// [`feature_dependencies`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureDependency {
    /// Canonical feature name (as accepted by `AnalysisConfig::features`).
    pub name: &'static str,
    /// Evidence classification — see [`DependencyClass`].
    pub class: DependencyClass,
    /// [`TrackAnalysis`] field names a decode-free recomputation of this
    /// feature reads. Non-empty exactly when `class` is
    /// [`DependencyClass::Scalars`] or [`DependencyClass::Embedding`]; empty
    /// for `Audio`/`FrameCurves`, whose recomputation needs the audio itself.
    pub required_evidence: &'static [&'static str],
    /// Routing: computing this feature fresh requires the extended analysis
    /// pass (beyond compact). Orthogonal to the evidence classification —
    /// a consumer planning a *fresh* run reads this; a consumer planning a
    /// decode-free recompute reads `class`/`required_evidence`.
    pub needs_extended: bool,
    /// Routing: left out of the `Compact` and `Playlist` defaults
    /// (performance-first policy); computed by `Full` mode or an explicit
    /// `features=[...]` request.
    pub opt_in_only: bool,
    /// Routing: among mode-driven defaults, computed by `Full` mode only
    /// (skipped by `Playlist`).
    pub full_only: bool,
}

/// The declared feature-dependency map: for every public feature, its
/// [`DependencyClass`] and the [`TrackAnalysis`] evidence fields a
/// decode-free recompute would read. Iterates in canonical registry order
/// (the same order as [`analysis_feature_names`]).
///
/// Consumers persisting analysis results can plan cache freshness on this
/// map: a `Scalars`/`Embedding` feature is recomputable from a stored record
/// that carries every field in
/// [`required_evidence`](FeatureDependency::required_evidence) (at a matching
/// [`AnalysisProvenance::schema_version`]), while `Audio`/`FrameCurves`
/// features always need the audio again.
///
/// Caveats consumers should know:
/// - [`AnalysisProvenance`] is implicitly required evidence for every class:
///   `sample_rate`/`hop_length` define the frame→seconds mapping for
///   frame-index evidence such as `beats`, and `schema_version` gates whether
///   stored fields still mean what the recompute expects.
/// - `vocalness`/`instrumentalness` list the built-in heuristic's inputs.
///   When an `AnalysisConfig::vocalness_model` is set, the value instead
///   comes from the model over the similarity embedding — freshness then
///   additionally keys on [`AnalysisProvenance::vocalness_model_id`] and the
///   `embedding` feature's evidence.
/// - The routing flags (`needs_extended`, `opt_in_only`, `full_only`) are
///   pass-routing concerns for planning a *fresh* run; they say nothing about
///   evidence requirements, which are `class`/`required_evidence`'s domain.
pub fn feature_dependencies() -> impl Iterator<Item = FeatureDependency> {
    FEATURE_REGISTRY.iter().map(FeatureDependency::from_spec)
}

/// Look up a single feature's dependency-map row by (case-insensitive) name.
/// `None` for unknown names. See [`feature_dependencies`].
pub fn feature_dependency(name: &str) -> Option<FeatureDependency> {
    feature_spec(name).map(FeatureDependency::from_spec)
}

impl FeatureDependency {
    fn from_spec(feature: &'static FeatureSpec) -> Self {
        FeatureDependency {
            name: feature.name,
            class: feature.class,
            required_evidence: feature.required_evidence,
            needs_extended: feature.needs_extended,
            opt_in_only: feature.opt_in_only,
            full_only: feature.full_only,
        }
    }
}

/// Configuration for a single analysis run.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Analysis depth — which feature groups to compute.
    pub mode: AnalysisMode,
    /// Optional: override which features to include, regardless of mode.
    /// When `Some`, only the listed feature groups are returned (plus the core
    /// signal scalars every run reports); groups computed internally for a
    /// requested feature, such as the spectral summaries behind `energy`, are not.
    /// One exception: `valence` also returns the key it is derived from
    /// (`key`, `key_confidence`, `key_camelot`), and so does augmenting it.
    /// Valid feature names (case-insensitive; `FEATURE_REGISTRY` is the
    /// authoritative list):
    ///
    /// **Core signal:**
    /// `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`
    ///
    /// **Spectral:**
    /// `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`,
    /// `spectral_shape`, `spectral_texture`, `mel_bands`
    ///
    /// **Tonal:**
    /// `chords`, `dissonance`, `tuning`
    ///
    /// **Perceptual:**
    /// `energy`, `danceability`, `key`, `valence`, `acousticness`
    ///
    /// **Rhythm levels (Playlist and Full):**
    /// `rhythm_levels`
    ///
    /// **Rhythm analysis (Full mode or explicit request):**
    /// `tempo_curve`, `time_signature`
    ///
    /// **Opt-in (Full mode or explicit request):**
    /// `beatgrid` (grid offset, downbeats, grid stability),
    /// `rhythmic_regularity` (how straight the pattern sits in the metric grid),
    /// `onset_bands` (onset strength per frequency band),
    /// `hpss` (percussive share of the energy),
    /// `structure` (energy curve, segments, intro/outro, energy level),
    /// `embedding` (similarity vector; auto-pulls the features it is built from),
    /// `aggression` (bundled model score; available with the `aggression` Cargo feature),
    /// `fingerprint` (duplicate-detection fingerprint),
    /// `loudness` (true peak, ReplayGain, loudness curve, momentary max, LRA),
    /// `silence` (leading/trailing silence offsets),
    /// `key_candidates` (top-3 keys), `vocalness` and `instrumentalness`
    /// (vocal presence and its inverse), `mood` (four mood affinities)
    ///
    /// **Opt-in, files only (`analyze_file` / `analyze_batch`):**
    /// `ebur128` (EBU R128 loudness of the file's own channels),
    /// `native_dynamics` (sample peak and crest factors),
    /// `amplitude_envelope`, `stereo_field` (Side share and L/R correlation),
    /// `tags` (file metadata)
    ///
    /// Note: `duration` is always included. Some features depend on others
    /// (e.g., `key` requires `chroma`, `valence` requires `key`); dependencies
    /// are resolved automatically.
    pub features: Option<HashSet<String>>,
    /// Optional lower bound for octave-folding tempo normalization.
    ///
    /// When both `bpm_min` and `bpm_max` are set, BPM values outside the range
    /// are doubled or halved by octaves until they fit the requested range.
    pub bpm_min: Option<Float>,
    /// Optional upper bound for octave-folding tempo normalization.
    pub bpm_max: Option<Float>,
    /// Optional user-supplied genre classifier over the similarity embedding
    /// (bring-your-own-model; sonara ships none). When `Some`, analysis computes
    /// the embedding, runs the model, and populates `genre` + `genre_confidence`.
    /// The model's `embedding_version` must match
    /// [`crate::similarity::SIMILARITY_VERSION`], else analysis fails fast with a
    /// [`SonaraError::ModelError`]. `Arc` so `Clone` (used per-file in batch)
    /// stays cheap. See [`crate::genre`].
    pub genre_model: Option<std::sync::Arc<crate::genre::GenreModel>>,
    /// Optional user-supplied vocal-presence classifier over the similarity
    /// embedding. When `Some`, analysis computes the embedding, runs the model,
    /// and **overrides** `vocalness` + `instrumentalness` with the calibrated
    /// P(vocal) (the built-in heuristic is not consulted). The model's required
    /// `id` is surfaced as [`AnalysisProvenance::vocalness_model_id`] and its
    /// `embedding_version` must match [`crate::similarity::SIMILARITY_VERSION`],
    /// else analysis fails fast with a [`SonaraError::ModelError`]. See
    /// [`crate::vocal_model`].
    pub vocalness_model: Option<std::sync::Arc<crate::vocal_model::VocalnessModel>>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::Compact,
            features: None,
            bpm_min: None,
            bpm_max: None,
            genre_model: None,
            vocalness_model: None,
        }
    }
}

impl AnalysisConfig {
    /// Check the settings that apply to every input, without reading audio:
    /// feature names, the tempo range and the embedding layout of supplied
    /// models. Every analysis entry point runs it first; call it once before
    /// [`analyze_batch`] to get one error instead of one per file.
    pub fn validate(&self) -> Result<()> {
        self.validate_features()?;
        // Only the range check is wanted here; the tempo is a placeholder.
        crate::beat::align_tempo_to_bpm_range(120.0, self.bpm_min, self.bpm_max)?;
        // Classifying on a mismatched embedding is silently wrong, so refuse
        // rather than produce a bogus label.
        if let Some(ref model) = self.genre_model {
            if model.embedding_version != crate::similarity::SIMILARITY_VERSION {
                return Err(SonaraError::ModelError(format!(
                    "genre model embedding_version {} does not match this build's embedding version {}; \
                     re-export the model against the current embedding",
                    model.embedding_version,
                    crate::similarity::SIMILARITY_VERSION
                )));
            }
        }
        if let Some(ref model) = self.vocalness_model {
            if model.embedding_version() != crate::similarity::SIMILARITY_VERSION {
                return Err(SonaraError::ModelError(format!(
                    "vocalness model embedding_version {} does not match this build's embedding version {}; \
                     re-export the model against the current embedding",
                    model.embedding_version(),
                    crate::similarity::SIMILARITY_VERSION
                )));
            }
        }
        Ok(())
    }

    fn validate_features(&self) -> Result<()> {
        let Some(features) = self.features.as_ref() else {
            return Ok(());
        };
        let mut invalid: Vec<&str> = features
            .iter()
            .map(String::as_str)
            .filter(|name| canonical_feature_name(name).is_none())
            .collect();
        if invalid.is_empty() {
            return Ok(());
        }
        invalid.sort_unstable();
        Err(SonaraError::InvalidParameter {
            param: "features",
            reason: format!(
                "unknown feature(s): {}; valid features: {}",
                invalid.join(", "),
                analysis_feature_names().collect::<Vec<_>>().join(", ")
            ),
        })
    }

    fn feature_requested(&self, name: &str) -> bool {
        self.features.as_ref().is_some_and(|features| {
            features
                .iter()
                .any(|feature| feature.eq_ignore_ascii_case(name))
        })
    }

    fn requested_feature_names(&self) -> Option<Vec<String>> {
        self.features.as_ref().map(|features| {
            let mut names: Vec<String> = features
                .iter()
                .filter_map(|name| canonical_feature_name(name))
                .map(str::to_owned)
                .collect();
            names.sort_unstable();
            names.dedup();
            names
        })
    }

    /// Check if a feature should be computed.
    fn wants(&self, name: &str) -> bool {
        let spec = feature_spec(name).expect("internal feature name must be registered");
        // An internal embedding consumer needs the tonal/perceptual inputs even
        // when those fields were not requested for public emission.
        if self.has_internal_embedding_consumer() && EMBEDDING_DEPS.contains(&name) {
            return true;
        }
        #[cfg(feature = "aggression")]
        if self.needs_aggression() && AGGRESSION_DEPS.contains(&name) {
            return true;
        }
        // Rhythmic regularity reads the banded onset envelopes and the bar
        // phase, so it computes them even when the caller did not ask for
        // those groups — they are then scrubbed from the result (mirrors how
        // an embedding model computes its inputs without leaking them).
        if self.needs_rhythmic_regularity() && RHYTHMIC_REGULARITY_DEPS.contains(&name) {
            return true;
        }
        // mood_aggressive weighs dissonance; computing it whenever mood is wanted
        // keeps the score independent of whether dissonance was also requested.
        if name == "dissonance" && self.wants("mood") {
            return true;
        }
        if self.features.is_some() {
            self.emits(name)
        } else {
            // Opt-in features stay out of Compact and Playlist (performance-
            // first policy); Full computes every feature.
            if spec.opt_in_only {
                return self.mode == AnalysisMode::Full;
            }
            // Mode-based defaults
            match self.mode {
                AnalysisMode::Compact => false,
                // `tempo_curve` and `time_signature` are Full-only.
                AnalysisMode::Playlist => !spec.full_only,
                AnalysisMode::Full => true,
            }
        }
    }

    /// Check whether a computed feature belongs in the public result. Internal
    /// model dependencies are deliberately excluded so they cannot leak merely
    /// because a classifier needed the embedding.
    fn emits(&self, name: &str) -> bool {
        let spec = feature_spec(name).expect("internal feature name must be registered");
        if self.features.is_some() {
            return self.feature_requested(name)
                || (self.feature_requested("embedding") && EMBEDDING_COMPONENTS.contains(&name));
        }
        if spec.opt_in_only {
            return self.mode == AnalysisMode::Full;
        }
        match self.mode {
            AnalysisMode::Compact => false,
            AnalysisMode::Playlist => !spec.full_only,
            AnalysisMode::Full => true,
        }
    }

    /// True if any embedding-consuming model (genre or vocalness) is set.
    fn has_embedding_model(&self) -> bool {
        self.genre_model.is_some() || self.vocalness_model.is_some()
    }

    fn has_internal_embedding_consumer(&self) -> bool {
        self.has_embedding_model()
    }

    #[cfg(feature = "aggression")]
    fn needs_aggression(&self) -> bool {
        self.emits("aggression")
    }

    /// True when `rhythmic_regularity` is computed: an explicit request or
    /// Full mode (it is opt-in, so no other mode turns it on).
    fn needs_rhythmic_regularity(&self) -> bool {
        self.emits("rhythmic_regularity")
    }

    /// True if the similarity embedding must be computed: either the feature
    /// is wanted (`features=["embedding"]` or Full mode) or an embedding model
    /// is set (genre/vocalness classify over the embedding). The
    /// `embedding`/`embedding_version` fields are emitted only in the first
    /// case — a model computes the vector without leaking it (mirrors how mood
    /// computes key silently).
    fn needs_embedding(&self) -> bool {
        self.wants("embedding") || self.has_internal_embedding_consumer()
    }

    /// Check if extended features (anything beyond compact) are needed.
    fn needs_extended(&self) -> bool {
        // An embedding model needs the embedding, which requires the extended
        // pass (mfcc/chroma/contrast/spectral scalars).
        if self.has_internal_embedding_consumer() {
            return true;
        }
        #[cfg(feature = "aggression")]
        if self.needs_aggression() {
            return true;
        }
        if let Some(ref features) = self.features {
            features.iter().any(|name| {
                feature_spec(name)
                    .map(|feature| feature.needs_extended)
                    .unwrap_or(false)
            })
        } else {
            self.mode != AnalysisMode::Compact
        }
    }
}

// --- similarity ---
/// Feature names the similarity embedding is assembled from. Requesting
/// "embedding" implies each of these (see `AnalysisConfig::wants`). The spectral
/// timbre features (mfcc/chroma/contrast/bandwidth/rolloff/flatness) are computed
/// automatically whenever extended analysis runs, so only the wants-gated tonal
/// and perceptual features need to be listed here.
const EMBEDDING_DEPS: &[&str] = &[
    "energy",
    "danceability",
    "key",
    "valence",
    "dissonance",
    "chords",
];

/// Optional result groups physically represented in the similarity vector.
/// An explicit embedding request continues to surface these for backward
/// compatibility; internal model consumers compute them without emission.
const EMBEDDING_COMPONENTS: &[&str] = &[
    "bandwidth",
    "rolloff",
    "flatness",
    "contrast",
    "mfcc",
    "chroma",
    "energy",
    "danceability",
    "key",
    "valence",
    "dissonance",
    "chords",
];

#[cfg(feature = "aggression")]
const AGGRESSION_DEPS: &[&str] = &["energy", "danceability", "dissonance"];

/// Feature groups `rhythmic_regularity` is measured from: the per-band onset
/// envelopes supply the accents, the beat grid supplies the bar phase and the
/// grid-quality term of its confidence. Requesting it implies computing these
/// (see `AnalysisConfig::wants`), but not emitting them — they are scrubbed
/// afterwards unless separately requested.
const RHYTHMIC_REGULARITY_DEPS: &[&str] = &["onset_bands", "beatgrid"];

/// Version of the `TrackAnalysis` result schema. Bump whenever the meaning,
/// unit, or time base of an existing field changes (e.g. a different
/// `HOP_LENGTH`, a re-derived curve, renamed/retyped fields) — additions of
/// new fields do NOT require a bump. Consumers persisting analysis results
/// compare this against `AnalysisProvenance::schema_version` to detect stale
/// records.
///
/// v2 (2026-07-17): chroma filterbank gained librosa-parity octave-domain
/// Gaussian weighting, changing all chroma-derived fields (chroma, key,
/// tonal features) at every sample rate — most visibly at sr > 22050.
///
/// v3 (2026-07-17): `vocalness`/`instrumentalness` re-derived from mid-band
/// spectral contrast (heuristic v2) instead of the mel-based v1 heuristic —
/// different semantics and values, and they now require the extended pass.
/// Also recalibrated the absolute scales of `acousticness` (added a brightness
/// term; electronic < 0.3, acoustic > 0.6 on real music) and `danceability`
/// (monotonic logistic remap spreading the output across `[0, 1]`) — values
/// change, orderings are largely preserved. Consumers with 0.2.2/0.2.3-era
/// cutoffs on these fields must re-derive.
///
/// v4 (2026-07-20): tied `predominant_chord` counts now resolve to the
/// lexicographically smallest label instead of depending on randomized map
/// iteration. Persisted analyses must be refreshed because an old tied result
/// may contain any of the equally frequent chord labels.
///
/// v5 (2026-07-22): `aggression_score` is now a fused physical/perceptual rank
/// with independent support and diagnostics, not the legacy similarity-vector
/// probability. Stored aggression values require an audio rescan.
///
/// v6 (2026-07-24): bundled aggression values are evaluated at the model's
/// canonical 22.05 kHz sample rate. Main-pass fields and frame provenance stay
/// in the caller-requested sample-rate domain.
///
/// v7 (2026-10-02): `bpm` and `bpm_raw` are the period of the tracked beats
/// (a hierarchical consensus of beat pairs) instead of the fractional ACF
/// peak, which remains the fallback; a steady track whose beats fit the
/// nearest integer tempo reports that integer. The beats are tracked on the
/// onset envelope of the mel bins below 3.2 kHz; the tempo estimate and
/// `bpm_candidates` still come from the broadband envelope. The tempo level is
/// re-checked against up to five other ACF levels by tracking each again at
/// its own tempo, whatever the caller's range. Same meaning and
/// units; stored `bpm`, `bpm_raw` and `beats`, and what derives from them,
/// differ.
///
/// v8 (2026-11-26): the chord HPCP weights a peak just below C like a peak
/// just below any other note, so stored `chord_sequence`, `chord_events`,
/// `chord_change_rate` and `predominant_chord` differ, and with the change
/// rate dimension 42 of `embedding`. `downbeats` and `rhythmic_regularity*`
/// always use a 4-beat bar, so a record computed together with
/// `time_signature` differs. `loudness_lufs` no longer reads long signals as
/// quieter (the mean square is summed in f64): stored values are low by a few
/// hundredths of a dB on a track and by about a dB on an hour-long mix, and
/// `replaygain_db` and dimension 39 of `embedding` follow. `spectral_flatness_*`
/// and `spectral_contrast_*` cover only the sounding frames, so on a track
/// with silence, a fade or a quiet tail they differ, and with them
/// `acousticness`, the heuristic `vocalness`/`instrumentalness` and dimensions
/// 25-30 and 34 of `embedding`. `tempo_curve`, `tempo_variability` and
/// `grid_stability` are measured between the beats' sub-frame positions
/// instead of whole frames, so all three differ, and with the grid stability
/// `rhythmic_regularity_confidence`; `tempo_curve` is no longer recomputed
/// from a record without the audio. Same meaning and units.
pub const ANALYSIS_SCHEMA_VERSION: u32 = 8;

/// STFT hop length (samples) used by the main analysis pass. All frame-index
/// fields on [`TrackAnalysis`] (`beats`, `onset_frames`, `downbeats`) convert
/// to time as `frame * HOP_LENGTH / sample_rate`.
pub(crate) const HOP_LENGTH: usize = 512;

/// Number of spectral contrast bands.
const N_CONTRAST_BANDS: usize = 6;
/// Number of MFCC coefficients.
const N_MFCC: usize = 13;
/// Dynamic range kept in the log-mel matrix below its loudest cell, in dB.
const MEL_TOP_DB: Float = 80.0;
/// Number of HPCP harmonics.
const N_HPCP_HARMONICS: usize = 4;
/// Max spectral peaks for HPCP/dissonance.
const MAX_PEAKS: usize = 50;

/// Cached mel filterbank, sparse chroma, DCT matrix, and analysis constants.
struct AnalysisCache {
    key: (u32, usize, usize), // (sr, n_fft, n_mels)
    sparse_mel: Vec<(usize, Vec<Float>)>,
    sparse_chroma: Vec<(usize, Vec<Float>)>,
    freqs: Array1<Float>,
    win_padded: Array1<Float>,
    /// Pre-computed DCT-II matrix (n_mfcc, n_mels) for fast MFCC.
    dct_matrix: Array2<Float>,
    /// Spectral contrast band boundaries (bin indices).
    contrast_bands: Vec<(usize, usize)>,
    /// HPCP harmonic weights: 1/(h+1).
    harmonic_weights: [Float; N_HPCP_HARMONICS],
}

thread_local! {
    static ANALYSIS_CACHE: RefCell<Option<AnalysisCache>> = const { RefCell::new(None) };
}

#[cfg(feature = "aggression")]
thread_local! {
    // A canonical aggression pass commonly alternates between the caller's
    // rate and 22.05 kHz. Keep the established primary-cache hot path and one
    // feature-gated secondary entry so neither table is rebuilt per track.
    static SECONDARY_ANALYSIS_CACHE: RefCell<Option<AnalysisCache>> = const { RefCell::new(None) };
}

#[cfg(feature = "bench-internals")]
thread_local! {
    // Armed by `analyze_file_with_rhythm_envelopes`: the fused pass on this
    // thread leaves its rhythm envelopes here for the bench runner.
    static RHYTHM_CAPTURE: RefCell<Option<Option<crate::onset::RhythmEnvelopes>>> =
        const { RefCell::new(None) };
}

#[cfg(feature = "bench-internals")]
fn capture_rhythm_envelopes(envelopes: &crate::onset::RhythmEnvelopes) {
    RHYTHM_CAPTURE.with(|slot| {
        if let Some(captured) = slot.borrow_mut().as_mut() {
            *captured = Some(envelopes.clone());
        }
    });
}

/// Provenance metadata: how an analysis result was produced.
///
/// Makes a persisted [`TrackAnalysis`] self-describing — a consumer can
/// convert frame indices to seconds (`frame * hop_length / sample_rate`) and
/// detect stale stored records by comparing `schema_version` against
/// [`ANALYSIS_SCHEMA_VERSION`] without out-of-band knowledge.
///
/// `sample_rate`/`hop_length` describe the main analysis pass only; the
/// `fingerprint` field lives in its own fixed internal sample-rate space and
/// carries its own version (see [`crate::fingerprint`]), as does `embedding`
/// (`embedding_version`).
// `Eq` was dropped (0.3.x) when the float-typed `bpm_min`/`bpm_max` fields
// were added — floats are only `PartialEq`. No consumer relied on `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisProvenance {
    /// Value of [`ANALYSIS_SCHEMA_VERSION`] at analysis time.
    pub schema_version: u32,
    /// Effective sample rate (Hz) the analyzed signal was at — after any
    /// resampling by `analyze_file`. Frame indices are valid against this
    /// rate only.
    pub sample_rate: u32,
    /// STFT hop length (samples) of the main pass.
    pub hop_length: usize,
    /// The configured [`AnalysisMode`]. Ignored by feature selection when
    /// `requested_features` is `Some` (an explicit list overrides the mode).
    pub mode: AnalysisMode,
    /// The explicit `features=[...]` request (sorted), if one was given.
    pub requested_features: Option<Vec<String>>,
    /// The octave-folding lower tempo bound (`AnalysisConfig::bpm_min`) in
    /// effect at analysis time; `None` when the config left it unset. Recorded
    /// because the configured range changes the reported `bpm` (out-of-range
    /// raw tempos are folded by octaves into it) — without it, two results for
    /// the same audio can silently diverge with no visible cause. Cache
    /// consumers should treat a change in the configured range as invalidating
    /// stored `bpm` (though `bpm_raw` stays comparable).
    pub bpm_min: Option<Float>,
    /// The octave-folding upper tempo bound (`AnalysisConfig::bpm_max`) in
    /// effect at analysis time; see [`bpm_min`](Self::bpm_min).
    pub bpm_max: Option<Float>,
    /// Identity of the genre model that produced `genre`/`genre_confidence`,
    /// when a model carrying an `id` was supplied. `None` when no genre model
    /// ran (or the model has no `id`). Cache consumers should treat a change
    /// in this field as invalidating stored `genre` fields.
    pub genre_model_id: Option<String>,
    /// Identity of the vocalness model that produced
    /// `vocalness`/`instrumentalness`. `None` means the built-in heuristic
    /// (schema-versioned) produced them. Cache consumers should treat a change
    /// in this field as invalidating stored vocalness/instrumentalness scores.
    pub vocalness_model_id: Option<String>,
    /// Identity of the bundled aggression model that produced
    /// [`TrackAnalysis::aggression_score`]. Present exactly when the score was
    /// requested. Cache consumers should invalidate the score when it changes.
    #[cfg(feature = "aggression")]
    pub aggression_model_id: Option<String>,
}

pub use crate::core::audio::TrackTags;
pub use crate::structure::SegmentEvent;

/// A chord with its time span, in seconds.
///
/// Derived from the same beat-aligned windows as `chord_sequence`, with runs
/// of consecutive identical labels merged into one event. Events are
/// contiguous and cover the track: the first starts at 0.0 and the last ends
/// at `duration_sec`. `label` uses the `chord_sequence` vocabulary (e.g.
/// "C", "Am"; "N" = no chord detected in that span).
#[derive(Debug, Clone, PartialEq)]
pub struct ChordEvent {
    pub label: String,
    pub start_sec: Float,
    pub end_sec: Float,
}

/// Complete analysis result for a single track.
///
/// Core fields are always populated. Extended/perceptual fields are `Some`
/// only when the selected mode or feature list includes them.
///
/// `Clone` is part of the public API: [`augment_analysis`] returns a patched
/// clone of a cached record, and consumers caching analyses may clone freely.
#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub struct TrackAnalysis {
    // -- Basic (always computed) --
    /// How this result was produced (schema version, effective sample rate,
    /// hop length, mode/features) — see [`AnalysisProvenance`].
    pub provenance: AnalysisProvenance,
    pub duration_sec: Float,
    pub bpm: Float,
    /// Selected tempo before optional `bpm_min`/`bpm_max` range alignment.
    pub bpm_raw: Float,
    /// How firmly the tempo estimate is anchored in the audio ([0,1]): combines
    /// the strength of the dominant autocorrelation tempo peak, agreement
    /// between that tempo and the tracked beat rate, and rhythmic onset density.
    /// High (>0.7) on steady percussive music; low (<0.45) on ambient, rubato,
    /// or sparse-onset material where the reported BPM should be treated with
    /// suspicion. A trust signal, not a probability.
    pub bpm_confidence: Float,
    /// Strongest tempo candidates as `(bpm, score)` pairs, sorted by score
    /// descending (up to the top 5).
    pub bpm_candidates: Vec<(Float, Float)>,
    /// Beat positions as main-pass frame indices; seconds =
    /// `frame * provenance.hop_length / provenance.sample_rate`
    /// (or use [`TrackAnalysis::beats_sec`]).
    pub beats: Vec<usize>,
    /// Onset positions as main-pass frame indices (see `beats` for the
    /// frame→seconds convention, or [`TrackAnalysis::onsets_sec`]).
    pub onset_frames: Vec<usize>,
    pub rms_mean: Float,
    pub rms_max: Float,
    pub loudness_lufs: Float,
    pub dynamic_range_db: Float,

    // --- loudness ---
    // Extended loudness / gain metrics (opt-in via `features=["loudness"]`).
    // `Some` only when the "loudness" group is computed (Full mode or a request).
    /// True peak in dBTP (4x oversampled, ITU-R BS.1770-4 Annex 2). ~0 dBTP is
    /// full scale; > 0 dBTP means inter-sample overs that can clip on playback.
    pub true_peak_db: Option<Float>,
    /// ReplayGain-style track gain in dB to reach the -18 LUFS reference:
    /// `-18 - loudness_lufs`.
    pub replaygain_db: Option<Float>,
    /// Short-term loudness curve: one LUFS value per 3 s window at a 1 s hop
    /// (ITU-R BS.1770 short-term integration). Empty for tracks under one window.
    pub loudness_curve: Option<Vec<Float>>,
    /// Maximum momentary (400 ms window) loudness, dB (EBU R128 momentary).
    pub loudness_momentary_max_db: Option<Float>,
    /// EBU R128 loudness range (LRA) in LU: gated 95th-10th percentile spread of
    /// the short-term loudness distribution. The standardized counterpart to the
    /// approximate `dynamic_range_db` (which is a raw p95-p5 of RMS).
    pub loudness_range_lu: Option<Float>,
    // --- end loudness ---
    // --- ebur128 / native_dynamics / amplitude_envelope ---
    // File-only, opt-in via `features=[...]`: measured by `analyze_file` /
    // `analyze_batch` on the file's own channels at its own sample rate
    // (see `crate::native_dynamics`). `None` when the feature was not
    // requested, when the analysis ran through `analyze_signal`, or when the
    // value is undefined (digital silence, file shorter than the window).
    // Unlike `loudness_lufs` / `true_peak_db` (mono mix at the analysis rate,
    // ungated), these follow the standards on the original channels.
    /// `ebur128`: integrated (programme) loudness in LUFS, ITU-R BS.1770-5 /
    /// EBU Tech 3341 with absolute (-70 LUFS) and relative (-10 LU) gating.
    /// Channels in WAV order (L R C LFE Ls Rs, LFE excluded); the loudness
    /// fields are `None` past six channels, where the layout is unknown.
    pub ebur128_integrated_lufs: Option<Float>,
    /// `ebur128`: maximum momentary (400 ms) loudness in LUFS, EBU Tech 3341.
    pub ebur128_momentary_max_lufs: Option<Float>,
    /// `ebur128`: maximum short-term (3 s) loudness in LUFS, EBU Tech 3341.
    pub ebur128_short_term_max_lufs: Option<Float>,
    /// `ebur128`: loudness range (LRA) in LU, EBU Tech 3342.
    pub ebur128_range_lu: Option<Float>,
    /// `ebur128`: true peak in dBTP (4x oversampled, BS.1770-5 Annex 2), the
    /// maximum over the file's channels. Accurate to the EBU Tech 3341
    /// tolerance (+0.2 / -0.4 dB): it can read up to 0.4 dB low.
    pub ebur128_true_peak_dbtp: Option<Float>,
    /// Sample rate of the file in Hz, the rate the file-only features are
    /// measured at. `Some` whenever any of `ebur128`, `native_dynamics`,
    /// `amplitude_envelope` or `stereo_field` was measured from a file, so it
    /// tells "measured, value undefined" (a level is `None`) from "never
    /// measured" (a signal).
    pub native_sample_rate: Option<u32>,
    /// `native_dynamics`: sample peak in dBFS, max |x| over the file's channels
    /// (not the mono mix).
    pub sample_peak_dbfs: Option<Float>,
    /// `native_dynamics`: crest factor in dB, track peak over the RMS of all
    /// samples of all channels.
    pub crest_factor_db: Option<Float>,
    /// `native_dynamics`: mean of the per-frame crest factor in dB over frames
    /// whose RMS is above -70 dBFS (frames of `amplitude_envelope`).
    pub crest_factor_frame_mean_db: Option<Float>,
    /// `amplitude_envelope`: peak envelope, max |x| over the file's channels in
    /// each frame. The grid is in seconds, not samples: frame 2048/22050 s, hop
    /// 512/22050 s, frame `k` centred at `k * amplitude_envelope_hop_sec`.
    pub amplitude_envelope: Option<Vec<Float>>,
    /// `amplitude_envelope`: hop between envelope frames in seconds.
    pub amplitude_envelope_hop_sec: Option<Float>,
    /// `amplitude_envelope`: envelope frame length in seconds.
    pub amplitude_envelope_frame_sec: Option<Float>,
    // --- stereo_field ---
    // File-only and opt-in like the group above, measured on the left and
    // right channels of the file. A mono file reads as L = R (shares 0,
    // correlation +1). `None` when not requested, for a signal, for a file
    // with more than two channels (unknown layout), and where noted below.
    /// `stereo_field`: share of the Side signal in the energy of the whole
    /// file, `E_S / (E_M + E_S)` with `M = (L + R) / 2`, `S = (L - R) / 2` and
    /// each channel's mean removed, in `[0, 1]`: 0 for identical channels, 0.5
    /// for unrelated ones, 1 for an inverted channel (Tardieu et al. 2011,
    /// eq. 3, in bounded form). `None` when the mean variance of the channels
    /// is below -70 dB.
    pub stereo_side_ratio: Option<Float>,
    /// `stereo_field`: the same share below 150 Hz, after a Butterworth
    /// low-pass of order 4 on both channels (means removed first). `None`
    /// when the filtered channels are below -70 dB or the file's Nyquist
    /// frequency is at most 150 Hz.
    pub stereo_side_ratio_low: Option<Float>,
    /// `stereo_field`: mean of the Pearson correlation of the left and right
    /// channels over frames of 2048/22050 s without overlap (Ziemer 2023), in
    /// `[-1, 1]`. A frame with the variance of both channels below -70 dB is
    /// left out; a frame with a channel of no variance counts as 0. `None`
    /// when no frame is left.
    pub stereo_correlation_mean: Option<Float>,
    /// `stereo_field`: 5th percentile of the same frame correlations (linear
    /// interpolation, as the `_p10` fields), `None` under the same condition.
    pub stereo_correlation_p05: Option<Float>,
    pub spectral_centroid_mean: Float,
    pub zero_crossing_rate: Float,
    pub onset_density: Float,

    // --- onset bands ---
    /// Frequency boundaries in Hz for `onset_strength_bands`. Opt-in via
    /// `features=["onset_bands"]` or Full mode; `None` in Compact and Playlist.
    pub onset_band_edges_hz: Option<Vec<Float>>,
    /// Raw mean positive log-mel flux per frequency band. Each inner vector is
    /// frame-aligned through `provenance.hop_length` and `sample_rate`.
    pub onset_strength_bands: Option<Vec<Vec<Float>>>,

    // -- Extended (extended or full) --
    pub spectral_bandwidth_mean: Option<Float>,
    pub spectral_rolloff_mean: Option<Float>,
    /// Mean over the sounding frames only: a frame counts when its RMS is
    /// above zero and within 60 dB of the track's loudest frame. 0.0 when no
    /// frame counts.
    pub spectral_flatness_mean: Option<Float>,
    /// 7 values, over the same sounding frames as `spectral_flatness_mean`.
    pub spectral_contrast_mean: Option<Vec<Float>>,
    pub mfcc_mean: Option<Vec<Float>>,
    pub chroma_mean: Option<Vec<Float>>,

    // --- frame statistics ---
    // Spread, frame-to-frame change and quantiles of the per-frame values behind
    // the `_mean` fields, over the same frames as that `_mean` (all main-pass
    // frames, silence included; for flatness and contrast the sounding frames
    // only). Each is computed with its base feature: `Some` exactly when
    // that feature is (`rms`, `centroid`, `zcr` in every mode; the rest with
    // their playlist/full feature). `_std` is the population standard
    // deviation; `_delta_mean` the mean absolute difference between consecutive
    // frames, in the base unit per hop; quantiles interpolate linearly between
    // order statistics and `_iqr` is p75 - p25. Vectors are per coefficient /
    // pitch class / contrast band, in the order of their `_mean`.
    pub rms_std: Option<Float>,
    pub rms_delta_mean: Option<Float>,
    pub rms_iqr: Option<Float>,
    pub rms_p10: Option<Float>,
    pub rms_p90: Option<Float>,
    /// Standard deviation of the per-frame zero-crossing rate (2048-sample
    /// frames on the main hop). `zero_crossing_rate` itself is whole-signal.
    pub zero_crossing_rate_std: Option<Float>,
    pub spectral_centroid_std: Option<Float>,
    pub spectral_centroid_delta_mean: Option<Float>,
    pub spectral_centroid_iqr: Option<Float>,
    pub spectral_bandwidth_std: Option<Float>,
    pub spectral_bandwidth_delta_mean: Option<Float>,
    pub spectral_rolloff_std: Option<Float>,
    pub spectral_rolloff_p10: Option<Float>,
    pub spectral_rolloff_p90: Option<Float>,
    pub spectral_flatness_std: Option<Float>,
    pub spectral_flatness_iqr: Option<Float>,
    pub spectral_flatness_p10: Option<Float>,
    pub spectral_flatness_p90: Option<Float>,
    /// 7 values, aligned with `spectral_contrast_mean`.
    pub spectral_contrast_std: Option<Vec<Float>>,
    /// 13 values each, aligned with `mfcc_mean`.
    pub mfcc_std: Option<Vec<Float>>,
    pub mfcc_delta_mean: Option<Vec<Float>>,
    /// 12 values each, aligned with `chroma_mean` (per-frame L-inf normalized chroma).
    pub chroma_std: Option<Vec<Float>>,
    pub chroma_delta_mean: Option<Vec<Float>>,
    // --- end frame statistics ---

    // --- spectral shape ---
    // Track means of per-frame spectral-shape descriptors (MIR_FEATURES group
    // 3.1), computed with `spectral_shape` (playlist/full, or
    // `features=["spectral_shape"]`). Each is the mean over only the frames
    // where that descriptor is defined: all-zero frames (and, for the moments,
    // zero-bandwidth frames) are skipped, by a rule of their own; with
    // no defined frame the value is 0.0. Frames are the main pass's
    // (|X_k| magnitudes, f_k = k·sr/n_fft over every bin, DC included).
    /// Dimensionless. Magnitude-weighted third standardized moment about the
    /// spectral centroid c: Σ|X_k|(f_k−c)³ / Σ|X_k| / bw³, with bw the
    /// per-frame spectral bandwidth.
    pub spectral_skewness_mean: Option<Float>,
    /// Dimensionless. Magnitude-weighted excess kurtosis:
    /// Σ|X_k|(f_k−c)⁴ / Σ|X_k| / bw⁴ − 3.
    pub spectral_kurtosis_excess_mean: Option<Float>,
    /// dB per octave. Least-squares slope of the octave-band power level
    /// 10·log10(mean |X_k|²) against log2 of the band centre, over 8 octave
    /// bands centred at 62.5·2^b Hz (b = 0..7, edges centre/√2..centre·√2,
    /// clipped at Nyquist; empty bands skipped, at least 2 needed). Pink
    /// noise reads ≈ −3 dB/oct, white noise ≈ 0; mixed music with bass about
    /// −3 to −7. Every band counts, so near-empty low octaves (a tonal sound
    /// whose fundamental is above ~90 Hz, no bass) read their noise floor and
    /// window leakage and push the slope up, to positive values.
    pub spectral_slope_db_per_octave_mean: Option<Float>,
    /// Dimensionless. Spectral decrease (Peeters, Timbre Toolbox) on the
    /// magnitude spectrum with the DC bin as reference:
    /// Σ_{k≥1} (|X_k| − |X_0|)/k / Σ_{k≥1} |X_k|. The DC bin is ≈0 for
    /// zero-mean music, so in practice this measures 1/k-weighted
    /// low-frequency emphasis. At most 1, usually positive; negative when the
    /// DC bin is strong (a DC offset in the signal, or the lowest bass leaking
    /// into it through the window).
    pub spectral_decrease_mean: Option<Float>,
    // --- end spectral shape ---

    // --- spectral texture ---
    // Per-frame spectral-texture descriptors (MIR_FEATURES group 3.2),
    // computed with `spectral_texture` (playlist/full, or
    // `features=["spectral_texture"]`) on the main pass's magnitudes |X_k|
    // (every bin, DC and Nyquist included). Each averages only the frames
    // where it is defined (silent frames skipped; 0.0 when none is).
    /// Mean of the per-frame spectral flux: the half-wave rectified L2
    /// distance sqrt(Σ max(0, p_t[k] − p_{t−1}[k])²) between consecutive
    /// L1-normalized magnitude spectra p = |X| / Σ|X| (Essentia
    /// Flux(halfRectify, L2) on a sum-normalized spectrum). Range 0..1,
    /// loudness-invariant; defined for frame pairs where neither is silent.
    pub spectral_flux_mean: Option<Float>,
    /// Population standard deviation (ddof 0) over the same flux values.
    pub spectral_flux_std: Option<Float>,
    /// Spectral crest on magnitude: max|X_k| / mean|X_k|, ≥ 1.
    pub spectral_crest_mean: Option<Float>,
    /// Spectral entropy of p = |X| / Σ|X|: −Σ p_k ln p_k / ln(n_bins), 0..1
    /// (MIRtoolbox mirentropy normalization).
    pub spectral_entropy_mean: Option<Float>,
    /// Jensen irregularity Σ(a_{i+1} − a_i)² / Σa_i² over the fused pass's
    /// spectral peaks (local maxima in 40–5000 Hz, parabolic interpolation,
    /// top 50 by magnitude) sorted by frequency; depends on that peak
    /// picking. Defined for frames with ≥ 2 peaks.
    pub spectral_irregularity_mean: Option<Float>,
    // --- end spectral texture ---

    // --- mel bands ---
    // Track summary of the main pass's log-mel spectrogram (MIR_FEATURES group
    // 4.1), computed with `mel_bands` (playlist/full, or
    // `features=["mel_bands"]`). It is the matrix the MFCCs and the onset
    // envelopes are taken from: 128 Slaney mel bands over 0..sr/2
    // (area-normalized triangles on |X_k|²), 10·log10 with a 1e-10 power
    // floor, then floored at the track's loudest cell − 80 dB. Statistics run
    // over every main-pass frame, silence included.
    /// dB, 128 values, low band first (centres `mel_frequencies(130, 0,
    /// sr/2)[1..129]`). Per-band mean of the dB values, not the dB of the mean
    /// power. `mfcc_mean` is the first 13 coefficients of this vector's
    /// orthonormal DCT-II.
    pub mel_bands_mean: Option<Vec<Float>>,
    /// dB, 128 values. Per-band population standard deviation (ddof 0) over
    /// the same frames. A gain change shifts every mean and the floor by the
    /// same dB and leaves this unchanged (while the floor is above −100 dB).
    pub mel_bands_std: Option<Vec<Float>>,
    /// dB. The lowest value a cell can take: the loudest cell − 80 dB, or
    /// −100 dB (the 1e-10 power floor) when that is higher, as on silence. A
    /// band whose mean is within about 1 dB of it holds nothing above the
    /// floor; exact equality is rare, edge frames lift the mean slightly.
    pub mel_bands_floor_db: Option<Float>,
    // --- end mel bands ---

    // --- tuning ---
    // Tuning deviation of the recording from A4 = 440 Hz, computed with
    // `tuning` (playlist/full, or `features=["tuning"]`) on the main pass's
    // magnitudes |X_k| (Dressler & Streich, ISMIR 2007). In each frame the 10
    // strongest spectral peaks (strict local maxima with bin centre in
    // 150–5000 Hz and more than −80 dB of the frame's strongest bin,
    // frequency refined by a parabola on ln|X|) vote with their
    // deviation from the nearest equal-tempered semitone; the votes of all
    // frames are averaged on the circle, weighted by sqrt|X|. Both fields are
    // `None` when the feature was not requested.
    /// Cents, in `[-50, 50)`, positive = sharp of A4 = 440 Hz. Circular mean,
    /// weighted by sqrt magnitude, of the deviation from the nearest
    /// equal-tempered semitone of the 10 strongest spectral peaks per frame in
    /// 150–5000 Hz, over all frames (Dressler & Streich, ISMIR 2007). `None`
    /// with a present `tuning_confidence` means the measure abstained:
    /// measured, could not tell — noise, silence or unpitched material. It
    /// abstains when the confidence is below 0.08, or below 1.5/√n when only
    /// n < 352 frames (about 8 s) carry a peak, where noise alone would clear
    /// 0.08. The value wraps: a track near ±50 lies between two
    /// semitones and can read as either sign. Chroma, HPCP, key and chords
    /// still assume 440 Hz and are not corrected by it.
    pub tuning_offset_cents: Option<Float>,
    /// Dimensionless, `[0, 1]`. Mean resultant length of the same votes:
    /// 0 = no agreement, 1 = every peak on the same grid; about 0.01–0.03 on
    /// noise of 5 s or more (noise reads about 0.4/√frames, so more on
    /// shorter signals), 0.05–0.6 on mixed music, 0.99 on clean tones.
    /// Reported whenever the feature ran, also when the offset abstained.
    pub tuning_confidence: Option<Float>,
    // --- end tuning ---

    // --- hpss ---
    // Opt-in (Full mode or features=["hpss"]); `None` otherwise. Median
    // filter harmonic/percussive separation of the main pass's magnitudes |X|
    // over all bins and frames, as `librosa.decompose.hpss(S, kernel_size=
    // hpss_kernel(sr, n_fft, hop), power=2.0, margin=1.0)`:
    // the harmonic estimate of a cell is the median of |X| over that many
    // frames within its bin, the percussive estimate the median over that
    // many bins within its frame, both windows mirrored at the edges
    // (`scipy.ndimage` `reflect`). The kernel is 0.72 s by 334 Hz: librosa's
    // default 31 frames by 31 bins at 22050 Hz, scaled with the sample rate.
    /// Dimensionless, `[0, 1]`: Σp² / (Σh² + Σp²) over every cell, the share
    /// of the spectrogram's energy in the percussive component, where
    /// h = |X|·H²/(H² + P²) and p = |X|·P²/(H² + P²) for the harmonic and
    /// percussive estimates H and P (a cell with H = P = 0 is split evenly).
    /// About 0 on sustained tones, 1 on isolated clicks. `0.0` on digital
    /// silence: there is no abstention.
    pub hpss_percussive_ratio: Option<Float>,
    // --- end hpss ---

    // -- Rhythm (extended or full) --
    /// Local BPM per inter-beat interval (median-smoothed): value `i` covers
    /// the span between `beats[i]` and `beats[i+1]`
    /// (length = `beats.len() - 1`). The intervals are measured between the
    /// beats' sub-frame positions, not between the whole frames in `beats`,
    /// so the curve is not `60 / (beats[i+1] - beats[i])`.
    pub tempo_curve: Option<Vec<Float>>,
    /// Coefficient of variation (std / mean) of `tempo_curve`. A sequenced
    /// track with one tempo still reads about 0.01: the beat positions carry
    /// an error of a fraction of a frame.
    pub tempo_variability: Option<Float>,
    pub time_signature: Option<String>,
    pub time_signature_confidence: Option<Float>,

    // -- Tonal (extended or full) --
    /// One chord label per beat-aligned window (`tonal::chord_boundaries`);
    /// "N" = no chord. For explicit time spans use `chord_events`.
    pub chord_sequence: Option<Vec<String>>,
    /// Time-spanned chord events (merged runs of `chord_sequence`); `Some`
    /// exactly when `chord_sequence` is. See [`ChordEvent`].
    pub chord_events: Option<Vec<ChordEvent>>,
    pub chord_change_rate: Option<Float>,
    pub predominant_chord: Option<String>,
    pub dissonance: Option<Float>,

    // -- Perceptual (extended or full) --
    pub energy: Option<Float>,
    pub danceability: Option<Float>,
    pub key: Option<String>,
    pub key_confidence: Option<Float>,
    /// Camelot wheel code for the detected key (e.g. "8A" for A minor), for DJ harmonic mixing.
    pub key_camelot: Option<String>,
    pub valence: Option<Float>,
    pub acousticness: Option<Float>,

    // -- Similarity embedding --
    /// Hand-crafted similarity vector ([`crate::similarity::EMBEDDING_DIM`]
    /// values in `[0, 1]`, not a learned embedding); see [`crate::similarity`].
    /// Opt-in via `features=["embedding"]` or Full mode.
    pub embedding: Option<Vec<Float>>,

    /// Bundled perceptual rank in `[0, 1]`, opt-in via
    /// `features=["aggression"]`. This is a rank, not a probability. `None`
    /// with a present confidence means the model abstained for insufficient
    /// musical evidence.
    #[cfg(feature = "aggression")]
    pub aggression_score: Option<Float>,
    /// Independent content/evidence support for `aggression_score`.
    #[cfg(feature = "aggression")]
    pub aggression_confidence: Option<Float>,
    #[cfg(feature = "aggression")]
    pub aggression_forcefulness: Option<Float>,
    #[cfg(feature = "aggression")]
    pub aggression_harshness: Option<Float>,
    #[cfg(feature = "aggression")]
    pub aggression_tension: Option<Float>,
    #[cfg(feature = "aggression")]
    pub aggression_rhythm: Option<Float>,

    // -- Mood + instrumentalness (heuristic v1, opt-in) --
    /// Mood affinities in `[0, 1]`, **heuristic v1 (not ML)**. Opt-in via
    /// `features=["mood"]` or Full mode; all four populate together.
    pub mood_happy: Option<Float>,
    pub mood_aggressive: Option<Float>,
    pub mood_relaxed: Option<Float>,
    pub mood_sad: Option<Float>,
    /// Inverse of the vocalness heuristic (`1 - vocalness`, clamped `[0, 1]`),
    /// **heuristic v2 (not ML)**. Opt-in via `features=["instrumentalness"]` or
    /// Full mode; requires the extended pass (shares `vocalness`'s
    /// contrast/flatness inputs). With `AnalysisConfig::vocalness_model` set it
    /// is `1 - P(vocal)` from that model instead, filled in every mode whether
    /// requested or not.
    /// Semantics changed in 0.2.4 — see [`vocalness`](Self::vocalness).
    pub instrumentalness: Option<Float>,
    /// Predicted genre label — populated only when a user-supplied genre model
    /// is set (`AnalysisConfig::genre_model`); `None` otherwise. sonara ships no
    /// model. See [`crate::genre`].
    pub genre: Option<String>,
    /// Confidence (softmax probability, `(0, 1]`) of the predicted `genre`.
    /// Populated only when a user-supplied genre model is set; `None` otherwise.
    pub genre_confidence: Option<Float>,

    // --- beat grid ---
    // Opt-in (Full mode or features=["beatgrid"]); `None` in the compact
    // and playlist modes.
    /// Time (seconds) of the first beat — the grid anchor.
    pub grid_offset_sec: Option<Float>,
    /// Frame indices of bar-starting beats (subset of `beats`).
    pub downbeats: Option<Vec<usize>>,
    /// How rigidly beats fit a constant-tempo grid, in `[0, 1]`:
    /// `1 - MAD / median` of the intervals between the beats' sub-frame
    /// positions. A sequenced track reads about 0.99, not exactly 1.
    pub grid_stability: Option<Float>,

    // --- rhythmic regularity ---
    // Opt-in (Full mode or features=["rhythmic_regularity"]). All four fields
    // populate together, and are `None` both in Compact and Playlist and
    // when the track carries no metric grid to measure against (see
    // [`crate::rhythmic_regularity::analyze`]).
    /// How straight the percussive pattern sits inside the metric grid, in
    /// `[0, 1]`: `1.0` a regular house/techno-style pattern on the pulse
    /// lattice, `0.0` a syncopated, polyrhythmic or breakbeat-driven one. This
    /// is a deterministic DSP measure, not a model output, and it is
    /// independent of `grid_stability` (which describes drift *of* the grid,
    /// not the distribution of events *within* it). See
    /// [`crate::rhythmic_regularity`].
    pub rhythmic_regularity: Option<Float>,
    /// `"regular"` when `rhythmic_regularity >= 0.5`, else `"irregular"`.
    pub rhythmic_regularity_label: Option<String>,
    /// Quality of the rhythmic evidence behind the score, in `[0, 1]` — how
    /// far the tempo, beat grid, observed bar count and accent contrast
    /// support the measurement. **Not** a class probability.
    pub rhythmic_regularity_confidence: Option<Float>,
    /// Both labels with their DSP scores, ranked by score descending. The
    /// scores are complementary measurements (`r` and `1 - r`), not model
    /// probabilities.
    pub rhythmic_regularity_candidates: Option<Vec<(String, Float)>>,

    // --- rhythm levels ---
    // Playlist, Full or features=["rhythm_levels"]; all three `None`
    // otherwise. The onset envelope the tempo is estimated from is cut into
    // frames of 8 s every 0.5 s (Peeters 2011); each frame has its mean
    // removed and is autocorrelated (biased, rectangular window), the result
    // divided by its value at lag 0, read at a fractional lag by linear
    // interpolation and set to zero where negative. With T the beat period of
    // `bpm`, the ratios divide the mean over frames at T/2 and at 2T/3 by the
    // mean at T. Both ratios are `None` with a present confidence when the
    // measure abstained: no tempo, less than 8 s, no variation in any frame,
    // or no repetition at the beat period at all.
    /// Dimensionless, `>= 0`: strength of the eighth-note level (lag T/2)
    /// relative to the beat. 0 = nothing repeats at half a beat.
    pub rhythm_level_double_ratio: Option<Float>,
    /// Dimensionless, `>= 0`: strength of the one-and-a-half level (lag 2T/3:
    /// triplets, shuffle, three against two) relative to the beat.
    pub rhythm_level_one_and_half_ratio: Option<Float>,
    /// Dimensionless, `[0, 1]`: mean normalized autocorrelation of the onset
    /// envelope at the beat period. Reported whenever the feature ran, `0.0`
    /// when nothing could be measured.
    pub rhythm_level_confidence: Option<Float>,
    // --- structure ---
    /// Time-resolved perceptual energy (0-1), one value per window.
    pub energy_curve: Option<Vec<Float>>,
    /// Seconds between successive `energy_curve` samples.
    pub energy_curve_hop_sec: Option<Float>,
    /// Contiguous structural sections covering the track. See [`SegmentEvent`].
    pub segments: Option<Vec<SegmentEvent>>,
    /// End of the initial low-energy / pre-first-drop region (seconds).
    pub intro_end_sec: Option<Float>,
    /// Start of the final fade / low-energy region (seconds).
    pub outro_start_sec: Option<Float>,
    /// Coarse 1-10 energy level derived from mean energy.
    pub energy_level: Option<u8>,
    // --- silence ---
    /// Leading silence duration in seconds — audio below the silence threshold
    /// (-60 dBFS relative to full scale) at the very start. Opt-in via
    /// `features=["silence"]` or Full mode; `None` otherwise.
    pub leading_silence_sec: Option<Float>,
    /// Trailing silence duration in seconds — audio below the silence threshold
    /// at the very end. Opt-in via `features=["silence"]` or Full mode; `None` otherwise.
    pub trailing_silence_sec: Option<Float>,

    // --- key candidates ---
    /// Top-3 ranked key candidates as `(key string, Camelot code, score)`.
    /// Opt-in via `features=["key_candidates"]` or Full mode. The first entry
    /// equals `key`.
    pub key_candidates: Option<Vec<(String, String, Float)>>,

    // --- vocalness ---
    /// Vocal-presence heuristic in `[0, 1]` (**heuristic v2**, not a classifier).
    /// Measures the prominence of vocal/broadband energy filling the ~0.8-5.6 kHz
    /// spectral valleys (low mid-band peak-to-valley contrast → high vocalness):
    /// it rises harsh > clean > instrumental (screamed metal ≈ 1.0, clean singing
    /// mid, solo sax/flute/violin low). Semantics changed in 0.2.4 (was a mel-based
    /// tonal+syllabic heuristic that inverted on distorted vocals). Known ambiguous
    /// cases: sparse voice+piano ballads read mid-low; a voice-mimicking solo violin
    /// reads borderline. Opt-in via `features=["vocalness"]` or Full mode; requires
    /// the extended pass (mid-band spectral contrast). `None` unless computed.
    /// With `AnalysisConfig::vocalness_model` set, the value is that model's
    /// calibrated P(vocal), filled in every mode whether requested or not; the
    /// heuristic is not consulted.
    pub vocalness: Option<Float>,
    // --- fingerprint ---
    /// Acoustic fingerprint (raw sub-fingerprint sequence, ~8 `u32`/sec) for
    /// duplicate detection. Opt-in: `Some` when the `"fingerprint"` feature is
    /// requested or in Full mode; `None` otherwise, and for audio too short to
    /// yield a fingerprint. See
    /// [`crate::fingerprint`]. Compare two with [`crate::fingerprint::match_score`].
    pub fingerprint: Option<Vec<u32>>,
    // --- similarity ---
    /// Version of the `embedding` layout + normalization (see
    /// `crate::similarity::SIMILARITY_VERSION`). `Some` iff `embedding` is `Some`.
    /// Present when the `"embedding"` feature is requested or in Full mode.
    pub embedding_version: Option<u32>,
    // --- tags ---
    /// Container/stream metadata tags (title, artist, album, genre, year,
    /// track number) read from the file. `Some` only when the `"tags"` feature
    /// is requested, or Full mode is used, via `analyze_file`/`analyze_batch`;
    /// always `None` for `analyze_signal` (no container) and when not
    /// requested. WAV inputs yield `None`: the WAV path reads no tags. See
    /// [`TrackTags`]. Note: `TrackTags::genre` (file metadata) is distinct from
    /// the `genre` field above, which a user-supplied genre model fills.
    pub tags: Option<TrackTags>,
}

impl TrackAnalysis {
    /// Convert a main-pass frame index (as in `beats`, `onset_frames`,
    /// `downbeats`) to seconds using the carried provenance:
    /// `frame * hop_length / sample_rate`.
    pub fn frame_to_sec(&self, frame: usize) -> Float {
        frame as Float * self.provenance.hop_length as Float / self.provenance.sample_rate as Float
    }

    /// Beat times in seconds (see [`Self::frame_to_sec`]).
    pub fn beats_sec(&self) -> Vec<Float> {
        self.beats.iter().map(|&f| self.frame_to_sec(f)).collect()
    }

    /// Onset times in seconds (see [`Self::frame_to_sec`]).
    pub fn onsets_sec(&self) -> Vec<Float> {
        self.onset_frames
            .iter()
            .map(|&f| self.frame_to_sec(f))
            .collect()
    }

    /// Downbeat times in seconds; `None` unless `features=["beatgrid"]`.
    pub fn downbeats_sec(&self) -> Option<Vec<Float>> {
        self.downbeats
            .as_ref()
            .map(|d| d.iter().map(|&f| self.frame_to_sec(f)).collect())
    }
}

/// Per-frame results from the fused FFT pass.
struct FrameResult {
    mel_col: Vec<Float>,
    centroid: Float,
    rms: Float,
    bandwidth: Float,
    rolloff: Float,
    flatness: Float,
    chroma_col: [Float; 12],
    // Fused tonal + contrast (only populated when extended)
    contrast_bands: [Float; N_CONTRAST_BANDS + 1],
    hpcp_col: [Float; 12],
    dissonance: Float,
    /// Skewness, excess kurtosis, slope, decrease; `None` where undefined.
    spectral_shape: [Option<f64>; 4],
    /// Crest, entropy, irregularity; `None` where undefined.
    spectral_texture: [Option<f64>; 3],
    /// Normalized spectrum for the flux; `None` when silent or unwanted.
    texture_norm: Option<Vec<Float>>,
    /// Tuning vote `[re, im, wsum]`; zeros when unwanted.
    tuning: [f64; 3],
    /// `(|X|, its median over frequency)` for `hpss`; `None` when unwanted.
    hpss: Option<(Vec<Float>, Vec<Float>)>,
    #[cfg(feature = "aggression")]
    aggression_crest_db: Float,
    #[cfg(feature = "aggression")]
    aggression_high_energy_ratio: Float,
    #[cfg(feature = "aggression")]
    aggression_high_flatness: Float,
    #[cfg(feature = "aggression")]
    aggression_high_total: Float,
    #[cfg(feature = "aggression")]
    aggression_peak_ratio: Float,
}

// ============================================================
// Public API
// ============================================================

/// Run one analysis and report a panic inside it as that call's error, so a
/// single input cannot abort a batch or reach a binding as a panic.
// The closure only reads its inputs, and the thread-local caches hold no
// borrow and no half-written entry after an unwind, so resuming is sound.
pub(crate) fn catch_analysis_panic<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|payload| {
        let reason = payload
            .downcast_ref::<&str>()
            .map(|message| message.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_owned());
        Err(SonaraError::Numerical(format!("analysis panicked: {reason}")))
    })
}

/// Analyze a track from a file path with the given configuration.
pub fn analyze_file(path: &Path, sr: u32, config: &AnalysisConfig) -> Result<TrackAnalysis> {
    catch_analysis_panic(|| analyze_file_unguarded(path, sr, config))
}

fn analyze_file_unguarded(path: &Path, sr: u32, config: &AnalysisConfig) -> Result<TrackAnalysis> {
    config.validate()?;
    // Tags are file-only metadata (see `TrackTags`): read them during load when
    // requested, then post-fill the result. When not requested, `load_with_tags`
    // does zero extra work — identical to the plain `load` fast path.
    let want_tags = config.wants("tags");
    // File-only loudness/dynamics/stereo need the file's own channels and rate: decode
    // once without downmixing, measure, then continue on the same mono mix
    // `load_with_tags` would have produced.
    let want_ebur128 = config.wants("ebur128");
    let want_dynamics = config.wants("native_dynamics");
    let want_envelope = config.wants("amplitude_envelope");
    let want_stereo = config.wants("stereo_field");
    if want_ebur128 || want_dynamics || want_envelope || want_stereo {
        let (channels, native_sr, tags) = audio::load_channels_with_tags(path, want_tags)?;
        let ebu = want_ebur128.then(|| crate::native_dynamics::ebur128(channels.view(), native_sr));
        let dynamics = (want_dynamics || want_envelope)
            .then(|| crate::native_dynamics::native_dynamics(channels.view(), native_sr));
        let stereo = want_stereo.then(|| stereo_field(channels.view(), native_sr));
        let native = if channels.nrows() == 1 {
            channels.row(0).to_owned()
        } else {
            audio::to_mono(channels.view())
        };
        drop(channels);
        let mut result = analyze_native_mono(native, native_sr, sr, config)?;
        result.tags = tags;
        // Marks the record as measured from a file: a missing level then
        // means "undefined", not "never measured".
        result.native_sample_rate = Some(native_sr);
        if let Some(ebu) = ebu {
            result.ebur128_integrated_lufs = ebu.integrated_lufs;
            result.ebur128_momentary_max_lufs = ebu.momentary_max_lufs;
            result.ebur128_short_term_max_lufs = ebu.short_term_max_lufs;
            result.ebur128_range_lu = ebu.range_lu;
            result.ebur128_true_peak_dbtp = ebu.true_peak_dbtp;
        }
        if let Some(dynamics) = dynamics {
            if want_dynamics {
                result.sample_peak_dbfs = dynamics.sample_peak_dbfs;
                result.crest_factor_db = dynamics.crest_factor_db;
                result.crest_factor_frame_mean_db = dynamics.crest_factor_frame_mean_db;
            }
            if want_envelope {
                result.amplitude_envelope = Some(dynamics.envelope);
                result.amplitude_envelope_hop_sec =
                    Some(crate::native_dynamics::ENVELOPE_HOP_SEC as Float);
                result.amplitude_envelope_frame_sec =
                    Some(crate::native_dynamics::ENVELOPE_FRAME_SEC as Float);
            }
        }
        if let Some(stereo) = stereo {
            result.stereo_side_ratio = stereo.side_ratio;
            result.stereo_side_ratio_low = stereo.side_ratio_low;
            result.stereo_correlation_mean = stereo.correlation_mean;
            result.stereo_correlation_p05 = stereo.correlation_p05;
        }
        return Ok(result);
    }
    // Decode at the file's own rate: `analyze_native_mono` rejects non-finite
    // samples before it resamples.
    let (native, native_sr, tags) = audio::load_with_tags(path, 0, true, 0.0, 0.0, want_tags)?;
    let mut result = analyze_native_mono(native, native_sr, sr, config)?;
    result.tags = tags;
    Ok(result)
}

/// Analyze the mono mix of a file decoded at its own rate `native_sr`; `sr` is
/// the caller's rate (0 = native). File-only fields are left to the caller.
fn analyze_native_mono(
    native: Array1<Float>,
    native_sr: u32,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    // The resampler panics on non-finite samples: reject them before it runs.
    if !native.is_empty() {
        validated_zero_crossing_rate(native.view())?;
    }
    let caller_sr = if sr == 0 { native_sr } else { sr };
    // At the model's own rate one lane serves both: the plain path below, which
    // also frees the native copy once it is resampled.
    #[cfg(feature = "aggression")]
    if config.needs_aggression() && caller_sr != crate::aggression::AGGRESSION_SAMPLE_RATE {
        // Decode once at the source rate, then derive the caller and model
        // lanes independently. This prevents a lossy caller-rate roundtrip,
        // preserves generic field semantics, and avoids a second file decode.
        let caller = (caller_sr != native_sr)
            .then(|| audio::resample(native.view(), native_sr, caller_sr))
            .transpose()?;
        let caller_view = caller
            .as_ref()
            .map(|audio| audio.view())
            .unwrap_or_else(|| native.view());
        let canonical = (native_sr != crate::aggression::AGGRESSION_SAMPLE_RATE)
            .then(|| {
                audio::resample(
                    native.view(),
                    native_sr,
                    crate::aggression::AGGRESSION_SAMPLE_RATE,
                )
            })
            .transpose()?;
        let canonical_view = canonical
            .as_ref()
            .map(|audio| audio.view())
            .unwrap_or_else(|| native.view());
        return analyze_signal_with_precomputed_aggression(
            caller_view,
            caller_sr,
            canonical_view,
            crate::aggression::AGGRESSION_SAMPLE_RATE,
            config,
        );
    }
    let y = if caller_sr != native_sr {
        let resampled = audio::resample(native.view(), native_sr, caller_sr)?;
        // The analysis reads only the resampled copy: free the native one first.
        drop(native);
        resampled
    } else {
        native
    };
    analyze_signal(y.view(), caller_sr, config)
}

/// [`analyze_file`] that also returns the rhythm envelopes of its fused pass by
/// name (`onset`, `beat`, `low`, `mid`; frame grid of the pass, hop
/// [`HOP_LENGTH`]). Bench tooling only: `None` when no fused pass ran.
#[doc(hidden)]
#[cfg(feature = "bench-internals")]
#[allow(clippy::type_complexity)]
pub fn analyze_file_with_rhythm_envelopes(
    path: &Path,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<(TrackAnalysis, Option<Vec<(&'static str, Array1<Float>)>>)> {
    RHYTHM_CAPTURE.with(|slot| *slot.borrow_mut() = Some(None));
    let result = analyze_file(path, sr, config);
    let captured = RHYTHM_CAPTURE.with(|slot| slot.borrow_mut().take()).flatten();
    result.map(|analysis| (analysis, captured.map(|envelopes| envelopes.named())))
}

/// Analyze a pre-loaded audio signal with the given configuration.
pub fn analyze_signal(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    catch_analysis_panic(|| analyze_signal_unguarded(y, sr, config))
}

fn analyze_signal_unguarded(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    config.validate()?;
    if sr == 0 {
        return Err(SonaraError::InvalidParameter {
            param: "sr",
            reason: "sample rate must be greater than zero".into(),
        });
    }
    if y.is_empty() {
        return Err(SonaraError::InvalidAudio(
            "signal must contain at least one sample".into(),
        ));
    }

    #[cfg(feature = "aggression")]
    if sr != crate::aggression::AGGRESSION_SAMPLE_RATE && config.needs_aggression() {
        return analyze_signal_with_canonical_aggression(y, sr, config);
    }

    let zero_crossing_rate = validated_zero_crossing_rate(y)?;
    let extended = config.needs_extended();
    analyze_signal_inner(y, sr, extended, zero_crossing_rate, config, FRAME_CHUNK)
}

#[cfg(feature = "aggression")]
#[inline(never)]
fn analyze_signal_with_canonical_aggression(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    // The resampler panics on non-finite samples: reject them before it runs.
    validated_zero_crossing_rate(y)?;
    // Normal API inputs are contiguous, so resample directly and allocate only
    // the canonical lane. Materialize unusual strided views only when needed by
    // the optimized 2:1 resampler.
    let source = (!y.is_standard_layout()).then(|| y.to_owned());
    let source_view = source.as_ref().map(|audio| audio.view()).unwrap_or(y);
    let canonical = audio::resample(source_view, sr, crate::aggression::AGGRESSION_SAMPLE_RATE)?;
    analyze_signal_with_precomputed_aggression(
        y,
        sr,
        canonical.view(),
        crate::aggression::AGGRESSION_SAMPLE_RATE,
        config,
    )
}

#[cfg(feature = "aggression")]
#[inline(never)]
fn analyze_signal_with_precomputed_aggression(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    canonical: ndarray::ArrayView1<Float>,
    canonical_sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    if canonical_sr != crate::aggression::AGGRESSION_SAMPLE_RATE {
        return Err(SonaraError::ModelError(format!(
            "aggression model requires {} Hz audio, got {canonical_sr} Hz",
            crate::aggression::AGGRESSION_SAMPLE_RATE
        )));
    }
    // Preserve every generic field in the caller's sample-rate domain. Merely
    // remove the opt-in aggression request so its trained DSP dependencies do
    // not force an unnecessary native-rate extended pass.
    let mut native_config = config.clone();
    match native_config.features.as_mut() {
        Some(features) => features.retain(|feature| !feature.eq_ignore_ascii_case("aggression")),
        // No list means Full mode turned aggression on: Full emits every
        // feature, so the same run without aggression is the full list minus it.
        None => {
            native_config.features = Some(
                analysis_feature_names()
                    .filter(|name| *name != "aggression")
                    .map(str::to_owned)
                    .collect(),
            )
        }
    }
    let native_zcr = validated_zero_crossing_rate(y)?;
    let mut result = analyze_signal_inner(
        y,
        sr,
        native_config.needs_extended(),
        native_zcr,
        &native_config,
        FRAME_CHUNK,
    )?;

    let aggression = analyze_canonical_aggression(canonical)?;
    result.aggression_score = aggression.score;
    result.aggression_confidence = Some(aggression.confidence);
    result.aggression_forcefulness = Some(aggression.forcefulness);
    result.aggression_harshness = Some(aggression.harshness);
    result.aggression_tension = Some(aggression.tension);
    result.aggression_rhythm = Some(aggression.rhythm);
    result.provenance.requested_features = config.requested_feature_names();
    result.provenance.aggression_model_id = Some(crate::aggression::AGGRESSION_MODEL_ID.to_owned());
    Ok(result)
}

#[cfg(feature = "aggression")]
pub(crate) fn analyze_canonical_aggression(
    canonical: ndarray::ArrayView1<Float>,
) -> Result<crate::aggression::AggressionAnalysis> {
    if canonical.is_empty() {
        return Err(SonaraError::InvalidAudio(
            "signal must contain at least one sample".into(),
        ));
    }
    validated_zero_crossing_rate(canonical)?;
    aggression_dsp::analyze_signal(canonical)
}

/// Analyze multiple files in parallel.
///
/// Failures are isolated per file: the returned vector has exactly one entry
/// per input path, in the same order as `paths`, and a decode/IO failure on one
/// file yields an `Err` for that entry only — it never aborts or poisons the
/// rest of the batch. This is the robustness contract the Python `analyze_batch`
/// binding relies on when analyzing large libraries.
pub fn analyze_batch(
    paths: &[&Path],
    sr: u32,
    config: &AnalysisConfig,
) -> Vec<Result<TrackAnalysis>> {
    analyze_batch_with(paths, sr, config, |_, _| {})
}

/// Like [`analyze_batch`], invoking `on_done(done, total)` after each file
/// completes (success or failure). `done` counts completions in completion
/// order — not input order; the returned Vec is input-ordered. The callback
/// runs on rayon worker threads: keep it cheap and non-blocking.
pub fn analyze_batch_with<F>(
    paths: &[&Path],
    sr: u32,
    config: &AnalysisConfig,
    on_done: F,
) -> Vec<Result<TrackAnalysis>>
where
    F: Fn(usize, usize) + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    let total = paths.len();
    let done = AtomicUsize::new(0);
    paths
        .par_iter()
        .map(|path| {
            let r = analyze_file(path, sr, config);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_done(n, total);
            r
        })
        .collect()
}

// ============================================================
// Core implementation
// ============================================================

fn analyze_signal_inner(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    extended: bool,
    zero_crossing_rate: Float,
    config: &AnalysisConfig,
    frame_chunk: usize,
) -> Result<TrackAnalysis> {
    #[cfg(feature = "aggression")]
    let wants_aggression = config.needs_aggression();

    let sr_f = sr as Float;
    let n_fft = 2048;
    let hop_length = HOP_LENGTH;
    let n_mels = 128;
    let n_bins = n_fft / 2 + 1;

    let duration_sec = y.len() as Float / sr_f;

    // ================================================================
    // SETUP: mel filterbank, window, padding (cached across calls)
    // ================================================================

    let cache_key = (sr, n_fft, n_mels);

    let cache_data = ANALYSIS_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(ref c) = *cache {
            if c.key == cache_key {
                return (
                    c.sparse_mel.clone(),
                    c.sparse_chroma.clone(),
                    c.freqs.clone(),
                    c.win_padded.clone(),
                    c.dct_matrix.clone(),
                    c.contrast_bands.clone(),
                    c.harmonic_weights,
                );
            }
        }
        #[cfg(feature = "aggression")]
        if let Some(data) = SECONDARY_ANALYSIS_CACHE.with(|secondary| {
            let secondary = secondary.borrow();
            secondary.as_ref().filter(|c| c.key == cache_key).map(|c| {
                (
                    c.sparse_mel.clone(),
                    c.sparse_chroma.clone(),
                    c.freqs.clone(),
                    c.win_padded.clone(),
                    c.dct_matrix.clone(),
                    c.contrast_bands.clone(),
                    c.harmonic_weights,
                )
            })
        }) {
            return data;
        }
        let mel_fb = filters::mel(sr_f, n_fft, n_mels, 0.0, sr_f / 2.0, false, "slaney");
        let sparse_mel: Vec<(usize, Vec<Float>)> = (0..n_mels)
            .map(|m| {
                let row = mel_fb.row(m);
                let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                if first > last {
                    (0, vec![])
                } else {
                    (first, row.slice(s![first..=last]).to_vec())
                }
            })
            .collect();
        let f = convert::fft_frequencies(sr_f, n_fft);
        let win = windows::get_window(&WindowSpec::Named("hann".into()), n_fft, true)
            .expect("hann window");
        let wp = utils::pad_center(win.view(), n_fft).expect("pad_center");

        // Sparse chroma filterbank
        let cfb = filters::chroma(sr_f, n_fft, 12, 0.0);
        let sparse_chroma: Vec<(usize, Vec<Float>)> = (0..12)
            .map(|c| {
                let row = cfb.row(c);
                let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                if first > last {
                    (0, vec![])
                } else {
                    (first, row.slice(s![first..=last]).to_vec())
                }
            })
            .collect();

        // Pre-computed DCT-II matrix (n_mfcc × n_mels)
        let dct_matrix = Array2::from_shape_fn((N_MFCC, n_mels), |(k, m)| {
            let norm = if k == 0 {
                (1.0 / n_mels as Float).sqrt()
            } else {
                (2.0 / n_mels as Float).sqrt()
            };
            norm * (std::f32::consts::PI * k as Float * (2 * m + 1) as Float
                / (2.0 * n_mels as Float))
                .cos()
        });

        // Spectral contrast band bin boundaries
        let fmin: Float = 200.0;
        let fmax = sr_f / 2.0;
        let freqs_slice = f.as_slice().unwrap();
        let mut band_edges = vec![fmin];
        for i in 1..=N_CONTRAST_BANDS {
            band_edges.push(fmin * (fmax / fmin).powf(i as Float / N_CONTRAST_BANDS as Float));
        }
        let contrast_bands: Vec<(usize, usize)> = (0..N_CONTRAST_BANDS)
            .map(|b| {
                let lo = band_edges[b];
                let hi = band_edges[b + 1];
                let start = freqs_slice.iter().position(|&freq| freq >= lo).unwrap_or(0);
                let end = freqs_slice
                    .iter()
                    .position(|&freq| freq >= hi)
                    .unwrap_or(n_bins);
                (start, end)
            })
            .collect();

        let harmonic_weights: [Float; N_HPCP_HARMONICS] =
            std::array::from_fn(|h| 1.0 / (h as Float + 1.0));

        let entry = AnalysisCache {
            key: cache_key,
            sparse_mel: sparse_mel.clone(),
            sparse_chroma: sparse_chroma.clone(),
            freqs: f.clone(),
            win_padded: wp.clone(),
            dct_matrix: dct_matrix.clone(),
            contrast_bands: contrast_bands.clone(),
            harmonic_weights,
        };
        #[cfg(feature = "aggression")]
        if cache.is_some() {
            SECONDARY_ANALYSIS_CACHE.with(|secondary| {
                *secondary.borrow_mut() = Some(entry);
            });
        } else {
            *cache = Some(entry);
        }
        #[cfg(not(feature = "aggression"))]
        {
            *cache = Some(entry);
        }

        (
            sparse_mel,
            sparse_chroma,
            f,
            wp,
            dct_matrix,
            contrast_bands,
            harmonic_weights,
        )
    });
    let (
        sparse_mel,
        sparse_chroma,
        freqs,
        win_padded,
        dct_matrix,
        contrast_bands,
        harmonic_weights,
    ) = cache_data;

    let pad = n_fft / 2;
    let mut y_padded = Array1::<Float>::zeros(y.len() + 2 * pad);
    y_padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
    let n = y_padded.len();
    if n < n_fft {
        return Err(SonaraError::InsufficientData {
            needed: n_fft,
            got: n,
        });
    }
    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = win_padded.as_slice().unwrap();

    // ================================================================
    // SINGLE PASS: FFT → mel + centroid + rms + (extended features)
    // Also computes chroma via proper filterbank and stores power spectrum
    // All extended features are fused into this single FFT pass.
    // ================================================================

    let n_frames = 1 + (n - n_fft) / hop_length;
    let mut mel_spec = Array2::<Float>::zeros((n_mels, n_frames));
    let mut centroids = Array1::<Float>::zeros(n_frames);
    let mut rms_frames = Array1::<Float>::zeros(n_frames);
    let mut bandwidths = if extended {
        Array1::<Float>::zeros(n_frames)
    } else {
        Array1::zeros(0)
    };
    let mut rolloffs = if extended {
        Array1::<Float>::zeros(n_frames)
    } else {
        Array1::zeros(0)
    };
    let mut flatnesses = if extended {
        Array1::<Float>::zeros(n_frames)
    } else {
        Array1::zeros(0)
    };
    let mut chroma_raw = if extended {
        Array2::<Float>::zeros((12, n_frames))
    } else {
        Array2::zeros((0, 0))
    };
    // Per-frame contrast, for `spectral_contrast_mean` and `spectral_contrast_std`.
    let mut contrast_frames = if extended {
        Array2::<Float>::zeros((N_CONTRAST_BANDS + 1, n_frames))
    } else {
        Array2::zeros((0, 0))
    };
    // Fused HPCP (accumulated per-frame, normalized+averaged post-loop)
    let mut hpcp_raw = if extended {
        Array2::<Float>::zeros((12, n_frames))
    } else {
        Array2::zeros((0, 0))
    };
    // Fused contrast + dissonance accumulators
    #[cfg(feature = "aggression")]
    let mut aggression_crest_db = wants_aggression.then(|| vec![0.0; n_frames]);
    #[cfg(feature = "aggression")]
    let mut aggression_dissonance = wants_aggression.then(|| vec![0.0; n_frames]);
    #[cfg(feature = "aggression")]
    let mut aggression_high_energy_ratio = wants_aggression.then(|| vec![0.0; n_frames]);
    #[cfg(feature = "aggression")]
    let mut aggression_high_flatness = wants_aggression.then(|| vec![0.0; n_frames]);
    #[cfg(feature = "aggression")]
    let mut aggression_high_total = wants_aggression.then(|| vec![0.0; n_frames]);
    #[cfg(feature = "aggression")]
    let mut aggression_peak_ratio = wants_aggression.then(|| vec![0.0; n_frames]);

    let wants_spectral_shape = extended && config.wants("spectral_shape");
    let shape_bands = if wants_spectral_shape {
        spectral_shape_octave_bands(sr, n_fft, n_bins)
    } else {
        Vec::new()
    };
    let mut shape_sum = [0.0_f64; 4];
    let mut shape_count = [0_usize; 4];

    let wants_spectral_texture = extended && config.wants("spectral_texture");
    // Flux, crest, entropy, irregularity: sum and count of defined frames.
    let mut texture_sum = [0.0_f64; 4];
    let mut texture_count = [0_usize; 4];
    let mut flux_sq_sum = 0.0_f64;
    // Previous frame's normalized spectrum, carried across chunks.
    let mut texture_prev: Option<Vec<Float>> = None;

    // Not gated on `extended`: it only reads the magnitude column.
    let wants_tuning = config.wants("tuning");
    // Σ re, Σ im, Σ weight of the frame votes, and how many frames voted.
    let mut tuning_sum = [0.0_f64; 3];
    let mut tuning_frames = 0_usize;

    // Opt-in and not gated on `extended`: it reads the magnitude column.
    let wants_hpss = config.wants("hpss");
    let (hpss_frames, hpss_bins) = hpss_kernel(sr, n_fft, hop_length);
    let mut hpss_stream = wants_hpss.then(|| HpssStream::new(n_bins, n_frames, hpss_frames));

    let freqs_raw = freqs.as_slice().unwrap();
    let roll_percent: Float = 0.85;
    let contrast_quantile: Float = 0.02;
    let c_ref: Float = 261.6256; // C4 reference for HPCP

    // Sethares (1998) dissonance model constants
    let diss_b1: Float = 3.5144;
    let diss_b2: Float = 5.7564;
    let diss_d_max: Float = 0.24;
    let diss_s1: Float = 0.0207;
    let diss_s2: Float = 18.96;

    let compute_frame = |t: usize| -> FrameResult {
        let start = t * hop_length;
        let mut fft_in = vec![0.0_f32; n_fft];
        for i in 0..n_fft {
            fft_in[i] = y_raw[start + i] * win_raw[i];
        }
        let mut fft_out = vec![num_complex::Complex::new(0.0, 0.0); n_bins];
        fft::rfft(&mut fft_in, &mut fft_out).expect("FFT failed");

        // Compute power and magnitude ONCE
        let mut power_col = vec![0.0_f32; n_bins];
        let mut mag_col = vec![0.0_f32; n_bins];
        let mut cent_num = 0.0_f32;
        let mut cent_den = 0.0_f32;

        for i in 0..n_bins {
            let pwr = fft_out[i].norm_sqr();
            let mag = pwr.sqrt();
            power_col[i] = pwr;
            mag_col[i] = mag;
            cent_num += freqs_raw[i] * mag;
            cent_den += mag;
        }

        let centroid = if cent_den > 0.0 {
            cent_num / cent_den
        } else {
            0.0
        };

        // RMS from time-domain
        let mut sum_sq = 0.0_f32;
        for i in 0..n_fft {
            sum_sq += y_raw[start + i] * y_raw[start + i];
        }
        let rms = (sum_sq / n_fft as Float).sqrt();

        #[cfg(feature = "aggression")]
        let aggression_crest_db = if wants_aggression {
            const EPS: Float = 1.0e-12;
            let peak = y_raw[start..start + n_fft]
                .iter()
                .copied()
                .map(Float::abs)
                .fold(0.0, Float::max);
            20.0 * ((peak + EPS) / (rms + EPS)).log10()
        } else {
            0.0
        };
        #[cfg(feature = "aggression")]
        let mut aggression_high_energy_ratio = 0.0;
        #[cfg(feature = "aggression")]
        let mut aggression_high_flatness = 0.0;
        #[cfg(feature = "aggression")]
        let mut aggression_high_total = 0.0;
        #[cfg(feature = "aggression")]
        let mut aggression_peak_ratio = 0.0;

        let (bandwidth, rolloff, flatness) = if extended {
            // Bandwidth — reuse mag_col
            let bw = if cent_den > 0.0 {
                let mut bw_num = 0.0_f32;
                for i in 0..n_bins {
                    let dev = freqs_raw[i] - centroid;
                    bw_num += mag_col[i] * dev * dev;
                }
                (bw_num / cent_den).sqrt()
            } else {
                0.0
            };

            // Rolloff — reuse mag_col
            let threshold = roll_percent * cent_den; // cent_den == sum of mag
            let mut cumsum = 0.0_f32;
            let mut ro = 0.0_f32;
            for i in 0..n_bins {
                cumsum += mag_col[i];
                if cumsum >= threshold {
                    ro = freqs_raw[i];
                    break;
                }
            }

            // Flatness — on power_col directly
            let amin: Float = 1e-10;
            let mut log_sum = 0.0_f32;
            let mut arith_sum = 0.0_f32;
            #[cfg(feature = "aggression")]
            let mut aggression_total = 0.0_f32;
            #[cfg(feature = "aggression")]
            let mut aggression_high_log_sum = 0.0_f32;
            #[cfg(feature = "aggression")]
            let mut aggression_high_arith_sum = 0.0_f32;
            #[cfg(feature = "aggression")]
            let mut aggression_high_count = 0_usize;
            #[cfg(feature = "aggression")]
            let mut aggression_strongest = [0.0_f32; 8];
            #[cfg(feature = "aggression")]
            if wants_aggression {
                for i in 0..n_bins {
                    let v = power_col[i].max(amin);
                    let log_v = v.ln();
                    log_sum += log_v;
                    arith_sum += v;
                    let power = power_col[i];
                    aggression_total += power;
                    if freqs_raw[i] >= 4_000.0 {
                        aggression_high_total += power;
                        aggression_high_log_sum += log_v;
                        aggression_high_arith_sum += v;
                        aggression_high_count += 1;
                    }
                    if power > aggression_strongest[0] {
                        aggression_strongest[0] = power;
                        aggression_strongest.sort_by(Float::total_cmp);
                    }
                }
            }
            #[cfg(feature = "aggression")]
            if !wants_aggression {
                for &power in power_col.iter().take(n_bins) {
                    let v = power.max(amin);
                    log_sum += v.ln();
                    arith_sum += v;
                }
            }
            #[cfg(not(feature = "aggression"))]
            for &power in power_col.iter().take(n_bins) {
                let v = power.max(amin);
                log_sum += v.ln();
                arith_sum += v;
            }
            let geo_mean = (log_sum / n_bins as Float).exp();
            let arith_mean = arith_sum / n_bins as Float;
            let fl = if arith_mean > 0.0 {
                geo_mean / arith_mean
            } else {
                0.0
            };
            #[cfg(feature = "aggression")]
            if wants_aggression {
                const EPS: Float = 1.0e-12;
                let high_mean = aggression_high_arith_sum / aggression_high_count.max(1) as Float;
                aggression_high_energy_ratio = aggression_high_total / (aggression_total + EPS);
                aggression_high_flatness = if high_mean > 0.0 {
                    (aggression_high_log_sum / aggression_high_count.max(1) as Float).exp()
                        / high_mean
                } else {
                    0.0
                };
                aggression_peak_ratio =
                    aggression_strongest.iter().sum::<Float>() / (aggression_total + EPS);
            }

            (bw, ro, fl)
        } else {
            (0.0, 0.0, 0.0)
        };

        let spectral_shape = if wants_spectral_shape {
            let (skewness, kurtosis) =
                spectral_shape_moments(&mag_col, freqs_raw, centroid, cent_den, bandwidth).unzip();
            [
                skewness,
                kurtosis,
                spectral_shape_slope(&power_col, &shape_bands),
                spectral_shape_decrease(&mag_col),
            ]
        } else {
            [None; 4]
        };

        let mut spectral_texture = [None; 3];
        let texture_norm = if wants_spectral_texture {
            spectral_texture_frame(&mag_col, cent_den).map(|f| {
                spectral_texture[0] = Some(f.crest);
                spectral_texture[1] = Some(f.entropy);
                f.p
            })
        } else {
            None
        };

        let tuning = if wants_tuning {
            tuning_frame(&mag_col, sr, n_fft)
        } else {
            [0.0; 3]
        };

        // Sparse mel projection
        let mel_col: Vec<Float> = sparse_mel
            .iter()
            .map(|(start_bin, weights)| {
                let mut sum = 0.0;
                for (k, &w) in weights.iter().enumerate() {
                    sum += w * power_col[start_bin + k];
                }
                sum
            })
            .collect();

        // Sparse chroma projection
        let mut chroma_col = [0.0_f32; 12];
        if extended {
            for (c, (sb, weights)) in sparse_chroma.iter().enumerate() {
                let mut sum = 0.0;
                for (k, &w) in weights.iter().enumerate() {
                    sum += w * power_col[sb + k];
                }
                chroma_col[c] = sum;
            }
        }

        // --- Fused spectral contrast (inline, using mag_col) ---
        let mut contrast_bands_out = [0.0_f32; N_CONTRAST_BANDS + 1];
        if extended {
            for (b, &(sb, eb)) in contrast_bands.iter().enumerate() {
                if sb >= eb {
                    continue;
                }
                let bn = eb - sb;
                // Collect magnitudes for this band into a small buffer
                let mut band_vals: Vec<Float> = (sb..eb).map(|f| mag_col[f].max(1e-10)).collect();
                // Partial sort: O(n) instead of O(n log n)
                let q_idx = ((bn as Float * contrast_quantile) as usize).min(bn - 1);
                band_vals.select_nth_unstable_by(q_idx, Float::total_cmp);
                let valley = band_vals[q_idx];
                let peak_idx = (bn - 1).saturating_sub(q_idx);
                band_vals.select_nth_unstable_by(peak_idx, Float::total_cmp);
                let peak = band_vals[peak_idx];
                contrast_bands_out[b] = peak.log10() - valley.log10();
            }
            let mean_mag = cent_den / n_bins as Float; // cent_den = sum of mag
            contrast_bands_out[N_CONTRAST_BANDS] = mean_mag.max(1e-10).log10();
        }

        // --- Fused HPCP + dissonance (inline, sharing peak detection) ---
        let mut hpcp_col = [0.0_f32; 12];
        let mut frame_diss = 0.0_f32;
        if extended {
            // Spectral peak detection with parabolic interpolation.
            // Collect ALL peaks, then keep top MAX_PEAKS by magnitude
            // (matches standalone tonal::detect_spectral_peaks behavior).
            let mut all_peaks_freq = Vec::new();
            let mut all_peaks_mag = Vec::new();

            for i in 1..n_bins - 1 {
                if mag_col[i] <= mag_col[i - 1] || mag_col[i] <= mag_col[i + 1] {
                    continue;
                }
                if freqs_raw[i] < 40.0 || freqs_raw[i] > 5000.0 {
                    continue;
                }

                // Parabolic interpolation
                let alpha = mag_col[i - 1];
                let beta = mag_col[i];
                let gamma = mag_col[i + 1];
                let denom = alpha - 2.0 * beta + gamma;
                let (freq, mag) = if denom.abs() > 1e-10 {
                    let p = 0.5 * (alpha - gamma) / denom;
                    let bin_frac = i as Float + p;
                    let f = if bin_frac >= 0.0 && (bin_frac as usize) < n_bins - 1 {
                        let lo = bin_frac as usize;
                        let frac = bin_frac - lo as Float;
                        freqs_raw[lo] * (1.0 - frac) + freqs_raw[lo + 1] * frac
                    } else {
                        freqs_raw[i]
                    };
                    (f, beta - 0.25 * (alpha - gamma) * p)
                } else {
                    (freqs_raw[i], beta)
                };

                all_peaks_freq.push(freq);
                all_peaks_mag.push(mag);
            }

            // Sort by magnitude descending, keep top MAX_PEAKS
            let mut indices: Vec<usize> = (0..all_peaks_freq.len()).collect();
            indices.sort_unstable_by(|&a, &b| all_peaks_mag[b].total_cmp(&all_peaks_mag[a]));
            indices.truncate(MAX_PEAKS);
            let n_peaks = indices.len();

            let peaks_freq: Vec<Float> = indices.iter().map(|&i| all_peaks_freq[i]).collect();
            let peaks_mag: Vec<Float> = indices.iter().map(|&i| all_peaks_mag[i]).collect();

            if wants_spectral_texture {
                spectral_texture[2] = spectral_texture_irregularity(&peaks_freq, &peaks_mag);
            }

            // HPCP from peaks
            for p in 0..n_peaks {
                let pmag_sq = peaks_mag[p] * peaks_mag[p];
                for h in 0..N_HPCP_HARMONICS {
                    let freq = peaks_freq[p] / (h as Float + 1.0);
                    if freq < 20.0 {
                        continue;
                    }
                    let semitones = 12.0 * (freq / c_ref).log2();
                    let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;
                    // Distance to the rounded value, not to the wrapped bin:
                    // 11.9 is 0.1 from C, not 11.9.
                    let nearest = pitch_class.round();
                    let center = nearest as usize % 12;
                    let dist = (pitch_class - nearest).abs();
                    if dist < 0.5 {
                        let w = (std::f32::consts::PI * dist).cos();
                        hpcp_col[center] += harmonic_weights[h] * pmag_sq * w;
                    }
                }
            }

            // Dissonance from same peaks (Sethares 1998)
            if n_peaks >= 2 {
                let mut diss_sum = 0.0_f32;
                let mut weight_sum = 0.0_f32;
                for i in 0..n_peaks {
                    for j in (i + 1)..n_peaks {
                        let f_min = peaks_freq[i].min(peaks_freq[j]);
                        let f_diff = (peaks_freq[i] - peaks_freq[j]).abs();
                        let s = diss_d_max / (diss_s1 * f_min + diss_s2);
                        let d = (-diss_b1 * s * f_diff).exp() - (-diss_b2 * s * f_diff).exp();
                        let d = d.max(0.0);
                        let w = peaks_mag[i] * peaks_mag[j];
                        diss_sum += w * d;
                        weight_sum += w;
                    }
                }
                if weight_sum > 0.0 {
                    frame_diss = (diss_sum / weight_sum).clamp(0.0, 1.0);
                }
            }
        }

        FrameResult {
            mel_col,
            centroid,
            rms,
            bandwidth,
            rolloff,
            flatness,
            chroma_col,
            contrast_bands: contrast_bands_out,
            hpcp_col,
            dissonance: frame_diss,
            spectral_shape,
            spectral_texture,
            texture_norm,
            tuning,
            // The median over frequency needs only this frame, so it runs here.
            hpss: wants_hpss.then(|| {
                let percussive = hpss_median(&mag_col, hpss_bins);
                (mag_col, percussive)
            }),
            #[cfg(feature = "aggression")]
            aggression_crest_db,
            #[cfg(feature = "aggression")]
            aggression_high_energy_ratio,
            #[cfg(feature = "aggression")]
            aggression_high_flatness,
            #[cfg(feature = "aggression")]
            aggression_high_total,
            #[cfg(feature = "aggression")]
            aggression_peak_ratio,
        }
    };

    // Scatter frame results into arrays, in frame order. Accumulators run
    // across chunks so every sum adds in the same order as one pass.
    let mut dissonance_acc = 0.0_f32;
    let mut scatter_results = |base: usize,
                               frame_results: Vec<FrameResult>,
                               fluxes: Vec<Option<f64>>| {
        let d_acc = &mut dissonance_acc;

        for (i, fr) in frame_results.into_iter().enumerate() {
            let t = base + i;
            centroids[t] = fr.centroid;
            rms_frames[t] = fr.rms;
            if extended {
                bandwidths[t] = fr.bandwidth;
                rolloffs[t] = fr.rolloff;
                flatnesses[t] = fr.flatness;
                for c in 0..12 {
                    chroma_raw[(c, t)] = fr.chroma_col[c];
                }
                for b in 0..N_CONTRAST_BANDS + 1 {
                    contrast_frames[(b, t)] = fr.contrast_bands[b];
                }
                for c in 0..12 {
                    hpcp_raw[(c, t)] = fr.hpcp_col[c];
                }
                *d_acc += fr.dissonance;
            }
            for (i, v) in fr.spectral_shape.into_iter().enumerate() {
                if let Some(v) = v {
                    shape_sum[i] += v;
                    shape_count[i] += 1;
                }
            }
            if wants_spectral_texture {
                if let Some(flux) = fluxes[i] {
                    texture_sum[0] += flux;
                    flux_sq_sum += flux * flux;
                    texture_count[0] += 1;
                }
                for (i, v) in fr.spectral_texture.into_iter().enumerate() {
                    if let Some(v) = v {
                        texture_sum[i + 1] += v;
                        texture_count[i + 1] += 1;
                    }
                }
            }
            tuning_frames += (fr.tuning[2] > 0.0) as usize;
            for (sum, v) in tuning_sum.iter_mut().zip(fr.tuning) {
                *sum += v;
            }
            #[cfg(feature = "aggression")]
            if wants_aggression {
                aggression_crest_db.as_mut().unwrap()[t] = fr.aggression_crest_db;
                aggression_dissonance.as_mut().unwrap()[t] = fr.dissonance;
                aggression_high_energy_ratio.as_mut().unwrap()[t] = fr.aggression_high_energy_ratio;
                aggression_high_flatness.as_mut().unwrap()[t] = fr.aggression_high_flatness;
                aggression_high_total.as_mut().unwrap()[t] = fr.aggression_high_total;
                aggression_peak_ratio.as_mut().unwrap()[t] = fr.aggression_peak_ratio;
            }
            for (m, val) in fr.mel_col.into_iter().enumerate() {
                mel_spec[(m, t)] = val;
            }
        }
    };

    // Bounded chunks keep at most `frame_chunk` frame results (and their
    // normalized spectra) alive at once.
    let frame_chunk = frame_chunk.max(1);
    let mut base = 0;
    while base < n_frames {
        let end = base.saturating_add(frame_chunk).min(n_frames);
        let mut frame_results: Vec<FrameResult> = if n_frames >= PARALLEL_THRESHOLD {
            (base..end)
                .into_par_iter()
                .map(|t| compute_frame(t))
                .collect()
        } else {
            (base..end).map(|t| compute_frame(t)).collect()
        };
        // The median over time looks ahead: the stream holds each frame back
        // until its window is complete, across chunks and at the track's end.
        if let Some(stream) = hpss_stream.as_mut() {
            let columns = frame_results.iter_mut().filter_map(|f| f.hpss.take());
            stream.push(columns, n_frames >= PARALLEL_THRESHOLD);
        }
        // Flux pairs each frame with its predecessor (the previous chunk's last
        // frame for the first one); computed in parallel, summed in frame order.
        let fluxes: Vec<Option<f64>> = if wants_spectral_texture {
            let flux_at = |i: usize| {
                let prev = match i {
                    0 => texture_prev.as_deref(),
                    _ => frame_results[i - 1].texture_norm.as_deref(),
                };
                Some(spectral_texture_flux(
                    prev?,
                    frame_results[i].texture_norm.as_deref()?,
                ))
            };
            let fluxes = if n_frames >= PARALLEL_THRESHOLD {
                (0..frame_results.len())
                    .into_par_iter()
                    .map(flux_at)
                    .collect()
            } else {
                (0..frame_results.len()).map(flux_at).collect()
            };
            texture_prev = frame_results.last().and_then(|f| f.texture_norm.clone());
            fluxes
        } else {
            Vec::new()
        };
        scatter_results(base, frame_results, fluxes);
        base = end;
    }
    let hpss_percussive_ratio = hpss_stream.map(|stream| stream.ratio());

    // ================================================================
    // ONSET STRENGTH from mel spectrogram (no additional FFT)
    // ================================================================

    let s_db = spectrum::power_to_db(mel_spec.view(), 1.0, 1e-10, Some(MEL_TOP_DB));
    // Everything below reads the dB copy: free the power one.
    drop(mel_spec);

    // Mel bands (group 4.1): per-band mean / std of the dB matrix over all
    // frames, plus the lowest value a cell can take: the `power_to_db` clamp
    // (loudest cell − top_db), or its −100 dB power floor when that is higher.
    let (mel_bands_mean, mel_bands_std, mel_bands_floor_db) = if config.wants("mel_bands") {
        let mean: Vec<Float> = s_db
            .rows()
            .into_iter()
            .map(|row| (row.iter().map(|&v| v as f64).sum::<f64>() / n_frames as f64) as Float)
            .collect();
        let std: Vec<Float> = s_db
            .rows()
            .into_iter()
            .map(|row| series_std(row.as_slice().unwrap()))
            .collect();
        let peak = s_db.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        (Some(mean), Some(std), Some((peak - MEL_TOP_DB).max(-100.0)))
    } else {
        (None, None, None)
    };

    // Tempo and its candidates follow the broadband envelope, the beat tracker
    // the mel bins below 3.2 kHz, and downbeats weigh the kick (below 200 Hz)
    // against the snare body (200-800 Hz): one loop, no onset bands needed.
    let rhythm = crate::onset::rhythm_envelopes_from_log_mel(s_db.view(), sr, n_fft, hop_length);
    #[cfg(feature = "bench-internals")]
    capture_rhythm_envelopes(&rhythm);
    let crate::onset::RhythmEnvelopes {
        onset: oenv_padded,
        beat: beat_env_padded,
        low: low_env_padded,
        mid: mid_env_padded,
    } = rhythm;

    let onset_bands = if config.wants("onset_bands") {
        Some(crate::onset::onset_strength_bands_from_log_mel(
            s_db.view(),
            sr,
            hop_length,
            n_fft,
            None,
        )?)
    } else {
        None
    };
    let onset_band_edges_hz = onset_bands
        .as_ref()
        .map(|bands| bands.band_edges_hz.clone());
    let onset_strength_bands = onset_bands.as_ref().map(|bands| {
        bands
            .envelopes
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect()
    });

    // ================================================================
    // BEAT TRACKING + ONSET DETECTION
    // ================================================================

    // The aggression model must not see the caller's range fold (the canonical
    // lane tracks without one), so with a range it also gets the range-free beats.
    #[cfg(feature = "aggression")]
    let want_range_free_beats = wants_aggression && config.bpm_min.is_some();
    #[cfg(not(feature = "aggression"))]
    let want_range_free_beats = false;
    // `beat_positions`: the beats at sub-frame precision, for the tempo curve
    // and the grid stability (whole frames quantize both).
    let (tempo_estimate, beats, range_free_beats, beat_positions) =
        crate::beat::beat_track_with_range_free_beats(
            None,
            Some(oenv_padded.view()),
            sr,
            hop_length,
            120.0,
            100.0,
            true,
            config.bpm_min,
            config.bpm_max,
            Some(beat_env_padded.view()),
            want_range_free_beats,
        )?;
    #[cfg(not(feature = "aggression"))]
    let _ = range_free_beats;
    let bpm = tempo_estimate.tempo;
    let bpm_raw = tempo_estimate.tempo_raw;
    let bpm_candidates = tempo_estimate.candidates;

    let onset_frames = crate::onset::onset_detect(
        None,
        Some(oenv_padded.view()),
        sr,
        hop_length,
        false,
        0.07,
        0,
    )?;

    // ================================================================
    // Zero crossings (trivial, time-domain)
    // ================================================================

    let zcr = zero_crossing_rate;

    // ================================================================
    // LUFS integrated loudness (ITU-R BS.1770-4, K-weighted)
    // ================================================================

    let loudness_lufs = perceptual::loudness_lufs(y, sr);

    // --- loudness ---
    // Extended loudness / gain metrics — opt-in (Full mode or `features=["loudness"]`).
    // Compact and Playlist skip this entirely, so they pay nothing.
    let (true_peak_db, replaygain_db, loudness_curve, loudness_momentary_max_db, loudness_range_lu) =
        if config.wants("loudness") {
            let tp = crate::loudness_ext::true_peak_db(y);
            let rg = crate::loudness_ext::replaygain_db(loudness_lufs);
            // Short-term curve: 3 s window, 1 s hop (ITU-R BS.1770 short-term).
            // One K-weighting pass feeds the curve, momentary max and LRA.
            let m = crate::loudness_ext::loudness_metrics(y, sr, 3.0, 1.0);
            (
                Some(tp),
                Some(rg),
                Some(m.curve),
                Some(m.momentary_max_db),
                Some(m.range_lu),
            )
        } else {
            (None, None, None, None, None)
        };
    // --- end loudness ---

    // ================================================================
    // EXTENDED: MFCCs via pre-computed DCT matrix (no per-frame cos())
    // ================================================================

    // Per-coefficient MFCC series (N_MFCC x n_frames) for the frame statistics.
    let mut mfcc_frames: Vec<Vec<Float>> = Vec::new();
    let mfcc_mean = if extended {
        // dct_matrix is (N_MFCC, n_mels), s_db is (n_mels, n_frames)
        // Compute MFCC mean = mean over frames of (dct_matrix @ s_db[:, t])
        let mut mfcc_avg = vec![0.0_f32; N_MFCC];
        mfcc_frames = vec![Vec::with_capacity(n_frames); N_MFCC];
        let dct_raw = dct_matrix.as_slice().unwrap();
        for t in 0..n_frames {
            for k in 0..N_MFCC {
                let mut sum = 0.0_f32;
                let row_start = k * n_mels;
                for m in 0..n_mels {
                    sum += dct_raw[row_start + m] * s_db[(m, t)];
                }
                mfcc_avg[k] += sum;
                mfcc_frames[k].push(sum);
            }
        }
        for v in mfcc_avg.iter_mut() {
            *v /= n_frames.max(1) as Float;
        }
        Some(mfcc_avg)
    } else {
        None
    };

    // ================================================================
    // CHROMA: proper filterbank (always, computed in the fused loop)
    // L-inf normalize per frame, then average across frames.
    // ================================================================

    // Per-class normalized chroma series (12 x n_frames) for the frame statistics.
    let mut chroma_frames: Vec<Vec<Float>> = Vec::new();
    let chroma_mean = if extended && n_frames > 0 {
        let mut chroma_avg = vec![0.0_f32; 12];
        chroma_frames = vec![Vec::with_capacity(n_frames); 12];
        for t in 0..n_frames {
            let mut frame_chroma = [0.0_f32; 12];
            for c in 0..12 {
                frame_chroma[c] = chroma_raw[(c, t)];
            }
            // L-inf normalize per frame
            let max_val = frame_chroma.iter().copied().fold(0.0_f32, Float::max);
            if max_val > 0.0 {
                for v in frame_chroma.iter_mut() {
                    *v /= max_val;
                }
            }
            for (i, &v) in frame_chroma.iter().enumerate() {
                chroma_avg[i] += v;
                chroma_frames[i].push(v);
            }
        }
        for v in chroma_avg.iter_mut() {
            *v /= n_frames as Float;
        }
        Some(chroma_avg)
    } else {
        None
    };

    // Frames that count for the flatness and contrast statistics: RMS above
    // zero and within 60 dB of the loudest frame (the non-silent rule of
    // `effects::split`, librosa's default `top_db`). Flatness and contrast are
    // undefined on silence (the floors make them 1.0 and 0.0 there) and read
    // the noise floor on near-silence.
    let sounding: Vec<usize> = if extended {
        let floor = rms_frames.iter().copied().fold(0.0_f32, Float::max) * 1e-3;
        (0..n_frames)
            .filter(|&t| rms_frames[t] > 0.0 && rms_frames[t] >= floor)
            .collect()
    } else {
        Vec::new()
    };

    // ================================================================
    // SPECTRAL CONTRAST: aggregated from fused frame loop
    // ================================================================

    let spectral_contrast_mean = if extended && n_frames > 0 {
        Some(
            contrast_frames
                .rows()
                .into_iter()
                .map(|row| {
                    sounding.iter().fold(0.0_f32, |sum, &t| sum + row[t])
                        / sounding.len().max(1) as Float
                })
                .collect::<Vec<Float>>(),
        )
    } else {
        None
    };

    // ================================================================
    // Aggregate results
    // ================================================================

    let rms_mean = rms_frames.iter().sum::<Float>() / rms_frames.len() as Float;
    let rms_max = rms_frames.iter().copied().fold(0.0_f32, Float::max);

    let rms_nonzero: Vec<Float> = rms_frames.iter().copied().filter(|&v| v > 1e-10).collect();
    let dynamic_range_db = if rms_nonzero.len() > 10 {
        let mut sorted = rms_nonzero.clone();
        sorted.sort_by(Float::total_cmp);
        let p5 = sorted[sorted.len() * 5 / 100];
        let p95 = sorted[sorted.len() * 95 / 100];
        if p5 > 0.0 {
            20.0 * (p95 / p5).log10()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let centroid_mean = centroids.iter().sum::<Float>() / centroids.len().max(1) as Float;
    let onset_density = onset_frames.len() as Float / duration_sec;

    // BPM confidence: a trust signal for the range-aligned `bpm` (not a
    // probability). Combines the dominant autocorrelation peak strength, the
    // agreement between `bpm` and the tracked beat rate (folded by one octave),
    // and rhythmic onset density.
    let bpm_confidence = {
        let s1 = bpm_candidates.first().map(|c| c.1).unwrap_or(0.0);
        let strength = s1 / (s1 + 1.2);
        let bpm_beats = if duration_sec > 0.0 {
            60.0 * beats.len() as Float / duration_sec
        } else {
            0.0
        };
        let agree = if bpm > 0.0 && bpm_beats > 0.0 {
            let mut d = (bpm / bpm_beats).log2().abs();
            d = d.min((d - 1.0).abs()); // fold one octave
            (-d / 0.10).exp()
        } else {
            0.0
        };
        let density = (onset_density / 4.0).clamp(0.0, 1.0);
        (0.50 * strength + 0.35 * agree + 0.15 * density).clamp(0.0, 1.0)
    };

    let spectral_bandwidth_mean = if extended {
        Some(bandwidths.iter().sum::<Float>() / bandwidths.len().max(1) as Float)
    } else {
        None
    };

    let spectral_rolloff_mean = if extended {
        Some(rolloffs.iter().sum::<Float>() / rolloffs.len().max(1) as Float)
    } else {
        None
    };

    let flatness_sounding: Vec<Float> = sounding.iter().map(|&t| flatnesses[t]).collect();
    let spectral_flatness_mean = if extended {
        Some(if flatness_sounding.is_empty() {
            0.0
        } else {
            flatness_sounding.iter().sum::<Float>() / flatness_sounding.len() as Float
        })
    } else {
        None
    };

    // Spectral shape (group 3.1): mean over the frames where each is defined.
    // Order: skewness, excess kurtosis, slope, decrease.
    let shape_means: [Option<Float>; 4] = std::array::from_fn(|i| {
        wants_spectral_shape.then(|| {
            if shape_count[i] > 0 {
                (shape_sum[i] / shape_count[i] as f64) as Float
            } else {
                0.0
            }
        })
    });

    // Spectral texture (group 3.2): flux mean, crest, entropy, irregularity
    // means over defined frames, plus the population std of the flux.
    let texture_means: [Option<Float>; 4] = std::array::from_fn(|i| {
        wants_spectral_texture.then(|| {
            if texture_count[i] > 0 {
                (texture_sum[i] / texture_count[i] as f64) as Float
            } else {
                0.0
            }
        })
    });
    let spectral_flux_std = wants_spectral_texture.then(|| {
        let n = texture_count[0] as f64;
        if n > 0.0 {
            let mean = texture_sum[0] / n;
            (flux_sq_sum / n - mean * mean).max(0.0).sqrt() as Float
        } else {
            0.0
        }
    });

    // Tuning: circular mean of the frame votes. The confidence is reported
    // whenever the feature ran; the offset abstains below the threshold.
    let tuning_confidence = wants_tuning.then(|| {
        let [re, im, wsum] = tuning_sum;
        if wsum > 0.0 {
            ((re * re + im * im).sqrt() / wsum) as Float
        } else {
            0.0
        }
    });
    // No voting frame gives an infinite bar: nothing to report.
    let tuning_bar =
        TUNING_MIN_CONFIDENCE.max((TUNING_SHORT_BAR / (tuning_frames as f64).sqrt()) as Float);
    let tuning_offset_cents = tuning_confidence
        .filter(|&confidence| confidence >= tuning_bar)
        .map(|_| {
            let [re, im, _] = tuning_sum;
            let cents = (100.0 * im.atan2(re) / std::f64::consts::TAU) as Float;
            // +50 (atan2 = π, or an f32 round-up) is the same grid as −50.
            if cents >= 50.0 {
                -50.0
            } else {
                cents
            }
        });

    // ================================================================
    // FRAME STATISTICS (MIR_FEATURES group 1.2): std / delta / quantiles
    // over the same frames as the `_mean` fields. Which statistic each
    // feature gets follows the spec, not a uniform set.
    // ================================================================

    let rms_slice = rms_frames.as_slice().unwrap();
    let rms_sorted = sorted_series(rms_slice);
    let rms_std = Some(series_std(rms_slice));
    let rms_delta_mean = Some(series_delta_mean(rms_slice));
    let rms_iqr = Some(sorted_iqr(&rms_sorted));
    let rms_p10 = Some(sorted_quantile(&rms_sorted, 0.10));
    let rms_p90 = Some(sorted_quantile(&rms_sorted, 0.90));

    let centroid_slice = centroids.as_slice().unwrap();
    let spectral_centroid_std = Some(series_std(centroid_slice));
    let spectral_centroid_delta_mean = Some(series_delta_mean(centroid_slice));
    let spectral_centroid_iqr = Some(sorted_iqr(&sorted_series(centroid_slice)));

    let zero_crossing_rate_std = Some(series_std(&frame_zero_crossing_rates(
        y_raw, n_fft, hop_length, n_frames,
    )));

    let bandwidth_slice = bandwidths.as_slice().unwrap();
    let spectral_bandwidth_std = extended.then(|| series_std(bandwidth_slice));
    let spectral_bandwidth_delta_mean = extended.then(|| series_delta_mean(bandwidth_slice));

    let rolloff_slice = rolloffs.as_slice().unwrap();
    let rolloff_sorted = sorted_series(rolloff_slice);
    let spectral_rolloff_std = extended.then(|| series_std(rolloff_slice));
    let spectral_rolloff_p10 = extended.then(|| sorted_quantile(&rolloff_sorted, 0.10));
    let spectral_rolloff_p90 = extended.then(|| sorted_quantile(&rolloff_sorted, 0.90));

    let flatness_slice = flatness_sounding.as_slice();
    let flatness_sorted = sorted_series(flatness_slice);
    let spectral_flatness_std = extended.then(|| series_std(flatness_slice));
    let spectral_flatness_iqr = extended.then(|| sorted_iqr(&flatness_sorted));
    let spectral_flatness_p10 = extended.then(|| sorted_quantile(&flatness_sorted, 0.10));
    let spectral_flatness_p90 = extended.then(|| sorted_quantile(&flatness_sorted, 0.90));

    // Vector features follow their `_mean` (`None` exactly when it is).
    let spectral_contrast_std = spectral_contrast_mean.as_ref().map(|_| {
        contrast_frames
            .rows()
            .into_iter()
            .map(|row| series_std(&sounding.iter().map(|&t| row[t]).collect::<Vec<Float>>()))
            .collect::<Vec<Float>>()
    });
    let mfcc_std = mfcc_mean.as_ref().map(|_| {
        mfcc_frames
            .iter()
            .map(|s| series_std(s))
            .collect::<Vec<Float>>()
    });
    let mfcc_delta_mean = mfcc_mean.as_ref().map(|_| {
        mfcc_frames
            .iter()
            .map(|s| series_delta_mean(s))
            .collect::<Vec<Float>>()
    });
    let chroma_std = chroma_mean.as_ref().map(|_| {
        chroma_frames
            .iter()
            .map(|s| series_std(s))
            .collect::<Vec<Float>>()
    });
    let chroma_delta_mean = chroma_mean.as_ref().map(|_| {
        chroma_frames
            .iter()
            .map(|s| series_delta_mean(s))
            .collect::<Vec<Float>>()
    });

    // ================================================================
    // RHYTHM: Tempo curve & time signature
    // ================================================================

    let (tempo_curve, tempo_variability) = if extended && config.wants("tempo_curve") {
        let tc = crate::beat::tempo_curve_at(&beat_positions, sr, hop_length, Some(5));
        let tv = crate::beat::tempo_variability(&tc);
        (Some(tc), Some(tv))
    } else {
        (None, None)
    };

    let (time_signature, time_signature_confidence) = if extended && config.wants("time_signature")
    {
        let win = 384.min(oenv_padded.len());
        if win >= 4 {
            match crate::feature::rhythm::metrogram(
                None,
                Some(oenv_padded.view()),
                sr,
                hop_length,
                win,
                None,
            ) {
                Ok(mg) => {
                    let (label, conf) =
                        crate::feature::rhythm::detect_time_signature(mg.view(), None);
                    (Some(label), Some(conf))
                }
                Err(_) => (None, None),
            }
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // ================================================================
    // BEAT GRID: offset, downbeats, stability (opt-in via features)
    // Reuses the already-computed beats + onset envelope (O(n_beats)).
    // Runs in Full mode or when requested via features=["beatgrid"].
    // ================================================================

    // Bars are always 4 beats: the meter detector is not reliable enough to
    // set the bar length (it reads plain four-on-the-floor as "2/4").
    // shortcut: 3/4 material gets a 4/4 grid; take the bar length from
    // `time_signature` once its detector is validated.
    let beats_per_bar = crate::beatgrid::DEFAULT_BEATS_PER_BAR;

    let (grid_offset_sec, downbeats, grid_stability) = if config.wants("beatgrid") {
        let grid = crate::beatgrid::analyze_grid_low_mid(
            &beats,
            low_env_padded.view(),
            mid_env_padded.view(),
            sr,
            hop_length,
            beats_per_bar,
        );
        (
            Some(grid.grid_offset_sec),
            Some(grid.downbeats),
            Some(crate::beatgrid::grid_stability_at(&beat_positions)),
        )
    } else {
        (None, None, None)
    };

    // ================================================================
    // RHYTHMIC REGULARITY: where accents land inside the metric grid
    // (opt-in via features). Reuses the banded onset envelopes, tracked
    // beats and bar phase computed above; the added work is one
    // bar-length DFT per band.
    // ================================================================

    let rhythmic_regularity_analysis = if config.wants("rhythmic_regularity") {
        onset_bands.as_ref().map(|bands| {
            crate::rhythmic_regularity::analyze(
                bands.envelopes.view(),
                &beats,
                downbeats.as_deref().unwrap_or(&[]),
                beats_per_bar,
                bpm_confidence,
                grid_stability
                    .unwrap_or_else(|| crate::beatgrid::grid_stability_at(&beat_positions)),
                // No meter evidence: the bar length above is assumed, not detected.
                None,
            )
        })
    } else {
        None
    };
    // The confidence is reported whenever the feature ran, including when the
    // measure abstained — that is what lets a caller tell "not regular" from
    // "could not tell".
    let rhythmic_regularity = rhythmic_regularity_analysis
        .as_ref()
        .and_then(|measured| measured.regularity);
    let rhythmic_regularity_label = rhythmic_regularity_analysis
        .as_ref()
        .and_then(|measured| measured.label)
        .map(str::to_owned);
    let rhythmic_regularity_confidence = rhythmic_regularity_analysis
        .as_ref()
        .map(|measured| measured.confidence);
    let rhythmic_regularity_candidates = rhythmic_regularity_analysis
        .as_ref()
        .and_then(|measured| measured.candidates.as_ref())
        .map(|candidates| {
            candidates
                .iter()
                .map(|&(label, score)| (label.to_owned(), score))
                .collect()
        });

    // ================================================================
    // RHYTHM LEVELS: strength of the eighth-note and one-and-a-half
    // levels relative to the beat (Playlist, Full or a request). Reuses the onset
    // envelope the tempo was estimated from and the reported tempo.
    // ================================================================

    let (rhythm_level_double_ratio, rhythm_level_one_and_half_ratio, rhythm_level_confidence) =
        if config.wants("rhythm_levels") {
            let oenv = oenv_padded
                .as_slice()
                .expect("owned envelope is contiguous");
            let (double, one_and_half, confidence) = rhythm_levels(oenv, bpm, sr, hop_length);
            (double, one_and_half, Some(confidence))
        } else {
            (None, None, None)
        };

    // ================================================================
    // TONAL: chords from fused HPCP, dissonance from fused accumulator
    // ================================================================

    let wants_chords = extended && config.wants("chords");
    let wants_diss = extended && config.wants("dissonance");

    let (chord_sequence, chord_events, chord_change_rate, predominant_chord, dissonance_val) =
        if (wants_chords || wants_diss) && n_frames > 0 {
            // L1-normalize HPCP per frame (in-place on hpcp_raw)
            for t in 0..n_frames {
                let sum: Float = (0..12).map(|c| hpcp_raw[(c, t)]).sum();
                if sum > 0.0 {
                    for c in 0..12 {
                        hpcp_raw[(c, t)] /= sum;
                    }
                }
            }

            let (cs, ce, ccr, pc) = if wants_chords {
                let chords = crate::tonal::chords_from_beats(hpcp_raw.view(), &beats);
                let desc = crate::tonal::chord_descriptors(&chords, duration_sec);
                let events = chord_events_from_labels(
                    &chords,
                    &beats,
                    n_frames,
                    sr_f,
                    hop_length,
                    duration_sec,
                );
                (
                    Some(chords),
                    Some(events),
                    Some(desc.change_rate),
                    Some(desc.predominant_chord),
                )
            } else {
                (None, None, None, None)
            };

            let dv = if wants_diss {
                Some(dissonance_acc / n_frames as Float)
            } else {
                None
            };

            (cs, ce, ccr, pc, dv)
        } else {
            (None, None, None, None, None)
        };

    // ================================================================
    // PERCEPTUAL FEATURES (from already-computed scalars, ~0 extra cost)
    // ================================================================

    let bw_mean = spectral_bandwidth_mean.unwrap_or(0.0);
    let fl_mean = spectral_flatness_mean.unwrap_or(0.0);
    let ro_mean = spectral_rolloff_mean.unwrap_or(0.0);

    let wants_energy = extended && config.wants("energy");
    let wants_dance = extended && config.wants("danceability");
    let wants_key = extended && config.wants("key");
    let wants_valence = extended && (config.wants("valence") || config.wants("key"));
    let wants_acoustic = extended && config.wants("acousticness");
    // mood (heuristic v1) is extended-gated (needs chroma/key).
    let wants_mood = extended && config.wants("mood");

    let energy = if wants_energy {
        Some(perceptual::energy(
            rms_mean,
            centroid_mean,
            onset_density,
            bw_mean,
        ))
    } else {
        None
    };

    let danceability = if wants_dance {
        Some(perceptual::danceability_heuristic(
            bpm,
            &beats,
            onset_density,
        ))
    } else {
        None
    };

    // Key detection requires chroma (resolved as dependency). mood also needs it.
    let key_result = if wants_key || wants_valence || wants_mood {
        chroma_mean.as_ref().map(|c| perceptual::detect_key(c))
    } else {
        None
    };

    let valence = if config.wants("valence") {
        key_result
            .as_ref()
            .map(|kr| perceptual::valence(kr, bpm, centroid_mean))
    } else {
        None
    };

    let acousticness = if wants_acoustic {
        Some(perceptual::acousticness(
            fl_mean,
            ro_mean,
            centroid_mean,
            onset_density,
        ))
    } else {
        None
    };

    // --- mood (heuristic v1) ---
    // Extended-gated; recomputes energy/danceability internally so it does not
    // depend on whether those fields were also requested. Only the four mood_*
    // fields are emitted — `key`/`valence` stay `None` unless individually wanted.
    let mood = if wants_mood {
        Some(perceptual::mood_scores(
            key_result.as_ref(),
            bpm,
            rms_mean,
            centroid_mean,
            onset_density,
            bw_mean,
            &beats,
            dissonance_val,
            dynamic_range_db,
        ))
    } else {
        None
    };

    #[cfg(feature = "aggression")]
    let aggression_analysis = if wants_aggression {
        let crest = aggression_crest_db.as_deref().unwrap();
        let diss = aggression_dissonance.as_deref().unwrap();
        let high_energy = aggression_high_energy_ratio.as_deref().unwrap();
        let high_flatness = aggression_high_flatness.as_deref().unwrap();
        let high_total = aggression_high_total.as_deref().unwrap();
        let peak_ratio = aggression_peak_ratio.as_deref().unwrap();

        let onset_p90 = aggression_quantile(oenv_padded.as_slice().unwrap(), 0.90);
        let onset_norm = oenv_padded
            .iter()
            .map(|value| (value / onset_p90.max(1.0e-12)).clamp(0.0, 4.0))
            .collect::<Vec<_>>();
        let onset_sorted = aggression_sorted(&onset_norm);
        let onset_p50 = aggression_quantile_sorted(&onset_sorted, 0.50);
        let onset_threshold = (onset_p50 + 0.25).max(0.30);
        let aggression_onsets = onset_norm
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value >= onset_threshold).then_some(index))
            .collect::<Vec<_>>();
        let aggression_onset_density = aggression_onsets.len() as Float / duration_sec;
        let high_flux = high_total
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).max(0.0) / pair[1].max(1.0e-12))
            .collect::<Vec<_>>();
        let rms_slice = rms_frames.as_slice().unwrap();
        let rms_sorted = aggression_sorted(rms_slice);
        let rms_p10 = aggression_quantile_sorted(&rms_sorted, 0.10);
        let rms_p90 = aggression_quantile_sorted(&rms_sorted, 0.90);
        let signal_rms =
            (y.iter().map(|value| value * value).sum::<Float>() / y.len() as Float).sqrt();
        let non_silent_threshold = 0.10 * rms_p90.max(1.0e-12);
        let non_silent = rms_slice
            .iter()
            .filter(|value| **value >= non_silent_threshold)
            .count() as Float
            / rms_slice.len().max(1) as Float;
        let peak_sorted = aggression_sorted(peak_ratio);
        let peak_p50 = aggression_quantile_sorted(&peak_sorted, 0.50);
        let high_flatness_sorted = aggression_sorted(high_flatness);
        let high_flatness_p50 = aggression_quantile_sorted(&high_flatness_sorted, 0.50);
        let content_support = if signal_rms <= 1.0e-6 {
            0.0
        } else {
            non_silent * (0.5 * peak_p50 + 0.5 * (1.0 - high_flatness_p50))
        }
        .clamp(0.0, 1.0);
        let (
            window_force_top2,
            window_harshness_top2,
            window_impact_persistence,
            window_impact_top2,
        ) = aggression_window_summaries(
            crest,
            high_energy,
            high_flatness,
            peak_ratio,
            &onset_norm,
            sr,
            hop_length,
        );
        let mfcc = mfcc_mean.as_deref().unwrap_or(&[]);
        // The model input is the contrast mean over every frame, as
        // `aggression_dsp` computes it, not `spectral_contrast_mean`.
        let contrast: Vec<Float> = contrast_frames
            .rows()
            .into_iter()
            .map(|row| row.iter().fold(0.0_f32, |sum, &v| sum + v) / n_frames as Float)
            .collect();
        // With a range, read the tempo and beats a range-free call reports, as the
        // canonical lane does; without one they are the pipeline's own.
        let (aggression_bpm, aggression_beats, aggression_danceability) =
            match range_free_beats.as_deref() {
                Some(free) => {
                    let free_bpm = bpm_raw.clamp(30.0, 320.0);
                    let dance = perceptual::danceability_heuristic(free_bpm, free, onset_density);
                    (free_bpm, free, dance)
                }
                None => (bpm, beats.as_slice(), danceability.unwrap_or(0.5)),
            };
        let crest_sorted = aggression_sorted(crest);
        let diss_sorted = aggression_sorted(&aggression_evenly_sample(diss, 48));
        let high_energy_sorted = aggression_sorted(high_energy);
        let high_flux_sorted = aggression_sorted(&high_flux);
        Some(crate::aggression::score_evidence(
            &crate::aggression::AggressionEvidence {
                crest_p50: aggression_quantile_sorted(&crest_sorted, 0.50),
                crest_p90: aggression_quantile_sorted(&crest_sorted, 0.90),
                dissonance_p50: aggression_quantile_sorted(&diss_sorted, 0.50),
                dissonance_p90: aggression_quantile_sorted(&diss_sorted, 0.90),
                mfcc_0: mfcc.first().copied().unwrap_or(-180.0),
                mfcc_2: mfcc.get(2).copied().unwrap_or(0.0),
                contrast: std::array::from_fn(|index| contrast.get(index).copied().unwrap_or(0.0)),
                centroid: centroid_mean,
                bandwidth: bw_mean,
                bpm: aggression_bpm,
                onset_density_embedding: onset_density,
                danceability: aggression_danceability,
                grid_regularity: aggression_grid_regularity(aggression_beats),
                dynamic_range_db,
                energy: energy.unwrap_or(0.5),
                high_energy_p50: aggression_quantile_sorted(&high_energy_sorted, 0.50),
                high_energy_p90: aggression_quantile_sorted(&high_energy_sorted, 0.90),
                high_flatness_p50,
                high_flux_p90: aggression_quantile_sorted(&high_flux_sorted, 0.90),
                onset_density: aggression_onset_density,
                onset_interval_cv: aggression_interval_cv(&aggression_onsets),
                onset_strength_p50: onset_p50,
                onset_strength_p90: aggression_quantile_sorted(&onset_sorted, 0.90),
                rms_dynamic_ratio: rms_p90 / rms_p10.max(1.0e-12),
                spectral_peak_ratio: peak_p50,
                window_force_top2,
                window_harshness_top2,
                window_impact_persistence,
                window_impact_top2,
                content_support,
            },
        )?)
    } else {
        None
    };

    // --- fingerprint ---
    // Opt-in (see FEATURE_REGISTRY): runs only in Full mode or when the caller
    // requested the "fingerprint" feature, so Compact and Playlist
    // pay exactly zero cost. Operates on its own downsampled mono copy of `y`.
    let fingerprint = if config.wants("fingerprint") {
        let fp = crate::fingerprint::compute(y, sr);
        if fp.is_empty() {
            None
        } else {
            Some(fp)
        }
    } else {
        None
    };

    // key_result may also have been computed solely to feed mood; only surface
    // the key fields when key/valence was actually requested (no mood leakage).
    let emit_key = wants_key || wants_valence;
    let key = if emit_key {
        key_result.as_ref().map(|kr| perceptual::format_key(kr))
    } else {
        None
    };
    let key_confidence = if emit_key {
        key_result.as_ref().map(|kr| kr.confidence)
    } else {
        None
    };
    let key_camelot = if emit_key {
        key_result
            .as_ref()
            .and_then(|kr| perceptual::camelot(kr.key, kr.mode))
            .map(|c| c.to_string())
    } else {
        None
    };

    // ================================================================
    // STRUCTURE (opt-in): energy curve + novelty segmentation
    // Reuses per-frame RMS/centroid/bandwidth and the mel dB spectrogram
    // already computed above — no extra decode or FFT pass.
    // ================================================================
    // --- structure ---
    let structure = if extended && config.wants("structure") && n_frames > 0 {
        let fps = sr_f / hop_length as Float;
        Some(crate::structure::analyze_structure(
            rms_frames.as_slice().unwrap(),
            centroids.as_slice().unwrap(),
            bandwidths.as_slice().unwrap_or(&[]),
            s_db.view(),
            dct_matrix.view(),
            &onset_frames,
            fps,
            duration_sec,
        ))
    } else {
        None
    };

    // ================================================================
    // OPT-IN FEATURES (computed by Full mode or an explicit `features=[...]`
    // request; never by Compact or Playlist — performance-first policy)
    // ================================================================

    // --- silence ---
    // Nearly free: pure arithmetic over the RMS frames already computed above.
    // Kept opt-in per the performance-first policy so Compact and Playlist are unchanged.
    let (leading_silence_sec, trailing_silence_sec) = if config.wants("silence") {
        let rms_slice = rms_frames.as_slice().unwrap();
        let (lead, trail) = silence_offsets(rms_slice, sr, hop_length, -60.0);
        (Some(lead), Some(trail))
    } else {
        (None, None)
    };

    // --- key candidates ---
    // Requires chroma (resolved as an extended-pass dependency).
    let key_candidates = if config.wants("key_candidates") {
        chroma_mean.as_ref().map(|c| {
            perceptual::detect_key_candidates(c)
                .into_iter()
                .map(|kc| (kc.key, kc.camelot.to_string(), kc.score))
                .collect::<Vec<_>>()
        })
    } else {
        None
    };

    // --- vocalness / instrumentalness (heuristic v2, 0.2.4) ---
    // Mid-band spectral contrast, from the extended pass. Voice and (especially)
    // screamed/broadband vocals fill the ~0.8-5.6 kHz spectral valleys → LOW
    // peak-to-valley contrast → HIGH vocalness; clean solo pitched instruments
    // leave deep valleys → HIGH contrast → LOW vocalness. A lift-only flatness
    // term nudges harsh/broadband material further up. instrumentalness (still
    // the inverse) shares the value; we compute it whenever either is wanted and
    // split the emission. Requires spectral_contrast_mean + spectral_flatness_mean
    // (extended pass) — hence both are extended in FEATURE_REGISTRY. See `crate::vocal`
    // for the superseded v1 mel-based heuristic.
    let vocalness_val = if config.wants("vocalness") || config.wants("instrumentalness") {
        vocalness_heuristic_v2(
            spectral_contrast_mean.as_deref(),
            spectral_flatness_mean,
            rms_mean,
        )
    } else {
        None
    };
    let vocalness = if config.wants("vocalness") {
        vocalness_val
    } else {
        None
    };
    let instrumentalness = if config.wants("instrumentalness") {
        vocalness_val.map(|v| (1.0 - v).clamp(0.0, 1.0))
    } else {
        None
    };

    let mut result = TrackAnalysis {
        provenance: AnalysisProvenance {
            schema_version: ANALYSIS_SCHEMA_VERSION,
            sample_rate: sr,
            hop_length,
            mode: config.mode,
            requested_features: config.requested_feature_names(),
            bpm_min: config.bpm_min,
            bpm_max: config.bpm_max,
            genre_model_id: config.genre_model.as_ref().and_then(|m| m.id.clone()),
            vocalness_model_id: config.vocalness_model.as_ref().map(|m| m.id().to_string()),
            #[cfg(feature = "aggression")]
            aggression_model_id: config
                .needs_aggression()
                .then(|| crate::aggression::AGGRESSION_MODEL_ID.to_owned()),
        },
        duration_sec,
        bpm,
        bpm_raw,
        bpm_confidence,
        bpm_candidates,
        beats,
        onset_frames,
        rms_mean,
        rms_max,
        loudness_lufs,
        dynamic_range_db,
        // --- loudness ---
        true_peak_db,
        replaygain_db,
        loudness_curve,
        loudness_momentary_max_db,
        loudness_range_lu,
        // --- end loudness ---
        // File-only: filled by analyze_file when requested; a bare signal has
        // no file channels or native rate.
        ebur128_integrated_lufs: None,
        ebur128_momentary_max_lufs: None,
        ebur128_short_term_max_lufs: None,
        ebur128_range_lu: None,
        ebur128_true_peak_dbtp: None,
        native_sample_rate: None,
        sample_peak_dbfs: None,
        crest_factor_db: None,
        crest_factor_frame_mean_db: None,
        amplitude_envelope: None,
        amplitude_envelope_hop_sec: None,
        amplitude_envelope_frame_sec: None,
        stereo_side_ratio: None,
        stereo_side_ratio_low: None,
        stereo_correlation_mean: None,
        stereo_correlation_p05: None,
        spectral_centroid_mean: centroid_mean,
        zero_crossing_rate: zcr,
        onset_density,
        onset_band_edges_hz,
        onset_strength_bands,
        spectral_bandwidth_mean,
        spectral_rolloff_mean,
        spectral_flatness_mean,
        spectral_contrast_mean,
        mfcc_mean,
        chroma_mean,
        rms_std,
        rms_delta_mean,
        rms_iqr,
        rms_p10,
        rms_p90,
        zero_crossing_rate_std,
        spectral_centroid_std,
        spectral_centroid_delta_mean,
        spectral_centroid_iqr,
        spectral_bandwidth_std,
        spectral_bandwidth_delta_mean,
        spectral_rolloff_std,
        spectral_rolloff_p10,
        spectral_rolloff_p90,
        spectral_flatness_std,
        spectral_flatness_iqr,
        spectral_flatness_p10,
        spectral_flatness_p90,
        spectral_contrast_std,
        mfcc_std,
        mfcc_delta_mean,
        chroma_std,
        chroma_delta_mean,
        spectral_skewness_mean: shape_means[0],
        spectral_kurtosis_excess_mean: shape_means[1],
        spectral_slope_db_per_octave_mean: shape_means[2],
        spectral_decrease_mean: shape_means[3],
        spectral_flux_mean: texture_means[0],
        spectral_flux_std,
        spectral_crest_mean: texture_means[1],
        spectral_entropy_mean: texture_means[2],
        spectral_irregularity_mean: texture_means[3],
        mel_bands_mean,
        mel_bands_std,
        mel_bands_floor_db,
        tuning_offset_cents,
        tuning_confidence,
        hpss_percussive_ratio,
        tempo_curve,
        tempo_variability,
        time_signature,
        time_signature_confidence,
        chord_sequence,
        chord_events,
        chord_change_rate,
        predominant_chord,
        dissonance: dissonance_val,
        energy,
        danceability,
        key,
        key_confidence,
        key_camelot,
        valence,
        acousticness,
        // Filled below when the similarity embedding is wanted.
        embedding: None,
        #[cfg(feature = "aggression")]
        aggression_score: aggression_analysis
            .as_ref()
            .and_then(|analysis| analysis.score),
        #[cfg(feature = "aggression")]
        aggression_confidence: aggression_analysis
            .as_ref()
            .map(|analysis| analysis.confidence),
        #[cfg(feature = "aggression")]
        aggression_forcefulness: aggression_analysis
            .as_ref()
            .map(|analysis| analysis.forcefulness),
        #[cfg(feature = "aggression")]
        aggression_harshness: aggression_analysis
            .as_ref()
            .map(|analysis| analysis.harshness),
        #[cfg(feature = "aggression")]
        aggression_tension: aggression_analysis
            .as_ref()
            .map(|analysis| analysis.tension),
        #[cfg(feature = "aggression")]
        aggression_rhythm: aggression_analysis.as_ref().map(|analysis| analysis.rhythm),
        // Mood + instrumentalness: heuristic v1 (opt-in), None unless computed.
        mood_happy: mood.as_ref().map(|m| m.happy),
        mood_aggressive: mood.as_ref().map(|m| m.aggressive),
        mood_relaxed: mood.as_ref().map(|m| m.relaxed),
        mood_sad: mood.as_ref().map(|m| m.sad),
        instrumentalness,
        // genre + genre_confidence: populated below iff a genre model is set.
        genre: None,
        genre_confidence: None,

        // --- beat grid ---
        grid_offset_sec,
        downbeats,
        grid_stability,
        // --- rhythmic regularity ---
        rhythmic_regularity,
        rhythmic_regularity_label,
        rhythmic_regularity_confidence,
        rhythmic_regularity_candidates,
        // --- rhythm levels ---
        rhythm_level_double_ratio,
        rhythm_level_one_and_half_ratio,
        rhythm_level_confidence,
        // --- structure ---
        energy_curve: structure.as_ref().map(|s| s.energy_curve.clone()),
        energy_curve_hop_sec: structure.as_ref().map(|s| s.energy_curve_hop_sec),
        segments: structure.as_ref().map(|s| s.segments.clone()),
        intro_end_sec: structure.as_ref().map(|s| s.intro_end_sec),
        outro_start_sec: structure.as_ref().map(|s| s.outro_start_sec),
        energy_level: structure.as_ref().map(|s| s.energy_level),
        // --- silence ---
        leading_silence_sec,
        trailing_silence_sec,
        // --- key candidates ---
        key_candidates,
        // --- vocalness ---
        vocalness,
        // --- fingerprint ---
        fingerprint,
        // --- similarity ---
        embedding_version: None,
        // --- tags ---
        // Populated by analyze_file when requested; analyze_signal has no file.
        tags: None,
    };

    // --- similarity ---
    // Populate the hand-crafted similarity vector only when opted in (Full mode
    // or `features=["embedding"]`). This keeps compact/playlist unchanged
    // and adds near-zero cost (the vector is assembled from features already
    // computed above). A future ML embedding can replace `embed` behind the same
    // version field.
    // Compute the embedding whenever it is needed — an explicit request OR a
    // genre model that classifies over it.
    if config.needs_embedding() {
        let emb = crate::similarity::embed(&result);
        // Run the user-supplied genre model, if any (the version match was
        // verified up front). Populate genre + genre_confidence.
        if let Some(ref model) = config.genre_model {
            let (label, conf) = model.try_predict(&emb)?;
            result.genre = Some(label);
            result.genre_confidence = Some(conf);
        }
        // Run the user-supplied vocalness model, if any: its calibrated
        // P(vocal) overrides the built-in heuristic (and its inverse).
        if let Some(ref model) = config.vocalness_model {
            let v = model.try_predict_vocalness(&emb)?;
            result.vocalness = Some(v);
            result.instrumentalness = Some((1.0 - v).clamp(0.0, 1.0));
        }
        // Surface the embedding fields only when the feature is wanted — a
        // model uses the vector internally without leaking it.
        if config.wants("embedding") {
            result.embedding = Some(emb);
            result.embedding_version = Some(crate::similarity::SIMILARITY_VERSION);
        }
    }

    // Internal models reuse computed components, but only explicitly requested
    // groups belong in the returned TrackAnalysis.
    if config.has_internal_embedding_consumer() || {
        #[cfg(feature = "aggression")]
        {
            config.needs_aggression()
        }
        #[cfg(not(feature = "aggression"))]
        {
            false
        }
    } {
        suppress_internal_embedding_components(&mut result, config);
    } else if config.features.is_some() {
        suppress_unrequested_spectral_summaries(&mut result, config);
        // Computed for mood even when not requested (see `AnalysisConfig::wants`).
        if !config.emits("dissonance") {
            result.dissonance = None;
        }
    }
    if config.needs_rhythmic_regularity() {
        suppress_internal_rhythm_components(&mut result, config);
    }

    Ok(result)
}

/// Drop the rhythm groups `rhythmic_regularity` computed for its own use.
///
/// Requesting the measure forces `onset_bands` and `beatgrid` through
/// `AnalysisConfig::wants`; only groups the caller actually asked for belong
/// in the returned result.
fn suppress_internal_rhythm_components(result: &mut TrackAnalysis, config: &AnalysisConfig) {
    if !config.emits("onset_bands") {
        result.onset_band_edges_hz = None;
        result.onset_strength_bands = None;
    }
    if !config.emits("beatgrid") {
        result.grid_offset_sec = None;
        result.downbeats = None;
        result.grid_stability = None;
    }
}

/// Drop the spectral summaries the extended pass computed but nobody requested.
///
/// The pass computes all six for its own features (`energy`, `key`,
/// `vocalness`, ...); only the groups the caller named belong in the result.
fn suppress_unrequested_spectral_summaries(result: &mut TrackAnalysis, config: &AnalysisConfig) {
    if !config.emits("bandwidth") {
        result.spectral_bandwidth_mean = None;
        result.spectral_bandwidth_std = None;
        result.spectral_bandwidth_delta_mean = None;
    }
    if !config.emits("rolloff") {
        result.spectral_rolloff_mean = None;
        result.spectral_rolloff_std = None;
        result.spectral_rolloff_p10 = None;
        result.spectral_rolloff_p90 = None;
    }
    if !config.emits("flatness") {
        result.spectral_flatness_mean = None;
        result.spectral_flatness_std = None;
        result.spectral_flatness_iqr = None;
        result.spectral_flatness_p10 = None;
        result.spectral_flatness_p90 = None;
    }
    if !config.emits("contrast") {
        result.spectral_contrast_mean = None;
        result.spectral_contrast_std = None;
    }
    if !config.emits("mfcc") {
        result.mfcc_mean = None;
        result.mfcc_std = None;
        result.mfcc_delta_mean = None;
    }
    if !config.emits("chroma") {
        result.chroma_mean = None;
        result.chroma_std = None;
        result.chroma_delta_mean = None;
    }
    if !config.emits("spectral_shape") {
        result.spectral_skewness_mean = None;
        result.spectral_kurtosis_excess_mean = None;
        result.spectral_slope_db_per_octave_mean = None;
        result.spectral_decrease_mean = None;
    }
    if !config.emits("spectral_texture") {
        result.spectral_flux_mean = None;
        result.spectral_flux_std = None;
        result.spectral_crest_mean = None;
        result.spectral_entropy_mean = None;
        result.spectral_irregularity_mean = None;
    }
    if !config.emits("mel_bands") {
        result.mel_bands_mean = None;
        result.mel_bands_std = None;
        result.mel_bands_floor_db = None;
    }
    if !config.emits("tuning") {
        result.tuning_offset_cents = None;
        result.tuning_confidence = None;
    }
}

fn suppress_internal_embedding_components(result: &mut TrackAnalysis, config: &AnalysisConfig) {
    suppress_unrequested_spectral_summaries(result, config);
    if !config.emits("chords") {
        result.chord_sequence = None;
        result.chord_events = None;
        result.chord_change_rate = None;
        result.predominant_chord = None;
    }
    if !config.emits("dissonance") {
        result.dissonance = None;
    }
    if !config.emits("energy") {
        result.energy = None;
    }
    if !config.emits("danceability") {
        result.danceability = None;
    }
    if !config.emits("key") && !config.emits("valence") {
        result.key = None;
        result.key_confidence = None;
        result.key_camelot = None;
    }
    if !config.emits("valence") {
        result.valence = None;
    }
}

#[cfg(feature = "aggression")]
fn aggression_quantile(values: &[Float], fraction: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(Float::total_cmp);
    let position = fraction.clamp(0.0, 1.0) * (sorted.len() - 1) as Float;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let weight = position - lower as Float;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

#[cfg(feature = "aggression")]
fn aggression_sorted(values: &[Float]) -> Vec<Float> {
    let mut sorted = values.to_vec();
    sorted.sort_by(Float::total_cmp);
    sorted
}

#[cfg(feature = "aggression")]
fn aggression_quantile_sorted(sorted: &[Float], fraction: Float) -> Float {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = fraction.clamp(0.0, 1.0) * (sorted.len() - 1) as Float;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let weight = position - lower as Float;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

#[cfg(feature = "aggression")]
fn aggression_evenly_sample(values: &[Float], limit: usize) -> Vec<Float> {
    if values.len() <= limit {
        return values.to_vec();
    }
    let last = values.len() - 1;
    (0..limit)
        .map(|index| {
            let position = index * last;
            values[(position + (limit - 1) / 2) / (limit - 1)]
        })
        .collect()
}

#[cfg(feature = "aggression")]
fn aggression_interval_cv(frames: &[usize]) -> Float {
    if frames.len() < 3 {
        return 0.0;
    }
    let intervals = frames
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as Float)
        .collect::<Vec<_>>();
    let mean = intervals.iter().sum::<Float>() / intervals.len() as Float;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = intervals
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<Float>()
        / intervals.len() as Float;
    variance.sqrt() / mean
}

#[cfg(feature = "aggression")]
fn aggression_grid_regularity(beats: &[usize]) -> Float {
    if beats.len() < 3 {
        return 0.0;
    }
    let intervals = beats
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as Float)
        .collect::<Vec<_>>();
    let mean = intervals.iter().sum::<Float>() / intervals.len() as Float;
    if mean <= 0.0 {
        return 0.0;
    }
    let variance = intervals
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<Float>()
        / intervals.len() as Float;
    (1.0 - variance.sqrt() / mean).clamp(0.0, 1.0)
}

#[cfg(feature = "aggression")]
fn aggression_window_summaries(
    crest: &[Float],
    high_energy: &[Float],
    high_flatness: &[Float],
    peak_ratio: &[Float],
    onset: &[Float],
    sr: u32,
    hop_length: usize,
) -> (Float, Float, Float, Float) {
    let length = crest
        .len()
        .min(high_energy.len())
        .min(high_flatness.len())
        .min(peak_ratio.len())
        .min(onset.len());
    if length == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let block = ((20.0 * sr as Float / hop_length as Float).round() as usize).max(1);
    let step = (block / 2).max(1);
    let mut starts = (0..=length.saturating_sub(block))
        .step_by(step)
        .collect::<Vec<_>>();
    let final_start = length.saturating_sub(block);
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    let mut windows = Vec::with_capacity(starts.len());
    for start in starts {
        let end = (start + block).min(length);
        let onset_slice = &onset[start..end];
        let threshold = (aggression_quantile(onset_slice, 0.50) + 0.25).max(0.30);
        let duration = (end - start) as Float * hop_length as Float / sr as Float;
        let onset_density = onset_slice
            .iter()
            .filter(|value| **value >= threshold)
            .count() as Float
            / duration.max(Float::EPSILON);
        let force = ((1.0
            - (aggression_quantile(&crest[start..end], 0.50) / 20.0).clamp(0.0, 1.0))
            + (onset_density / 15.0).clamp(0.0, 1.0)
            + (aggression_quantile(onset_slice, 0.50) / 2.0).clamp(0.0, 1.0))
            / 3.0;
        let harshness = ((aggression_quantile(&high_energy[start..end], 0.50) / 0.35)
            .clamp(0.0, 1.0)
            + aggression_quantile(&high_flatness[start..end], 0.50).clamp(0.0, 1.0)
            + 1.0
            - aggression_quantile(&peak_ratio[start..end], 0.50).clamp(0.0, 1.0))
            / 3.0;
        windows.push((force, harshness, force * harshness));
    }
    let top_two = |index: usize| {
        let mut values = windows
            .iter()
            .map(|window| [window.0, window.1, window.2][index])
            .collect::<Vec<_>>();
        values.sort_by(|left, right| right.total_cmp(left));
        values.iter().take(2).sum::<Float>() / values.len().min(2) as Float
    };
    let impact_persistence =
        windows.iter().filter(|window| window.2 >= 0.25).count() as Float / windows.len() as Float;
    (top_two(0), top_two(1), impact_persistence, top_two(2))
}

// ============================================================
// Silence offsets (opt-in)
// ============================================================

/// Leading/trailing silence duration (seconds) from per-frame RMS.
///
/// A frame counts as silent when its RMS is below `threshold_db` dBFS relative to
/// full scale (amplitude `10^(threshold_db/20)`; default -60 dBFS ≈ 0.001).
///
/// Hysteresis rule: leading silence ends at the first frame that *begins a
/// sustained run* of at least `HYST_FRAMES` consecutive above-threshold frames.
/// A single loud click surrounded by silence is shorter than the run and is
/// therefore ignored — it does not terminate the silence. Trailing silence is
/// the symmetric quantity measured from the end.
///
/// Returns `(leading_sec, trailing_sec)`, each clamped to `[0, duration]`.
fn silence_offsets(
    rms: &[Float],
    sr: u32,
    hop_length: usize,
    threshold_db: Float,
) -> (Float, Float) {
    /// Consecutive above-threshold frames required to count as real audio onset.
    const HYST_FRAMES: usize = 3;

    let n = rms.len();
    let sec_per_frame = hop_length as Float / sr as Float;
    if n == 0 {
        return (0.0, 0.0);
    }
    let thresh = 10.0_f32.powf(threshold_db / 20.0);
    let need = HYST_FRAMES.min(n);

    // Leading: first index that starts a sustained above-threshold run.
    let mut lead_frames = n; // all-silence fallback
    for i in 0..n {
        if rms[i] >= thresh {
            let end = (i + need).min(n);
            if (end - i) >= need && (i..end).all(|k| rms[k] >= thresh) {
                lead_frames = i;
                break;
            }
        }
    }

    // Trailing: last index that ends a sustained above-threshold run.
    let mut trail_frames = n; // all-silence fallback
    for i in (0..n).rev() {
        // A run ending at `i` needs `i + 1 >= need`; check it before `i + 1 - need`
        // so a loud frame near the start cannot underflow.
        if rms[i] >= thresh && i + 1 >= need && (i + 1 - need..=i).all(|k| rms[k] >= thresh) {
            trail_frames = n - 1 - i;
            break;
        }
    }

    let dur = n as Float * sec_per_frame;
    let lead = (lead_frames as Float * sec_per_frame).clamp(0.0, dur);
    let trail = (trail_frames as Float * sec_per_frame).clamp(0.0, dur);
    (lead, trail)
}

// ============================================================
// Convenience constructors
// ============================================================

/// Shorthand for compact mode analysis.
pub fn compact() -> AnalysisConfig {
    AnalysisConfig {
        mode: AnalysisMode::Compact,
        ..Default::default()
    }
}

/// Shorthand for playlist mode analysis.
pub fn playlist() -> AnalysisConfig {
    AnalysisConfig {
        mode: AnalysisMode::Playlist,
        ..Default::default()
    }
}

/// Shorthand for full mode analysis.
pub fn full() -> AnalysisConfig {
    AnalysisConfig {
        mode: AnalysisMode::Full,
        ..Default::default()
    }
}

/// Build merged [`ChordEvent`]s from per-window labels. Windows are the
/// `tonal::chord_boundaries` spans that produced `chords` (so the two stay
/// aligned by construction); runs of identical labels merge, the first event
/// starts at 0.0 and the last ends at `duration_sec` (the STFT tail past the
/// final frame belongs to the last chord).
fn chord_events_from_labels(
    chords: &[String],
    beats: &[usize],
    n_frames: usize,
    sr_f: Float,
    hop_length: usize,
    duration_sec: Float,
) -> Vec<ChordEvent> {
    if chords.is_empty() || beats.is_empty() {
        return Vec::new();
    }
    let boundaries = crate::tonal::chord_boundaries(beats, n_frames);
    debug_assert_eq!(boundaries.len(), chords.len() + 1);
    let to_sec = |frame: usize| frame as Float * hop_length as Float / sr_f;

    let mut events: Vec<ChordEvent> = Vec::new();
    for (i, label) in chords.iter().enumerate() {
        let start_sec = if i == 0 { 0.0 } else { to_sec(boundaries[i]) };
        let end_sec = to_sec(boundaries[i + 1]);
        match events.last_mut() {
            Some(last) if last.label == *label => last.end_sec = end_sec,
            _ => events.push(ChordEvent {
                label: label.clone(),
                start_sec,
                end_sec,
            }),
        }
    }
    if let Some(last) = events.last_mut() {
        last.end_sec = duration_sec;
    }
    events
}

/// Vocal-presence heuristic v2 (0.2.4) — the single implementation shared by
/// the fused pipeline and [`augment_analysis`], so a decode-free recompute is
/// bit-identical to a pipeline run by construction.
///
/// Peak-to-valley contrast at `C_HI` reads as fully instrumental, at `C_LO` as
/// fully vocal. Bands 2..=4 cover ~0.8-5.6 kHz (geometric edges over
/// [200, sr/2], see the contrast-band setup in `analyze_signal_inner`).
/// Returns `None` without usable contrast (defensive; the extended pass always
/// computes it when vocalness is wanted) — no vocalness rather than a wrong one.
fn vocalness_heuristic_v2(
    spectral_contrast_mean: Option<&[Float]>,
    spectral_flatness_mean: Option<Float>,
    rms_mean: Float,
) -> Option<Float> {
    const C_HI: Float = 2.05;
    const C_LO: Float = 1.35;
    spectral_contrast_mean.and_then(|c| {
        if c.len() < 5 {
            return None;
        }
        // Degenerate guard: on (near-)silence every band floors to the same
        // noise value, so mid-band contrast collapses to ~0 and the formula
        // below would read it as maximally vocal. Silence carries no vocal
        // energy → report 0.0 (instrumentalness then 1.0), not a wrong high.
        if rms_mean < 1e-5 {
            return Some(0.0);
        }
        let c_mid = (c[2] + c[3] + c[4]) / 3.0;
        let v_contrast = ((C_HI - c_mid) / (C_HI - C_LO)).clamp(0.0, 1.0);
        // Lift-only harsh/broadband boost from spectral flatness.
        let flat = spectral_flatness_mean.unwrap_or(0.0);
        let lift = 0.15 * ((flat - 0.02) / 0.05).clamp(0.0, 1.0);
        Some((v_contrast + lift).min(1.0))
    })
}

// ============================================================
// Augment: recompute named features onto a cached record
// ============================================================

/// Why a feature cannot be recomputed decode-free from a given cached
/// [`TrackAnalysis`] — the machine-readable reason behind a `false` from
/// [`can_augment`]. See [`augment_blocker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentBlocker {
    /// The name is not in the public feature registry (see
    /// [`analysis_feature_names`]).
    UnknownFeature,
    /// The feature's [`DependencyClass`] is `Audio` or `FrameCurves`: its
    /// inputs are not persisted on any record, so recomputation needs the
    /// decoded audio again (the [`augment_analysis`] audio fallback).
    NeedsAudio(DependencyClass),
    /// The record was produced under a different result schema; its stored
    /// fields may not mean what a recompute expects. Re-analyze instead of
    /// augmenting.
    SchemaVersionMismatch { record: u32, current: u32 },
    /// The record carries an embedding of a different layout version than
    /// this build's [`crate::similarity::SIMILARITY_VERSION`];
    /// embedding-consuming recomputation refuses to silently mix eras.
    EmbeddingVersionMismatch { record: u32, current: u32 },
    /// Decode-free evidence fields absent/empty on this record. Evidence is
    /// per-record, not per-feature: emission suppression means two records of
    /// the same schema version can differ in what they carry.
    MissingEvidence(Vec<&'static str>),
}

impl AugmentBlocker {
    /// Human-readable reason, prefixed with the feature name (used by
    /// [`augment_analysis`]'s no-audio error).
    fn describe(&self, feature: &str) -> String {
        match self {
            Self::UnknownFeature => format!("{feature}: unknown feature"),
            Self::NeedsAudio(class) => {
                format!("{feature}: {class:?}-class feature needs the decoded audio")
            }
            Self::SchemaVersionMismatch { record, current } => {
                format!("{feature}: record schema_version {record} != current {current}")
            }
            Self::EmbeddingVersionMismatch { record, current } => {
                format!("{feature}: record embedding_version {record} != current {current}")
            }
            Self::MissingEvidence(fields) => {
                format!("{feature}: missing evidence field(s) {}", fields.join(", "))
            }
        }
    }
}

/// Is a declared evidence field actually present on this record?
///
/// `Option` fields must be `Some`; `Option<Vec<_>>` fields must additionally
/// be non-empty (a `Some(vec![])` carries no evidence). Always-computed core
/// fields (`bpm`, `beats`, `onset_frames`, the scalar means) count as present
/// — an empty `beats`/`onset_frames` is legitimate data (e.g. a drone), not
/// absence. Unknown names fail closed (`false`); the registry tripwire test
/// pins every declared evidence name to a real field.
fn evidence_present(record: &TrackAnalysis, field: &str) -> bool {
    match field {
        // Always-computed core fields.
        "duration_sec"
        | "bpm"
        | "beats"
        | "onset_frames"
        | "onset_density"
        | "rms_mean"
        | "dynamic_range_db"
        | "loudness_lufs"
        | "spectral_centroid_mean" => true,
        "spectral_bandwidth_mean" => record.spectral_bandwidth_mean.is_some(),
        "spectral_rolloff_mean" => record.spectral_rolloff_mean.is_some(),
        "spectral_flatness_mean" => record.spectral_flatness_mean.is_some(),
        "spectral_contrast_mean" => record
            .spectral_contrast_mean
            .as_ref()
            .is_some_and(|v| !v.is_empty()),
        "mfcc_mean" => record.mfcc_mean.as_ref().is_some_and(|v| !v.is_empty()),
        "chroma_mean" => record.chroma_mean.as_ref().is_some_and(|v| !v.is_empty()),
        "dissonance" => record.dissonance.is_some(),
        "chord_change_rate" => record.chord_change_rate.is_some(),
        "key" => record.key.is_some(),
        "energy" => record.energy.is_some(),
        "danceability" => record.danceability.is_some(),
        "valence" => record.valence.is_some(),
        _ => false,
    }
}

/// Blocker for assembling the similarity embedding from this record — the
/// evidence + version check shared by the `embedding` feature and the
/// embedding-consuming model paths (genre / vocalness models).
fn embedding_evidence_blocker(record: &TrackAnalysis) -> Option<AugmentBlocker> {
    if let Some(version) = record.embedding_version {
        if version != crate::similarity::SIMILARITY_VERSION {
            return Some(AugmentBlocker::EmbeddingVersionMismatch {
                record: version,
                current: crate::similarity::SIMILARITY_VERSION,
            });
        }
    }
    let missing: Vec<&'static str> = EMBEDDING_EVIDENCE
        .iter()
        .copied()
        .filter(|field| !evidence_present(record, field))
        .collect();
    if missing.is_empty() {
        None
    } else {
        Some(AugmentBlocker::MissingEvidence(missing))
    }
}

/// Why `feature` cannot be recomputed decode-free from `cached`, or `None`
/// when [`augment_analysis`] can do it without audio.
///
/// The check mirrors the double-check pattern of
/// [`crate::aggression::score_versioned`]: class and evidence are validated
/// against this record AND the record's versions are validated against this
/// build (`schema_version` vs [`ANALYSIS_SCHEMA_VERSION`]; a recorded
/// `embedding_version` vs [`crate::similarity::SIMILARITY_VERSION`] for the
/// embedding class).
///
/// This answers for the *built-in* computation of the feature. A configured
/// `vocalness_model` changes `vocalness`/`instrumentalness` evidence to the
/// `embedding` feature's (the model classifies the embedding) — that
/// config-dependent variant is evaluated inside [`augment_analysis`].
pub fn augment_blocker(cached: &TrackAnalysis, feature: &str) -> Option<AugmentBlocker> {
    let Some(spec) = feature_spec(feature) else {
        return Some(AugmentBlocker::UnknownFeature);
    };
    if cached.provenance.schema_version != ANALYSIS_SCHEMA_VERSION {
        return Some(AugmentBlocker::SchemaVersionMismatch {
            record: cached.provenance.schema_version,
            current: ANALYSIS_SCHEMA_VERSION,
        });
    }
    match spec.class {
        DependencyClass::Audio | DependencyClass::FrameCurves => {
            Some(AugmentBlocker::NeedsAudio(spec.class))
        }
        DependencyClass::Embedding => embedding_evidence_blocker(cached),
        DependencyClass::Scalars => {
            let missing: Vec<&'static str> = spec
                .required_evidence
                .iter()
                .copied()
                .filter(|field| !evidence_present(cached, field))
                .collect();
            if missing.is_empty() {
                None
            } else {
                Some(AugmentBlocker::MissingEvidence(missing))
            }
        }
    }
}

/// Can `feature` be recomputed decode-free from this cached record?
///
/// `true` iff the feature's [`DependencyClass`] is `Scalars`/`Embedding`,
/// every [`FeatureDependency::required_evidence`] field is actually present
/// on **this** record (evidence is per-record: emission suppression means two
/// same-version records can differ), and the record's versions match this
/// build. `false` for unknown names. [`augment_blocker`] returns the reason.
pub fn can_augment(cached: &TrackAnalysis, feature: &str) -> bool {
    augment_blocker(cached, feature).is_none()
}

/// The three key fields exactly as the pipeline formats them.
fn write_key_fields(out: &mut TrackAnalysis, kr: &perceptual::KeyResult) {
    out.key = Some(perceptual::format_key(kr));
    out.key_confidence = Some(kr.confidence);
    out.key_camelot = perceptual::camelot(kr.key, kr.mode).map(|c| c.to_string());
}

/// Decode-free recompute of one feature onto `out` (evidence pre-checked by
/// the caller). Calls the same pure functions the fused pipeline calls, on
/// the cached fields, at the record's own rate — so results are bit-identical
/// to a direct pipeline run over the same audio.
fn recompute_decode_free(
    out: &mut TrackAnalysis,
    name: &str,
    config: &AnalysisConfig,
) -> Result<()> {
    match name {
        "onset_density" => {
            out.onset_density = out.onset_frames.len() as Float / out.duration_sec;
        }
        "energy" => {
            out.energy = Some(perceptual::energy(
                out.rms_mean,
                out.spectral_centroid_mean,
                out.onset_density,
                out.spectral_bandwidth_mean.expect("evidence checked"),
            ));
        }
        "danceability" => {
            // Uses the record's range-folded `bpm` directly, so the recorded
            // bpm_min/bpm_max fold is inherited by construction.
            out.danceability = Some(perceptual::danceability_heuristic(
                out.bpm,
                &out.beats,
                out.onset_density,
            ));
        }
        "key" => {
            let chroma = out.chroma_mean.as_ref().expect("evidence checked");
            let kr = perceptual::detect_key(chroma);
            write_key_fields(out, &kr);
        }
        "valence" => {
            let chroma = out.chroma_mean.as_ref().expect("evidence checked");
            let kr = perceptual::detect_key(chroma);
            out.valence = Some(perceptual::valence(
                &kr,
                out.bpm,
                out.spectral_centroid_mean,
            ));
            // A pipeline run emits the key valence is derived from (`emit_key`).
            write_key_fields(out, &kr);
        }
        "acousticness" => {
            out.acousticness = Some(perceptual::acousticness(
                out.spectral_flatness_mean.expect("evidence checked"),
                out.spectral_rolloff_mean.expect("evidence checked"),
                out.spectral_centroid_mean,
                out.onset_density,
            ));
        }
        "key_candidates" => {
            let chroma = out.chroma_mean.as_ref().expect("evidence checked");
            out.key_candidates = Some(
                perceptual::detect_key_candidates(chroma)
                    .into_iter()
                    .map(|kc| (kc.key, kc.camelot.to_string(), kc.score))
                    .collect(),
            );
        }
        "vocalness" | "instrumentalness" => {
            // One shared value + one provenance field → both fields update
            // together, and the config decides model vs heuristic exactly
            // like `analyze_*` does.
            if let Some(ref model) = config.vocalness_model {
                let emb = crate::similarity::embed(out);
                let v = model.try_predict_vocalness(&emb)?;
                out.vocalness = Some(v);
                out.instrumentalness = Some((1.0 - v).clamp(0.0, 1.0));
                out.provenance.vocalness_model_id = Some(model.id().to_string());
            } else {
                let v = vocalness_heuristic_v2(
                    out.spectral_contrast_mean.as_deref(),
                    out.spectral_flatness_mean,
                    out.rms_mean,
                );
                out.vocalness = v;
                out.instrumentalness = v.map(|value| (1.0 - value).clamp(0.0, 1.0));
                out.provenance.vocalness_model_id = None;
            }
        }
        "mood" => {
            let chroma = out.chroma_mean.as_ref().expect("evidence checked");
            let kr = perceptual::detect_key(chroma);
            // Raw scalar inputs, never the cached energy/danceability fields —
            // mood_scores recomputes those internally, and bit-equality with
            // the pipeline depends on that. The record's `dissonance` is fed
            // as-is (that is why it is in mood's evidence list): the result
            // equals a pipeline run that co-requests `dissonance`.
            let m = perceptual::mood_scores(
                Some(&kr),
                out.bpm,
                out.rms_mean,
                out.spectral_centroid_mean,
                out.onset_density,
                out.spectral_bandwidth_mean.expect("evidence checked"),
                &out.beats,
                out.dissonance,
                out.dynamic_range_db,
            );
            out.mood_happy = Some(m.happy);
            out.mood_aggressive = Some(m.aggressive);
            out.mood_relaxed = Some(m.relaxed);
            out.mood_sad = Some(m.sad);
        }
        "embedding" => {
            let emb = crate::similarity::embed(out);
            out.embedding = Some(emb);
            out.embedding_version = Some(crate::similarity::SIMILARITY_VERSION);
        }
        other => unreachable!("decode-free recompute for non-Scalars/Embedding feature {other}"),
    }
    Ok(())
}

/// Copy the fields a feature emits from a fresh fallback analysis into the
/// augmented clone. Only the named feature's fields move — everything else on
/// `out` is preserved.
fn merge_feature_fields(out: &mut TrackAnalysis, fresh: &TrackAnalysis, name: &str) {
    match name {
        "bpm" => {
            out.bpm = fresh.bpm;
            out.bpm_raw = fresh.bpm_raw;
            out.bpm_confidence = fresh.bpm_confidence;
            out.bpm_candidates = fresh.bpm_candidates.clone();
            // The recomputed bpm was folded with the fallback's effective
            // range; keep the provenance describing the value it carries.
            out.provenance.bpm_min = fresh.provenance.bpm_min;
            out.provenance.bpm_max = fresh.provenance.bpm_max;
        }
        "beats" => out.beats = fresh.beats.clone(),
        "onsets" => out.onset_frames = fresh.onset_frames.clone(),
        "rms" => {
            out.rms_mean = fresh.rms_mean;
            out.rms_max = fresh.rms_max;
            out.rms_std = fresh.rms_std;
            out.rms_delta_mean = fresh.rms_delta_mean;
            out.rms_iqr = fresh.rms_iqr;
            out.rms_p10 = fresh.rms_p10;
            out.rms_p90 = fresh.rms_p90;
        }
        "dynamic_range" => out.dynamic_range_db = fresh.dynamic_range_db,
        "centroid" => {
            out.spectral_centroid_mean = fresh.spectral_centroid_mean;
            out.spectral_centroid_std = fresh.spectral_centroid_std;
            out.spectral_centroid_delta_mean = fresh.spectral_centroid_delta_mean;
            out.spectral_centroid_iqr = fresh.spectral_centroid_iqr;
        }
        "zcr" => {
            out.zero_crossing_rate = fresh.zero_crossing_rate;
            out.zero_crossing_rate_std = fresh.zero_crossing_rate_std;
        }
        "onset_density" => out.onset_density = fresh.onset_density,
        "bandwidth" => {
            out.spectral_bandwidth_mean = fresh.spectral_bandwidth_mean;
            out.spectral_bandwidth_std = fresh.spectral_bandwidth_std;
            out.spectral_bandwidth_delta_mean = fresh.spectral_bandwidth_delta_mean;
        }
        "rolloff" => {
            out.spectral_rolloff_mean = fresh.spectral_rolloff_mean;
            out.spectral_rolloff_std = fresh.spectral_rolloff_std;
            out.spectral_rolloff_p10 = fresh.spectral_rolloff_p10;
            out.spectral_rolloff_p90 = fresh.spectral_rolloff_p90;
        }
        "flatness" => {
            out.spectral_flatness_mean = fresh.spectral_flatness_mean;
            out.spectral_flatness_std = fresh.spectral_flatness_std;
            out.spectral_flatness_iqr = fresh.spectral_flatness_iqr;
            out.spectral_flatness_p10 = fresh.spectral_flatness_p10;
            out.spectral_flatness_p90 = fresh.spectral_flatness_p90;
        }
        "contrast" => {
            out.spectral_contrast_mean = fresh.spectral_contrast_mean.clone();
            out.spectral_contrast_std = fresh.spectral_contrast_std.clone();
        }
        "mfcc" => {
            out.mfcc_mean = fresh.mfcc_mean.clone();
            out.mfcc_std = fresh.mfcc_std.clone();
            out.mfcc_delta_mean = fresh.mfcc_delta_mean.clone();
        }
        "chroma" => {
            out.chroma_mean = fresh.chroma_mean.clone();
            out.chroma_std = fresh.chroma_std.clone();
            out.chroma_delta_mean = fresh.chroma_delta_mean.clone();
        }
        "spectral_shape" => {
            out.spectral_skewness_mean = fresh.spectral_skewness_mean;
            out.spectral_kurtosis_excess_mean = fresh.spectral_kurtosis_excess_mean;
            out.spectral_slope_db_per_octave_mean = fresh.spectral_slope_db_per_octave_mean;
            out.spectral_decrease_mean = fresh.spectral_decrease_mean;
        }
        "spectral_texture" => {
            out.spectral_flux_mean = fresh.spectral_flux_mean;
            out.spectral_flux_std = fresh.spectral_flux_std;
            out.spectral_crest_mean = fresh.spectral_crest_mean;
            out.spectral_entropy_mean = fresh.spectral_entropy_mean;
            out.spectral_irregularity_mean = fresh.spectral_irregularity_mean;
        }
        "mel_bands" => {
            out.mel_bands_mean = fresh.mel_bands_mean.clone();
            out.mel_bands_std = fresh.mel_bands_std.clone();
            out.mel_bands_floor_db = fresh.mel_bands_floor_db;
        }
        "tuning" => {
            out.tuning_offset_cents = fresh.tuning_offset_cents;
            out.tuning_confidence = fresh.tuning_confidence;
        }
        "chords" => {
            out.chord_sequence = fresh.chord_sequence.clone();
            out.chord_events = fresh.chord_events.clone();
            out.chord_change_rate = fresh.chord_change_rate;
            out.predominant_chord = fresh.predominant_chord.clone();
        }
        "dissonance" => out.dissonance = fresh.dissonance,
        "energy" => out.energy = fresh.energy,
        "danceability" => out.danceability = fresh.danceability,
        "key" => {
            out.key = fresh.key.clone();
            out.key_confidence = fresh.key_confidence;
            out.key_camelot = fresh.key_camelot.clone();
        }
        "valence" => {
            // A fresh valence run also emits the key it is derived from.
            out.valence = fresh.valence;
            out.key = fresh.key.clone();
            out.key_confidence = fresh.key_confidence;
            out.key_camelot = fresh.key_camelot.clone();
        }
        "acousticness" => out.acousticness = fresh.acousticness,
        "tempo_curve" => {
            out.tempo_curve = fresh.tempo_curve.clone();
            out.tempo_variability = fresh.tempo_variability;
        }
        "time_signature" => {
            out.time_signature = fresh.time_signature.clone();
            out.time_signature_confidence = fresh.time_signature_confidence;
        }
        "onset_bands" => {
            out.onset_band_edges_hz = fresh.onset_band_edges_hz.clone();
            out.onset_strength_bands = fresh.onset_strength_bands.clone();
        }
        "beatgrid" => {
            out.grid_offset_sec = fresh.grid_offset_sec;
            out.downbeats = fresh.downbeats.clone();
            out.grid_stability = fresh.grid_stability;
        }
        "rhythmic_regularity" => {
            out.rhythmic_regularity = fresh.rhythmic_regularity;
            out.rhythmic_regularity_label = fresh.rhythmic_regularity_label.clone();
            out.rhythmic_regularity_confidence = fresh.rhythmic_regularity_confidence;
            out.rhythmic_regularity_candidates = fresh.rhythmic_regularity_candidates.clone();
        }
        "hpss" => out.hpss_percussive_ratio = fresh.hpss_percussive_ratio,
        "rhythm_levels" => {
            out.rhythm_level_double_ratio = fresh.rhythm_level_double_ratio;
            out.rhythm_level_one_and_half_ratio = fresh.rhythm_level_one_and_half_ratio;
            out.rhythm_level_confidence = fresh.rhythm_level_confidence;
        }
        "structure" => {
            out.energy_curve = fresh.energy_curve.clone();
            out.energy_curve_hop_sec = fresh.energy_curve_hop_sec;
            out.segments = fresh.segments.clone();
            out.intro_end_sec = fresh.intro_end_sec;
            out.outro_start_sec = fresh.outro_start_sec;
            out.energy_level = fresh.energy_level;
        }
        "embedding" => {
            out.embedding = fresh.embedding.clone();
            out.embedding_version = fresh.embedding_version;
        }
        #[cfg(feature = "aggression")]
        "aggression" => {
            out.aggression_score = fresh.aggression_score;
            out.aggression_confidence = fresh.aggression_confidence;
            out.aggression_forcefulness = fresh.aggression_forcefulness;
            out.aggression_harshness = fresh.aggression_harshness;
            out.aggression_tension = fresh.aggression_tension;
            out.aggression_rhythm = fresh.aggression_rhythm;
            out.provenance.aggression_model_id = fresh.provenance.aggression_model_id.clone();
        }
        "fingerprint" => out.fingerprint = fresh.fingerprint.clone(),
        "loudness" => {
            out.true_peak_db = fresh.true_peak_db;
            out.replaygain_db = fresh.replaygain_db;
            out.loudness_curve = fresh.loudness_curve.clone();
            out.loudness_momentary_max_db = fresh.loudness_momentary_max_db;
            out.loudness_range_lu = fresh.loudness_range_lu;
        }
        "ebur128" => {
            out.native_sample_rate = fresh.native_sample_rate;
            out.ebur128_integrated_lufs = fresh.ebur128_integrated_lufs;
            out.ebur128_momentary_max_lufs = fresh.ebur128_momentary_max_lufs;
            out.ebur128_short_term_max_lufs = fresh.ebur128_short_term_max_lufs;
            out.ebur128_range_lu = fresh.ebur128_range_lu;
            out.ebur128_true_peak_dbtp = fresh.ebur128_true_peak_dbtp;
        }
        "native_dynamics" => {
            out.native_sample_rate = fresh.native_sample_rate;
            out.sample_peak_dbfs = fresh.sample_peak_dbfs;
            out.crest_factor_db = fresh.crest_factor_db;
            out.crest_factor_frame_mean_db = fresh.crest_factor_frame_mean_db;
        }
        "amplitude_envelope" => {
            out.native_sample_rate = fresh.native_sample_rate;
            out.amplitude_envelope = fresh.amplitude_envelope.clone();
            out.amplitude_envelope_hop_sec = fresh.amplitude_envelope_hop_sec;
            out.amplitude_envelope_frame_sec = fresh.amplitude_envelope_frame_sec;
        }
        "stereo_field" => {
            out.native_sample_rate = fresh.native_sample_rate;
            out.stereo_side_ratio = fresh.stereo_side_ratio;
            out.stereo_side_ratio_low = fresh.stereo_side_ratio_low;
            out.stereo_correlation_mean = fresh.stereo_correlation_mean;
            out.stereo_correlation_p05 = fresh.stereo_correlation_p05;
        }
        "silence" => {
            out.leading_silence_sec = fresh.leading_silence_sec;
            out.trailing_silence_sec = fresh.trailing_silence_sec;
        }
        "key_candidates" => out.key_candidates = fresh.key_candidates.clone(),
        "vocalness" | "instrumentalness" => {
            // The pair is one shared value; the fallback set always contains
            // both names when either was requested (see augment_analysis).
            out.vocalness = fresh.vocalness;
            out.instrumentalness = fresh.instrumentalness;
        }
        "mood" => {
            out.mood_happy = fresh.mood_happy;
            out.mood_aggressive = fresh.mood_aggressive;
            out.mood_relaxed = fresh.mood_relaxed;
            out.mood_sad = fresh.mood_sad;
        }
        "tags" => out.tags = fresh.tags.clone(),
        other => unreachable!("merge for unregistered feature {other}"),
    }
}

/// Recompute the named features onto a clone of `cached` — decode-free where
/// the record's evidence allows, and via one full audio re-analysis fallback
/// otherwise.
///
/// Semantics (the contract the Python lane binds against):
/// - **Names** resolve case-insensitively through the registry; any unknown
///   name is a hard error (no silent fallback). The list may be empty, which
///   is useful with a genre model (below).
/// - **Decode-free recompute** (`Scalars`/`Embedding` class with all
///   [`FeatureDependency::required_evidence`] present — see [`can_augment`])
///   calls the same pure functions the fused pipeline calls, on the cached
///   fields, at the record's own `sample_rate`/`hop_length`, reproducing the
///   *standalone* meaning of the feature. Features are processed in canonical
///   registry order against the progressively augmented clone, so one call
///   can both recompute `energy` and build the `embedding` that reads it.
/// - **`mood`** is fed the record's `dissonance` value (that is why it is in
///   mood's evidence list); the result equals a pipeline run co-requesting
///   `dissonance`. Its energy/danceability inputs are recomputed internally
///   from raw scalars, never read from the cached `energy`/`danceability`.
/// - **`vocalness`/`instrumentalness`** are one shared value with one
///   provenance field, so requesting either writes *both* fields plus
///   [`AnalysisProvenance::vocalness_model_id`] (the config's model id, or
///   `None` for the built-in heuristic — the config decides, exactly like
///   `analyze_*`). With a `vocalness_model` set, the evidence requirement
///   becomes the `embedding` feature's (the model classifies the embedding).
/// - **Genre** has no registry feature name: carrying
///   [`AnalysisConfig::genre_model`] *is* the request, as in the pipeline.
///   The model predicts over the (freshly assembled) embedding and populates
///   `genre`/`genre_confidence` + `provenance.genre_model_id`. Both models'
///   `embedding_version`s are checked fail-fast up front.
/// - **Audio fallback**: requested features of `Audio`/`FrameCurves` class,
///   or whose evidence is missing on this record, need the audio again. With
///   `audio: Some(path)`, one `analyze_file` run at the **record's**
///   `provenance.sample_rate` (never a caller default — frame-domain fields
///   must stay in the record's rate) computes exactly those features (an
///   `aggression` request thereby routes through its dedicated 22.05 kHz
///   lane automatically), and only the requested fields are merged into the
///   clone. The fallback inherits the record's `bpm_min`/`bpm_max`, falling
///   back to `config`'s range only when the record's provenance carries
///   `None` (records predating range recording). With `audio: None` the call
///   fails, naming each blocked feature and why.
/// - **Version gates**: a `schema_version` mismatch is a hard error for the
///   whole call — patching a stale-schema record would mix field eras;
///   re-analyze instead. A recorded `embedding_version` differing from this
///   build's [`crate::similarity::SIMILARITY_VERSION`] is a hard error for
///   embedding-consuming requests.
/// - **Result**: a clone of `cached` with the recomputed/new fields set;
///   fields the caller did not ask about are never cleared. When any feature
///   was requested, `provenance.requested_features` becomes the sorted union
///   of the record's recorded request and the augmented names (a mode-driven
///   `None` is first expanded to the mode's default feature set, which
///   describes the same emitted fields — an explicit list overrides the mode,
///   so the expansion preserves meaning).
pub fn augment_analysis(
    cached: &TrackAnalysis,
    features: &[&str],
    audio: Option<&Path>,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    // Unknown names are hard errors, mirroring AnalysisConfig::validate_features.
    let mut invalid: Vec<&str> = features
        .iter()
        .copied()
        .filter(|name| canonical_feature_name(name).is_none())
        .collect();
    if !invalid.is_empty() {
        invalid.sort_unstable();
        invalid.dedup();
        return Err(SonaraError::InvalidParameter {
            param: "features",
            reason: format!(
                "unknown feature(s): {}; valid features: {}",
                invalid.join(", "),
                analysis_feature_names().collect::<Vec<_>>().join(", ")
            ),
        });
    }
    // Canonical request set, in registry order (deduplicated by construction).
    let requested: Vec<&'static str> = analysis_feature_names()
        .filter(|canonical| {
            features
                .iter()
                .any(|name| name.eq_ignore_ascii_case(canonical))
        })
        .collect();

    // Fail fast on a bad tempo range or model layout, exactly like the pipeline.
    config.validate()?;
    if cached.provenance.schema_version != ANALYSIS_SCHEMA_VERSION {
        return Err(SonaraError::InvalidParameter {
            param: "cached",
            reason: format!(
                "record schema_version {} does not match this build's ANALYSIS_SCHEMA_VERSION {}; \
                 augmenting would mix field eras — re-analyze the audio instead",
                cached.provenance.schema_version, ANALYSIS_SCHEMA_VERSION
            ),
        });
    }

    let mut out = cached.clone();
    let mut fallback: Vec<(&'static str, AugmentBlocker)> = Vec::new();

    for name in &requested {
        // A configured vocalness model swaps the vocalness/instrumentalness
        // evidence to the embedding's (the model classifies the embedding).
        let blocker = if config.vocalness_model.is_some()
            && matches!(*name, "vocalness" | "instrumentalness")
        {
            embedding_evidence_blocker(&out)
        } else {
            augment_blocker(&out, name)
        };
        match blocker {
            None => recompute_decode_free(&mut out, name, config)?,
            Some(AugmentBlocker::EmbeddingVersionMismatch { record, current }) => {
                return Err(SonaraError::InvalidParameter {
                    param: "cached",
                    reason: format!(
                        "{name}: record embedding_version {record} does not match this build's \
                         SIMILARITY_VERSION {current}; re-analyze the audio instead"
                    ),
                });
            }
            Some(blocker) => fallback.push((name, blocker)),
        }
    }

    // Genre: carrying the model is the request (no registry name exists).
    let mut genre_via_fallback = false;
    if let Some(ref model) = config.genre_model {
        match embedding_evidence_blocker(&out) {
            None => {
                let emb = crate::similarity::embed(&out);
                let (label, confidence) = model.try_predict(&emb)?;
                out.genre = Some(label);
                out.genre_confidence = Some(confidence);
                out.provenance.genre_model_id = model.id.clone();
            }
            Some(AugmentBlocker::EmbeddingVersionMismatch { record, current }) => {
                return Err(SonaraError::InvalidParameter {
                    param: "cached",
                    reason: format!(
                        "genre model: record embedding_version {record} does not match this \
                         build's SIMILARITY_VERSION {current}; re-analyze the audio instead"
                    ),
                });
            }
            Some(_) => genre_via_fallback = true,
        }
    }

    if !fallback.is_empty() || genre_via_fallback {
        let Some(path) = audio else {
            let mut reasons: Vec<String> = fallback
                .iter()
                .map(|(name, blocker)| blocker.describe(name))
                .collect();
            if genre_via_fallback {
                reasons.push("genre model: missing embedding evidence".to_string());
            }
            return Err(SonaraError::InvalidParameter {
                param: "audio",
                reason: format!(
                    "cannot recompute decode-free ({}); pass the audio path to enable the \
                     re-analysis fallback",
                    reasons.join("; ")
                ),
            });
        };
        let mut fallback_names: HashSet<String> = fallback
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect();
        // The vocalness/instrumentalness pair is one shared value with one
        // provenance field: compute + merge both together.
        let wants_vocalness_fallback =
            fallback_names.contains("vocalness") || fallback_names.contains("instrumentalness");
        if wants_vocalness_fallback {
            fallback_names.insert("vocalness".to_string());
            fallback_names.insert("instrumentalness".to_string());
        }
        let merge_names: Vec<String> = {
            let mut names: Vec<String> = fallback_names.iter().cloned().collect();
            names.sort_unstable();
            names
        };
        let fallback_config = AnalysisConfig {
            mode: config.mode,
            features: Some(fallback_names),
            // Inherit the record's folding range so the recomputed bpm keeps
            // the record's meaning; the caller's range applies only when the
            // record predates range recording (provenance None).
            bpm_min: cached.provenance.bpm_min.or(config.bpm_min),
            bpm_max: cached.provenance.bpm_max.or(config.bpm_max),
            // Models ride along only when their output is still needed from
            // this run — otherwise they would force an extended pass for
            // nothing (and re-do work already done decode-free above).
            genre_model: if genre_via_fallback {
                config.genre_model.clone()
            } else {
                None
            },
            vocalness_model: if wants_vocalness_fallback {
                config.vocalness_model.clone()
            } else {
                None
            },
        };
        // The record's sample rate, NOT a caller default: frame-index fields
        // and frame-derived values must stay in the record's rate domain.
        let fresh = analyze_file(path, cached.provenance.sample_rate, &fallback_config)?;
        for name in &merge_names {
            merge_feature_fields(&mut out, &fresh, name);
        }
        if genre_via_fallback {
            out.genre = fresh.genre.clone();
            out.genre_confidence = fresh.genre_confidence;
            out.provenance.genre_model_id = fresh.provenance.genre_model_id.clone();
        }
        if wants_vocalness_fallback {
            out.provenance.vocalness_model_id = fresh.provenance.vocalness_model_id.clone();
        }
    }

    // Provenance: record the union of what this record now carries. A pure
    // genre-model call (empty `features`) changes no feature set, so the
    // recorded request stays untouched (`genre_model_id` records the model).
    if !requested.is_empty() {
        let mut names: Vec<String> = match &cached.provenance.requested_features {
            Some(list) => list.clone(),
            None => {
                // Mode-driven record: expand to the mode's default feature
                // set, which describes the same emitted fields (an explicit
                // list overrides the mode, so the expansion preserves
                // meaning).
                let probe = AnalysisConfig {
                    mode: cached.provenance.mode,
                    ..Default::default()
                };
                // Only a record measured from a file (it then carries
                // `native_sample_rate`) holds the file-only groups.
                const FILE_ONLY: [&str; 5] = [
                    "ebur128",
                    "native_dynamics",
                    "amplitude_envelope",
                    "stereo_field",
                    "tags",
                ];
                let from_file = cached.native_sample_rate.is_some();
                analysis_feature_names()
                    .filter(|name| probe.emits(name) && (from_file || !FILE_ONLY.contains(name)))
                    .map(str::to_owned)
                    .collect()
            }
        };
        names.extend(requested.iter().map(|name| (*name).to_string()));
        names.sort_unstable();
        names.dedup();
        out.provenance.requested_features = Some(names);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    #[test]
    fn test_analyze_compact() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.duration_sec > 1.9 && result.duration_sec < 2.1);
        assert!(result.bpm > 30.0 && result.bpm < 320.0);
        assert!(result.rms_mean > 0.0);
        assert!(result.spectral_centroid_mean > 0.0);
        // Compact: no extended features
        assert!(result.spectral_bandwidth_mean.is_none());
        assert!(result.mfcc_mean.is_none());
        assert!(result.energy.is_none());
    }

    #[test]
    fn test_public_signal_validation_rejects_invalid_audio_in_every_mode() {
        for mode in [AnalysisMode::Compact, AnalysisMode::Full] {
            let config = AnalysisConfig {
                mode,
                ..Default::default()
            };
            let empty = Array1::<Float>::zeros(0);
            assert!(matches!(
                analyze_signal(empty.view(), 22050, &config),
                Err(SonaraError::InvalidAudio(_))
            ));

            let valid = Array1::<Float>::zeros(2048);
            assert!(matches!(
                analyze_signal(valid.view(), 0, &config),
                Err(SonaraError::InvalidParameter { param: "sr", .. })
            ));

            for invalid in [Float::NAN, Float::INFINITY, Float::NEG_INFINITY] {
                let mut signal = valid.clone();
                signal[17] = invalid;
                assert!(matches!(
                    analyze_signal(signal.view(), 22050, &config),
                    Err(SonaraError::InvalidAudio(_))
                ));
            }
        }
    }

    #[test]
    fn test_validated_zero_crossing_rate_preserves_existing_semantics() {
        let signals = [
            vec![0.0],
            vec![-1.0, 0.0, 1.0, 0.0, -1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ];
        for samples in signals {
            let signal = Array1::from_vec(samples);
            let expected = audio::zero_crossings(signal.view(), 0.0)
                .iter()
                .filter(|&&crossing| crossing)
                .count() as Float
                / signal.len() as Float;
            assert_eq!(
                validated_zero_crossing_rate(signal.view()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn test_feature_registry_validates_routes_and_canonicalizes() {
        let mut expected = vec![
            "bpm",
            "beats",
            "onsets",
            "rms",
            "dynamic_range",
            "centroid",
            "zcr",
            "onset_density",
            "bandwidth",
            "rolloff",
            "flatness",
            "contrast",
            "mfcc",
            "chroma",
            "spectral_shape",
            "spectral_texture",
            "mel_bands",
            "tuning",
            "chords",
            "dissonance",
            "energy",
            "danceability",
            "key",
            "valence",
            "acousticness",
            "tempo_curve",
            "time_signature",
            "onset_bands",
            "beatgrid",
            "rhythmic_regularity",
            "hpss",
            "rhythm_levels",
            "structure",
            "embedding",
            "fingerprint",
            "loudness",
            "ebur128",
            "native_dynamics",
            "amplitude_envelope",
            "stereo_field",
            "silence",
            "key_candidates",
            "vocalness",
            "mood",
            "instrumentalness",
            "tags",
        ];
        if cfg!(feature = "aggression") {
            let embedding = expected.iter().position(|&name| name == "embedding").unwrap();
            expected.insert(embedding + 1, "aggression");
        }
        assert_eq!(analysis_feature_names().collect::<Vec<_>>(), expected);
        let unique: HashSet<_> = analysis_feature_names().collect();
        assert_eq!(unique.len(), expected.len());

        let config = AnalysisConfig {
            features: Some(expected.iter().map(|name| name.to_string()).collect()),
            ..Default::default()
        };
        config.validate_features().unwrap();
        for &name in &expected {
            assert!(config.wants(name), "feature did not route: {name}");
        }
        assert_eq!(config.requested_feature_names().unwrap(), {
            let mut names: Vec<String> = expected.iter().map(|name| name.to_string()).collect();
            names.sort_unstable();
            names
        });

        let mixed = AnalysisConfig {
            features: Some(
                ["KeY", "EnErGy"]
                    .iter()
                    .map(|name| name.to_string())
                    .collect(),
            ),
            ..Default::default()
        };
        mixed.validate_features().unwrap();
        assert!(mixed.wants("key"));
        assert!(mixed.wants("energy"));
        assert_eq!(
            mixed.requested_feature_names().as_deref(),
            Some(&["energy".to_string(), "key".to_string()][..])
        );

        // The dependency map is the same registry viewed through another lens:
        // it must cover exactly the feature names, in canonical order, and its
        // lookup must canonicalize case like the rest of the registry.
        assert_eq!(
            feature_dependencies().map(|d| d.name).collect::<Vec<_>>(),
            expected
        );
        let key_dep = feature_dependency("KeY").expect("case-insensitive lookup");
        assert_eq!(key_dep.name, "key");
        assert!(feature_dependency("keyy").is_none());
    }

    #[test]
    fn test_feature_dependency_map_classes_and_evidence() {
        // Representative features of each class (the full table is encoded in
        // the registry; these pin one of each so a class flip cannot pass).
        let class_of = |name: &str| feature_dependency(name).unwrap().class;
        assert_eq!(class_of("zcr"), DependencyClass::Audio);
        assert_eq!(class_of("fingerprint"), DependencyClass::Audio);
        assert_eq!(class_of("loudness"), DependencyClass::Audio);
        assert_eq!(class_of("ebur128"), DependencyClass::Audio);
        assert_eq!(class_of("native_dynamics"), DependencyClass::Audio);
        assert_eq!(class_of("amplitude_envelope"), DependencyClass::Audio);
        assert_eq!(class_of("stereo_field"), DependencyClass::Audio);
        assert_eq!(class_of("tags"), DependencyClass::Audio);
        assert_eq!(class_of("bpm"), DependencyClass::FrameCurves);
        assert_eq!(class_of("chroma"), DependencyClass::FrameCurves);
        assert_eq!(class_of("silence"), DependencyClass::FrameCurves);
        assert_eq!(class_of("structure"), DependencyClass::FrameCurves);
        assert_eq!(class_of("onset_bands"), DependencyClass::FrameCurves);
        assert_eq!(
            class_of("rhythmic_regularity"),
            DependencyClass::FrameCurves
        );
        assert_eq!(class_of("key"), DependencyClass::Scalars);
        assert_eq!(class_of("energy"), DependencyClass::Scalars);
        assert_eq!(class_of("vocalness"), DependencyClass::Scalars);
        assert_eq!(class_of("mood"), DependencyClass::Scalars);
        assert_eq!(class_of("embedding"), DependencyClass::Embedding);
        #[cfg(feature = "aggression")]
        assert_eq!(class_of("aggression"), DependencyClass::Audio);

        // The routing flags surface the registry's booleans verbatim; pin one
        // representative of each flag so a registry flip cannot pass silently.
        let dep_of = |name: &str| feature_dependency(name).unwrap();
        assert!(dep_of("key").needs_extended);
        assert!(!dep_of("bpm").needs_extended);
        assert!(dep_of("beatgrid").opt_in_only && !dep_of("beatgrid").needs_extended);
        assert!(dep_of("onset_bands").opt_in_only && !dep_of("onset_bands").needs_extended);
        assert!(
            dep_of("rhythmic_regularity").opt_in_only
                && !dep_of("rhythmic_regularity").needs_extended
                && !dep_of("rhythmic_regularity").full_only
        );
        assert!(
            !dep_of("rhythm_levels").opt_in_only
                && !dep_of("rhythm_levels").needs_extended
                && !dep_of("rhythm_levels").full_only
        );
        assert!(!dep_of("energy").opt_in_only);
        assert!(dep_of("tempo_curve").full_only);
        assert!(!dep_of("key").full_only);
        assert!(dep_of("embedding").needs_extended && dep_of("embedding").opt_in_only);

        // Spot-pin two evidence lists the plan's consumers key on.
        assert_eq!(
            feature_dependency("key").unwrap().required_evidence,
            &["chroma_mean"]
        );
        assert_eq!(
            feature_dependency("vocalness").unwrap().required_evidence,
            &[
                "spectral_contrast_mean",
                "spectral_flatness_mean",
                "rms_mean"
            ]
        );

        // Evidence lists are non-empty exactly for the decode-free classes,
        // and every listed name is a real TrackAnalysis field (typo tripwire).
        let known_fields: HashSet<&str> = [
            "duration_sec",
            "bpm",
            "beats",
            "onset_frames",
            "onset_density",
            "rms_mean",
            "dynamic_range_db",
            "loudness_lufs",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean",
            "spectral_flatness_mean",
            "spectral_contrast_mean",
            "mfcc_mean",
            "chroma_mean",
            "dissonance",
            "chord_change_rate",
            "key",
            "energy",
            "danceability",
            "valence",
        ]
        .into_iter()
        .collect();
        for dep in feature_dependencies() {
            let decode_free = matches!(
                dep.class,
                DependencyClass::Scalars | DependencyClass::Embedding
            );
            assert_eq!(
                !dep.required_evidence.is_empty(),
                decode_free,
                "evidence must be non-empty exactly for Scalars/Embedding: {}",
                dep.name
            );
            for field in dep.required_evidence {
                assert!(
                    known_fields.contains(field),
                    "unknown evidence field {field} on {}",
                    dep.name
                );
            }
        }
    }

    #[test]
    fn test_unknown_feature_reports_canonical_allowed_list() {
        let config = AnalysisConfig {
            features: Some(["keyy".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let y = Array1::<Float>::zeros(2048);
        match analyze_signal(y.view(), 22050, &config) {
            Err(SonaraError::InvalidParameter { param, reason }) => {
                assert_eq!(param, "features");
                assert!(reason.contains("unknown feature(s): keyy"), "{reason}");
                assert!(reason.contains("valid features: bpm, beats"), "{reason}");
            }
            Err(other) => panic!("expected invalid feature error, got {other}"),
            Ok(_) => panic!("expected invalid feature error, got Ok"),
        }
    }

    #[test]
    fn test_mixed_case_features_emit_finite_results() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            features: Some(
                ["KeY", "EnErGy"]
                    .iter()
                    .map(|name| name.to_string())
                    .collect(),
            ),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        assert!(result.energy.is_some_and(Float::is_finite));
        assert!(result.key.is_some());
        assert!(result.duration_sec.is_finite());
        assert!(result.bpm.is_finite());
        assert!(result.rms_mean.is_finite());
        assert_eq!(
            result.provenance.requested_features.as_deref(),
            Some(&["energy".to_string(), "key".to_string()][..])
        );
    }

    #[test]
    fn onset_bands_are_absent_by_default_and_aligned_when_requested() {
        let y = sine(440.0, 22050, 2.0);
        for config in [compact(), playlist()] {
            let result = analyze_signal(y.view(), 22050, &config).unwrap();
            assert!(result.onset_band_edges_hz.is_none());
            assert!(result.onset_strength_bands.is_none());
        }

        let config = AnalysisConfig {
            features: Some(["onset_bands".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        let edges = result.onset_band_edges_hz.unwrap();
        let envelopes = result.onset_strength_bands.unwrap();
        assert_eq!(edges, vec![0.0, 200.0, 800.0, 3200.0, 8000.0, 11025.0]);
        assert_eq!(envelopes.len(), edges.len() - 1);
        assert!(envelopes
            .iter()
            .all(|band| band.len() == envelopes[0].len()));
        assert!(envelopes
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert_eq!(
            result.provenance.requested_features.as_deref(),
            Some(&["onset_bands".to_string()][..])
        );
    }

    /// Four-on-the-floor kick with offbeat hats at a fixed tempo — a straight
    /// pattern whose kick and hat land in different onset bands.
    fn four_on_the_floor(sr: u32, bpm: Float, bars: usize) -> Array1<Float> {
        let sr_f = sr as Float;
        let beat_sec = 60.0 / bpm;
        let total = ((bars * 4) as Float * beat_sec * sr_f) as usize + sr as usize / 4;
        let mut y = Array1::<Float>::zeros(total);
        let mut lcg = 0x2545_f491_u32;
        for beat in 0..bars * 4 {
            let onset = (beat as Float * beat_sec * sr_f) as usize;
            for k in 0..(0.12 * sr_f) as usize {
                if onset + k >= total {
                    break;
                }
                let t = k as Float / sr_f;
                y[onset + k] += (2.0 * PI * 55.0 * t).sin() * (-t / 0.045).exp();
            }
            let offbeat = onset + (0.5 * beat_sec * sr_f) as usize;
            for k in 0..(0.03 * sr_f) as usize {
                if offbeat + k >= total {
                    break;
                }
                // Deterministic broadband burst: a fixed LCG, never `rand`.
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let t = k as Float / sr_f;
                let sample = (lcg >> 16) as Float / 32_768.0 - 1.0;
                y[offbeat + k] += 0.25 * sample * (-t / 0.006).exp();
            }
        }
        y
    }

    #[test]
    fn rhythmic_regularity_is_absent_by_default_and_complete_when_requested() {
        let y = four_on_the_floor(22050, 128.0, 16);
        for config in [compact(), playlist()] {
            let result = analyze_signal(y.view(), 22050, &config).unwrap();
            assert!(result.rhythmic_regularity.is_none());
            assert!(result.rhythmic_regularity_label.is_none());
            assert!(result.rhythmic_regularity_confidence.is_none());
            assert!(result.rhythmic_regularity_candidates.is_none());
        }

        let config = AnalysisConfig {
            features: Some(["rhythmic_regularity".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();

        let score = result.rhythmic_regularity.expect("score");
        let label = result.rhythmic_regularity_label.clone().expect("label");
        let confidence = result.rhythmic_regularity_confidence.expect("confidence");
        let candidates = result
            .rhythmic_regularity_candidates
            .clone()
            .expect("candidates");
        assert!((0.0..=1.0).contains(&score), "score {score} out of range");
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence {confidence} out of range"
        );
        assert_eq!(
            label,
            if score >= 0.5 { "regular" } else { "irregular" }
        );
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].1 >= candidates[1].1, "candidates must rank");
        assert_eq!(candidates[0].0, label);
        assert_eq!(
            label, "regular",
            "a four-on-the-floor kick must read as regular, got {score}"
        );

        assert_eq!(
            result.provenance.requested_features.as_deref(),
            Some(&["rhythmic_regularity".to_string()][..])
        );
    }

    // Bars and regularity are computed in 4/4 whether or not `time_signature`
    // runs in the same pass (it always does in Full): the meter detector reads
    // four-on-the-floor as "2/4" and does not set the bar length.
    #[test]
    fn beat_grid_and_regularity_do_not_follow_the_detected_meter() {
        let y = four_on_the_floor(22050, 128.0, 16);
        let alone = analyze_signal(
            y.view(),
            22050,
            &feature_config(&["beatgrid", "rhythmic_regularity"]),
        )
        .unwrap();
        assert!(alone.downbeats.as_ref().is_some_and(|d| d.len() >= 2));
        assert!(alone.rhythmic_regularity.is_some());
        for with_meter in [
            feature_config(&["beatgrid", "rhythmic_regularity", "time_signature"]),
            full(),
        ] {
            let r = analyze_signal(y.view(), 22050, &with_meter).unwrap();
            assert!(r.time_signature.is_some(), "the detector must have run");
            assert_eq!(r.downbeats, alone.downbeats);
            assert_eq!(r.rhythmic_regularity, alone.rhythmic_regularity);
            assert_eq!(r.rhythmic_regularity_label, alone.rhythmic_regularity_label);
            assert_eq!(
                r.rhythmic_regularity_confidence,
                alone.rhythmic_regularity_confidence
            );
            assert_eq!(
                r.rhythmic_regularity_candidates,
                alone.rhythmic_regularity_candidates
            );
        }
    }

    #[test]
    fn rhythmic_regularity_abstains_on_drumless_audio_but_reports_confidence() {
        // A pure tone has no percussive structure to distribute in a grid.
        let y = sine(440.0, 22050, 20.0);
        let config = AnalysisConfig {
            features: Some(["rhythmic_regularity".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        assert!(
            result.rhythmic_regularity.is_none(),
            "a drumless tone must not be scored: {:?}",
            result.rhythmic_regularity
        );
        assert!(result.rhythmic_regularity_label.is_none());
        assert!(result.rhythmic_regularity_candidates.is_none());
        let confidence = result
            .rhythmic_regularity_confidence
            .expect("confidence is reported even when the measure abstains");
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence {confidence} out of range"
        );
        assert!(
            confidence < 0.5,
            "abstention must come with low confidence, got {confidence}"
        );
    }

    #[test]
    fn rhythmic_regularity_does_not_leak_the_groups_it_reads() {
        let y = four_on_the_floor(22050, 128.0, 16);
        let alone = AnalysisConfig {
            features: Some(["rhythmic_regularity".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &alone).unwrap();
        assert!(result.rhythmic_regularity.is_some());
        assert!(
            result.onset_strength_bands.is_none() && result.onset_band_edges_hz.is_none(),
            "onset bands are an internal input here, not a requested output"
        );
        assert!(
            result.grid_stability.is_none()
                && result.downbeats.is_none()
                && result.grid_offset_sec.is_none(),
            "the beat grid is an internal input here, not a requested output"
        );

        let co_requested = AnalysisConfig {
            features: Some(
                [
                    "rhythmic_regularity".to_string(),
                    "onset_bands".to_string(),
                    "beatgrid".to_string(),
                ]
                .into_iter()
                .collect(),
            ),
            ..Default::default()
        };
        let both = analyze_signal(y.view(), 22050, &co_requested).unwrap();
        assert!(both.onset_strength_bands.is_some() && both.grid_stability.is_some());
        assert_eq!(
            both.rhythmic_regularity, result.rhythmic_regularity,
            "emitting the inputs must not change the measurement"
        );
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures")).join(name)
    }

    #[test]
    fn test_analyze_signal_strided_view_matches_contiguous_copy() {
        // NumPy `analyze_signal(y[::2])` panicked inside Rust instead of analysing.
        let (y, sr) = crate::core::audio::load(&fixture("tagged.flac"), 0, true, 0.0, 0.0).unwrap();
        let config = AnalysisConfig::default();
        for view in [y.slice(ndarray::s![..;2]), y.slice(ndarray::s![..;-1])] {
            let copy = ndarray::Array1::from(view.to_vec());
            assert_eq!(
                analyze_signal(view, sr, &config).unwrap(),
                analyze_signal(copy.view(), sr, &config).unwrap()
            );
        }
    }

    #[test]
    fn test_analyze_file_tags_populated() {
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(["tags"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let result = analyze_file(&fixture("tagged.flac"), 22050, &config).unwrap();
        let t = result.tags.expect("tags requested → Some");
        assert_eq!(t.title.as_deref(), Some("Test Title"));
        assert_eq!(t.artist.as_deref(), Some("Test Artist"));
        assert_eq!(t.album.as_deref(), Some("Test Album"));
        assert_eq!(t.genre.as_deref(), Some("Electronic"));
        assert_eq!(t.year, Some(2024));
        assert_eq!(t.track_no, Some(3));
        // tags must NOT trigger the extended DSP pass.
        assert!(
            result.mfcc_mean.is_none(),
            "tags must not enable extended features"
        );
        assert!(result.energy.is_none());
        assert!(result.chroma_mean.is_none());
        // The model-predicted `genre` field is distinct from the tag genre.
        assert!(result.genre.is_none());
    }

    #[test]
    fn test_analyze_file_default_no_tags() {
        let result = analyze_file(&fixture("tagged.flac"), 22050, &compact()).unwrap();
        assert!(result.tags.is_none(), "tags not requested → None");
        // Full reads them, along with the other file-only groups.
        let result = analyze_file(&fixture("tagged.flac"), 22050, &full()).unwrap();
        assert_eq!(result.tags.unwrap().title.as_deref(), Some("Test Title"));
        assert!(result.native_sample_rate.is_some());
    }

    fn tiny_vocalness_model(id: &str, ev: u32) -> crate::vocal_model::VocalnessModel {
        // 48 → 2 softmax with all-zero weights: P(vocal) is exactly 0.5 for
        // any input, which no heuristic ever emits for a pure sine — an
        // unambiguous marker that the model produced the score.
        let row: Vec<String> = (0..48).map(|_| "0.0".to_string()).collect();
        let json = format!(
            r#"{{"format_version":1,"embedding_version":{ev},"id":"{id}",
                 "labels":["instrumental","vocal"],
                 "layers":[{{"weights":[[{r}],[{r}]],"bias":[0.0,0.0],"activation":"softmax"}}]}}"#,
            r = row.join(","),
        );
        crate::vocal_model::from_json_str(&json).unwrap()
    }

    #[test]
    fn test_vocalness_model_overrides_and_stamps_provenance() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            vocalness_model: Some(std::sync::Arc::new(tiny_vocalness_model(
                "vocal-test-v1",
                crate::similarity::SIMILARITY_VERSION,
            ))),
            ..Default::default()
        };
        // Setting the model must force the extended pass on its own.
        assert!(config.needs_extended());
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        assert_eq!(result.vocalness, Some(0.5));
        assert_eq!(result.instrumentalness, Some(0.5));
        assert_eq!(
            result.provenance.vocalness_model_id.as_deref(),
            Some("vocal-test-v1")
        );
        // The embedding itself must not leak without an explicit request.
        assert!(result.embedding.is_none());
    }

    #[test]
    fn test_vocalness_model_version_mismatch_fails_fast() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            vocalness_model: Some(std::sync::Arc::new(tiny_vocalness_model(
                "stale",
                crate::similarity::SIMILARITY_VERSION + 1,
            ))),
            ..Default::default()
        };
        match analyze_signal(y.view(), 22050, &config) {
            Err(SonaraError::ModelError(msg)) => {
                assert!(msg.contains("embedding_version"), "got: {msg}")
            }
            Err(e) => panic!("expected ModelError, got {e}"),
            Ok(_) => panic!("expected fail-fast ModelError, got Ok"),
        }
    }

    #[test]
    fn test_no_model_leaves_provenance_ids_none() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.provenance.vocalness_model_id.is_none());
        assert!(result.provenance.genre_model_id.is_none());
        #[cfg(feature = "aggression")]
        assert!(result.provenance.aggression_model_id.is_none());
    }

    #[cfg(feature = "aggression")]
    #[test]
    fn test_aggression_ignores_the_bpm_range_fold() {
        // The in-pass lane (22050 Hz) fed the model the range-folded tempo and the
        // beats retracked at it, while the canonical lane tracks without a range,
        // so a folding range moved the score at 22050 Hz only.
        let (y, sr) = audio::load(&fixture("corrupt.mp3"), 22050, true, 0.0, 0.0).unwrap();
        let free = analyze_signal(y.view(), sr, &feature_config(&["aggression"])).unwrap();
        let mut folding = feature_config(&["aggression"]);
        folding.bpm_min = Some(60.0);
        folding.bpm_max = Some(119.0);
        let folded = analyze_signal(y.view(), sr, &folding).unwrap();
        // The fixture tracks at ~35 BPM; 60-119 doubles it and retracks the beats.
        assert!(folded.bpm != free.bpm, "the range must fold the tempo");
        assert!(
            folded.beats != free.beats,
            "the folded tempo must retrack the beats"
        );
        assert_eq!(folded.aggression_score, free.aggression_score);
        assert_eq!(folded.aggression_rhythm, free.aggression_rhythm);
        assert_eq!(folded.aggression_forcefulness, free.aggression_forcefulness);
        assert_eq!(folded.aggression_harshness, free.aggression_harshness);
        assert_eq!(folded.aggression_tension, free.aggression_tension);
    }

    #[cfg(feature = "aggression")]
    #[test]
    fn test_aggression_uses_fused_evidence_without_dependency_leakage() {
        let y = sine(440.0, 22050, 2.0);
        let aggression_only = AnalysisConfig {
            features: Some(HashSet::from(["aggression".to_owned()])),
            ..Default::default()
        };
        assert!(aggression_only.needs_extended());
        let result = analyze_signal(y.view(), 22050, &aggression_only).unwrap();
        let score = result.aggression_score.expect("aggression requested");
        assert!((0.0..=1.0).contains(&score));
        assert!(result.aggression_confidence.is_some());
        assert!(result.aggression_forcefulness.is_some());
        assert!(result.aggression_harshness.is_some());
        assert!(result.aggression_tension.is_some());
        assert!(result.aggression_rhythm.is_some());
        assert_eq!(
            result.provenance.aggression_model_id.as_deref(),
            Some(crate::aggression::AGGRESSION_MODEL_ID)
        );
        assert!(result.embedding.is_none());
        assert!(result.embedding_version.is_none());
        assert!(result.mfcc_mean.is_none());
        assert!(result.chroma_mean.is_none());
        assert!(result.spectral_contrast_mean.is_none());
        assert!(result.energy.is_none());
        assert!(result.danceability.is_none());
        assert!(result.key.is_none());
        assert!(result.valence.is_none());
        assert!(result.dissonance.is_none());
        assert!(result.chord_sequence.is_none());

        let with_embedding = AnalysisConfig {
            features: Some(HashSet::from([
                "aggression".to_owned(),
                "embedding".to_owned(),
            ])),
            ..Default::default()
        };
        let both = analyze_signal(y.view(), 22050, &with_embedding).unwrap();
        assert_eq!(
            score.to_bits(),
            both.aggression_score
                .expect("aggression requested")
                .to_bits()
        );
        // The retained embedding scorer is explicitly legacy-v1 and does not
        // define the fused audio rank.
        assert!((0.0..=1.0)
            .contains(&crate::aggression::score(both.embedding.as_deref().unwrap()).unwrap()));
        assert_eq!(
            both.embedding_version,
            Some(crate::similarity::SIMILARITY_VERSION)
        );
    }

    #[cfg(feature = "aggression")]
    fn same_lane_file_config() -> AnalysisConfig {
        AnalysisConfig {
            mode: AnalysisMode::Playlist,
            features: Some(HashSet::from([
                "aggression".to_owned(),
                "embedding".to_owned(),
                "key".to_owned(),
                "key_candidates".to_owned(),
                "loudness".to_owned(),
                "structure".to_owned(),
                "tags".to_owned(),
                "vocalness".to_owned(),
            ])),
            ..Default::default()
        }
    }

    #[cfg(feature = "aggression")]
    fn legacy_same_lane_file(path: &Path, config: &AnalysisConfig) -> TrackAnalysis {
        let (native, native_sr, tags) =
            audio::load_with_tags(path, 0, true, 0.0, 0.0, true).unwrap();
        let canonical = (native_sr != crate::aggression::AGGRESSION_SAMPLE_RATE)
            .then(|| {
                audio::resample(
                    native.view(),
                    native_sr,
                    crate::aggression::AGGRESSION_SAMPLE_RATE,
                )
            })
            .transpose()
            .unwrap();
        let canonical_view = canonical
            .as_ref()
            .map(|audio| audio.view())
            .unwrap_or_else(|| native.view());
        let mut result = analyze_signal_with_precomputed_aggression(
            canonical_view,
            crate::aggression::AGGRESSION_SAMPLE_RATE,
            canonical_view,
            crate::aggression::AGGRESSION_SAMPLE_RATE,
            config,
        )
        .unwrap();
        result.tags = tags;
        result
    }

    #[cfg(feature = "aggression")]
    #[test]
    fn test_aggression_file_same_lane_is_bit_identical_to_two_pass_route() {
        let config = same_lane_file_config();
        let native_canonical = fixture("tagged.flac");
        assert_eq!(
            analyze_file(&native_canonical, 0, &config).unwrap(),
            legacy_same_lane_file(&native_canonical, &config),
        );

        let noncanonical = std::env::temp_dir().join(format!(
            "sonara-aggression-same-lane-{}.wav",
            std::process::id()
        ));
        let specification = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&noncanonical, specification).unwrap();
        for index in 0..44_100 * 2 {
            let time = index as Float / 44_100.0;
            let sample =
                0.4 * (2.0 * PI * 220.0 * time).sin() + 0.2 * (2.0 * PI * 2_317.0 * time).sin();
            writer
                .write_sample((sample * i16::MAX as Float) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        assert_eq!(
            analyze_file(
                &noncanonical,
                crate::aggression::AGGRESSION_SAMPLE_RATE,
                &config,
            )
            .unwrap(),
            legacy_same_lane_file(&noncanonical, &config),
        );
        std::fs::remove_file(noncanonical).unwrap();
    }

    #[test]
    fn test_analyze_signal_never_has_tags() {
        let y = sine(440.0, 22050, 2.0);
        // Even when "tags" is (meaninglessly) requested for a bare signal.
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(["tags"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        assert!(
            result.tags.is_none(),
            "analyze_signal has no file → tags None"
        );
    }

    #[test]
    fn test_analyze_playlist() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert!(result.spectral_bandwidth_mean.unwrap() > 0.0);
        assert!(result.mfcc_mean.unwrap().len() == 13);
        assert!(result.chroma_mean.unwrap().len() == 12);
        assert!(result.energy.unwrap() >= 0.0);
        assert!(result.danceability.unwrap() >= 0.0);
        assert!(result.key.is_some());
        assert!(result.valence.unwrap() >= 0.0);
        assert!(result.acousticness.unwrap() >= 0.0);
    }

    #[test]
    fn test_analyze_accurate_chroma() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        // Chroma should map A440 to bin 9
        let chroma = result.chroma_mean.unwrap();
        assert_eq!(chroma.len(), 12);
        let max_bin = chroma
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_bin, 9,
            "A440 should map to chroma bin 9 (A), got {}",
            max_bin
        );
    }

    /// Sum of equal-amplitude pure sines over `dur` seconds at `sr`.
    fn chord(freqs: &[Float], sr: u32, dur: Float) -> Vec<Float> {
        let n = (sr as Float * dur) as usize;
        (0..n)
            .map(|i| {
                let t = i as Float / sr as Float;
                freqs
                    .iter()
                    .map(|&f| (2.0 * PI * f * t).sin())
                    .sum::<Float>()
                    / freqs.len() as Float
            })
            .collect()
    }

    /// A C-major triad progression — the authentic cadence I–IV–V–I, repeated
    /// `reps` times, as pure sines. The tonic-triad pitch classes dominate:
    /// C appears in 3 of 4 chords, G in 3, E in 2, while every non-tonic-triad
    /// class (A, F, B, D) appears only once — so chroma top-3 is unambiguously
    /// {C, E, G} and the cadence pins the key to C major.
    fn c_major_progression(sr: u32, reps: usize) -> Array1<Float> {
        // I: C5 E5 G5 / IV: F4 A4 C5 / V: G4 B4 D5 / I: C5 E5 G5 — voiced in the
        // well-resolved mid register near the octave-weighting centre (~880 Hz)
        // so a fixed 2048-pt FFT resolves each note at 48 kHz without smearing
        // into neighbouring pitch classes.
        let chords: [&[Float]; 4] = [
            &[523.25, 659.26, 783.99],
            &[349.23, 440.00, 523.25],
            &[392.00, 493.88, 587.33],
            &[523.25, 659.26, 783.99],
        ];
        let mut samples: Vec<Float> = Vec::new();
        for _ in 0..reps {
            for c in &chords {
                samples.extend(chord(c, sr, 1.0));
            }
        }
        Array1::from(samples)
    }

    /// THE regression test for the chroma sr-bias bug: without the librosa
    /// octave-domain weighting in `filters::chroma`, the >11 kHz broadband band
    /// floods chroma at 44.1k/48k and real-music tonality collapses (e.g. every
    /// track reads as F major). A C-major progression must yield chroma top-3
    /// {C,E,G} == bins {0,4,7} and key "C major" at every sample rate.
    #[test]
    fn test_chroma_key_multirate_c_major() {
        for &sr in &[22050u32, 44100, 48000] {
            let y = c_major_progression(sr, 3);
            let r = analyze_signal(y.view(), sr, &feature_config(&["key", "chroma"])).unwrap();

            let chroma = r.chroma_mean.clone().expect("chroma_mean populated");
            let mut idx: Vec<usize> = (0..12).collect();
            idx.sort_by(|&a, &b| chroma[b].partial_cmp(&chroma[a]).unwrap());
            let top3: HashSet<usize> = idx[..3].iter().copied().collect();
            let expected: HashSet<usize> = [0usize, 4, 7].into_iter().collect();
            assert_eq!(
                top3, expected,
                "sr={sr}: chroma top-3 should be C/E/G bins {{0,4,7}}, got {top3:?} (chroma={chroma:?})"
            );

            assert_eq!(
                r.key.clone().unwrap(),
                "C major",
                "sr={sr}: key should be C major, got {:?}",
                r.key
            );
        }
    }

    /// Single-sine invariance: a 440 Hz tone maps to chroma bin 9 (A) at every
    /// supported sample rate — the octave weighting must not shift a pure tone.
    #[test]
    fn test_chroma_single_sine_invariance_multirate() {
        for &sr in &[22050u32, 44100, 48000] {
            let y = sine(440.0, sr, 2.0);
            let r = analyze_signal(y.view(), sr, &feature_config(&["chroma"])).unwrap();
            let chroma = r.chroma_mean.expect("chroma_mean populated");
            let max_bin = chroma
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                max_bin, 9,
                "sr={sr}: A440 should map to bin 9 (A), got {max_bin}"
            );
        }
    }

    #[test]
    fn test_analysis_schema_version_pinned() {
        // Bump deliberately (with a changelog note), never accidentally.
        assert_eq!(ANALYSIS_SCHEMA_VERSION, 8);
    }

    #[test]
    fn test_provenance_populated() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let p = &result.provenance;
        assert_eq!(p.schema_version, ANALYSIS_SCHEMA_VERSION);
        assert_eq!(p.sample_rate, 22050);
        assert_eq!(p.hop_length, HOP_LENGTH);
        assert_eq!(p.mode, AnalysisMode::Compact);
        assert!(p.requested_features.is_none());
        // Frame→seconds via provenance must agree with the beatgrid convention.
        if let Some(&f) = result.beats.first() {
            let via_prov = f as Float * p.hop_length as Float / p.sample_rate as Float;
            let via_grid = crate::beatgrid::grid_offset(&result.beats, 22050, HOP_LENGTH);
            assert!((via_prov - via_grid).abs() < 1e-9);
        }
    }

    #[test]
    fn test_provenance_records_feature_override() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            mode: AnalysisMode::Playlist,
            features: Some(
                ["key", "energy", "chroma"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        let p = &result.provenance;
        assert_eq!(p.mode, AnalysisMode::Playlist);
        // Sorted, so the recorded list is deterministic across runs.
        assert_eq!(
            p.requested_features.as_deref(),
            Some(
                &[
                    "chroma".to_string(),
                    "energy".to_string(),
                    "key".to_string()
                ][..]
            )
        );
    }

    #[test]
    fn test_frame_to_sec_helpers_match_beatgrid() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let beats_sec = result.beats_sec();
        assert_eq!(beats_sec.len(), result.beats.len());
        if !result.beats.is_empty() {
            let via_grid = crate::beatgrid::grid_offset(&result.beats, 22050, HOP_LENGTH);
            assert!((beats_sec[0] - via_grid).abs() < 1e-9);
        }
        for (sec, &frame) in result.onsets_sec().iter().zip(&result.onset_frames) {
            assert!((sec - result.frame_to_sec(frame)).abs() < 1e-9);
            assert!(*sec >= 0.0 && *sec <= result.duration_sec + 0.1);
        }
        assert!(result.downbeats_sec().is_none(), "beatgrid not requested");
    }

    #[test]
    fn test_chord_events_merge_and_cover() {
        let chords: Vec<String> = ["Am", "Am", "C", "C", "C", "G"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // beats[0] > 0 and last beat < n_frames → boundaries = [0] + beats + [n_frames]
        let beats = vec![10, 20, 30, 40, 50];
        let n_frames = 60;
        let (sr_f, hop) = (22050.0, 512);
        let dur = 2.0;
        let events = chord_events_from_labels(&chords, &beats, n_frames, sr_f, hop, dur);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].label, "Am");
        assert_eq!(events[1].label, "C");
        assert_eq!(events[2].label, "G");
        // Contiguous, covering [0, dur]
        assert_eq!(events[0].start_sec, 0.0);
        assert_eq!(events[2].end_sec, dur);
        for w in events.windows(2) {
            assert!(
                (w[0].end_sec - w[1].start_sec).abs() < 1e-9,
                "not contiguous"
            );
        }
        // Merge boundary: Am ends where C starts = boundary frame 20
        let expect = 20.0 * hop as Float / sr_f;
        assert!((events[0].end_sec - expect).abs() < 1e-9);
    }

    #[test]
    fn test_chord_events_populated_with_sequence() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(["chords"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        let seq = result.chord_sequence.expect("chord_sequence requested");
        let events = result
            .chord_events
            .expect("chord_events mirror chord_sequence");
        if seq.is_empty() {
            assert!(events.is_empty());
            return;
        }
        // Events are the run-length merge of the sequence.
        let mut merged: Vec<&String> = Vec::new();
        for label in &seq {
            if merged.last() != Some(&label) {
                merged.push(label);
            }
        }
        let labels: Vec<&String> = events.iter().map(|e| &e.label).collect();
        assert_eq!(labels, merged);
        // Spans are monotone, contiguous, and cover [0, duration].
        assert_eq!(events[0].start_sec, 0.0);
        assert!((events.last().unwrap().end_sec - result.duration_sec).abs() < 1e-6);
        for e in &events {
            assert!(e.end_sec > e.start_sec, "empty span {:?}", e);
        }
        for w in events.windows(2) {
            assert!((w[0].end_sec - w[1].start_sec).abs() < 1e-9);
        }
    }

    #[test]
    fn test_analyze_custom_features() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(
                ["energy", "key", "chroma"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            ..Default::default()
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        // Requested features should be present
        assert!(result.energy.is_some());
        assert!(result.key.is_some());
        assert!(result.chroma_mean.is_some());
        // Non-requested extended features should be absent
        assert!(result.danceability.is_none());
        assert!(result.acousticness.is_none());
    }

    #[test]
    fn test_analyze_config_accepts_runtime_bpm_range() {
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: None,
            bpm_min: Some(79.0),
            bpm_max: Some(192.0),
            genre_model: None,
            vocalness_model: None,
        };
        assert_eq!(config.bpm_min, Some(79.0));
        assert_eq!(config.bpm_max, Some(192.0));
    }

    #[test]
    fn test_provenance_records_configured_bpm_range() {
        let y = sine(440.0, 22050, 2.0);

        // Default config: no range configured, provenance says so.
        let unbounded = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert_eq!(unbounded.provenance.bpm_min, None);
        assert_eq!(unbounded.provenance.bpm_max, None);

        // Configured range round-trips through a real analysis run, so a
        // persisted record can no longer silently diverge on folding bounds.
        let config = AnalysisConfig {
            bpm_min: Some(79.0),
            bpm_max: Some(192.0),
            ..Default::default()
        };
        let bounded = analyze_signal(y.view(), 22050, &config).unwrap();
        assert_eq!(bounded.provenance.bpm_min, Some(79.0));
        assert_eq!(bounded.provenance.bpm_max, Some(192.0));
        // And the recorded range is honored by the folded tempo itself.
        assert!(
            bounded.bpm >= 79.0 && bounded.bpm <= 192.0,
            "bpm {} outside recorded fold range",
            bounded.bpm
        );
    }

    #[test]
    fn test_analyze_exposes_bpm_candidates() {
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let result = analyze_signal(y.view(), sr, &compact()).unwrap();
        assert!(
            !result.bpm_candidates.is_empty(),
            "expected tempo candidates"
        );
        assert!(result.bpm_candidates.len() <= 5);
        // Candidates are sorted by score descending.
        for w in result.bpm_candidates.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "candidates must be sorted by score descending"
            );
        }
        assert!(result.bpm_raw > 30.0 && result.bpm_raw < 320.0);
        // Without a bpm range, the final bpm equals the raw selection.
        assert!((result.bpm - result.bpm_raw).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_click_train() {
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let result = analyze_signal(y.view(), sr, &compact()).unwrap();
        assert!(result.bpm > 50.0 && result.bpm < 250.0);
        assert!(result.onset_frames.len() >= 3);
    }

    #[test]
    fn test_bpm_confidence_present_and_click_vs_drone() {
        let sr = 22050u32;
        // Steady 120-BPM click train: strong, agreeing tempo + dense onsets.
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut clicks = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                clicks[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let r_click = analyze_signal(clicks.view(), sr, &compact()).unwrap();
        // Always present + bounded.
        assert!(
            (0.0..=1.0).contains(&r_click.bpm_confidence),
            "bpm_confidence out of [0,1]: {}",
            r_click.bpm_confidence
        );
        // Steady percussive material should read as reasonably anchored.
        assert!(
            r_click.bpm_confidence > 0.5,
            "click train bpm_confidence should be > 0.5, got {}",
            r_click.bpm_confidence
        );

        // Slow sustained sine drone: sparse onsets, weak tempo evidence.
        let drone = sine(220.0, sr, 4.0);
        let r_drone = analyze_signal(drone.view(), sr, &compact()).unwrap();
        assert!(
            (0.0..=1.0).contains(&r_drone.bpm_confidence),
            "drone bpm_confidence out of [0,1]: {}",
            r_drone.bpm_confidence
        );
        // Comparative: sparse-onset drone is less anchored than the click train.
        assert!(
            r_drone.bpm_confidence < r_click.bpm_confidence,
            "drone ({}) should be less anchored than click train ({})",
            r_drone.bpm_confidence,
            r_click.bpm_confidence
        );
    }

    #[test]
    fn test_analyze_features_reasonable() {
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * PI * 440.0 * i as Float / 22050.0).sin() * 0.5
        });
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(
            result.rms_mean > 0.1 && result.rms_mean < 0.6,
            "RMS {} unexpected",
            result.rms_mean
        );
        assert!(
            result.spectral_centroid_mean > 300.0 && result.spectral_centroid_mean < 600.0,
            "Centroid {} unexpected",
            result.spectral_centroid_mean
        );
    }

    #[test]
    fn test_analyze_playlist_sine_vs_noise() {
        let sine_sig = sine(440.0, 22050, 2.0);
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let noise = Array1::from_shape_fn(44100, |i| {
            let mut h = DefaultHasher::new();
            (i as u64 ^ 0xDEADBEEF).hash(&mut h);
            (h.finish() as Float / u64::MAX as Float) * 2.0 - 1.0
        });

        let cfg = playlist();
        let r_sine = analyze_signal(sine_sig.view(), 22050, &cfg).unwrap();
        let r_noise = analyze_signal(noise.view(), 22050, &cfg).unwrap();

        assert!(r_sine.spectral_flatness_mean.unwrap() < r_noise.spectral_flatness_mean.unwrap());
        assert!(r_sine.spectral_bandwidth_mean.unwrap() < r_noise.spectral_bandwidth_mean.unwrap());
    }

    // ---- structure (opt-in) ----

    fn structure_config() -> AnalysisConfig {
        AnalysisConfig {
            mode: AnalysisMode::Playlist,
            features: Some(["structure"].iter().map(|s| s.to_string()).collect()),
            ..AnalysisConfig::default()
        }
    }

    // ---- opt-in features: silence, key candidates, vocalness ----

    fn feature_config(names: &[&str]) -> AnalysisConfig {
        AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: Some(names.iter().map(|s| s.to_string()).collect()),
            ..AnalysisConfig::default()
        }
    }

    const SR: u32 = 22050;
    const HOP: usize = 512;
    const SPF: Float = HOP as Float / SR as Float; // seconds per frame

    #[test]
    fn test_silence_offsets_leading_trailing() {
        // 65 silent frames, 100 loud, 97 silent → 1.5s lead, 2.25s trail (approx).
        let mut rms = vec![0.0_f32; 65];
        rms.extend(std::iter::repeat(0.5).take(100));
        rms.extend(std::iter::repeat(0.0).take(97));
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert!(
            (lead - 65.0 * SPF).abs() < SPF,
            "lead {} vs {}",
            lead,
            65.0 * SPF
        );
        assert!(
            (trail - 97.0 * SPF).abs() < SPF,
            "trail {} vs {}",
            trail,
            97.0 * SPF
        );
        assert!((lead - 1.5).abs() < 2.0 * SPF);
        assert!((trail - 2.25).abs() < 2.0 * SPF);
    }

    #[test]
    fn test_silence_offsets_no_silence() {
        let rms = vec![0.5_f32; 200];
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert_eq!(lead, 0.0);
        assert_eq!(trail, 0.0);
    }

    #[test]
    fn test_silence_offsets_all_silence() {
        let rms = vec![0.0_f32; 200];
        let dur = 200.0 * SPF;
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        assert!(
            (lead - dur).abs() < 1e-4,
            "leading {} should ~= duration {}",
            lead,
            dur
        );
        assert!((trail - dur).abs() < 1e-4);
    }

    #[test]
    fn test_silence_offsets_click_hysteresis() {
        // 30 silent, 1 loud click, 30 silent, then 100 sustained loud, then silent.
        let mut rms = vec![0.0_f32; 30];
        rms.push(0.9); // isolated click
        rms.extend(std::iter::repeat(0.0).take(30));
        rms.extend(std::iter::repeat(0.5).take(100));
        rms.extend(std::iter::repeat(0.0).take(20));
        let (lead, _trail) = silence_offsets(&rms, SR, HOP, -60.0);
        // Sustained audio starts at frame 61, not at the click (frame 30).
        assert!(
            (lead - 61.0 * SPF).abs() < SPF,
            "click should not end silence: lead {} expected ~{}",
            lead,
            61.0 * SPF
        );
    }

    #[test]
    fn test_silence_offsets_loud_first_frame_does_not_underflow() {
        // The trailing scan reached frame 0 and computed `0 + 1 - 3` before checking
        // that a run could end there: a debug build panicked.
        let rms = [0.9_f32, 0.0, 0.0, 0.0, 0.0];
        let (lead, trail) = silence_offsets(&rms, SR, HOP, -60.0);
        let dur = rms.len() as Float * SPF;
        assert_eq!((lead, trail), (dur, dur), "a lone click is not audio");
    }

    #[test]
    fn test_silence_pipeline_optin_and_bounds() {
        // Real pipeline: 1.5s leading + 2.25s trailing silence around a tone.
        let sr = SR;
        let lead_n = (1.5 * sr as Float) as usize;
        let trail_n = (2.25 * sr as Float) as usize;
        let mid_n = (3.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(lead_n + mid_n + trail_n);
        for i in 0..mid_n {
            y[lead_n + i] = 0.5 * (2.0 * PI * 440.0 * i as Float / sr as Float).sin();
        }
        let r = analyze_signal(y.view(), sr, &feature_config(&["silence"])).unwrap();
        let lead = r.leading_silence_sec.unwrap();
        let trail = r.trailing_silence_sec.unwrap();
        assert!((lead - 1.5).abs() < 0.05, "lead {}", lead);
        assert!((trail - 2.25).abs() < 0.05, "trail {}", trail);
        assert!(lead >= 0.0 && lead <= r.duration_sec);
        assert!(trail >= 0.0 && trail <= r.duration_sec);
    }

    #[test]
    fn test_full_mode_computes_every_feature() {
        let y = sine(440.0, SR, 15.0);
        let r = analyze_signal(y.view(), SR, &full()).unwrap();
        assert!(r.leading_silence_sec.is_some(), "silence");
        assert!(r.key_candidates.is_some(), "key_candidates");
        assert!(r.vocalness.is_some(), "vocalness");
        assert!(r.instrumentalness.is_some(), "instrumentalness");
        assert!(r.mood_happy.is_some(), "mood");
        assert!(r.energy_curve.is_some(), "structure");
        assert!(r.embedding.is_some(), "embedding");
        assert!(r.fingerprint.is_some(), "fingerprint");
        assert!(r.true_peak_db.is_some(), "loudness");
        assert!(r.onset_strength_bands.is_some(), "onset_bands");
        assert!(r.hpss_percussive_ratio.is_some(), "hpss");
        assert!(r.rhythm_level_confidence.is_some(), "rhythm_levels");
        assert!(
            r.rhythmic_regularity_confidence.is_some(),
            "rhythmic_regularity"
        );
        // A mode-driven run records no feature list.
        assert!(r.provenance.requested_features.is_none());

        // Playlist gains rhythm_levels; the opt-in features stay out of it.
        let p = analyze_signal(y.view(), SR, &playlist()).unwrap();
        assert!(p.rhythm_level_confidence.is_some());
        assert!(p.hpss_percussive_ratio.is_none());
        assert!(analyze_signal(y.view(), SR, &compact())
            .unwrap()
            .rhythm_level_confidence
            .is_none());
    }

    // Full turns aggression on without a feature list: in the fused pass at
    // the model's rate, through the separate 22.05 kHz lane off it.
    #[cfg(feature = "aggression")]
    #[test]
    fn test_full_mode_computes_aggression_at_any_rate() {
        for sr in [SR, 44100] {
            let y = sine(440.0, sr, 15.0);
            let r = analyze_signal(y.view(), sr, &full()).unwrap();
            assert!(r.aggression_confidence.is_some(), "{sr}");
            assert!(r.provenance.aggression_model_id.is_some(), "{sr}");
            assert!(r.provenance.requested_features.is_none(), "{sr}");
            assert!(r.time_signature_confidence.is_some(), "full-only kept");
            assert!(r.embedding.is_some(), "opt-in kept");
        }
    }

    #[test]
    fn test_optin_absent_by_default() {
        let y = sine(440.0, SR, 3.0);
        for cfg in [compact(), playlist()] {
            let r = analyze_signal(y.view(), SR, &cfg).unwrap();
            assert!(r.leading_silence_sec.is_none(), "silence must be opt-in");
            assert!(r.trailing_silence_sec.is_none());
            assert!(r.key_candidates.is_none(), "key_candidates must be opt-in");
            assert!(r.vocalness.is_none(), "vocalness must be opt-in");
            assert!(r.mood_happy.is_none(), "mood must be opt-in");
            assert!(r.mood_aggressive.is_none());
            assert!(r.mood_relaxed.is_none());
            assert!(r.mood_sad.is_none());
            assert!(
                r.instrumentalness.is_none(),
                "instrumentalness must be opt-in"
            );
        }
    }

    #[test]
    fn test_mood_pipeline_optin_and_no_leak() {
        let y = sine(440.0, SR, 3.0);
        let r = analyze_signal(y.view(), SR, &feature_config(&["mood"])).unwrap();
        for v in [r.mood_happy, r.mood_aggressive, r.mood_relaxed, r.mood_sad] {
            let v = v.expect("mood_* must be Some when mood requested");
            assert!(
                (0.0..=1.0).contains(&v) && v.is_finite(),
                "mood out of range {}",
                v
            );
        }
        // Requesting mood must NOT leak the key / valence / dissonance fields.
        assert!(r.key.is_none(), "mood must not leak key");
        assert!(r.valence.is_none(), "mood must not leak valence");
        assert!(r.dissonance.is_none(), "mood must not leak dissonance");
    }

    #[test]
    fn test_instrumentalness_pipeline_and_inverse() {
        let y = sine(440.0, SR, 3.0);
        // instrumentalness alone: Some in range, vocalness field stays None.
        let r = analyze_signal(y.view(), SR, &feature_config(&["instrumentalness"])).unwrap();
        let inst = r.instrumentalness.expect("instrumentalness must be Some");
        assert!(
            (0.0..=1.0).contains(&inst) && inst.is_finite(),
            "instrumentalness {}",
            inst
        );
        assert!(r.vocalness.is_none(), "vocalness field must stay None");
        // Both together: instrumentalness == 1 - vocalness.
        let r2 = analyze_signal(
            y.view(),
            SR,
            &feature_config(&["vocalness", "instrumentalness"]),
        )
        .unwrap();
        let v = r2.vocalness.expect("vocalness Some");
        let i = r2.instrumentalness.expect("instrumentalness Some");
        assert!(
            (i - (1.0 - v)).abs() < 1e-5,
            "inst {} should equal 1 - vocalness {}",
            i,
            v
        );
    }

    #[test]
    fn test_structure_is_opt_in() {
        let y = sine(440.0, 22050, 15.0);
        // Playlist mode must NOT compute structure (Full computes everything).
        let r = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert!(
            r.energy_curve.is_none(),
            "structure must be absent by default"
        );
        assert!(r.segments.is_none());
        assert!(r.energy_level.is_none());
        // Compact obviously not.
        let rc = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(rc.energy_curve.is_none());
        // Opt-in via features=["structure"] turns it on.
        let rs = analyze_signal(y.view(), 22050, &structure_config()).unwrap();
        assert!(rs.energy_curve.as_ref().unwrap().len() > 0);
        assert!(rs.segments.is_some());
        assert!(rs.energy_curve_hop_sec.unwrap() > 0.0);
        let lvl = rs.energy_level.unwrap();
        assert!((1..=10).contains(&lvl));
    }

    #[test]
    fn test_structure_pipeline_known_shape() {
        // Synthetic audio with known structure:
        // 30s quiet 200 Hz sine -> 60s loud broadband -> 30s quiet sine.
        let sr = 22050u32;
        let seg = |dur: Float, loud: bool| -> Vec<Float> {
            let n = (dur * sr as Float) as usize;
            (0..n)
                .map(|i| {
                    if loud {
                        // Broadband: sum of several partials, high amplitude.
                        let t = i as Float / sr as Float;
                        0.5 * ((2.0 * PI * 200.0 * t).sin()
                            + (2.0 * PI * 1500.0 * t).sin()
                            + (2.0 * PI * 4000.0 * t).sin())
                            / 3.0
                            * 3.0
                    } else {
                        0.04 * (2.0 * PI * 200.0 * i as Float / sr as Float).sin()
                    }
                })
                .collect()
        };
        let mut samples = seg(30.0, false);
        samples.extend(seg(60.0, true));
        samples.extend(seg(30.0, false));
        let y = Array1::from(samples);

        let r = analyze_signal(y.view(), sr, &structure_config()).unwrap();
        let segs = r.segments.as_ref().unwrap();
        // Covering + ordered + non-overlapping.
        assert!(
            segs.first().unwrap().start_sec.abs() < 1e-2,
            "first segment must start at 0"
        );
        assert!(
            (segs.last().unwrap().end_sec - r.duration_sec).abs() < 0.5,
            "last must end at duration"
        );
        for w in segs.windows(2) {
            assert!(
                (w[0].end_sec - w[1].start_sec).abs() < 1e-2,
                "segments must be contiguous"
            );
        }
        // Boundaries near 30s and 90s.
        let interior: Vec<Float> = segs.iter().skip(1).map(|s| s.start_sec).collect();
        let near = |target: Float| interior.iter().any(|&b| (b - target).abs() < 8.0);
        assert!(
            near(30.0),
            "expected boundary near 30s, interior={:?}",
            interior
        );
        assert!(
            near(90.0),
            "expected boundary near 90s, interior={:?}",
            interior
        );
        // Intro/outro land in the quiet regions.
        assert!(r.intro_end_sec.unwrap() < 45.0);
        assert!(r.outro_start_sec.unwrap() > 80.0);
        // Middle (loud) segment has clearly higher mean energy than the ends.
        let mid = segs
            .iter()
            .find(|s| s.start_sec < 60.0 && s.end_sec > 60.0)
            .map(|s| s.energy)
            .unwrap_or(0.0);
        assert!(
            mid > segs.first().unwrap().energy + 0.15,
            "loud section should be more energetic"
        );
    }

    #[test]
    fn test_key_candidates_pipeline_a_minor() {
        // Synthesized A-minor triad: A(220), C(~261.6), E(~329.6).
        let sr = SR;
        let n = (4.0 * sr as Float) as usize;
        let y = Array1::from_shape_fn(n, |i| {
            let t = i as Float / sr as Float;
            0.5 * (2.0 * PI * 220.0 * t).sin()
                + 0.4 * (2.0 * PI * 261.63 * t).sin()
                + 0.35 * (2.0 * PI * 329.63 * t).sin()
        });
        let r = analyze_signal(y.view(), sr, &feature_config(&["key_candidates"])).unwrap();
        let cands = r.key_candidates.unwrap();
        assert_eq!(cands.len(), 3, "exactly 3 candidates");
        // Scores descending, finite, in [0,1].
        for (_, _, s) in &cands {
            assert!(*s >= 0.0 && *s <= 1.0 && s.is_finite());
        }
        assert!(cands[0].2 >= cands[1].2 && cands[1].2 >= cands[2].2);
        // Camelot codes valid.
        let valid: HashSet<&str> = [
            "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "11A", "12A", "1B", "2B",
            "3B", "4B", "5B", "6B", "7B", "8B", "9B", "10B", "11B", "12B",
        ]
        .into_iter()
        .collect();
        for (_, cam, _) in &cands {
            assert!(valid.contains(cam.as_str()), "invalid camelot {}", cam);
        }
        // First candidate is A minor and matches the separately requested `key`.
        assert_eq!(cands[0].0, "A minor", "got {}", cands[0].0);
        let r2 = analyze_signal(y.view(), sr, &feature_config(&["key", "key_candidates"])).unwrap();
        assert_eq!(r2.key_candidates.unwrap()[0].0, r2.key.unwrap());
    }

    #[test]
    fn test_vocalness_pipeline_in_range() {
        let y = sine(440.0, SR, 3.0);
        let r = analyze_signal(y.view(), SR, &feature_config(&["vocalness"])).unwrap();
        // vocalness now requires the extended pass (mid-band spectral contrast);
        // requesting it alone must trigger that pass and yield Some in [0, 1].
        // This Some doubles as the proof the extended pass ran.
        let v = r
            .vocalness
            .expect("vocalness must be Some (extended pass runs for it)");
        assert!(v >= 0.0 && v <= 1.0 && v.is_finite(), "vocalness {}", v);
    }

    #[test]
    fn test_vocalness_requesting_triggers_extended() {
        // features=["vocalness"] alone must flip needs_extended() on (v2 is
        // contrast-based), yet return only vocalness: the spectral summaries the
        // pass computed for it are not part of the request.
        let cfg = feature_config(&["vocalness"]);
        assert!(
            cfg.needs_extended(),
            "vocalness must require the extended pass"
        );
        let y = sine(440.0, SR, 3.0);
        let r = analyze_signal(y.view(), SR, &cfg).unwrap();
        assert!(r.vocalness.is_some());
        assert!(r.spectral_flatness_mean.is_none() && r.spectral_contrast_mean.is_none());
        assert!(r.mfcc_mean.is_none() && r.chroma_mean.is_none());
        assert!(r.spectral_bandwidth_mean.is_none() && r.spectral_rolloff_mean.is_none());

        // A requested summary is still returned next to the feature it feeds.
        let r = analyze_signal(y.view(), SR, &feature_config(&["vocalness", "flatness"])).unwrap();
        assert!(r.spectral_flatness_mean.is_some() && r.spectral_contrast_mean.is_none());
    }

    #[test]
    fn test_vocalness_broadband_high_tonal_low() {
        // v2 semantics: broadband/flat content fills the mid-band spectral valleys
        // → LOW contrast → HIGH vocalness; harmonically rich tonal content leaves
        // deep valleys → HIGH contrast → LOW vocalness.
        //
        // A pure sine is degenerate for spectral contrast (a single line has no
        // valley structure), so we use a richly harmonic sawtooth-like tone for the
        // "tonal/instrumental" pole — many sharp harmonic peaks over quiet valleys,
        // exactly the deep-valley case the contrast score is built to catch — and
        // white noise for the "broadband/vocal-like" pole. This pair reliably
        // demonstrates the ordering; see the acceptance means in the plan.
        let sr = SR;
        let dur = 4.0;
        let n = (sr as Float * dur) as usize;

        // White noise: flat spectrum, shallow valleys → low contrast → high vocalness.
        let mut noise = Array1::<Float>::zeros(n);
        let mut state: u32 = 0x1234_5678;
        for s in noise.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *s = (state as Float / u32::MAX as Float) * 2.0 - 1.0;
        }
        let rn = analyze_signal(noise.view(), sr, &feature_config(&["vocalness"])).unwrap();
        let v_noise = rn.vocalness.unwrap();

        // Harmonic-rich tone: many sharp partials with deep valleys between →
        // high contrast → low vocalness.
        let mut tone = Array1::<Float>::zeros(n);
        let f0 = 220.0_f32;
        for (i, s) in tone.iter_mut().enumerate() {
            let t = i as Float / sr as Float;
            let mut acc = 0.0;
            for h in 1..=20 {
                acc += (1.0 / h as Float) * (2.0 * PI * f0 * h as Float * t).sin();
            }
            *s = 0.3 * acc;
        }
        let rt = analyze_signal(tone.view(), sr, &feature_config(&["vocalness"])).unwrap();
        let v_tone = rt.vocalness.unwrap();

        assert!(
            v_noise > v_tone,
            "broadband {v_noise} should out-score tonal {v_tone}"
        );
        assert!(
            v_noise > 0.6,
            "broadband vocalness {v_noise} should be high"
        );
        assert!(
            v_tone < 0.5,
            "harmonic-tone vocalness {v_tone} should be low"
        );
    }

    #[test]
    fn test_analyze_batch_with_progress() {
        use std::sync::Mutex;
        // Mix a real fixture (Ok) with nonexistent paths (Err). Errors still
        // count as completions, so the callback must fire for every input.
        let real = fixture("tagged.flac");
        let paths: Vec<&Path> = vec![
            real.as_path(),
            Path::new("/nonexistent/one.flac"),
            Path::new("/nonexistent/two.flac"),
        ];
        let len = paths.len();
        // Fn + Sync: record (done, total) pairs from the worker threads.
        let calls: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
        let results = analyze_batch_with(&paths, 22050, &compact(), |done, total| {
            calls.lock().unwrap().push((done, total));
        });

        // One result per input, in input order.
        assert_eq!(results.len(), len);
        assert!(results[0].is_ok(), "real fixture should analyze");
        assert!(results[1].is_err(), "nonexistent path should error");
        assert!(results[2].is_err(), "nonexistent path should error");

        // Callback fired exactly once per input; total constant == len.
        let mut recorded = calls.into_inner().unwrap();
        assert_eq!(recorded.len(), len, "callback fires once per file");
        assert!(
            recorded.iter().all(|&(_, total)| total == len),
            "total is constant == len"
        );

        // `done` values are a permutation of 1..=len (completion order varies).
        let mut dones: Vec<usize> = recorded.drain(..).map(|(d, _)| d).collect();
        dones.sort_unstable();
        assert_eq!(
            dones,
            (1..=len).collect::<Vec<_>>(),
            "done values are 1..=len"
        );
    }

    // ---- genre (bring-your-own model) ----

    /// A trivial 2-class model (48-dim, zero weights + bias favoring "b") at a
    /// given embedding_version. Class "b" wins for any embedding.
    fn genre_model_json(embedding_version: u32) -> String {
        let zeros48 = "0,".repeat(48);
        let zeros48 = zeros48.trim_end_matches(',');
        format!(
            r#"{{"format_version": 1, "embedding_version": {embedding_version},
                "labels": ["a", "b"],
                "layers": [{{"weights": [[{zeros48}],[{zeros48}]], "bias": [0.0, 2.0], "activation": "softmax"}}]}}"#
        )
    }

    fn genre_config(features: Option<&[&str]>) -> AnalysisConfig {
        let model =
            crate::genre::from_json_str(&genre_model_json(crate::similarity::SIMILARITY_VERSION))
                .unwrap();
        AnalysisConfig {
            mode: AnalysisMode::Compact,
            features: features.map(|f| f.iter().map(|s| s.to_string()).collect()),
            genre_model: Some(std::sync::Arc::new(model)),
            ..AnalysisConfig::default()
        }
    }

    #[test]
    fn test_genre_model_predicts_without_leaking_embedding() {
        let y = sine(440.0, 22050, 3.0);
        // No feature list → embedding computed internally but NOT emitted.
        let r = analyze_signal(y.view(), 22050, &genre_config(None)).unwrap();
        assert_eq!(r.genre.as_deref(), Some("b"));
        let conf = r.genre_confidence.expect("genre_confidence populated");
        assert!(conf > 0.5 && conf <= 1.0, "confidence {conf}");
        // Embedding fields must NOT leak when features didn't request them.
        assert!(r.embedding.is_none(), "embedding must not be emitted");
        assert!(r.embedding_version.is_none());
    }

    #[test]
    fn test_genre_model_with_embedding_feature_emits_both() {
        let y = sine(440.0, 22050, 3.0);
        let r = analyze_signal(y.view(), 22050, &genre_config(Some(&["embedding"]))).unwrap();
        assert_eq!(r.genre.as_deref(), Some("b"));
        assert!(r.genre_confidence.is_some());
        // Now the embedding fields ARE emitted (explicitly requested).
        assert!(r.embedding.is_some(), "embedding emitted when requested");
        assert_eq!(
            r.embedding_version,
            Some(crate::similarity::SIMILARITY_VERSION)
        );
    }

    #[test]
    fn test_genre_model_version_mismatch_fails_fast() {
        let y = sine(440.0, 22050, 3.0);
        let model = crate::genre::from_json_str(&genre_model_json(999)).unwrap();
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            genre_model: Some(std::sync::Arc::new(model)),
            ..AnalysisConfig::default()
        };
        match analyze_signal(y.view(), 22050, &config) {
            Err(SonaraError::ModelError(msg)) => {
                assert!(
                    msg.contains("999"),
                    "message should name the model version: {msg}"
                );
            }
            Err(other) => panic!("expected ModelError, got {other:?}"),
            Ok(_) => panic!("expected ModelError, got Ok"),
        }
    }

    #[test]
    fn test_genre_none_without_model() {
        let y = sine(440.0, 22050, 3.0);
        for cfg in [compact(), playlist(), full()] {
            let r = analyze_signal(y.view(), 22050, &cfg).unwrap();
            assert!(r.genre.is_none(), "genre None without a model");
            assert!(r.genre_confidence.is_none());
        }
    }

    // ---- augment (decode-free recompute onto cached records) ----

    /// Tonal + rhythmic deterministic signal: the C-major progression under a
    /// 120-BPM exponential amplitude pulse, so chroma/key/chords AND
    /// beats/onsets all carry real structure.
    fn augment_rich_signal() -> Array1<Float> {
        let base = c_major_progression(SR, 2);
        Array1::from_shape_fn(base.len(), |i| {
            let t = i as Float / SR as Float;
            let beat_phase = (t * 2.0).fract();
            base[i] * (0.35 + 0.65 * (-8.0 * beat_phase).exp())
        })
    }

    /// A feature request whose record carries every evidence field any
    /// decode-free feature (incl. the embedding) reads.
    fn evidence_complete_config() -> AnalysisConfig {
        feature_config(&[
            "bpm",
            "beats",
            "onsets",
            "rms",
            "dynamic_range",
            "centroid",
            "onset_density",
            "bandwidth",
            "rolloff",
            "flatness",
            "contrast",
            "mfcc",
            "chroma",
            "chords",
            "dissonance",
            "energy",
            "danceability",
            "key",
            "valence",
            "acousticness",
        ])
    }

    /// THE headline P2 test: for every decode-free feature, augmenting a rich
    /// cached record must reproduce, bit-for-bit, the field a direct pipeline
    /// run over the same audio produces. Failures are collected per feature so
    /// one red run exposes every broken recompute path at once.
    #[test]
    fn test_augment_decode_free_features_match_direct_runs() {
        let y = augment_rich_signal();
        let rich = analyze_signal(y.view(), SR, &evidence_complete_config()).unwrap();
        let cfg = AnalysisConfig::default();

        // (feature, direct-run request reproducing its standalone meaning)
        let cases: &[(&str, &[&str])] = &[
            ("onset_density", &["onset_density"]),
            ("energy", &["energy"]),
            ("danceability", &["danceability"]),
            ("key", &["key"]),
            ("valence", &["valence"]),
            ("acousticness", &["acousticness"]),
            ("key_candidates", &["key_candidates"]),
            ("vocalness", &["vocalness"]),
            ("instrumentalness", &["instrumentalness"]),
            // mood's declared evidence includes dissonance: augment feeds the
            // record's value, and a mood-only run computes it as well.
            ("mood", &["mood"]),
            ("embedding", &["embedding"]),
        ];
        let mut mismatches: Vec<String> = Vec::new();
        for (feature, direct_features) in cases {
            assert!(can_augment(&rich, feature), "{feature} must be augmentable");
            let aug = augment_analysis(&rich, &[feature], None, &cfg).unwrap();
            let direct = analyze_signal(y.view(), SR, &feature_config(direct_features)).unwrap();
            let mut check = |field: &str, equal: bool| {
                if !equal {
                    mismatches.push(format!("{feature}.{field}"));
                }
            };
            match *feature {
                "onset_density" => {
                    check("onset_density", aug.onset_density == direct.onset_density)
                }
                "energy" => check("energy", aug.energy == direct.energy),
                "danceability" => check("danceability", aug.danceability == direct.danceability),
                "key" => {
                    check("key", aug.key == direct.key);
                    check(
                        "key_confidence",
                        aug.key_confidence == direct.key_confidence,
                    );
                    check("key_camelot", aug.key_camelot == direct.key_camelot);
                }
                "valence" => check("valence", aug.valence == direct.valence),
                "acousticness" => check("acousticness", aug.acousticness == direct.acousticness),
                "key_candidates" => check(
                    "key_candidates",
                    aug.key_candidates == direct.key_candidates,
                ),
                "vocalness" => check("vocalness", aug.vocalness == direct.vocalness),
                "instrumentalness" => check(
                    "instrumentalness",
                    aug.instrumentalness == direct.instrumentalness,
                ),
                "mood" => {
                    check("mood_happy", aug.mood_happy == direct.mood_happy);
                    check(
                        "mood_aggressive",
                        aug.mood_aggressive == direct.mood_aggressive,
                    );
                    check("mood_relaxed", aug.mood_relaxed == direct.mood_relaxed);
                    check("mood_sad", aug.mood_sad == direct.mood_sad);
                }
                "embedding" => {
                    check("embedding", aug.embedding == direct.embedding);
                    check(
                        "embedding_version",
                        aug.embedding_version == direct.embedding_version,
                    );
                }
                other => panic!("unhandled case {other}"),
            }
            // Untouched fields must be preserved from the cached record.
            check("bpm-preserved", aug.bpm == rich.bpm);
            check("chroma-preserved", aug.chroma_mean == rich.chroma_mean);
        }
        assert!(
            mismatches.is_empty(),
            "augment != direct run for: {mismatches:?}"
        );
    }

    #[test]
    fn test_augment_valence_writes_the_key_like_a_direct_run() {
        // A direct ["valence"] run emits the key valence is derived from; augmenting
        // a record with the evidence (chroma) but no key must emit it too.
        let y = augment_rich_signal();
        let record = analyze_signal(y.view(), SR, &feature_config(&["chroma"])).unwrap();
        assert!(record.key.is_none() && can_augment(&record, "valence"));
        let cfg = AnalysisConfig::default();
        let aug = augment_analysis(&record, &["valence"], None, &cfg).unwrap();
        let direct = analyze_signal(y.view(), SR, &feature_config(&["valence"])).unwrap();
        assert!(direct.key.is_some());
        assert_eq!(aug.valence, direct.valence);
        assert_eq!(aug.key, direct.key);
        assert_eq!(aug.key_confidence, direct.key_confidence);
        assert_eq!(aug.key_camelot, direct.key_camelot);
    }

    #[test]
    fn test_augment_evidence_missing_blocks_and_errors_without_audio() {
        let y = sine(440.0, SR, 2.0);
        let compact_record = analyze_signal(y.view(), SR, &compact()).unwrap();
        // Compact records carry no chroma → key is not decode-free on THIS record.
        assert!(!can_augment(&compact_record, "key"));
        assert!(matches!(
            augment_blocker(&compact_record, "key"),
            Some(AugmentBlocker::MissingEvidence(fields)) if fields == vec!["chroma_mean"]
        ));
        // Audio/FrameCurves classes are never decode-free, on any record.
        assert!(!can_augment(&compact_record, "chroma"));
        assert!(!can_augment(&compact_record, "zcr"));
        assert!(can_augment(&compact_record, "onset_density"));
        // The same feature on an evidence-complete record IS augmentable —
        // augmentability is per-record, not per-feature.
        let rich = analyze_signal(y.view(), SR, &evidence_complete_config()).unwrap();
        assert!(can_augment(&rich, "key"));

        // Without audio, augment must fail naming the feature and the reason.
        let err = augment_analysis(&compact_record, &["key"], None, &AnalysisConfig::default())
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("key"), "{msg}");
        assert!(msg.contains("chroma_mean"), "{msg}");
        let err = augment_analysis(
            &compact_record,
            &["chroma"],
            None,
            &AnalysisConfig::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("chroma"), "{err}");
    }

    #[test]
    fn test_augment_schema_version_mismatch_fails_fast() {
        let y = sine(440.0, SR, 2.0);
        let mut stale = analyze_signal(y.view(), SR, &evidence_complete_config()).unwrap();
        assert!(can_augment(&stale, "energy"));
        stale.provenance.schema_version = ANALYSIS_SCHEMA_VERSION - 1;
        assert!(!can_augment(&stale, "energy"));
        let err =
            augment_analysis(&stale, &["energy"], None, &AnalysisConfig::default()).unwrap_err();
        assert!(err.to_string().contains("schema_version"), "{err}");
    }

    #[test]
    fn test_augment_embedding_version_mismatch_fails_fast() {
        let y = sine(440.0, SR, 2.0);
        let mut record = analyze_signal(y.view(), SR, &evidence_complete_config()).unwrap();
        record.embedding_version = Some(crate::similarity::SIMILARITY_VERSION + 1);
        assert!(!can_augment(&record, "embedding"));
        // Non-embedding features are unaffected by the stale embedding marker.
        assert!(can_augment(&record, "energy"));
        let err = augment_analysis(&record, &["embedding"], None, &AnalysisConfig::default())
            .unwrap_err();
        assert!(err.to_string().contains("embedding_version"), "{err}");
    }

    #[test]
    fn test_augment_unknown_feature_is_hard_error() {
        let y = sine(440.0, SR, 2.0);
        let record = analyze_signal(y.view(), SR, &compact()).unwrap();
        match augment_analysis(&record, &["keyy"], None, &AnalysisConfig::default()) {
            Err(SonaraError::InvalidParameter { param, reason }) => {
                assert_eq!(param, "features");
                assert!(reason.contains("unknown feature(s): keyy"), "{reason}");
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn test_augment_danceability_reflects_recorded_bpm_range() {
        // 120-BPM click train; a (130, 260) fold range doubles the reported
        // bpm to ~240, which danceability's tempo term is sensitive to — so
        // this test bites if augment fed anything but the record's folded bpm.
        let n = (4.0 * SR as Float) as usize;
        let interval = (60.0 / 120.0 * SR as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / SR as Float).sin();
            }
            pos += interval;
        }
        let ranged = AnalysisConfig {
            bpm_min: Some(130.0),
            bpm_max: Some(260.0),
            ..Default::default()
        };
        let record = analyze_signal(y.view(), SR, &ranged).unwrap();
        assert!(
            record.bpm >= 130.0 && record.bpm <= 260.0,
            "fold range not applied: {}",
            record.bpm
        );
        assert!(
            (record.bpm - record.bpm_raw).abs() > 1.0,
            "fold must actually trigger for this test to bite: bpm {} raw {}",
            record.bpm,
            record.bpm_raw
        );

        // The augment config carries NO range — the record's is inherited.
        let aug =
            augment_analysis(&record, &["danceability"], None, &AnalysisConfig::default()).unwrap();
        let direct = analyze_signal(
            y.view(),
            SR,
            &AnalysisConfig {
                features: Some(HashSet::from(["danceability".to_string()])),
                bpm_min: Some(130.0),
                bpm_max: Some(260.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(aug.danceability, direct.danceability);
        // The recorded fold range survives augmentation.
        assert_eq!(aug.provenance.bpm_min, Some(130.0));
        assert_eq!(aug.provenance.bpm_max, Some(260.0));
    }

    /// Deterministic 2 s pulsed-tone WAV at native 44.1 kHz for fallback tests.
    fn write_augment_wav(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for index in 0..44_100 * 2 {
            let t = index as Float / 44_100.0;
            let beat_phase = (t * 2.0).fract();
            let env = 0.3 + 0.7 * (-9.0 * beat_phase).exp();
            let sample = env
                * (0.4 * (2.0 * PI * 220.0 * t).sin()
                    + 0.25 * (2.0 * PI * 523.25 * t).sin()
                    + 0.15 * (2.0 * PI * 2_093.0 * t).sin());
            writer
                .write_sample((sample * i16::MAX as Float) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn test_augment_audio_fallback_runs_at_recorded_rate_and_range() {
        let path = std::env::temp_dir().join(format!(
            "sonara-augment-fallback-{}.wav",
            std::process::id()
        ));
        write_augment_wav(&path);
        // Record analyzed at 22050 (not the file's native 44100), with a fold
        // range the augment config below does NOT carry.
        let ranged = AnalysisConfig {
            bpm_min: Some(130.0),
            bpm_max: Some(260.0),
            ..Default::default()
        };
        let record = analyze_file(&path, 22050, &ranged).unwrap();
        assert_eq!(record.provenance.sample_rate, 22050);

        // chroma is FrameCurves: decode-free impossible, audio fallback required.
        // So is tempo_curve: it reads beat positions a record does not keep.
        assert!(!can_augment(&record, "chroma"));
        assert!(!can_augment(&record, "tempo_curve"));
        let aug = augment_analysis(
            &record,
            &["chroma", "bpm", "tempo_curve"],
            Some(&path),
            &AnalysisConfig::default(),
        )
        .unwrap();
        // The fallback must have run at the RECORD's sample rate with the
        // RECORD's fold range — i.e. match a direct file run configured so.
        let direct = analyze_file(
            &path,
            22050,
            &AnalysisConfig {
                features: Some(HashSet::from([
                    "chroma".to_string(),
                    "bpm".to_string(),
                    "tempo_curve".to_string(),
                ])),
                bpm_min: Some(130.0),
                bpm_max: Some(260.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(aug.chroma_mean, direct.chroma_mean);
        // One value per interval of the beats the same run reports.
        assert!(direct.beats.len() > 1);
        assert_eq!(
            direct.tempo_curve.as_ref().map(Vec::len),
            Some(direct.beats.len() - 1)
        );
        assert_eq!(aug.tempo_curve, direct.tempo_curve);
        assert_eq!(aug.tempo_variability, direct.tempo_variability);
        assert_eq!(aug.bpm, direct.bpm);
        assert_eq!(aug.bpm_raw, direct.bpm_raw);
        assert_eq!(aug.bpm_candidates, direct.bpm_candidates);
        // Frame-domain provenance unchanged; untouched cached fields preserved.
        assert_eq!(aug.provenance.sample_rate, 22050);
        assert_eq!(aug.rms_mean, record.rms_mean);
        assert_eq!(aug.duration_sec, record.duration_sec);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_augment_vocalness_model_and_genre_model_decode_free() {
        let y = augment_rich_signal();
        let rich = analyze_signal(y.view(), SR, &evidence_complete_config()).unwrap();

        // Vocalness model over the recomputed embedding: id stamped, pair
        // coupled (both fields update together).
        let model_cfg = AnalysisConfig {
            vocalness_model: Some(std::sync::Arc::new(tiny_vocalness_model(
                "vocal-aug-v1",
                crate::similarity::SIMILARITY_VERSION,
            ))),
            ..Default::default()
        };
        let aug = augment_analysis(&rich, &["vocalness"], None, &model_cfg).unwrap();
        assert_eq!(aug.vocalness, Some(0.5));
        assert_eq!(aug.instrumentalness, Some(0.5));
        assert_eq!(
            aug.provenance.vocalness_model_id.as_deref(),
            Some("vocal-aug-v1")
        );

        // Genre: carrying the model IS the request; predicts decode-free.
        let genre_cfg = AnalysisConfig {
            genre_model: Some(std::sync::Arc::new(
                crate::genre::from_json_str(&genre_model_json(
                    crate::similarity::SIMILARITY_VERSION,
                ))
                .unwrap(),
            )),
            ..Default::default()
        };
        let aug = augment_analysis(&rich, &[], None, &genre_cfg).unwrap();
        assert_eq!(aug.genre.as_deref(), Some("b"));
        assert!(aug.genre_confidence.is_some_and(|c| c > 0.5));
        // An empty feature list changes no recorded request.
        assert_eq!(
            aug.provenance.requested_features,
            rich.provenance.requested_features
        );

        // Stale model versions fail fast, mirroring the pipeline.
        let stale_cfg = AnalysisConfig {
            genre_model: Some(std::sync::Arc::new(
                crate::genre::from_json_str(&genre_model_json(999)).unwrap(),
            )),
            ..Default::default()
        };
        assert!(matches!(
            augment_analysis(&rich, &[], None, &stale_cfg),
            Err(SonaraError::ModelError(_))
        ));
        // Model-driven vocalness on an embedding-evidence-poor record needs audio.
        let compact_record = analyze_signal(y.view(), SR, &compact()).unwrap();
        assert!(augment_analysis(&compact_record, &["vocalness"], None, &model_cfg).is_err());
    }

    #[test]
    fn test_augment_updates_requested_features_provenance() {
        let y = sine(440.0, SR, 2.0);
        // Explicit-list record: union with the augmented names.
        let record = analyze_signal(y.view(), SR, &feature_config(&["chroma", "key"])).unwrap();
        let aug = augment_analysis(
            &record,
            &["valence", "key"],
            None,
            &AnalysisConfig::default(),
        )
        .unwrap();
        assert_eq!(
            aug.provenance.requested_features.as_deref(),
            Some(
                &[
                    "chroma".to_string(),
                    "key".to_string(),
                    "valence".to_string()
                ][..]
            )
        );
        // Mode-driven record (None): expanded to the mode's default feature
        // set (empty for Compact) plus the augmented names.
        let compact_record = analyze_signal(y.view(), SR, &compact()).unwrap();
        let aug = augment_analysis(
            &compact_record,
            &["onset_density"],
            None,
            &AnalysisConfig::default(),
        )
        .unwrap();
        assert_eq!(
            aug.provenance.requested_features.as_deref(),
            Some(&["onset_density".to_string()][..])
        );
        // A Full record from a signal expands to every feature it could hold:
        // the opt-in ones, but not the file-only groups it never measured.
        let full_record = analyze_signal(y.view(), SR, &full()).unwrap();
        let aug = augment_analysis(
            &full_record,
            &["onset_density"],
            None,
            &AnalysisConfig::default(),
        )
        .unwrap();
        let names = aug.provenance.requested_features.unwrap();
        assert!(names.contains(&"hpss".to_string()));
        for file_only in ["ebur128", "native_dynamics", "amplitude_envelope", "stereo_field", "tags"] {
            assert!(!names.contains(&file_only.to_string()), "{file_only}");
        }
    }

    #[test]
    fn test_non_finite_audio_is_an_error_before_resampling() {
        let mut y = sine(440.0, 48000, 1.0);
        y[1000] = Float::NAN;
        // Full computes aggression in a separate 22.05 kHz lane off its rate.
        assert!(matches!(
            analyze_signal(y.view(), 48000, &full()),
            Err(SonaraError::InvalidAudio(_))
        ));
        // A file decoded at 48 kHz and analyzed at the default rate.
        assert!(matches!(
            analyze_native_mono(y, 48000, 22050, &playlist()),
            Err(SonaraError::InvalidAudio(_))
        ));
    }

    /// 1 s float WAV at 48 kHz; `nan_at` plants one non-finite sample.
    fn write_float_wav(path: &Path, nan_at: Option<usize>) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for (index, &sample) in sine(440.0, 48000, 1.0).iter().enumerate() {
            let sample = if Some(index) == nan_at { Float::NAN } else { sample };
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    // A file the analysis cannot process is that file's error: the rest of the
    // batch keeps its results, in input order.
    #[test]
    fn test_batch_isolates_a_file_with_non_finite_samples() {
        let dir = std::env::temp_dir();
        let good = dir.join(format!("sonara-batch-good-{}.wav", std::process::id()));
        let bad = dir.join(format!("sonara-batch-nan-{}.wav", std::process::id()));
        write_float_wav(&good, None);
        write_float_wav(&bad, Some(1000));
        let paths: Vec<&Path> = vec![good.as_path(), bad.as_path(), good.as_path()];
        let results = analyze_batch(&paths, 22050, &playlist());
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok() && results[2].is_ok());
        let err = results[1].as_ref().unwrap_err();
        assert!(matches!(err, SonaraError::InvalidAudio(_)), "{err}");
    }

    #[test]
    fn test_panic_inside_analysis_is_reported_as_an_error() {
        let err = catch_analysis_panic(|| -> Result<()> { panic!("boom") }).unwrap_err();
        assert!(
            matches!(&err, SonaraError::Numerical(reason) if reason.contains("boom")),
            "{err}"
        );
        // A formatted panic carries a `String` payload, not a `&str`.
        let frame = std::hint::black_box(3);
        let err =
            catch_analysis_panic(|| -> Result<()> { panic!("boom at frame {frame}") }).unwrap_err();
        assert!(
            matches!(&err, SonaraError::Numerical(reason) if reason.contains("frame 3")),
            "{err}"
        );
    }

    // The standalone model lane resamples inside the loader, so the file with
    // a non-finite sample panics in the resampler: the batch still returns one
    // entry per path and keeps the results of the other files.
    #[cfg(feature = "aggression")]
    #[test]
    fn test_batch_survives_a_panic_inside_one_file() {
        let dir = std::env::temp_dir();
        let good = dir.join(format!("sonara-panic-good-{}.wav", std::process::id()));
        let bad = dir.join(format!("sonara-panic-nan-{}.wav", std::process::id()));
        write_float_wav(&good, None);
        write_float_wav(&bad, Some(1000));
        let paths: Vec<&Path> = vec![good.as_path(), bad.as_path(), good.as_path()];
        let results = crate::aggression::analyze_batch(&paths, 22050);
        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok() && results[2].is_ok());
        assert!(results[1].is_err());
    }

    // The file route (decode at the file's rate, validate, resample) gives the
    // same analysis as the loader's own resampling followed by the signal entry.
    #[test]
    fn test_file_route_matches_loader_resampling() {
        let dir = std::env::temp_dir();
        let at_44k = dir.join(format!("sonara-route-44k-{}.wav", std::process::id()));
        let at_48k = dir.join(format!("sonara-route-48k-{}.wav", std::process::id()));
        write_augment_wav(&at_44k);
        write_float_wav(&at_48k, None);
        for path in [&at_44k, &at_48k] {
            for sr in [0, 22050] {
                let (y, loaded_sr) = audio::load(path, sr, true, 0.0, 0.0).unwrap();
                let from_signal = analyze_signal(y.view(), loaded_sr, &playlist()).unwrap();
                let from_file = analyze_file(path, sr, &playlist()).unwrap();
                assert!(from_file == from_signal, "{} sr={sr}", path.display());
            }
        }
        let _ = std::fs::remove_file(&at_44k);
        let _ = std::fs::remove_file(&at_48k);
    }

    #[test]
    fn test_empty_signal_is_an_error_not_a_panic() {
        let empty = Array1::<Float>::zeros(0);
        assert!(matches!(
            validated_zero_crossing_rate(empty.view()),
            Err(SonaraError::InvalidAudio(_))
        ));
    }

    #[cfg(feature = "aggression")]
    #[test]
    fn test_augment_aggression_routes_through_audio_fallback_lane() {
        let path = std::env::temp_dir().join(format!(
            "sonara-augment-aggression-{}.wav",
            std::process::id()
        ));
        write_augment_wav(&path);
        let record = analyze_file(&path, 22050, &compact()).unwrap();
        assert!(
            !can_augment(&record, "aggression"),
            "aggression is Audio-class"
        );
        // No audio → error naming the feature.
        let err = augment_analysis(&record, &["aggression"], None, &AnalysisConfig::default())
            .unwrap_err();
        assert!(err.to_string().contains("aggression"), "{err}");
        // With audio, the fallback routes through the dedicated 22.05 kHz lane.
        let aug = augment_analysis(
            &record,
            &["aggression"],
            Some(&path),
            &AnalysisConfig::default(),
        )
        .unwrap();
        let direct = analyze_file(
            &path,
            22050,
            &AnalysisConfig {
                features: Some(HashSet::from(["aggression".to_string()])),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(aug.aggression_score, direct.aggression_score);
        assert_eq!(aug.aggression_confidence, direct.aggression_confidence);
        assert_eq!(aug.aggression_harshness, direct.aggression_harshness);
        assert_eq!(
            aug.provenance.aggression_model_id.as_deref(),
            Some(crate::aggression::AGGRESSION_MODEL_ID)
        );
        // Cached generic fields stay in the caller-rate domain, untouched.
        assert_eq!(aug.bpm, record.bpm);
        assert_eq!(aug.rms_mean, record.rms_mean);
        std::fs::remove_file(path).unwrap();
    }

    // ---- Frame statistics (MIR_FEATURES group 1.2) ----

    #[test]
    fn test_frame_stat_helpers_match_numpy_definitions() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert!((series_std(&x) - 1.25_f32.sqrt()).abs() < 1e-6); // np.std, ddof=0
        assert!((series_delta_mean(&[1.0, 3.0, 2.0]) - 1.5).abs() < 1e-6);
        assert!((sorted_quantile(&x, 0.10) - 1.3).abs() < 1e-6); // np.quantile default
        assert!((sorted_quantile(&x, 0.90) - 3.7).abs() < 1e-6);
        assert!((sorted_iqr(&x) - 1.5).abs() < 1e-6);
        let ramp: Vec<Float> = (0..=10).map(|v| v as Float).collect();
        assert_eq!(sorted_quantile(&ramp, 0.10), 1.0);
        assert_eq!(sorted_quantile(&ramp, 0.90), 9.0);
        assert_eq!(sorted_series(&[3.0, 1.0, 2.0]), vec![1.0, 2.0, 3.0]);
        // Degenerate series stay finite.
        assert_eq!(series_std(&[]), 0.0);
        assert_eq!(series_std(&[5.0]), 0.0);
        assert_eq!(series_delta_mean(&[5.0]), 0.0);
        assert_eq!(sorted_quantile(&[], 0.5), 0.0);
    }

    #[test]
    fn test_frame_zero_crossing_rates() {
        // Alternating sign: every adjacent pair crosses; 3 pairs in a 4-sample frame.
        let alt = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert_eq!(frame_zero_crossing_rates(&alt, 4, 2, 3), vec![0.75; 3]);
        assert_eq!(frame_zero_crossing_rates(&[0.5; 8], 4, 2, 3), vec![0.0; 3]);
        // Crossings are counted per frame, not shared across frames.
        let one = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        assert_eq!(
            frame_zero_crossing_rates(&one, 4, 2, 3),
            vec![0.0, 0.25, 0.25]
        );
    }

    #[test]
    fn test_frame_stats_follow_their_base_features() {
        let y = sine(440.0, 22050, 3.0);
        let c = analyze_signal(y.view(), 22050, &compact()).unwrap();
        for v in [c.rms_std, c.rms_delta_mean, c.rms_iqr, c.rms_p10, c.rms_p90] {
            assert!(v.is_some_and(Float::is_finite));
        }
        assert!(c.zero_crossing_rate_std.is_some());
        assert!(c.spectral_centroid_std.is_some() && c.spectral_centroid_iqr.is_some());
        assert!(c.spectral_bandwidth_std.is_none() && c.spectral_flatness_p10.is_none());
        assert!(c.mfcc_std.is_none() && c.chroma_delta_mean.is_none());
        assert!(c.spectral_contrast_std.is_none());

        let p = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert_eq!(p.mfcc_std.as_ref().unwrap().len(), N_MFCC);
        assert_eq!(p.mfcc_delta_mean.as_ref().unwrap().len(), N_MFCC);
        assert_eq!(p.chroma_std.as_ref().unwrap().len(), 12);
        assert_eq!(p.chroma_delta_mean.as_ref().unwrap().len(), 12);
        assert_eq!(
            p.spectral_contrast_std.as_ref().unwrap().len(),
            N_CONTRAST_BANDS + 1
        );
        for v in [
            p.spectral_bandwidth_std,
            p.spectral_bandwidth_delta_mean,
            p.spectral_rolloff_std,
            p.spectral_flatness_std,
            p.spectral_flatness_iqr,
        ] {
            assert!(v.is_some_and(|v| v.is_finite() && v >= 0.0));
        }
        assert!(p.spectral_rolloff_p10.unwrap() <= p.spectral_rolloff_p90.unwrap());
        assert!(p.spectral_flatness_p10.unwrap() <= p.spectral_flatness_p90.unwrap());

        // A narrow request keeps only the named group's statistics.
        let narrow = AnalysisConfig {
            features: Some(["mfcc".to_string()].into_iter().collect()),
            ..Default::default()
        };
        let n = analyze_signal(y.view(), 22050, &narrow).unwrap();
        assert!(n.mfcc_std.is_some() && n.mfcc_delta_mean.is_some());
        assert!(n.chroma_std.is_none() && n.spectral_contrast_std.is_none());
        assert!(n.spectral_rolloff_p90.is_none() && n.spectral_flatness_iqr.is_none());
    }

    #[test]
    fn test_frame_stats_values_on_an_amplitude_step() {
        // 2 s at amplitude 0.1, then 2 s at 0.9: per-frame RMS takes two levels.
        let sr = 22050;
        let mut y = sine(440.0, sr, 4.0);
        let half = y.len() / 2;
        y.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v *= if i < half { 0.1 } else { 0.9 });
        let r = analyze_signal(y.view(), sr, &compact()).unwrap();
        let (lo, hi) = (0.1 / 2.0_f32.sqrt(), 0.9 / 2.0_f32.sqrt());
        // Two equal groups: population std is half their gap.
        let std = r.rms_std.unwrap();
        assert!((std - (hi - lo) / 2.0).abs() < 0.03 * hi, "rms_std {std}");
        assert!(
            (r.rms_p10.unwrap() - lo).abs() < 0.05 * lo,
            "p10 {:?}",
            r.rms_p10
        );
        assert!(
            (r.rms_p90.unwrap() - hi).abs() < 0.05 * hi,
            "p90 {:?}",
            r.rms_p90
        );
        // One jump over ~170 frames: the mean change per hop stays small.
        assert!(
            r.rms_delta_mean.unwrap() < 0.02,
            "delta {:?}",
            r.rms_delta_mean
        );
        // The pitch never moves, so the centroid's core spread stays tight.
        assert!(
            r.spectral_centroid_iqr.unwrap() < 20.0,
            "{:?}",
            r.spectral_centroid_iqr
        );
    }

    #[test]
    fn test_augment_merge_moves_frame_stats_with_their_feature() {
        let y = sine(440.0, 22050, 2.0);
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        for name in [
            "mfcc",
            "chroma",
            "contrast",
            "rolloff",
            "flatness",
            "bandwidth",
        ] {
            merge_feature_fields(&mut out, &fresh, name);
        }
        assert_eq!(out.mfcc_std, fresh.mfcc_std);
        assert_eq!(out.mfcc_delta_mean, fresh.mfcc_delta_mean);
        assert_eq!(out.chroma_delta_mean, fresh.chroma_delta_mean);
        assert_eq!(out.spectral_contrast_std, fresh.spectral_contrast_std);
        assert_eq!(out.spectral_rolloff_p90, fresh.spectral_rolloff_p90);
        assert_eq!(out.spectral_flatness_iqr, fresh.spectral_flatness_iqr);
        assert_eq!(
            out.spectral_bandwidth_delta_mean,
            fresh.spectral_bandwidth_delta_mean
        );
    }

    // ---- Spectral shape (MIR_FEATURES group 3.1) ----

    #[test]
    fn test_spectral_shape_moments_two_symmetric_lines() {
        // Equal lines at 100 and 300 Hz: c = 200, bw = 100.
        let freqs: Vec<Float> = (0..5).map(|k| k as Float * 100.0).collect();
        let mag = [0.0, 1.0, 0.0, 1.0, 0.0];
        let (skew, kurt) = spectral_shape_moments(&mag, &freqs, 200.0, 2.0, 100.0).unwrap();
        assert!(skew.abs() < 1e-12, "skew {skew}");
        assert!((kurt + 2.0).abs() < 1e-12, "kurt {kurt}");
        assert!(spectral_shape_moments(&mag, &freqs, 200.0, 0.0, 100.0).is_none());
        assert!(spectral_shape_moments(&mag, &freqs, 200.0, 2.0, 0.0).is_none());
    }

    #[test]
    fn test_spectral_shape_slope_white_and_pink() {
        let (sr, n_fft) = (22050, 2048);
        let n_bins = n_fft / 2 + 1;
        let bands = spectral_shape_octave_bands(sr, n_fft, n_bins);
        assert_eq!(bands.len(), 8);
        // Lowest band 44.2..88.4 Hz → bins 5..=8; top band clipped below Nyquist.
        assert_eq!((bands[0].1, bands[0].2), (5, 9));
        assert_eq!(bands[7].2, n_bins - 1);
        let white = vec![1.0; n_bins];
        let s = spectral_shape_slope(&white, &bands).unwrap();
        assert!(s.abs() < 1e-9, "white {s}");
        let pink: Vec<Float> = (0..n_bins)
            .map(|k| if k == 0 { 0.0 } else { 1.0 / k as Float })
            .collect();
        let s = spectral_shape_slope(&pink, &bands).unwrap();
        assert!((s + 3.0103).abs() < 0.05, "pink {s}");
        assert!(spectral_shape_slope(&vec![0.0; n_bins], &bands).is_none());
        assert!(spectral_shape_slope(&white, &bands[..1]).is_none());
        // A low rate drops the bands above Nyquist instead of leaving them empty.
        assert_eq!(spectral_shape_octave_bands(8000, 2048, 1025).len(), 7);
    }

    #[test]
    fn test_spectral_shape_decrease_hand_example() {
        // ((2-1)/1 + (3-1)/2) / (2 + 3) = 0.4
        let d = spectral_shape_decrease(&[1.0, 2.0, 3.0]).unwrap();
        assert!((d - 0.4).abs() < 1e-12, "{d}");
        assert!(spectral_shape_decrease(&[0.0; 8]).is_none());
    }

    #[test]
    fn test_spectral_shape_silent_frame_and_signal() {
        let n_bins = 1025;
        let zeros = vec![0.0; n_bins];
        let freqs: Vec<Float> = (0..n_bins).map(|k| k as Float * 22050.0 / 2048.0).collect();
        let bands = spectral_shape_octave_bands(22050, 2048, n_bins);
        assert!(spectral_shape_moments(&zeros, &freqs, 0.0, 0.0, 0.0).is_none());
        assert!(spectral_shape_slope(&zeros, &bands).is_none());
        assert!(spectral_shape_decrease(&zeros).is_none());

        let y = Array1::<Float>::zeros(22050 * 2);
        let r = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert_eq!(r.spectral_skewness_mean, Some(0.0));
        assert_eq!(r.spectral_kurtosis_excess_mean, Some(0.0));
        assert_eq!(r.spectral_slope_db_per_octave_mean, Some(0.0));
        assert_eq!(r.spectral_decrease_mean, Some(0.0));
    }

    fn spectral_shape_fields(r: &TrackAnalysis) -> [Option<Float>; 4] {
        [
            r.spectral_skewness_mean,
            r.spectral_kurtosis_excess_mean,
            r.spectral_slope_db_per_octave_mean,
            r.spectral_decrease_mean,
        ]
    }

    #[test]
    fn test_spectral_shape_presence_follows_requests() {
        let y = sine(440.0, 22050, 2.0);
        let c = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert_eq!(spectral_shape_fields(&c), [None; 4]);
        let p = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        for v in spectral_shape_fields(&p) {
            assert!(v.is_some_and(Float::is_finite), "{v:?}");
        }
        let m = analyze_signal(y.view(), 22050, &feature_config(&["mfcc"])).unwrap();
        assert_eq!(spectral_shape_fields(&m), [None; 4]);
        let s = analyze_signal(y.view(), 22050, &feature_config(&["spectral_shape"])).unwrap();
        assert_eq!(spectral_shape_fields(&s), spectral_shape_fields(&p));
    }

    #[test]
    fn test_spectral_shape_augment_merge_moves_fields() {
        let y = sine(440.0, 22050, 2.0);
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        merge_feature_fields(&mut out, &fresh, "spectral_shape");
        assert_eq!(spectral_shape_fields(&out), spectral_shape_fields(&fresh));
        assert!(out.spectral_decrease_mean.is_some());
    }

    // ---- Spectral texture (MIR_FEATURES group 3.2) ----

    fn spectral_texture_flux_of(prev: &[Float], cur: &[Float]) -> Option<f64> {
        let frame = |m: &[Float]| spectral_texture_frame(m, m.iter().sum());
        Some(spectral_texture_flux(&frame(prev)?.p, &frame(cur)?.p))
    }

    #[test]
    fn test_spectral_texture_flux() {
        let a = [0.0, 3.0, 1.0, 0.5];
        assert_eq!(spectral_texture_flux_of(&a, &a), Some(0.0));
        // All energy moves from bin 1 to bin 2: p_t[2] = 1, p_{t-1}[2] = 0.
        let f = spectral_texture_flux_of(&[0.0, 2.0, 0.0], &[0.0, 0.0, 5.0]).unwrap();
        assert!((f - 1.0).abs() < 1e-12, "{f}");
        // Loudness-invariant.
        let b = [1.0, 0.2, 4.0, 0.5];
        let b10: Vec<Float> = b.iter().map(|v| v * 10.0).collect();
        let (f1, f2) = (
            spectral_texture_flux_of(&a, &b).unwrap(),
            spectral_texture_flux_of(&a, &b10).unwrap(),
        );
        assert!((f1 - f2).abs() < 1e-7 && f1 > 0.0, "{f1} {f2}");
        assert_eq!(spectral_texture_flux_of(&[0.0; 4], &b), None);
        assert_eq!(spectral_texture_flux_of(&b, &[0.0; 4]), None);
    }

    #[test]
    fn test_spectral_texture_crest() {
        let crest = |m: &[Float]| spectral_texture_frame(m, m.iter().sum()).map(|f| f.crest);
        assert!((crest(&[2.0; 8]).unwrap() - 1.0).abs() < 1e-6);
        let mut one = [0.0; 16];
        one[5] = 3.0;
        assert!((crest(&one).unwrap() - 16.0).abs() < 1e-6);
        assert_eq!(crest(&[0.0; 8]), None);
    }

    #[test]
    fn test_spectral_texture_entropy() {
        let entropy = |m: &[Float]| spectral_texture_frame(m, m.iter().sum()).map(|f| f.entropy);
        assert!((entropy(&[0.7; 1025]).unwrap() - 1.0).abs() < 1e-5);
        let mut one = [0.0; 1025];
        one[9] = 2.0;
        assert_eq!(entropy(&one), Some(0.0));
        assert!(entropy(&[0.0; 1025]).is_none());
    }

    #[test]
    fn test_spectral_texture_irregularity() {
        let f = [100.0, 200.0, 300.0, 400.0];
        assert_eq!(spectral_texture_irregularity(&f, &[2.0; 4]), Some(0.0));
        // [1, 0, 1, 0]: 3 squared steps of 1 over energy 2.
        let r = spectral_texture_irregularity(&f, &[1.0, 0.0, 1.0, 0.0]).unwrap();
        assert!((r - 1.5).abs() < 1e-12, "{r}");
        // Unsorted input (magnitude order) is sorted by frequency first:
        // by frequency the amplitudes are [1, 2, 3] → (1 + 1) / 14.
        let r = spectral_texture_irregularity(&[300.0, 200.0, 100.0], &[3.0, 2.0, 1.0]).unwrap();
        assert!((r - 2.0 / 14.0).abs() < 1e-12, "{r}");
        assert_eq!(spectral_texture_irregularity(&[100.0], &[1.0]), None);
        assert_eq!(spectral_texture_irregularity(&[], &[]), None);
        assert_eq!(spectral_texture_irregularity(&f, &[0.0; 4]), None);
    }

    fn spectral_texture_fields(r: &TrackAnalysis) -> [Option<Float>; 5] {
        [
            r.spectral_flux_mean,
            r.spectral_flux_std,
            r.spectral_crest_mean,
            r.spectral_entropy_mean,
            r.spectral_irregularity_mean,
        ]
    }

    #[test]
    fn test_spectral_texture_presence_follows_requests() {
        let y = sine(440.0, 22050, 2.0);
        let c = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert_eq!(spectral_texture_fields(&c), [None; 5]);
        let p = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        for v in spectral_texture_fields(&p) {
            assert!(v.is_some_and(Float::is_finite), "{v:?}");
        }
        let m = analyze_signal(y.view(), 22050, &feature_config(&["mfcc"])).unwrap();
        assert_eq!(spectral_texture_fields(&m), [None; 5]);
        let s = analyze_signal(y.view(), 22050, &feature_config(&["spectral_texture"])).unwrap();
        assert_eq!(spectral_texture_fields(&s), spectral_texture_fields(&p));
        // Silence: nothing defined, all 0.0.
        let z = Array1::<Float>::zeros(22050 * 2);
        let r = analyze_signal(z.view(), 22050, &playlist()).unwrap();
        assert_eq!(spectral_texture_fields(&r), [Some(0.0); 5]);
    }

    #[test]
    fn test_spectral_texture_chunking_is_bit_identical() {
        // ~6 s of a gliding, tremolo tone with a silent gap: ~260 frames.
        let sr = 22050;
        let y = Array1::from_shape_fn(sr as usize * 6, |i| {
            let t = i as Float / sr as Float;
            if (2.0..2.5).contains(&t) {
                return 0.0;
            }
            let f = 220.0 + 150.0 * t;
            (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin()) * (2.0 * PI * f * t).sin()
                + 0.3 * (2.0 * PI * 1250.0 * t).sin() * (t % 0.5)
        });
        let zcr = validated_zero_crossing_rate(y.view()).unwrap();
        let run = |config: &AnalysisConfig, chunk| {
            analyze_signal_inner(y.view(), sr, config.needs_extended(), zcr, config, chunk).unwrap()
        };
        let config = playlist();
        let whole = run(&config, usize::MAX);
        assert!(whole.spectral_flux_mean.is_some_and(|f| f > 0.0));
        // Every output, not just the texture fields, matches across chunkings.
        assert_eq!(run(&config, 7), whole);
        assert_eq!(run(&config, FRAME_CHUNK), whole);
        let only = feature_config(&["spectral_texture"]);
        assert_eq!(
            spectral_texture_fields(&run(&only, 1)),
            spectral_texture_fields(&whole)
        );
    }

    #[test]
    fn test_spectral_texture_augment_merge_moves_fields() {
        let y = sine(440.0, 22050, 2.0);
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        merge_feature_fields(&mut out, &fresh, "spectral_texture");
        assert_eq!(
            spectral_texture_fields(&out),
            spectral_texture_fields(&fresh)
        );
        assert!(out.spectral_crest_mean.is_some());
    }

    // --- mel bands ---

    /// Two tones with a loud second half: bands that move and bands at the floor.
    fn mel_bands_signal() -> Array1<Float> {
        let sr = 22050.0;
        Array1::from_shape_fn(22050 * 3, |i| {
            let t = i as Float / sr;
            let gain = if t < 1.5 { 0.05 } else { 0.8 };
            gain * ((2.0 * PI * 220.0 * t).sin() + 0.5 * (2.0 * PI * 3000.0 * t).sin())
        })
    }

    #[test]
    fn test_mel_bands_presence_follows_requests() {
        let y = mel_bands_signal();
        let c = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(c.mel_bands_mean.is_none() && c.mel_bands_std.is_none());
        assert!(c.mel_bands_floor_db.is_none());
        let p = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        let (mean, std) = (
            p.mel_bands_mean.as_ref().unwrap(),
            p.mel_bands_std.as_ref().unwrap(),
        );
        assert_eq!((mean.len(), std.len()), (128, 128));
        let floor = p.mel_bands_floor_db.unwrap();
        assert!(mean.iter().all(|&v| v.is_finite() && v >= floor));
        assert!(std.iter().all(|&v| v.is_finite() && v >= 0.0));
        let m = analyze_signal(y.view(), 22050, &feature_config(&["mfcc"])).unwrap();
        assert!(m.mel_bands_mean.is_none() && m.mel_bands_floor_db.is_none());
        // On its own it needs no extended pass and reads the same matrix.
        let only = feature_config(&["mel_bands"]);
        assert!(!only.needs_extended());
        let s = analyze_signal(y.view(), 22050, &only).unwrap();
        assert_eq!(s.mel_bands_mean, p.mel_bands_mean);
        assert_eq!(s.mel_bands_std, p.mel_bands_std);
        assert_eq!(s.mel_bands_floor_db, p.mel_bands_floor_db);
        assert!(s.mfcc_mean.is_none());
    }

    #[test]
    fn test_mel_bands_mean_is_the_pre_dct_mfcc_mean() {
        let r = analyze_signal(mel_bands_signal().view(), 22050, &playlist()).unwrap();
        let mean = r.mel_bands_mean.unwrap();
        let n = mean.len() as f64;
        // Orthonormal DCT-II of the band means reproduces mfcc_mean.
        for (k, &mfcc) in r.mfcc_mean.unwrap().iter().enumerate() {
            let scale = if k == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            let c: f64 = mean
                .iter()
                .enumerate()
                .map(|(m, &v)| {
                    let angle =
                        std::f64::consts::PI * k as f64 * (2.0 * m as f64 + 1.0) / (2.0 * n);
                    v as f64 * angle.cos()
                })
                .sum::<f64>()
                * scale;
            assert!((c - mfcc as f64).abs() < 0.02, "mfcc[{k}]: {c} vs {mfcc}");
        }
        // 13 of 128 orthonormal coefficients cannot hold more variance than the bands.
        let energy = |v: &[Float]| v.iter().map(|&s| (s as f64).powi(2)).sum::<f64>();
        let (mfcc_var, mel_var) = (
            energy(&r.mfcc_std.unwrap()),
            energy(&r.mel_bands_std.unwrap()),
        );
        assert!(
            mfcc_var > 0.0 && mfcc_var <= mel_var * 1.0001,
            "{mfcc_var} vs {mel_var}"
        );
    }

    #[test]
    fn test_mel_bands_gain_shifts_mean_and_floor_not_std() {
        let y = mel_bands_signal();
        let half = y.mapv(|v| 0.5 * v);
        let config = feature_config(&["mel_bands"]);
        let a = analyze_signal(y.view(), 22050, &config).unwrap();
        let b = analyze_signal(half.view(), 22050, &config).unwrap();
        let shift = 20.0 * 0.5_f32.log10(); // -6.02 dB
        let floor_shift = b.mel_bands_floor_db.unwrap() - a.mel_bands_floor_db.unwrap();
        assert!((floor_shift - shift).abs() < 1e-3, "{floor_shift}");
        let (am, bm) = (a.mel_bands_mean.unwrap(), b.mel_bands_mean.unwrap());
        let (asd, bsd) = (a.mel_bands_std.unwrap(), b.mel_bands_std.unwrap());
        for m in 0..128 {
            assert!((bm[m] - am[m] - shift).abs() < 1e-3, "mean[{m}]");
            assert!((bsd[m] - asd[m]).abs() < 1e-3, "std[{m}]");
        }
        // The level step moves the tone bands; bands with no content sit at the floor.
        assert!(asd.iter().any(|&s| s > 5.0));
        let floor = a.mel_bands_floor_db.unwrap();
        assert!(am.iter().zip(&asd).any(|(&m, &s)| m == floor && s == 0.0));
    }

    #[test]
    fn test_mel_bands_silence_and_augment_merge() {
        // Silence: every cell is the 1e-10 power floor (-100 dB); the reported floor is
        // that value, not the lower -180 dB clamp, so empty bands still sit on it.
        let z = Array1::<Float>::zeros(22050 * 2);
        let r = analyze_signal(z.view(), 22050, &playlist()).unwrap();
        assert_eq!(r.mel_bands_mean, Some(vec![-100.0; 128]));
        assert_eq!(r.mel_bands_std, Some(vec![0.0; 128]));
        assert_eq!(r.mel_bands_floor_db, Some(-100.0));

        let y = mel_bands_signal();
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        merge_feature_fields(&mut out, &fresh, "mel_bands");
        assert!(out.mel_bands_mean.is_some());
        assert_eq!(out.mel_bands_mean, fresh.mel_bands_mean);
        assert_eq!(out.mel_bands_std, fresh.mel_bands_std);
        assert_eq!(out.mel_bands_floor_db, fresh.mel_bands_floor_db);
    }

    // --- tuning ---

    /// 4 s of sustained harmonic triads (8 harmonics at 1/k per note, MIDI
    /// 52..72), one per second, tuned to A4 = 440·2^(cents/1200).
    fn tuning_chords(cents: f64) -> Array1<Float> {
        const CHORDS: [[i32; 3]; 4] = [[57, 61, 64], [52, 59, 68], [62, 65, 69], [55, 62, 71]];
        let a4 = 440.0 * 2f64.powf(cents / 1200.0);
        Array1::from_shape_fn(22050 * 4, |i| {
            let t = i as f64 / 22050.0;
            let mut v = 0.0;
            for note in CHORDS[(t as usize).min(3)] {
                let f0 = a4 * 2f64.powf((note - 69) as f64 / 12.0);
                for k in 1..=8 {
                    v += (std::f64::consts::TAU * f0 * k as f64 * t).sin() / k as f64;
                }
            }
            (0.1 * v) as Float
        })
    }

    fn tuning_fields(y: &Array1<Float>, config: &AnalysisConfig) -> (Option<Float>, Option<Float>) {
        let r = analyze_signal(y.view(), 22050, config).unwrap();
        (r.tuning_offset_cents, r.tuning_confidence)
    }

    #[test]
    fn test_tuning_presence_follows_requests() {
        let y = tuning_chords(0.0);
        assert_eq!(tuning_fields(&y, &compact()), (None, None));
        let p = tuning_fields(&y, &playlist());
        assert!(p.1.is_some_and(Float::is_finite), "{p:?}");
        assert_eq!(tuning_fields(&y, &feature_config(&["mfcc"])), (None, None));
        // On its own it needs no extended pass and reads the same magnitudes.
        let only = feature_config(&["tuning"]);
        assert!(!only.needs_extended());
        assert_eq!(tuning_fields(&y, &only), p);
    }

    #[test]
    fn test_tuning_recovers_known_detuning() {
        for cents in [20.0, -45.0, 0.0] {
            let (offset, confidence) =
                tuning_fields(&tuning_chords(cents), &feature_config(&["tuning"]));
            let (offset, confidence) = (offset.unwrap(), confidence.unwrap());
            assert!((offset as f64 - cents).abs() < 1.0, "{cents}: {offset}");
            assert!((-50.0..50.0).contains(&offset));
            assert!(confidence > 0.9, "{cents}: {confidence}");
        }
    }

    #[test]
    fn test_tuning_single_sine() {
        let y = sine(440.0 * 2f32.powf(-30.0 / 1200.0), 22050, 3.0);
        let (offset, confidence) = tuning_fields(&y, &feature_config(&["tuning"]));
        assert!((offset.unwrap() + 30.0).abs() < 1.5, "{offset:?}");
        assert!(confidence.unwrap() > 0.9, "{confidence:?}");
    }

    #[test]
    fn test_tuning_abstains_on_noise_and_silence() {
        // Deterministic white noise from a 32-bit LCG.
        let mut state = 12345_u32;
        let noise = Array1::from_shape_fn(22050 * 5, |_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as Float / 8_388_608.0 - 1.0
        });
        let (offset, confidence) = tuning_fields(&noise, &feature_config(&["tuning"]));
        assert!(
            confidence.unwrap() < TUNING_MIN_CONFIDENCE,
            "{confidence:?}"
        );
        assert_eq!(offset, None);

        // A tone below the band leaves only FFT round-off there: no vote, no offset.
        // f64 phase: the f32 phase ramp of `sine` adds −80 dB noise of its own.
        let low = Array1::from_shape_fn(22050 * 3, |i| {
            (std::f64::consts::TAU * 110.0 * i as f64 / 22050.0).sin() as Float
        });
        let (offset, confidence) = tuning_fields(&low, &feature_config(&["tuning"]));
        assert_eq!(offset, None, "{confidence:?}");
        assert!(confidence.unwrap() < TUNING_MIN_CONFIDENCE);

        let z = Array1::<Float>::zeros(22050 * 2);
        assert_eq!(tuning_fields(&z, &playlist()), (None, Some(0.0)));
    }

    #[test]
    fn test_tuning_bar_rises_for_short_signals() {
        // Half a second (22 frames) of noise often reads above the fixed 0.08
        // bar; the 1.5/√frames bar keeps every draw from reporting an offset.
        let config = feature_config(&["tuning"]);
        let mut above_fixed_bar = 0;
        for seed in 1..=12_u32 {
            let mut state = seed.wrapping_mul(2_654_435_761);
            let noise = Array1::from_shape_fn(22050 / 2, |_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as Float / 8_388_608.0 - 1.0
            });
            let (offset, confidence) = tuning_fields(&noise, &config);
            assert_eq!(offset, None, "seed {seed}: {confidence:?}");
            above_fixed_bar += (confidence.unwrap() >= TUNING_MIN_CONFIDENCE) as usize;
        }
        assert!(
            above_fixed_bar > 0,
            "no draw exercised the short-signal bar"
        );
        // A clean chord of the same length still clears it.
        let short = tuning_chords(20.0).slice(s![..22050 / 2]).to_owned();
        let (offset, confidence) = tuning_fields(&short, &config);
        assert!(
            (offset.unwrap() - 20.0).abs() < 1.5,
            "{offset:?} {confidence:?}"
        );
    }

    #[test]
    fn test_tuning_augment_merge_moves_fields() {
        let y = tuning_chords(20.0);
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        merge_feature_fields(&mut out, &fresh, "tuning");
        assert!(out.tuning_offset_cents.is_some() && out.tuning_confidence.is_some());
        assert_eq!(out.tuning_offset_cents, fresh.tuning_offset_cents);
        assert_eq!(out.tuning_confidence, fresh.tuning_confidence);
    }

    // --- hpss ---

    /// A click every half second, the first at 0.25 s.
    fn hpss_click_positions(len: usize) -> impl Iterator<Item = usize> {
        (22050 / 4..len).step_by(22050 / 2)
    }

    /// Two tones, a click every half second and white noise (32-bit LCG).
    fn hpss_mix(secs: usize) -> Array1<Float> {
        let mut state = 20_261_122_u32;
        let mut y = Array1::from_shape_fn(22050 * secs, |i| {
            let t = i as Float / 22050.0;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as Float / 8_388_608.0 - 1.0;
            0.4 * (2.0 * PI * 330.0 * t).sin() + 0.2 * (2.0 * PI * 1870.0 * t).sin() + 0.05 * noise
        });
        for i in hpss_click_positions(y.len()) {
            y[i] += 0.9;
        }
        y
    }

    fn hpss_of(y: &Array1<Float>, config: &AnalysisConfig) -> Option<Float> {
        let r = analyze_signal(y.view(), 22050, config).unwrap();
        r.hpss_percussive_ratio
    }

    /// `features=["hpss"]` alone, the fused pass cut into chunks of `chunk` frames.
    fn hpss_chunked(y: &Array1<Float>, chunk: usize) -> Float {
        let config = feature_config(&["hpss"]);
        let zcr = validated_zero_crossing_rate(y.view()).unwrap();
        analyze_signal_inner(y.view(), 22050, false, zcr, &config, chunk)
            .unwrap()
            .hpss_percussive_ratio
            .unwrap()
    }

    /// The ratio of `y` straight from the definition with a `(frames, bins)`
    /// kernel, and the frame count, on the full magnitude matrix of the
    /// library STFT (same framing as the main pass).
    fn hpss_reference(y: &Array1<Float>, kernel: (usize, usize)) -> (Float, usize) {
        let hann = WindowSpec::Named("hann".into());
        let stft = spectrum::stft(
            y.view(),
            2048,
            Some(HOP_LENGTH),
            None,
            &hann,
            true,
            PadMode::Constant,
        )
        .unwrap();
        let mag = stft.mapv(|c| c.norm_sqr().sqrt());
        (hpss_reference_of(&mag, kernel), mag.ncols())
    }

    /// The ratio of a `[bins x frames]` magnitude matrix with a `(frames,
    /// bins)` kernel: every window collected with indices mirrored by
    /// repeated folding and sorted whole.
    fn hpss_reference_of(mag: &Array2<Float>, kernel: (usize, usize)) -> Float {
        let (bins, frames) = mag.dim();
        let fold = |mut i: isize, n: usize| {
            let n = n as isize;
            while !(0..n).contains(&i) {
                i = if i < 0 { -i - 1 } else { 2 * n - 1 - i };
            }
            i as usize
        };
        let median = |mut window: Vec<Float>| {
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            window[window.len() / 2]
        };
        let over_frames = (kernel.0 / 2) as isize;
        let over_bins = (kernel.1 / 2) as isize;
        let (mut sum_h, mut sum_p) = (0.0_f64, 0.0_f64);
        for t in 0..frames {
            let (mut frame_h, mut frame_p) = (0.0_f64, 0.0_f64);
            for b in 0..bins {
                let (ti, bi) = (t as isize, b as isize);
                let hm = median(
                    (ti - over_frames..=ti + over_frames)
                        .map(|i| mag[(b, fold(i, frames))])
                        .collect(),
                );
                let pm = median(
                    (bi - over_bins..=bi + over_bins)
                        .map(|i| mag[(fold(i, bins), t)])
                        .collect(),
                );
                let (s, wh, wp) = (mag[(b, t)], hm * hm, pm * pm);
                let tot = wh + wp;
                let (h, p) = if tot > 0.0 {
                    (s * wh / tot, s * wp / tot)
                } else {
                    (0.5 * s, 0.5 * s)
                };
                frame_h += (h as f64).powi(2);
                frame_p += (p as f64).powi(2);
            }
            sum_h += frame_h;
            sum_p += frame_p;
        }
        let total = sum_h + sum_p;
        let ratio = if total > 0.0 { sum_p / total } else { 0.0 };
        ratio as Float
    }

    #[test]
    fn test_hpss_kernel_scales_with_sample_rate() {
        assert_eq!(hpss_kernel(16000, 2048, 512), (23, 43));
        // librosa's default kernel at librosa's default resolution.
        assert_eq!(hpss_kernel(22050, 2048, 512), (31, 31));
        assert_eq!(hpss_kernel(44100, 2048, 512), (63, 17));
        assert_eq!(hpss_kernel(48000, 2048, 512), (69, 15));
    }

    #[test]
    fn test_hpss_stream_matches_reference_on_sparse_matrices() {
        // 60 % zeros and two repeated levels: ties in every window, cells
        // whose two medians are both 0 (split evenly), axes shorter than a
        // kernel, chunks shorter than the look-ahead. The kernels of 22050
        // and 44100 Hz.
        let mut state = 7_u32;
        for (bins, frames) in [(40, 1), (3, 50), (64, 45), (33, 70)] {
            let mag = Array2::from_shape_fn((bins, frames), |_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let u = (state >> 8) as Float / 16_777_216.0;
                if u < 0.6 {
                    0.0
                } else {
                    (u * 4.0).floor() * 0.25
                }
            });
            for kernel in [(31, 31), (63, 17)] {
                let want = hpss_reference_of(&mag, kernel);
                assert!(
                    want > 0.0 && want < 1.0,
                    "{bins}x{frames} {kernel:?}: {want}"
                );
                for chunk in [frames, 11, 4, 1] {
                    for parallel in [false, true] {
                        let mut stream = HpssStream::new(bins, frames, kernel.0);
                        for base in (0..frames).step_by(chunk) {
                            let columns = (base..(base + chunk).min(frames)).map(|t| {
                                let column = mag.column(t).to_vec();
                                let percussive = hpss_median(&column, kernel.1);
                                (column, percussive)
                            });
                            stream.push(columns, parallel);
                        }
                        let got = stream.ratio();
                        assert!(
                            (got - want).abs() < 1e-6,
                            "{bins}x{frames} {kernel:?}, chunk {chunk}, parallel {parallel}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_hpss_presence_follows_requests() {
        let y = hpss_mix(3);
        // Opt-in: Compact and Playlist leave it out, Full computes it.
        for config in [compact(), playlist()] {
            assert_eq!(hpss_of(&y, &config), None);
        }
        assert_eq!(hpss_of(&y, &feature_config(&["mfcc"])), None);
        // On its own it needs no extended pass and reads the same magnitudes.
        let only = feature_config(&["hpss"]);
        assert!(!only.needs_extended());
        let r = hpss_of(&y, &only).unwrap();
        assert!(r > 0.0 && r < 1.0, "{r}");
        assert_eq!(hpss_of(&y, &full()), Some(r));
        assert_eq!(hpss_of(&y, &feature_config(&["hpss", "mfcc"])), Some(r));
    }

    #[test]
    fn test_hpss_known_answers() {
        let config = feature_config(&["hpss"]);
        let tone = sine(440.0, 22050, 3.0);
        let r = hpss_of(&tone, &config).unwrap();
        assert!(r < 0.01, "sine: {r}");

        let stack = Array1::from_shape_fn(22050 * 3, |i| {
            let t = i as Float / 22050.0;
            (1..=8)
                .map(|k| 0.2 * (2.0 * PI * 220.0 * k as Float * t).sin() / k as Float)
                .sum::<Float>()
        });
        let r = hpss_of(&stack, &config).unwrap();
        assert!(r < 0.01, "harmonic stack: {r}");

        let mut clicks = Array1::<Float>::zeros(tone.len());
        for i in hpss_click_positions(clicks.len()) {
            clicks[i] = 1.0;
        }
        let r = hpss_of(&clicks, &config).unwrap();
        assert!(r > 0.99, "clicks: {r}");

        // Clicks carrying as much time-domain energy as the sine.
        let energy: f64 = tone.iter().map(|&v| (v as f64).powi(2)).sum();
        let n_clicks = hpss_click_positions(tone.len()).count();
        let mut both = tone.clone();
        for i in hpss_click_positions(both.len()) {
            both[i] += (energy / n_clicks as f64).sqrt() as Float;
        }
        let r = hpss_of(&both, &config).unwrap();
        assert!((r - 0.5).abs() < 0.02, "sine + clicks: {r}");

        // Digital silence: no energy to split, 0.0 rather than an abstention.
        let z = Array1::<Float>::zeros(22050 * 2);
        assert_eq!(hpss_of(&z, &config), Some(0.0));
    }

    #[test]
    fn test_hpss_matches_brute_force_reference() {
        let y = hpss_mix(3);
        let (want, frames) = hpss_reference(&y, (31, 31));
        assert_eq!(frames, 1 + y.len() / HOP_LENGTH);
        let got = hpss_of(&y, &feature_config(&["hpss"])).unwrap();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }

    #[test]
    fn test_hpss_at_44100_matches_brute_force_reference() {
        // The same samples read at 44100 Hz: twice the frames, half the bins.
        let y = hpss_mix(3);
        let (want, _) = hpss_reference(&y, (63, 17));
        // The 22050 Hz kernel would not pass the comparison below.
        let (other, _) = hpss_reference(&y, (31, 31));
        assert!((want - other).abs() > 1e-4, "{want} vs {other}");
        let r = analyze_signal(y.view(), 44100, &feature_config(&["hpss"])).unwrap();
        let got = r.hpss_percussive_ratio.unwrap();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }

    #[test]
    fn test_hpss_chunking_is_bit_identical() {
        // 13 s is 560 frames: more than one chunk of FRAME_CHUNK.
        let y = hpss_mix(13);
        let zcr = validated_zero_crossing_rate(y.view()).unwrap();
        let config = feature_config(&["hpss", "spectral_texture", "tuning"]);
        let extended = config.needs_extended();
        let run =
            |chunk| analyze_signal_inner(y.view(), 22050, extended, zcr, &config, chunk).unwrap();
        let whole = run(usize::MAX);
        assert!(whole
            .hpss_percussive_ratio
            .is_some_and(|r| r > 0.0 && r < 1.0));
        // Chunks longer and shorter than the look-ahead; every output matches.
        for chunk in [FRAME_CHUNK, 7, 1] {
            assert_eq!(run(chunk), whole, "chunk {chunk}");
        }
        assert_eq!(
            hpss_chunked(&y, 1).to_bits(),
            whole.hpss_percussive_ratio.unwrap().to_bits()
        );
    }

    #[test]
    fn test_hpss_short_inputs_match_reference() {
        // 1, 2, 5 and 12 frames are fewer than the look-ahead, 20 fewer than
        // the kernel: the time window folds back, more than once on the shortest.
        let mix = hpss_mix(1);
        for (len, frames) in [(300, 1), (700, 2), (2300, 5), (6000, 12), (10000, 20)] {
            // The slice holds the click at 5512.
            let y = mix.slice(s![5400..5400 + len]).to_owned();
            let (want, n) = hpss_reference(&y, (31, 31));
            assert_eq!(n, frames);
            assert!(want > 0.0 && want < 1.0, "{frames} frames: {want}");
            for chunk in [usize::MAX, 3, 1] {
                let got = hpss_chunked(&y, chunk);
                assert!(
                    (got - want).abs() < 1e-6,
                    "{frames} frames, chunk {chunk}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn test_hpss_augment_merge_moves_field() {
        let y = hpss_mix(3);
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &feature_config(&["hpss"])).unwrap();
        merge_feature_fields(&mut out, &fresh, "hpss");
        assert!(out.hpss_percussive_ratio.is_some());
        assert_eq!(out.hpss_percussive_ratio, fresh.hpss_percussive_ratio);
    }

    #[test]
    fn test_hpss_request_leaves_other_fields_untouched() {
        let y = hpss_mix(3);
        let requests: [&[&str]; 2] = [
            &["tuning", "mel_bands"],
            &[
                "bpm",
                "mfcc",
                "chroma",
                "spectral_shape",
                "spectral_texture",
                "tuning",
                "key",
                "onset_bands",
            ],
        ];
        for names in requests {
            let without = analyze_signal(y.view(), 22050, &feature_config(names)).unwrap();
            let with_hpss = feature_config(&[names, &["hpss"]].concat());
            let mut with = analyze_signal(y.view(), 22050, &with_hpss).unwrap();
            assert!(with.hpss_percussive_ratio.take().is_some());
            with.provenance.requested_features = without.provenance.requested_features.clone();
            assert_eq!(with, without);
        }
    }

    // --- rhythm_levels ---

    /// Frames per second of the main pass at 22050 Hz.
    const RHYTHM_LEVELS_FR: f64 = 22050.0 / 512.0;

    /// 30 s onset curve at 120 BPM: a unit impulse at `round((k + offset) * T)`
    /// for every beat `k` and every `offset` (in beats).
    fn rhythm_levels_curve(offsets: &[f64]) -> Vec<Float> {
        rhythm_levels_curve_at(RHYTHM_LEVELS_FR, offsets)
    }

    /// The same curve at `fr` frames per second.
    fn rhythm_levels_curve_at(fr: f64, offsets: &[f64]) -> Vec<Float> {
        let period = fr * 60.0 / 120.0;
        let mut o = vec![0.0; (30.0 * fr) as usize + 1];
        for &offset in offsets {
            for k in 0.. {
                let i = ((k as f64 + offset) * period).round() as usize;
                if i >= o.len() {
                    break;
                }
                o[i] = 1.0;
            }
        }
        o
    }

    fn rhythm_levels_at_120(o: &[Float]) -> (Option<Float>, Option<Float>, Float) {
        rhythm_levels(o, 120.0, 22050, 512)
    }

    #[test]
    fn test_rhythm_levels_synthetic_patterns() {
        // Beats only: nothing repeats at half or two thirds of a beat.
        let (double, one_and_half, confidence) = rhythm_levels_at_120(&rhythm_levels_curve(&[0.0]));
        assert_eq!((double, one_and_half), (Some(0.0), Some(0.0)));
        // The period is 21.53 frames, so the rounded impulses lie 21 or 22
        // frames apart and the interpolated value is about 0.45, not 1.
        assert!(confidence > 0.4, "{confidence}");
        // Eighth notes.
        let (double, one_and_half, _) = rhythm_levels_at_120(&rhythm_levels_curve(&[0.0, 0.5]));
        assert!(double.unwrap() > 0.5, "{double:?}");
        assert!(one_and_half.unwrap() < 0.05, "{one_and_half:?}");
        // A hit two thirds of a beat after every beat.
        let (_, one_and_half, _) = rhythm_levels_at_120(&rhythm_levels_curve(&[0.0, 2.0 / 3.0]));
        assert!(one_and_half.unwrap() > 0.2, "{one_and_half:?}");
        // The frame is 8 s at any frame rate: the same patterns at 44100 Hz.
        let at_44100 = |offsets: &[f64]| {
            let o = rhythm_levels_curve_at(44100.0 / 512.0, offsets);
            rhythm_levels(&o, 120.0, 44100, 512)
        };
        assert_eq!(at_44100(&[0.0]).0, Some(0.0));
        let double = at_44100(&[0.0, 0.5]).0;
        assert!(double.unwrap() > 0.5, "{double:?}");
    }

    #[test]
    fn test_rhythm_levels_ignore_a_constant_offset() {
        let o = rhythm_levels_curve(&[0.0, 0.5, 2.0 / 3.0]);
        let raised: Vec<Float> = o.iter().map(|&v| v + 3.0).collect();
        let (a, b) = (rhythm_levels_at_120(&o), rhythm_levels_at_120(&raised));
        assert!((a.0.unwrap() - b.0.unwrap()).abs() < 1e-4, "{a:?} {b:?}");
        assert!((a.1.unwrap() - b.1.unwrap()).abs() < 1e-4, "{a:?} {b:?}");
        assert!((a.2 - b.2).abs() < 1e-4, "{a:?} {b:?}");
    }

    #[test]
    fn test_rhythm_levels_abstain() {
        let o = rhythm_levels_curve(&[0.0]);
        let window = (8.0 * RHYTHM_LEVELS_FR).round() as usize;
        // Shorter than one frame, no tempo, no variation.
        assert_eq!(rhythm_levels_at_120(&o[..window - 1]), (None, None, 0.0));
        assert_eq!(rhythm_levels(&o, 0.0, 22050, 512), (None, None, 0.0));
        assert_eq!(rhythm_levels(&o, Float::NAN, 22050, 512), (None, None, 0.0));
        assert_eq!(rhythm_levels_at_120(&vec![0.7; o.len()]), (None, None, 0.0));
        // A beat period that does not fit the frame is not measured either.
        assert_eq!(rhythm_levels(&o, 1.0, 22050, 512), (None, None, 0.0));
    }

    /// The same measure written out naively: every lag of every frame.
    fn rhythm_levels_reference(o: &[Float], bpm: f64) -> [f64; 3] {
        let (w, h) = (345, 22);
        let period = RHYTHM_LEVELS_FR * 60.0 / bpm;
        let mut means = [0.0; 3];
        let mut frames = 0.0;
        let mut start = 0;
        while start + w <= o.len() {
            let mean = o[start..start + w].iter().map(|&v| v as f64).sum::<f64>() / w as f64;
            let x: Vec<f64> = (0..w).map(|n| o[start + n] as f64 - mean).collect();
            let acf: Vec<f64> = (0..w)
                .map(|l| (0..w - l).map(|n| x[n] * x[n + l]).sum())
                .collect();
            for (m, lag) in means
                .iter_mut()
                .zip([period, period / 2.0, period * 2.0 / 3.0])
            {
                let i = lag as usize;
                let v = acf[i] + (lag - i as f64) * (acf[i + 1] - acf[i]);
                *m += (v / acf[0]).max(0.0);
            }
            frames += 1.0;
            start += h;
        }
        [means[1] / means[0], means[2] / means[0], means[0] / frames]
    }

    #[test]
    fn test_rhythm_levels_match_naive_reference() {
        // 20 s of non-negative noise (32-bit LCG).
        let mut state = 20_261_125_u32;
        let o: Vec<Float> = (0..(20.0 * RHYTHM_LEVELS_FR) as usize)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as Float / 16_777_216.0
            })
            .collect();
        let want = rhythm_levels_reference(&o, 128.0);
        let (double, one_and_half, confidence) = rhythm_levels(&o, 128.0, 22050, 512);
        let got = [double.unwrap(), one_and_half.unwrap(), confidence];
        for (g, w) in got.iter().zip(want) {
            assert!((*g as f64 - w).abs() < 1e-5, "{got:?} {want:?}");
        }
    }

    fn rhythm_levels_of(
        y: &Array1<Float>,
        config: &AnalysisConfig,
    ) -> (Option<Float>, Option<Float>, Option<Float>) {
        let r = analyze_signal(y.view(), 22050, config).unwrap();
        (
            r.rhythm_level_double_ratio,
            r.rhythm_level_one_and_half_ratio,
            r.rhythm_level_confidence,
        )
    }

    #[test]
    fn test_rhythm_levels_presence_follows_requests() {
        // A click every half second: 12 s hold a full frame.
        let y = hpss_mix(12);
        // Playlist and Full compute it; Compact does not.
        assert_eq!(rhythm_levels_of(&y, &compact()), (None, None, None));
        assert_eq!(
            rhythm_levels_of(&y, &feature_config(&["mfcc"])),
            (None, None, None)
        );
        // On its own it needs no extended pass; the tempo is always tracked.
        let only = feature_config(&["rhythm_levels"]);
        assert!(!only.needs_extended());
        let got = rhythm_levels_of(&y, &only);
        assert!(got.0.is_some() && got.1.is_some(), "{got:?}");
        assert!(got.2.unwrap() > 0.0 && got.2.unwrap() <= 1.0, "{got:?}");
        for config in [playlist(), full()] {
            assert_eq!(rhythm_levels_of(&y, &config), got);
        }
        assert_eq!(
            rhythm_levels_of(&y, &feature_config(&["rhythm_levels", "mfcc"])),
            got
        );
        // Too short to measure: the confidence stays, the ratios abstain.
        let short = rhythm_levels_of(&hpss_mix(3), &only);
        assert_eq!(short, (None, None, Some(0.0)));
        // Augment moves all three fields.
        let mut out = analyze_signal(y.view(), 22050, &compact()).unwrap();
        let fresh = analyze_signal(y.view(), 22050, &only).unwrap();
        merge_feature_fields(&mut out, &fresh, "rhythm_levels");
        assert_eq!(
            (
                out.rhythm_level_double_ratio,
                out.rhythm_level_one_and_half_ratio,
                out.rhythm_level_confidence
            ),
            got
        );
    }

    // --- ebur128 / native_dynamics / amplitude_envelope (file-only) ---

    const NATIVE_FEATURES: [&str; 4] = [
        "ebur128",
        "native_dynamics",
        "amplitude_envelope",
        "stereo_field",
    ];

    /// Stereo 16-bit 48 kHz WAV: 1 kHz sine at `peak_dbfs` (silence if `None`),
    /// in phase on both channels.
    fn write_native_dynamics_wav(
        tag: &str,
        peak_dbfs: Option<f64>,
        secs: usize,
    ) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sonara-native-dynamics-{tag}-{}.wav",
            std::process::id()
        ));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let amp = peak_dbfs.map_or(0.0, |db| 10f64.powf(db / 20.0));
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48_000 * secs {
            let phase = 2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 48_000.0;
            let sample = (amp * phase.sin() * 32768.0).round() as i16;
            writer.write_sample(sample).unwrap();
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        path
    }

    fn ebur128_fields(r: &TrackAnalysis) -> [Option<Float>; 5] {
        [
            r.ebur128_integrated_lufs,
            r.ebur128_momentary_max_lufs,
            r.ebur128_short_term_max_lufs,
            r.ebur128_range_lu,
            r.ebur128_true_peak_dbtp,
        ]
    }

    fn native_dynamics_fields(r: &TrackAnalysis) -> [Option<Float>; 3] {
        [
            r.sample_peak_dbfs,
            r.crest_factor_db,
            r.crest_factor_frame_mean_db,
        ]
    }

    fn amplitude_envelope_fields(r: &TrackAnalysis) -> (Option<Vec<Float>>, [Option<Float>; 2]) {
        (
            r.amplitude_envelope.clone(),
            [r.amplitude_envelope_hop_sec, r.amplitude_envelope_frame_sec],
        )
    }

    /// `r` with the 16 file-only fields and the recorded request cleared.
    fn without_native_fields(mut r: TrackAnalysis) -> TrackAnalysis {
        let blank = analyze_signal(sine(440.0, 22050, 0.5).view(), 22050, &compact()).unwrap();
        for name in NATIVE_FEATURES {
            merge_feature_fields(&mut r, &blank, name);
        }
        r.provenance.requested_features = None;
        r
    }

    #[test]
    fn test_native_dynamics_reference_tone() {
        let path = write_native_dynamics_wav("tone", Some(-23.0), 5);
        let r = analyze_file(&path, 22050, &feature_config(&NATIVE_FEATURES)).unwrap();
        std::fs::remove_file(&path).unwrap();

        // EBU Tech 3341 case: a -23 dBFS 1 kHz stereo sine reads -23.0 LUFS.
        let lufs = r.ebur128_integrated_lufs.unwrap();
        assert!((lufs + 23.0).abs() < 0.1, "{lufs}");
        for v in ebur128_fields(&r) {
            assert!(v.is_some_and(Float::is_finite), "{v:?}");
        }
        assert_eq!(r.native_sample_rate, Some(48_000));
        let peak = r.sample_peak_dbfs.unwrap();
        assert!((peak + 23.0).abs() < 0.05, "{peak}");
        let crest = r.crest_factor_db.unwrap();
        assert!((crest - 3.01).abs() < 0.05, "{crest}");
        let frame_crest = r.crest_factor_frame_mean_db.unwrap();
        assert!((frame_crest - 3.01).abs() < 0.1, "{frame_crest}");

        let n = 48_000 * 5;
        let envelope = r.amplitude_envelope.as_ref().unwrap();
        assert_eq!(envelope.len(), 1 + (n * 22050) / (512 * 48_000));
        let amp = 10f32.powf(-23.0 / 20.0);
        assert!(envelope.iter().all(|&v| (v - amp).abs() < 1e-3));
        let hop = crate::native_dynamics::ENVELOPE_HOP_SEC as Float;
        let frame = crate::native_dynamics::ENVELOPE_FRAME_SEC as Float;
        assert_eq!(r.amplitude_envelope_hop_sec, Some(hop));
        assert_eq!(r.amplitude_envelope_frame_sec, Some(frame));
        assert!((hop - 0.02322).abs() < 1e-5 && (frame - 0.09288).abs() < 1e-5);
    }

    #[test]
    fn test_native_dynamics_presence_follows_requests() {
        let path = write_native_dynamics_wav("presence", Some(-23.0), 5);
        let run = |names: &[&str]| analyze_file(&path, 22050, &feature_config(names)).unwrap();
        let none = run(&["rms"]);
        let all = run(&NATIVE_FEATURES);
        // Each feature fills its own group only.
        let groups = |r: &TrackAnalysis| {
            [
                ebur128_fields(r) != ebur128_fields(&none),
                native_dynamics_fields(r) != native_dynamics_fields(&none),
                amplitude_envelope_fields(r) != amplitude_envelope_fields(&none),
                stereo_field_fields(r) != stereo_field_fields(&none),
            ]
        };
        assert_eq!(ebur128_fields(&none), [None; 5]);
        assert_eq!(native_dynamics_fields(&none), [None; 3]);
        assert_eq!(none.native_sample_rate, None);
        assert_eq!(amplitude_envelope_fields(&none), (None, [None; 2]));
        assert_eq!(stereo_field_fields(&none), [None; 4]);
        assert_eq!(groups(&all), [true; 4]);
        for (i, name) in NATIVE_FEATURES.iter().enumerate() {
            let one = run(&[*name]);
            let mut expected = [false; 4];
            expected[i] = true;
            assert_eq!(groups(&one), expected, "{name}");
            // Any of the four marks the record as measured from a file.
            assert_eq!(one.native_sample_rate, Some(48_000), "{name}");
            // The envelope run shares the dynamics pass but emits the same values.
            assert_eq!(
                [
                    ebur128_fields(&one) == ebur128_fields(&all),
                    native_dynamics_fields(&one) == native_dynamics_fields(&all),
                    amplitude_envelope_fields(&one) == amplitude_envelope_fields(&all),
                    stereo_field_fields(&one) == stereo_field_fields(&all),
                ],
                expected,
                "{name}"
            );
        }

        // A bare signal has no file: requested or not, the fields stay None.
        let y = sine(440.0, 22050, 2.0);
        let signal = analyze_signal(y.view(), 22050, &feature_config(&NATIVE_FEATURES)).unwrap();
        assert_eq!(groups(&signal), [false; 4]);
        assert_eq!(signal.native_sample_rate, None);
        std::fs::remove_file(&path).unwrap();

        // Digital silence: levels are undefined, the envelope is all zeros.
        let silent = write_native_dynamics_wav("silence", None, 1);
        let z = analyze_file(&silent, 22050, &feature_config(&NATIVE_FEATURES)).unwrap();
        std::fs::remove_file(&silent).unwrap();
        assert_eq!(ebur128_fields(&z), [None; 5]);
        assert_eq!(z.native_sample_rate, Some(48_000));
        assert_eq!(native_dynamics_fields(&z), [None; 3]);
        assert_eq!(stereo_field_fields(&z), [None; 4]);
        assert!(z
            .amplitude_envelope
            .is_some_and(|e| !e.is_empty() && e.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_native_dynamics_leave_other_fields_bit_identical() {
        let wav = write_native_dynamics_wav("identical", Some(-6.0), 3);
        let mut bases = vec![vec!["rms", "loudness"], vec!["rms", "loudness", "tags"]];
        if cfg!(feature = "aggression") {
            bases.push(vec!["rms", "loudness", "aggression"]);
        }
        for path in [wav.clone(), fixture("tagged.flac")] {
            for base in &bases {
                // 0 = native rate; 22050 and 16000 resample (and split the
                // aggression lanes).
                for sr in [0, 22050, 16000] {
                    let mut names = base.clone();
                    names.extend(NATIVE_FEATURES);
                    let with = analyze_file(&path, sr, &feature_config(&names)).unwrap();
                    let plain = analyze_file(&path, sr, &feature_config(base)).unwrap();
                    assert!(with.native_sample_rate.is_some());
                    assert!(
                        without_native_fields(with) == without_native_fields(plain),
                        "{} sr={sr} {base:?}",
                        path.display()
                    );
                }
            }
        }
        std::fs::remove_file(&wav).unwrap();
    }

    #[test]
    fn test_native_dynamics_augment_merge_moves_fields() {
        let path = write_native_dynamics_wav("merge", Some(-23.0), 5);
        let fresh = analyze_file(&path, 22050, &feature_config(&NATIVE_FEATURES)).unwrap();
        let cached = analyze_file(&path, 22050, &feature_config(&["rms"])).unwrap();

        let mut out = cached.clone();
        merge_feature_fields(&mut out, &fresh, "ebur128");
        assert_eq!(ebur128_fields(&out), ebur128_fields(&fresh));
        assert_eq!(native_dynamics_fields(&out), [None; 3]);
        assert_eq!(out.native_sample_rate, Some(48_000));
        merge_feature_fields(&mut out, &fresh, "native_dynamics");
        assert_eq!(native_dynamics_fields(&out), native_dynamics_fields(&fresh));
        assert_eq!(amplitude_envelope_fields(&out), (None, [None; 2]));
        merge_feature_fields(&mut out, &fresh, "amplitude_envelope");
        assert_eq!(
            amplitude_envelope_fields(&out),
            amplitude_envelope_fields(&fresh)
        );
        assert_eq!(stereo_field_fields(&out), [None; 4]);
        merge_feature_fields(&mut out, &fresh, "stereo_field");
        assert_eq!(stereo_field_fields(&out), stereo_field_fields(&fresh));

        // End to end: the audio fallback of augment fills the same values.
        let augmented =
            augment_analysis(&cached, &NATIVE_FEATURES, Some(&path), &compact()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(without_native_fields(augmented.clone()) == without_native_fields(cached));
        assert_eq!(ebur128_fields(&augmented), ebur128_fields(&fresh));
        assert_eq!(
            native_dynamics_fields(&augmented),
            native_dynamics_fields(&fresh)
        );
        assert_eq!(
            amplitude_envelope_fields(&augmented),
            amplitude_envelope_fields(&fresh)
        );
        assert_eq!(stereo_field_fields(&augmented), stereo_field_fields(&fresh));
    }

    // --- stereo_field (file-only) ---

    const STEREO_RATES: [u32; 4] = [22050, 44100, 48000, 96000];
    /// Ten cycles per correlation frame, inside the low band.
    const STEREO_TONE_HZ: f64 = 10.0 * 22050.0 / 2048.0;

    fn stereo_field_fields(r: &TrackAnalysis) -> [Option<Float>; 4] {
        [
            r.stereo_side_ratio,
            r.stereo_side_ratio_low,
            r.stereo_correlation_mean,
            r.stereo_correlation_p05,
        ]
    }

    fn stereo_values(s: StereoField) -> [Option<Float>; 4] {
        [
            s.side_ratio,
            s.side_ratio_low,
            s.correlation_mean,
            s.correlation_p05,
        ]
    }

    fn wave(freq: f64, t: f64) -> f64 {
        (2.0 * std::f64::consts::PI * freq * t).sin()
    }

    /// 60 Hz plus 1 kHz: energy in the low band and above it.
    fn two_tones(t: f64) -> f64 {
        0.4 * (wave(60.0, t) + wave(1000.0, t))
    }

    /// Channels `(l, r)` as functions of time in seconds.
    fn stereo_pair(sr: u32, secs: f64, f: impl Fn(f64) -> (f64, f64)) -> Array2<Float> {
        let n = (sr as f64 * secs) as usize;
        let mut out = Array2::zeros((2, n));
        for i in 0..n {
            let (l, r) = f(i as f64 / sr as f64);
            out[[0, i]] = l as Float;
            out[[1, i]] = r as Float;
        }
        out
    }

    /// `[side_ratio, side_ratio_low, correlation_mean, correlation_p05]` within `tol`.
    fn assert_stereo(x: &Array2<Float>, sr: u32, want: [f64; 4], tol: f64, label: &str) {
        let got = stereo_values(stereo_field(x.view(), sr));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.is_some_and(|g| (g as f64 - w).abs() <= tol),
                "{label} sr={sr} field {i}: {got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn test_stereo_field_exact_cases() {
        for sr in STEREO_RATES {
            let same = stereo_pair(sr, 4.0, |t| (two_tones(t), two_tones(t)));
            assert_stereo(&same, sr, [0.0, 0.0, 1.0, 1.0], 0.01, "L = R");
            // A mono file plays as L = R.
            let mono = same.slice(s![..1, ..]).to_owned();
            assert_stereo(&mono, sr, [0.0, 0.0, 1.0, 1.0], 0.01, "mono");

            let inverted = stereo_pair(sr, 4.0, |t| (two_tones(t), -two_tones(t)));
            assert_stereo(&inverted, sr, [1.0, 1.0, -1.0, -1.0], 0.01, "R = -L");

            // Panning R = a L: side share (1 - 2a / (1 + a^2)) / 2 = 0.1 at a = 0.5.
            let panned = stereo_pair(sr, 4.0, |t| (two_tones(t), 0.5 * two_tones(t)));
            assert_stereo(&panned, sr, [0.1, 0.1, 1.0, 1.0], 0.01, "R = 0.5 L");

            // One tone shifted by phi: correlation cos(phi), side share (1 - cos(phi)) / 2.
            for degrees in [60.0_f64, 120.0] {
                let phi = degrees.to_radians();
                let shifted = stereo_pair(sr, 4.0, |t| {
                    let phase = 2.0 * std::f64::consts::PI * STEREO_TONE_HZ * t;
                    (0.5 * phase.sin(), 0.5 * (phase + phi).sin())
                });
                let (c, side) = (phi.cos(), (1.0 - phi.cos()) / 2.0);
                assert_stereo(&shifted, sr, [side, side, c, c], 0.01, "phase shift");
            }

            // A constant offset in one channel is not stereo content.
            let offset = stereo_pair(sr, 4.0, |t| (two_tones(t) + 0.1, two_tones(t)));
            assert_stereo(&offset, sr, [0.0, 0.0, 1.0, 1.0], 0.01, "L = R + 0.1");
        }
    }

    #[test]
    fn test_stereo_field_independent_noise() {
        for sr in STEREO_RATES {
            // Knuth's MMIX generator; rows are filled one after the other, so
            // the channels are independent.
            let mut state = 1_u64;
            let noise = Array2::from_shape_fn((2, sr as usize * 4), |_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f64 / (1_u64 << 24) as f64 - 0.5) as Float
            });
            let [side, low, mean, p05] =
                stereo_values(stereo_field(noise.view(), sr)).map(Option::unwrap);
            assert!((side - 0.5).abs() < 0.02, "sr={sr} side {side}");
            // About 2 * 150 Hz * 4 s independent values in the low band.
            assert!((low - 0.5).abs() < 0.1, "sr={sr} low {low}");
            assert!(mean.abs() < 0.02, "sr={sr} mean {mean}");
            assert!(p05 < mean && p05 > -0.1, "sr={sr} p05 {p05}");
        }
    }

    #[test]
    fn test_stereo_field_low_band_differs_from_the_whole() {
        for sr in STEREO_RATES {
            let wide_top = stereo_pair(sr, 4.0, |t| {
                let (low, high) = (0.4 * wave(60.0, t), 0.4 * wave(2000.0, t));
                (low + high, low - high)
            });
            let got = stereo_field(wide_top.view(), sr);
            assert!(got.side_ratio_low.unwrap() < 0.01, "sr={sr} {got:?}");
            assert!(
                (got.side_ratio.unwrap() - 0.5).abs() < 0.01,
                "sr={sr} {got:?}"
            );

            let wide_bass = stereo_pair(sr, 4.0, |t| {
                let (low, high) = (0.4 * wave(60.0, t), 0.4 * wave(2000.0, t));
                (low + high, high - low)
            });
            let got = stereo_field(wide_bass.view(), sr);
            assert!(got.side_ratio_low.unwrap() > 0.99, "sr={sr} {got:?}");
            assert!(
                (got.side_ratio.unwrap() - 0.5).abs() < 0.01,
                "sr={sr} {got:?}"
            );
        }
    }

    #[test]
    fn test_stereo_field_edges() {
        for sr in STEREO_RATES {
            let n = sr as usize * 4;
            let silence = Array2::<Float>::zeros((2, n));
            assert_eq!(stereo_values(stereo_field(silence.view(), sr)), [None; 4]);
            let empty = Array2::<Float>::zeros((2, 0));
            assert_eq!(stereo_values(stereo_field(empty.view(), sr)), [None; 4]);

            // Past two channels the layout is unknown.
            let three =
                Array2::from_shape_fn((3, n), |(_, i)| two_tones(i as f64 / sr as f64) as Float);
            assert_eq!(stereo_values(stereo_field(three.view(), sr)), [None; 4]);

            // Shorter than one correlation frame: the share is defined, the correlation is not.
            let short = stereo_pair(sr, 0.05, |t| (two_tones(t), two_tones(t)));
            let got = stereo_field(short.view(), sr);
            assert_eq!(got.side_ratio, Some(0.0));
            assert_eq!([got.correlation_mean, got.correlation_p05], [None; 2]);

            // One channel empty: correlation 0 by convention, half the energy is Side.
            let one_sided = stereo_pair(sr, 4.0, |t| (two_tones(t), 0.0));
            assert_stereo(&one_sided, sr, [0.5, 0.5, 0.0, 0.0], 0.01, "R = 0");
            // A constant channel has no variance either, whatever its level.
            let flat = stereo_pair(sr, 4.0, |t| (two_tones(t), 0.3));
            assert_stereo(&flat, sr, [0.5, 0.5, 0.0, 0.0], 0.01, "R = 0.3");

            // Silent frames stay out of the statistics, with or without a
            // constant offset: counted in, they would pull the mean to -0.5.
            for offset in [0.0, 0.01] {
                let half_silent = stereo_pair(sr, 4.0, |t| {
                    let x = if t < 2.0 { two_tones(t) } else { 0.0 };
                    (x + offset, offset - x)
                });
                assert_stereo(
                    &half_silent,
                    sr,
                    [1.0, 1.0, -1.0, -1.0],
                    0.01,
                    "half silent",
                );
            }

            // Only a constant: nothing varies, so nothing is defined.
            let constant = stereo_pair(sr, 4.0, |_| (0.1, 0.3));
            assert_eq!(
                stereo_values(stereo_field(constant.view(), sr)),
                [None; 4]
            );
            // An offset on a signal without bass does not read as wide bass.
            let no_bass = stereo_pair(sr, 4.0, |t| {
                let x = 0.5 * wave(1000.0, t);
                (x + 0.1, x)
            });
            let got = stereo_field(no_bass.view(), sr);
            assert!(got.side_ratio_low.unwrap_or(0.0) < 0.01, "sr={sr} {got:?}");
            assert!(got.side_ratio.unwrap() < 0.01, "sr={sr} {got:?}");
        }
        // 43 frames at 22050 Hz, the first three inverted: the mean is 37/43 and
        // the 5th percentile lies 0.1 of the way from -1 to +1.
        let mostly_same = stereo_pair(22050, 4.0, |t| {
            let x = two_tones(t);
            (x, if t < 3.0 * 2048.0 / 22050.0 { -x } else { x })
        });
        let got = stereo_field(mostly_same.view(), 22050);
        assert!(
            (got.correlation_mean.unwrap() - 37.0 / 43.0).abs() < 1e-3,
            "{got:?}"
        );
        assert!((got.correlation_p05.unwrap() + 0.8).abs() < 1e-3, "{got:?}");
        // The low band needs its 150 Hz edge below Nyquist.
        let slow = stereo_pair(300, 4.0, |t| (wave(20.0, t), -wave(20.0, t)));
        let got = stereo_field(slow.view(), 300);
        assert_eq!(got.side_ratio_low, None);
        assert_eq!(got.side_ratio, Some(1.0));
    }

    /// Stereo 16-bit 48 kHz WAV, 3 s of `two_tones` with the right channel at half level.
    fn write_stereo_field_wav(tag: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "sonara-stereo-field-{tag}-{}.wav",
            std::process::id()
        ));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48_000 * 3 {
            let x = two_tones(i as f64 / 48_000.0);
            for level in [1.0, 0.5] {
                writer
                    .write_sample((level * x * 32768.0).round() as i16)
                    .unwrap();
            }
        }
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn test_stereo_field_presence_follows_requests() {
        let path = write_stereo_field_wav("presence");
        let with = analyze_file(&path, 22050, &feature_config(&["stereo_field"])).unwrap();
        let without = analyze_file(&path, 22050, &feature_config(&["rms"])).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(with.native_sample_rate, Some(48_000));
        for (got, want) in stereo_field_fields(&with).iter().zip([0.1, 0.1, 1.0, 1.0]) {
            assert!(
                got.is_some_and(|g| (g - want).abs() < 0.01),
                "{got:?} vs {want}"
            );
        }
        assert_eq!(stereo_field_fields(&without), [None; 4]);

        // A bare signal has no file channels: requested or not, the fields stay None.
        let y = sine(440.0, 22050, 2.0);
        let signal = analyze_signal(y.view(), 22050, &feature_config(&["stereo_field"])).unwrap();
        assert_eq!(stereo_field_fields(&signal), [None; 4]);
        assert_eq!(signal.native_sample_rate, None);
    }
}
