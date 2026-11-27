//! Rhythm features — tempogram, tempo estimation.
//!
//! Tempogram, Fourier tempogram, tempo estimation, tempogram ratio, metrogram,
//! and time signature detection.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex;
#[cfg(test)]
use std::f32::consts::PI;

use crate::core::{audio, fft};
use crate::dsp::windows;
use crate::error::{Result, SonaraError};
use crate::onset;
use crate::types::*;
use crate::util::utils;

/// Compute autocorrelation tempogram.
///
/// Returns shape `(win_length, n_frames)` — lag × time.
pub fn tempogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<Float>> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(SonaraError::InsufficientData {
            needed: win_length,
            got: n,
        });
    }

    // Frame the onset envelope
    let frames = utils::frame(oenv.view(), win_length, 1)?;
    let n_frames = frames.ncols();

    // Window
    let window = windows::hann(win_length, true);

    let mut tgram = Array2::<Float>::zeros((win_length, n_frames));

    for t in 0..n_frames {
        // Window the frame
        let mut windowed = vec![0.0; win_length];
        for i in 0..win_length {
            windowed[i] = frames[(i, t)] * window[i];
        }

        // Autocorrelate
        let windowed_arr = Array1::from_vec(windowed);
        let acf = audio::autocorrelate(windowed_arr.view(), Some(win_length))?;

        // Normalize
        let max_val = acf.iter().copied().fold(0.0_f32, Float::max).max(1e-10);
        for i in 0..win_length {
            tgram[(i, t)] = acf[i] / max_val;
        }
    }

    Ok(tgram)
}

/// Compute Fourier tempogram.
///
/// Returns complex-valued shape `(win_length/2 + 1, n_frames)`.
pub fn fourier_tempogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
) -> Result<Array2<ComplexFloat>> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    let n = oenv.len();
    if n < win_length {
        return Err(SonaraError::InsufficientData {
            needed: win_length,
            got: n,
        });
    }

    let frames = utils::frame(oenv.view(), win_length, 1)?;
    let n_frames = frames.ncols();
    let n_bins = win_length / 2 + 1;

    let window = windows::hann(win_length, true);

    let mut ftgram = Array2::<ComplexFloat>::zeros((n_bins, n_frames));

    let mut fft_in = vec![0.0_f32; win_length];
    let mut fft_out = vec![Complex::new(0.0, 0.0); n_bins];

    for t in 0..n_frames {
        for i in 0..win_length {
            fft_in[i] = frames[(i, t)] * window[i];
        }
        fft::rfft(&mut fft_in, &mut fft_out)?;
        for i in 0..n_bins {
            ftgram[(i, t)] = fft_out[i];
        }
    }

    Ok(ftgram)
}

/// Estimate tempo from onset envelope.
///
/// Returns estimated tempo in BPM.
pub fn tempo(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    start_bpm: Float,
    max_tempo: Float,
) -> Result<Float> {
    let oenv = match onset_envelope {
        Some(env) => env.to_owned(),
        None => {
            let y = y.ok_or(SonaraError::InvalidParameter {
                param: "y",
                reason: "provide y or onset_envelope".into(),
            })?;
            onset::onset_strength(y, sr, hop_length)?
        }
    };

    // Compute tempogram
    let win_length = 384.min(oenv.len());
    if win_length < 4 {
        return Ok(start_bpm);
    }

    let tg = tempogram(None, Some(oenv.view()), sr, hop_length, win_length)?;

    // Aggregate across time (mean)
    let avg: Array1<Float> = tg
        .mean_axis(ndarray::Axis(1))
        .unwrap_or(Array1::zeros(win_length));

    // Convert lag to BPM and find peak
    let sr_f = sr as Float;
    let frame_rate = sr_f / hop_length as Float;

    let mut best_bpm = start_bpm;
    let mut best_score = Float::NEG_INFINITY;

    for lag in 1..avg.len() {
        let bpm = 60.0 * frame_rate / lag as Float;
        if bpm < 30.0 || bpm > max_tempo {
            continue;
        }

        // Log-normal prior centered at start_bpm
        let log_prior = -0.5 * ((bpm.log2() - start_bpm.log2()) / 1.0).powi(2);
        let score = avg[lag] * (1.0 + log_prior.exp());

        if score > best_score {
            best_score = score;
            best_bpm = bpm;
        }
    }

    Ok(best_bpm)
}

/// Compute tempogram ratio features.
///
/// Ratio of energy at different tempo multiples.
pub fn tempogram_ratio(
    tg: ArrayView2<Float>,
    _sr: u32,
    _hop_length: usize,
) -> Result<Array2<Float>> {
    let n_lags = tg.nrows();
    let n_frames = tg.ncols();

    // Compute ratio at 1x, 2x, 3x tempo
    let mut ratios = Array2::<Float>::zeros((3, n_frames));

    for t in 0..n_frames {
        // Find peak lag
        let mut peak_lag = 1;
        let mut peak_val = 0.0;
        for lag in 1..n_lags {
            if tg[(lag, t)] > peak_val {
                peak_val = tg[(lag, t)];
                peak_lag = lag;
            }
        }

        if peak_val > 0.0 {
            ratios[(0, t)] = tg[(peak_lag, t)] / peak_val;
            if peak_lag * 2 < n_lags {
                ratios[(1, t)] = tg[(peak_lag * 2, t)] / peak_val;
            }
            if peak_lag * 3 < n_lags {
                ratios[(2, t)] = tg[(peak_lag * 3, t)] / peak_val;
            }
        }
    }

    Ok(ratios)
}

// ============================================================
// Time signature detection (metrogram)
// ============================================================

/// Default subharmonic ratios mapping to common time signatures.
///
/// `[0.5, 1/3, 0.25, 0.2, 1/6, 1/7]` → meters `[2, 3, 4, 5, 6, 7]`
const DEFAULT_METER_RATIOS: &[Float] = &[
    0.5,       // 2/4
    1.0 / 3.0, // 3/4
    0.25,      // 4/4
    0.2,       // 5/4
    1.0 / 6.0, // 6/4 (or 6/8)
    1.0 / 7.0, // 7/4
];

/// Numerators corresponding to each ratio in `DEFAULT_METER_RATIOS`.
const METER_NUMERATORS: &[usize] = &[2, 3, 4, 5, 6, 7];

/// Compute a metrogram — a time-varying meter estimate.
///
/// The metrogram measures the strength of different metric subdivisions
/// (2, 3, 4, 5, 6, 7 beats per bar) over time by analysing subharmonic
/// ratios in the Fourier tempogram.
///
/// Returns shape `(n_ratios, n_frames)` where each row is the meter
/// strength for that ratio over time.
pub fn metrogram(
    y: Option<ArrayView1<Float>>,
    onset_envelope: Option<ArrayView1<Float>>,
    sr: u32,
    hop_length: usize,
    win_length: usize,
    ratios: Option<&[Float]>,
) -> Result<Array2<Float>> {
    let ratios = ratios.unwrap_or(DEFAULT_METER_RATIOS);
    let n_ratios = ratios.len();

    // Compute Fourier tempogram magnitude
    let ft = fourier_tempogram(y, onset_envelope, sr, hop_length, win_length)?;
    let n_bins = ft.nrows();
    let n_frames = ft.ncols();
    let ft_mag = ft.mapv(|c| c.norm());

    // Compute Fourier tempogram frequencies (BPM)
    let freqs =
        crate::core::convert::fourier_tempo_frequencies(sr as Float, win_length, hop_length);

    // Interpolate at subharmonic frequencies and compute product
    let mut meter_scores = Array2::<Float>::zeros((n_ratios, n_frames));

    for (ri, &ratio) in ratios.iter().enumerate() {
        // Where each bin's subharmonic sits depends on the ratio and the bin,
        // not on the frame: locate it once per ratio. `frac` is `None` when the
        // subharmonic falls on or past the last bin and is read without
        // interpolation.
        let taps: Vec<(usize, usize, usize, Option<Float>)> = (1..n_bins)
            .filter_map(|b| {
                let target_freq = freqs[b] * ratio;
                if target_freq <= 0.0 {
                    return None;
                }
                let mut lo = 0;
                for k in 0..n_bins {
                    if freqs[k] <= target_freq {
                        lo = k;
                    }
                }
                let hi = (lo + 1).min(n_bins - 1);
                let frac = (hi > lo && freqs[hi] > freqs[lo]).then(|| {
                    ((target_freq - freqs[lo]) / (freqs[hi] - freqs[lo])).clamp(0.0, 1.0)
                });
                Some((b, lo, hi, frac))
            })
            .collect();

        for t in 0..n_frames {
            let mut score = 0.0;
            for &(b, lo, hi, frac) in &taps {
                // Linear interpolation in the frequency axis
                let interp_val = match frac {
                    Some(frac) => (1.0 - frac) * ft_mag[(lo, t)] + frac * ft_mag[(hi, t)],
                    None => ft_mag[(lo, t)],
                };

                // Product: original magnitude × subharmonic magnitude
                score += ft_mag[(b, t)] * interp_val;
            }
            meter_scores[(ri, t)] = score;
        }
    }

    // Normalize each frame by its max value (column-wise)
    for t in 0..n_frames {
        let max_val = (0..n_ratios)
            .map(|r| meter_scores[(r, t)])
            .fold(0.0_f32, Float::max);
        if max_val > 0.0 {
            for r in 0..n_ratios {
                meter_scores[(r, t)] /= max_val;
            }
        }
    }

    Ok(meter_scores)
}

/// Detect time signature from a metrogram.
///
/// Aggregates the metrogram over time and picks the dominant meter.
/// Returns `(label, confidence)` e.g. `("4/4", 0.85)`.
pub fn detect_time_signature(
    metrogram: ArrayView2<Float>,
    ratios: Option<&[Float]>,
) -> (String, Float) {
    let nums = if ratios.is_some() {
        // If custom ratios, just use indices
        (0..metrogram.nrows()).collect::<Vec<_>>()
    } else {
        METER_NUMERATORS.to_vec()
    };

    if metrogram.ncols() == 0 || metrogram.nrows() == 0 {
        return ("4/4".to_string(), 0.0);
    }

    // Aggregate: mean score per ratio across all frames
    let n_ratios = metrogram.nrows();
    let mut scores = vec![0.0_f32; n_ratios];
    for r in 0..n_ratios {
        scores[r] = metrogram.row(r).sum() / metrogram.ncols() as Float;
    }

    // Find best and second-best
    let mut best_idx = 0;
    let mut best_score = scores[0];
    let mut second_score = 0.0_f32;
    // Start at 1: index 0 already seeds `best`, and revisiting it would make
    // a winning first meter its own runner-up (confidence 0).
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            second_score = best_score;
            best_score = s;
            best_idx = i;
        } else if s > second_score {
            second_score = s;
        }
    }

    let confidence = if best_score > 0.0 {
        ((best_score - second_score) / best_score).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let numerator = if best_idx < nums.len() {
        nums[best_idx]
    } else {
        4
    };
    let label = format!("{}/4", numerator);

    (label, confidence)
}

/// Beats one autocorrelation window of `meter_from_beats` spans.
const METER_INTEGRATION: usize = 10;
/// Longest beat lag `meter_from_beats` compares.
const METER_MAX_LAG: usize = 8;
/// Fewest usable beats: one integration window shifted by the longest lag.
const METER_MIN_BEATS: usize = METER_INTEGRATION + METER_MAX_LAG;
/// Beat lags at which a duple or quadruple metre repeats.
const METER_DUPLE_LAGS: [usize; 3] = [2, 4, 8];
/// Beat lags at which a triple metre repeats.
const METER_TRIPLE_LAGS: [usize; 2] = [3, 6];
/// Frames are 1/50 s long (20 ms) and half overlapped (a 10 ms hop).
const METER_FRAME_RATE: u32 = 50;
/// Half of the 120 ms region around a beat its onset descriptors are read from.
const METER_ONSET_HALF_WIDTH_SEC: f64 = 0.06;
/// Band of the low-frequency energy descriptor, Hz (inclusive).
const METER_LOW_BAND_HZ: std::ops::RangeInclusive<f64> = 50.0..=100.0;
/// Floor of the power spectrum under the flatness logarithm.
const METER_POWER_FLOOR: f64 = 1e-12;

/// Decide between a duple and a triple metre from the recurrence of
/// beat-segment descriptors (Gouyon & Herrera, "Determination of the meter of
/// musical audio signals: Seeking recurrences in beat segment descriptors",
/// AES 114th Convention, 2003).
///
/// Every interior beat gets four descriptors: the spread of the frame energy
/// over the segment between the midpoints to its neighbours, the mean spectral
/// flatness and the mean 50-100 Hz energy within 60 ms of the beat, and the
/// temporal centroid of the segment's energy. Each descriptor series is
/// standardised and autocorrelated over beat lags in 10-beat windows;
/// `M = mean(acf[2, 4, 8]) - mean(acf[3, 6])`, averaged over the descriptors.
///
/// Returns `(beats per bar, confidence)`: `4` when `M >= 0`, which stands for
/// the duple and quadruple metres (2/4, 4/4), `3` otherwise, which stands for
/// the triple ones (3/4, and 6/8 tracked at the eighth note); the confidence
/// is `|M|` capped at 1. `beat_times` are beat positions in seconds,
/// increasing.
///
/// The answer is as good as the beats: the descriptors compare whole beats, so
/// a tracker on another metrical level changes what recurs. A pulse whose
/// beats all sound alike (a bare metronome) differs between beats only by how
/// the frames fall on them, and still gets an answer with a non-zero
/// confidence.
///
/// `None` when the signal is shorter than one 20 ms frame or `sr` is below
/// 100 Hz, when `beat_times` is not strictly increasing, when fewer than 18
/// beats yield descriptors (the first and the last
/// beat never do, and a beat is skipped when its segment is silent, holds
/// fewer than two frames or has no frame within 60 ms), or when `M` is not
/// finite.
pub fn meter_from_beats(
    y: ArrayView1<Float>,
    sr: u32,
    beat_times: &[f64],
) -> Option<(usize, Float)> {
    let frame = (sr / METER_FRAME_RATE) as usize;
    let hop = frame / 2;
    if hop == 0 || y.len() < frame {
        return None;
    }
    // Also rejects a NaN: an unordered list would pair beats with wrong segments.
    if beat_times.windows(2).any(|w| !(w[1] > w[0])) {
        return None;
    }
    let sr = sr as f64;
    let n_frames = 1 + (y.len() - frame) / hop;

    let window: Vec<f64> = (0..frame)
        .map(|n| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * n as f64 / (frame - 1) as f64).cos())
        .collect();
    let low_bins: Vec<usize> = (0..=frame / 2)
        .filter(|&bin| METER_LOW_BAND_HZ.contains(&(bin as f64 * sr / frame as f64)))
        .collect();
    // f64 like the reference; `core::fft` is single precision.
    let fft = realfft::RealFftPlanner::<f64>::new().plan_fft_forward(frame);
    let mut windowed = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();

    // Per frame: energy, spectral flatness, low-band energy, centre time.
    let mut energy = Vec::with_capacity(n_frames);
    let mut flatness = Vec::with_capacity(n_frames);
    let mut low = Vec::with_capacity(n_frames);
    let mut centres = Vec::with_capacity(n_frames);
    for k in 0..n_frames {
        let start = k * hop;
        let mut sum_sq = 0.0;
        for (i, slot) in windowed.iter_mut().enumerate() {
            let x = y[start + i] as f64;
            sum_sq += x * x;
            *slot = x * window[i];
        }
        fft.process_with_scratch(&mut windowed, &mut spectrum, &mut scratch)
            .ok()?;
        let (mut log_sum, mut power_sum) = (0.0, 0.0);
        for c in &spectrum {
            let p = c.norm_sqr().max(METER_POWER_FLOOR);
            log_sum += p.ln();
            power_sum += p;
        }
        let n_bins = spectrum.len() as f64;
        energy.push(sum_sq);
        flatness.push((log_sum / n_bins).exp() / (power_sum / n_bins));
        low.push(low_bins.iter().map(|&b| spectrum[b].norm_sqr()).sum::<f64>());
        centres.push((start as f64 + frame as f64 / 2.0) / sr);
    }

    let mean_std = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
        (mean, var.sqrt())
    };

    let mut rows: Vec<[f64; 4]> = Vec::new();
    for w in beat_times.windows(3) {
        let beat = w[1];
        let lo = 0.5 * (w[0] + beat);
        let hi = 0.5 * (beat + w[2]);
        let segment = centres.partition_point(|&c| c < lo)..centres.partition_point(|&c| c < hi);
        let onset = centres.partition_point(|&c| c < beat - METER_ONSET_HALF_WIDTH_SEC)
            ..centres.partition_point(|&c| c <= beat + METER_ONSET_HALF_WIDTH_SEC);
        let i0 = ((lo * sr) as usize).min(y.len());
        let i1 = ((hi * sr) as usize).min(y.len());
        if segment.len() < 2 || onset.is_empty() || i1 <= i0 {
            continue;
        }
        let (mut total, mut moment) = (0.0, 0.0);
        for (i, &x) in y.slice(ndarray::s![i0..i1]).iter().enumerate() {
            let e = (x as f64).powi(2);
            total += e;
            moment += i as f64 * e;
        }
        if total <= 0.0 {
            continue;
        }
        let onset_len = onset.len() as f64;
        rows.push([
            mean_std(&energy[segment]).1,
            flatness[onset.clone()].iter().sum::<f64>() / onset_len,
            low[onset].iter().sum::<f64>() / onset_len,
            moment / total / (i1 - i0) as f64,
        ]);
    }
    if rows.len() < METER_MIN_BEATS {
        return None;
    }

    let n_windows = rows.len() - METER_MIN_BEATS + 1;
    let mut m = 0.0;
    for d in 0..4 {
        let column: Vec<f64> = rows.iter().map(|r| r[d]).collect();
        let (mean, std) = mean_std(&column);
        let std = if std == 0.0 { 1.0 } else { std };
        let x: Vec<f64> = column.iter().map(|v| (v - mean) / std).collect();

        // Normalised autocorrelation per lag, summed over the windows.
        let mut acf = [0.0; METER_MAX_LAG + 1];
        for s in 0..n_windows {
            let a = &x[s..s + METER_INTEGRATION];
            let a_sq: f64 = a.iter().map(|v| v * v).sum();
            for (lag, slot) in acf.iter_mut().enumerate() {
                let b = &x[s + lag..s + lag + METER_INTEGRATION];
                let den = (a_sq * b.iter().map(|v| v * v).sum::<f64>()).sqrt();
                if den > 0.0 {
                    *slot += a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>() / den;
                }
            }
        }
        let mean_at = |lags: &[usize]| {
            lags.iter().map(|&l| acf[l]).sum::<f64>() / (lags.len() * n_windows) as f64
        };
        m += (mean_at(&METER_DUPLE_LAGS) - mean_at(&METER_TRIPLE_LAGS)) / 4.0;
    }
    if !m.is_finite() {
        return None;
    }
    Some((if m >= 0.0 { 4 } else { 3 }, m.abs().min(1.0) as Float))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_tempogram_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let tg = tempogram(None, Some(oenv.view()), 22050, 512, 384.min(oenv.len())).unwrap();
        assert!(tg.nrows() > 0);
        assert!(tg.ncols() > 0);
    }

    #[test]
    fn test_tempo_120bpm() {
        let y = click_train(22050, 4.0, 120.0);
        let t = tempo(Some(y.view()), None, 22050, 512, 120.0, 320.0).unwrap();
        assert!(t > 80.0 && t < 180.0, "tempo {t} should be near 120");
    }

    #[test]
    fn test_fourier_tempogram_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let ft = fourier_tempogram(None, Some(oenv.view()), 22050, 512, win).unwrap();
        assert_eq!(ft.nrows(), win / 2 + 1);
    }

    #[test]
    fn test_tempogram_ratio_shape() {
        let y = click_train(22050, 3.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let tg = tempogram(None, Some(oenv.view()), 22050, 512, 384.min(oenv.len())).unwrap();
        let ratios = tempogram_ratio(tg.view(), 22050, 512).unwrap();
        assert_eq!(ratios.nrows(), 3);
    }

    #[test]
    fn test_metrogram_shape() {
        let y = click_train(22050, 4.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let mg = metrogram(None, Some(oenv.view()), 22050, 512, win, None).unwrap();
        assert_eq!(mg.nrows(), DEFAULT_METER_RATIOS.len());
        assert!(mg.ncols() > 0);
    }

    /// The metrogram formula written out frame by frame, with no shortcuts.
    fn metrogram_per_frame(
        envelope: ArrayView1<Float>,
        sr: u32,
        hop_length: usize,
        win_length: usize,
        ratios: &[Float],
    ) -> Array2<Float> {
        let magnitude = fourier_tempogram(None, Some(envelope), sr, hop_length, win_length)
            .unwrap()
            .mapv(|c| c.norm());
        let freqs =
            crate::core::convert::fourier_tempo_frequencies(sr as Float, win_length, hop_length);
        let (n_bins, n_frames) = magnitude.dim();
        let mut scores = Array2::<Float>::zeros((ratios.len(), n_frames));
        for (ri, &ratio) in ratios.iter().enumerate() {
            for t in 0..n_frames {
                let mut score = 0.0;
                for b in 1..n_bins {
                    let target = freqs[b] * ratio;
                    if target <= 0.0 {
                        continue;
                    }
                    let mut lo = 0;
                    for k in 0..n_bins {
                        if freqs[k] <= target {
                            lo = k;
                        }
                    }
                    let hi = (lo + 1).min(n_bins - 1);
                    let subharmonic = if hi > lo && freqs[hi] > freqs[lo] {
                        let frac = ((target - freqs[lo]) / (freqs[hi] - freqs[lo])).clamp(0.0, 1.0);
                        (1.0 - frac) * magnitude[(lo, t)] + frac * magnitude[(hi, t)]
                    } else {
                        magnitude[(lo, t)]
                    };
                    score += magnitude[(b, t)] * subharmonic;
                }
                scores[(ri, t)] = score;
            }
        }
        for t in 0..n_frames {
            let max = scores.column(t).iter().copied().fold(0.0_f32, Float::max);
            if max > 0.0 {
                scores.column_mut(t).mapv_inplace(|v| v / max);
            }
        }
        scores
    }

    #[test]
    fn test_metrogram_equals_the_per_frame_formula() {
        // A deterministic envelope: accents every 11 frames over fixed noise.
        let mut state = 0x2545_f491_u32;
        let envelope = Array1::from_iter((0..400).map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 16) as Float / 65_536.0;
            noise + if i % 11 == 0 { 3.0 } else { 0.0 }
        }));
        // 2.0 puts subharmonics past the last bin (no interpolation); 0.0 makes
        // every bin skip.
        for (win_length, ratios) in [
            (96, DEFAULT_METER_RATIOS),
            (64, &[0.5, 0.3, 0.9, 2.0, 0.0][..]),
        ] {
            let fast =
                metrogram(None, Some(envelope.view()), 22050, 512, win_length, Some(ratios))
                    .unwrap();
            let direct = metrogram_per_frame(envelope.view(), 22050, 512, win_length, ratios);
            assert!(fast.iter().any(|&v| v > 0.0), "scores must not be all zero");
            assert_eq!(fast, direct, "win_length {win_length}");
        }
    }

    #[test]
    fn test_detect_time_signature_basic() {
        let y = click_train(22050, 4.0, 120.0);
        let oenv = onset::onset_strength(y.view(), 22050, 512).unwrap();
        let win = 384.min(oenv.len());
        let mg = metrogram(None, Some(oenv.view()), 22050, 512, win, None).unwrap();
        let (label, confidence) = detect_time_signature(mg.view(), None);
        // A simple click train is ambiguous in meter, but should return a valid label
        assert!(
            label.contains('/'),
            "label should be N/4 format, got {label}"
        );
        assert!(
            confidence >= 0.0 && confidence <= 1.0,
            "confidence {confidence} out of range"
        );
    }

    #[test]
    fn test_detect_time_signature_first_meter_wins_with_margin() {
        // Rows follow METER_NUMERATORS (2, 3, 4, ...): the first meter wins by a
        // clear margin, so the confidence is (1.0 - 0.5) / 1.0, not 0.
        let mg = ndarray::array![[1.0, 1.0], [0.5, 0.5], [0.2, 0.2]];
        let (label, confidence) = detect_time_signature(mg.view(), None);
        assert_eq!(label, "2/4");
        assert!(
            (confidence - 0.5).abs() < 1e-6,
            "confidence {confidence}, expected 0.5"
        );

        // A later winner keeps the same margin rule.
        let mg = ndarray::array![[0.5, 0.5], [0.2, 0.2], [1.0, 1.0]];
        let (label, confidence) = detect_time_signature(mg.view(), None);
        assert_eq!(label, "4/4");
        assert!(
            (confidence - 0.5).abs() < 1e-6,
            "confidence {confidence}, expected 0.5"
        );
    }

    #[test]
    fn test_meter_from_beats_tells_triple_from_duple_drums() {
        let sr = 22050_u32;
        let beat = sr as usize / 2; // 120 BPM
        let n_beats = 120; // 60 s
        let drums = |pattern: &[bool]| {
            let mut state = 0x2545_f491_u32;
            let mut y = Array1::<Float>::zeros(n_beats * beat);
            for k in 0..n_beats {
                let kick = pattern[k % pattern.len()];
                for i in 0..beat / 2 {
                    let t = i as Float / sr as Float;
                    y[k * beat + i] = if kick {
                        (2.0 * PI * 60.0 * t).sin() * (-t / 0.1).exp()
                    } else {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        0.7 * ((state >> 16) as Float / 32_768.0 - 1.0) * (-t / 0.03).exp()
                    };
                }
            }
            y
        };
        let beat_times: Vec<f64> = (0..n_beats).map(|k| k as f64 * 0.5).collect();

        let triple = drums(&[true, false, false]);
        let (meter, confidence) = meter_from_beats(triple.view(), sr, &beat_times).unwrap();
        assert_eq!(meter, 3, "kick, snare, snare (confidence {confidence})");
        assert_eq!(confidence, 1.0, "a clean triple pattern reaches the cap");

        // The value, not only the decision: one broken descriptor moves the
        // mean of the four M by about a quarter.
        let duple = drums(&[true, false, true, false]);
        let (meter, confidence) = meter_from_beats(duple.view(), sr, &beat_times).unwrap();
        assert_eq!(meter, 4, "kick, snare, kick, snare (confidence {confidence})");
        assert!((confidence - 0.9985).abs() < 1e-3, "{confidence}");

        assert_eq!(meter_from_beats(duple.view(), sr, &beat_times[..10]), None);

        let mut unordered = beat_times.clone();
        unordered.swap(40, 41);
        assert_eq!(meter_from_beats(duple.view(), sr, &unordered), None);
    }
}
