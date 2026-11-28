//! True peak and ReplayGain-style track gain, the two values the loudness
//! measurement in [`crate::native_dynamics`] builds on.
//!
//! - [`true_peak_db`] — true peak (dBTP) via 4x oversampling (BS.1770-4 Annex 2).
//! - [`replaygain_db`] — ReplayGain-style track gain to a -18 LUFS reference.

use ndarray::ArrayView1;

use crate::types::Float;

/// EBU R128 absolute gate (LUFS). Blocks quieter than this are treated as silence.
const ABS_GATE_LUFS: Float = -70.0;

/// Reference loudness for ReplayGain-style track gain, in LUFS.
///
/// -18 LUFS is a widely used replay-gain reference target for music playback.
pub const REPLAYGAIN_REFERENCE_LUFS: Float = -18.0;

// ============================================================
// ReplayGain-style track gain
// ============================================================

/// ReplayGain-style track gain in dB.
///
/// This is the gain a player should apply to bring the track to the
/// [`REPLAYGAIN_REFERENCE_LUFS`] (-18 LUFS) reference:
///
/// ```text
/// replaygain_db = REPLAYGAIN_REFERENCE_LUFS - integrated_loudness_lufs
///               = -18 - loudness_lufs
/// ```
///
/// A track measured at -9 LUFS (loud master) yields `-18 - (-9) = -9 dB`
/// (turn it down 9 dB); a quiet -24 LUFS track yields `+6 dB`. Pure arithmetic
/// on the already-computed integrated loudness — zero extra signal processing.
#[inline]
pub fn replaygain_db(loudness_lufs: Float) -> Float {
    REPLAYGAIN_REFERENCE_LUFS - loudness_lufs
}

// ============================================================
// True peak (dBTP)
// ============================================================

// 4x oversampling, windowed-sinc polyphase FIR interpolation.
//
// ITU-R BS.1770-4 Annex 2 specifies a >=4x oversampled peak measurement and
// gives a reference 48-tap polyphase coefficient table defined for 48 kHz. sonara
// works at arbitrary sample rates (22050 Hz by default), so instead of that
// fixed table we build an equivalent 4x interpolator on the fly: a 48-tap
// (12 taps/phase) sinc lowpass, Blackman-windowed, with each polyphase branch
// normalised to unity DC gain. Phase 0 reproduces the input samples exactly, so
// the oversampled peak is guaranteed >= the raw sample peak.
const TP_L: usize = 4; // oversampling factor
const TP_HALF: usize = 6; // taps-per-phase = 2 * TP_HALF = 12; total taps = TP_L * 12 = 48

#[inline]
fn sinc(x: Float) -> Float {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let px = std::f32::consts::PI * x;
        px.sin() / px
    }
}

#[inline]
fn blackman(i: usize, m: usize) -> Float {
    // Symmetric Blackman window over indices 0..m-1.
    let d = (m - 1) as Float;
    let a = 2.0 * std::f32::consts::PI * i as Float / d;
    0.42 - 0.5 * a.cos() + 0.08 * (2.0 * a).cos()
}

/// True peak of the signal in dBTP (decibels relative to full scale, true peak).
///
/// Computed via 4x oversampling per ITU-R BS.1770-4 Annex 2 (windowed-sinc
/// polyphase interpolation — see the module notes for why a windowed sinc is
/// used instead of the spec's 48 kHz-specific table). The returned value is
/// `20*log10(max |oversampled sample|)`; a full-scale sine is ~0 dBTP, half
/// scale ~-6.02 dBTP. Values above 0 dBTP indicate inter-sample overs that can
/// clip a downstream reconstruction filter / DAC.
///
/// Empty input returns [`ABS_GATE_LUFS`] as a finite floor (no peak to report).
pub fn true_peak_db(y: ArrayView1<Float>) -> Float {
    let n = y.len();
    if n == 0 {
        return ABS_GATE_LUFS;
    }

    // Fast contiguous access (only copies for the rare non-contiguous view).
    let owned: Option<Vec<Float>> = match y.as_slice() {
        Some(_) => None,
        None => Some(y.iter().copied().collect()),
    };
    let x: &[Float] = match owned {
        Some(ref v) => v.as_slice(),
        None => y.as_slice().unwrap(),
    };

    // Raw sample peak (this is exactly phase 0 of the interpolator).
    let mut peak: Float = 0.0;
    for &v in x {
        peak = peak.max(v.abs());
    }

    // Build the polyphase prototype: h[idx] = sinc((idx-center)/L) * blackman.
    let taps = 2 * TP_HALF; // per phase
    let m = TP_L * taps; // total prototype length
    let center = (TP_HALF * TP_L) as Float;
    let mut proto = vec![0.0 as Float; m];
    for idx in 0..m {
        let arg = (idx as Float - center) / TP_L as Float;
        proto[idx] = sinc(arg) * blackman(idx, m);
    }
    // Normalise each polyphase branch (phase p = taps p, p+L, p+2L, ...) to unit
    // DC gain, so a constant / low-frequency input is reconstructed at gain 1.
    for p in 0..TP_L {
        let mut s = 0.0 as Float;
        let mut t = 0;
        while t < taps {
            s += proto[t * TP_L + p];
            t += 1;
        }
        if s.abs() > 1e-12 {
            let mut t = 0;
            while t < taps {
                proto[t * TP_L + p] /= s;
                t += 1;
            }
        }
    }

    // Scan every inter-sample position. Phase 0 == raw samples (already in `peak`),
    // so only phases 1..L need reconstruction. The window centered on sample `i`
    // reads x[i+TP_HALF-t] for t in 0..taps, i.e. x[i-(TP_HALF-1)..=i+TP_HALF].
    //
    // Interior samples (where that range is fully in-bounds) take a tight loop
    // with no bounds handling; the few edge samples fall back to a clamped path.
    let lo_edge = TP_HALF.saturating_sub(1); // first fully-interior i
    let hi_edge = n.saturating_sub(TP_HALF); // one past last fully-interior i
    let ni = n as isize;

    // Interior fast path (no clamping).
    if hi_edge > lo_edge {
        for i in lo_edge..hi_edge {
            let base = i + TP_HALF; // x index for t = 0
            for p in 1..TP_L {
                let mut acc = 0.0 as Float;
                for t in 0..taps {
                    acc += proto[t * TP_L + p] * x[base - t];
                }
                let a = acc.abs();
                if a > peak {
                    peak = a;
                }
            }
        }
    }

    // Edge samples (clamped reflection at the boundaries).
    let edge_iter = (0..lo_edge.min(n)).chain(hi_edge.max(lo_edge)..n);
    for i in edge_iter {
        for p in 1..TP_L {
            let mut acc = 0.0 as Float;
            for t in 0..taps {
                let xi = i as isize + TP_HALF as isize - t as isize;
                let xc = xi.clamp(0, ni - 1) as usize;
                acc += proto[t * TP_L + p] * x[xc];
            }
            let a = acc.abs();
            if a > peak {
                peak = a;
            }
        }
    }

    if peak <= 1e-12 {
        ABS_GATE_LUFS
    } else {
        20.0 * peak.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float, amp: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| {
            amp * (2.0 * PI * freq * i as Float / sr as Float).sin()
        })
    }

    #[test]
    fn test_true_peak_full_scale_sine() {
        // Full-scale 997 Hz sine -> ~0 dBTP.
        let y = sine(997.0, 48000, 1.0, 1.0);
        let tp = true_peak_db(y.view());
        assert!(tp.is_finite());
        assert!((tp - 0.0).abs() < 0.1, "expected ~0 dBTP, got {tp}");
    }

    #[test]
    fn test_true_peak_half_scale_sine() {
        // Half-scale -> ~-6.02 dBTP.
        let y = sine(997.0, 48000, 1.0, 0.5);
        let tp = true_peak_db(y.view());
        assert!(
            (tp - (-6.0206)).abs() < 0.1,
            "expected ~-6.02 dBTP, got {tp}"
        );
    }

    #[test]
    fn test_true_peak_intersample_over() {
        // A sine near fs/4 sampled so the raw samples straddle — but never land on —
        // the true crest. The oversampled true peak must exceed the raw sample peak.
        let sr = 48000u32;
        let f = sr as Float / 4.0; // fs/4
        let n = 4000usize;
        // Phase pi/4 -> samples at +/-sin(pi/4)=0.707 crests, true peak at 1.0.
        let phase = PI / 4.0;
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * PI * f * i as Float / sr as Float + phase).sin()
        });
        let sample_peak = y.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let sample_peak_db = 20.0 * sample_peak.log10();
        let tp = true_peak_db(y.view());
        assert!(
            tp > sample_peak_db + 0.5,
            "true peak {tp} should exceed sample peak {sample_peak_db} for inter-sample over"
        );
        // True crest is at 0 dBTP; interpolation should recover close to it.
        assert!(
            tp > -0.5 && tp < 0.2,
            "inter-sample true peak {tp} not near 0 dBTP"
        );
    }

    #[test]
    fn test_replaygain_formula() {
        // A loud master is turned down, a quiet track up.
        assert_eq!(replaygain_db(-9.0), -9.0);
        assert_eq!(replaygain_db(-24.0), 6.0);
    }

    #[test]
    fn test_silence_and_empty_are_finite() {
        let silence = Array1::<Float>::zeros(22050);
        assert!(true_peak_db(silence.view()).is_finite());
        let empty = Array1::<Float>::zeros(0);
        assert!(true_peak_db(empty.view()).is_finite());
    }
}
