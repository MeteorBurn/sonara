//! Loudness and dynamics measured on the file's own channels and sample rate.
//!
//! - [`ebur128`] — ITU-R BS.1770-5 / EBU Tech 3341 + 3342: integrated,
//!   momentary max, short-term max, loudness range, true peak.
//! - [`native_dynamics`] — sample peak, crest factor and a peak envelope on the
//!   analyzer's frame grid.
//!
//! Pure functions: no file access, no dependence on the analysis sample rate.

use ndarray::{s, ArrayView2};

use crate::loudness_ext::true_peak_db;
use crate::types::Float;

/// Envelope grid: frame and hop of the main pass (2048 and 512 samples at 22050 Hz).
pub const ENVELOPE_FRAME_SEC: f64 = 2048.0 / 22050.0;
pub const ENVELOPE_HOP_SEC: f64 = 512.0 / 22050.0;

const ABS_GATE_LUFS: f64 = -70.0;
/// Peaks at or below this are digital silence: the dB value is undefined.
const SILENCE_PEAK: Float = 1e-12;
/// Sub-blocks per momentary (400 ms) and short-term (3 s) window.
const MOMENTARY_SUBS: usize = 40;
const SHORT_TERM_SUBS: usize = 300;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ebur128 {
    pub integrated_lufs: Option<Float>,
    pub momentary_max_lufs: Option<Float>,
    pub short_term_max_lufs: Option<Float>,
    pub range_lu: Option<Float>,
    pub true_peak_dbtp: Option<Float>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeDynamics {
    pub sample_peak_dbfs: Option<Float>,
    pub crest_factor_db: Option<Float>,
    pub crest_factor_frame_mean_db: Option<Float>,
    pub envelope: Vec<Float>,
}

struct Biquad {
    b: [f64; 3],
    a1: f64,
    a2: f64,
}

impl Biquad {
    // Transposed direct form II.
    #[inline]
    fn step(&self, x: f64, st: &mut [f64; 2]) -> f64 {
        let y = self.b[0] * x + st[0];
        st[0] = self.b[1] * x - self.a1 * y + st[1];
        st[1] = self.b[2] * x - self.a2 * y;
        y
    }
}

/// K-weighting (shelf, high-pass) re-derived for `fs`, as libebur128 does;
/// reproduces the BS.1770 table at 48 kHz.
fn k_weighting(fs: f64) -> [Biquad; 2] {
    let k = (std::f64::consts::PI * 1681.974450955533 / fs).tan();
    let q = 0.7071752369554196;
    let vh = 10f64.powf(3.999843853973347 / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let a0 = 1.0 + k / q + k * k;
    let shelf = Biquad {
        b: [
            (vh + vb * k / q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / q + k * k) / a0,
        ],
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
    };

    let k = (std::f64::consts::PI * 38.13547087602444 / fs).tan();
    let q = 0.5003270373238773;
    let a0 = 1.0 + k / q + k * k;
    let high_pass = Biquad {
        // Numerator stays unnormalised, matching the standard's table.
        b: [1.0, -2.0, 1.0],
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / q + k * k) / a0,
    };
    [shelf, high_pass]
}

fn channel_weights(n_channels: usize) -> Vec<f64> {
    // shortcut: layout guessed from channel count, upgrade when the decoder exposes a channel map
    let layout: &[f64] = match n_channels {
        0..=3 => &[1.0, 1.0, 1.0],
        4 => &[1.0, 1.0, 1.41, 1.41],
        5 => &[1.0, 1.0, 1.0, 1.41, 1.41],
        _ => &[1.0, 1.0, 1.0, 0.0, 1.41, 1.41],
    };
    (0..n_channels)
        .map(|c| layout.get(c).copied().unwrap_or(0.0))
        .collect()
}

/// First sample of 10 ms sub-block `j`: round(j * sr / 100), so the grid never drifts.
#[inline]
fn sub_start(j: usize, sr: u32) -> usize {
    ((j as u64 * sr as u64 + 50) / 100) as usize
}

fn full_sub_blocks(n: usize, sr: u32) -> usize {
    let mut j = (n as u64 * 100 / sr as u64) as usize;
    while j > 0 && sub_start(j, sr) > n {
        j -= 1;
    }
    while sub_start(j + 1, sr) <= n {
        j += 1;
    }
    j
}

/// Channel-weighted K-filtered energy of every full 10 ms sub-block. Streams
/// each channel so memory is 100 values per second, not a filtered copy.
fn sub_block_energy(channels: ArrayView2<Float>, sr: u32) -> Vec<f64> {
    let mut energy = vec![0.0f64; full_sub_blocks(channels.ncols(), sr)];
    if energy.is_empty() {
        return energy;
    }
    let [shelf, high_pass] = k_weighting(sr as f64);
    let weights = channel_weights(channels.nrows());
    for (row, &w) in channels.outer_iter().zip(&weights) {
        if w == 0.0 {
            continue;
        }
        let (mut st_shelf, mut st_hp) = ([0.0f64; 2], [0.0f64; 2]);
        let mut samples = row.iter();
        for (j, e) in energy.iter_mut().enumerate() {
            let len = sub_start(j + 1, sr) - sub_start(j, sr);
            let mut acc = 0.0f64;
            for &x in samples.by_ref().take(len) {
                let y = high_pass.step(shelf.step(x as f64, &mut st_shelf), &mut st_hp);
                acc += y * y;
            }
            *e += w * acc;
        }
    }
    energy
}

/// Mean square of every `subs`-long window, one position per sub-block.
fn window_mean_sq(prefix: &[f64], sr: u32, subs: usize) -> Vec<f64> {
    let n_sub = prefix.len() - 1;
    if n_sub < subs {
        return Vec::new();
    }
    (0..=n_sub - subs)
        .map(|j| {
            let count = sub_start(j + subs, sr) - sub_start(j, sr);
            if count == 0 {
                0.0
            } else {
                (prefix[j + subs] - prefix[j]) / count as f64
            }
        })
        .collect()
}

#[inline]
fn lufs(mean_sq: f64) -> f64 {
    -0.691 + 10.0 * mean_sq.log10()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn finite(v: f64) -> Option<Float> {
    let f = v as Float;
    f.is_finite().then_some(f)
}

fn max_lufs(mean_sq: &[f64]) -> Option<Float> {
    let z = mean_sq.iter().copied().fold(0.0f64, f64::max);
    if z > 0.0 {
        finite(lufs(z))
    } else {
        None
    }
}

/// Blocks above the absolute gate and above (their own mean - `relative_lu`).
fn gate(mean_sq: impl Iterator<Item = f64>, relative_lu: f64) -> Vec<f64> {
    let abs_gate = 10f64.powf((ABS_GATE_LUFS + 0.691) / 10.0);
    let mut kept: Vec<f64> = mean_sq.filter(|&z| z > abs_gate).collect();
    if !kept.is_empty() {
        let rel_gate = mean(&kept) * 10f64.powf(-relative_lu / 10.0);
        kept.retain(|&z| z > rel_gate);
    }
    kept
}

/// EBU R128 loudness of a multichannel signal. `channels` is
/// (n_channels, n_samples) at the file's own sample rate `sr`.
pub fn ebur128(channels: ArrayView2<Float>, sr: u32) -> Ebur128 {
    let mut out = Ebur128 {
        integrated_lufs: None,
        momentary_max_lufs: None,
        short_term_max_lufs: None,
        range_lu: None,
        true_peak_dbtp: None,
    };
    if sr == 0 || channels.is_empty() {
        return out;
    }

    // BS.1770 weights follow speaker positions; past 5.1 the channel count
    // alone does not say which channel sits where, so loudness stays undefined.
    let energy = if channels.nrows() <= 6 {
        sub_block_energy(channels, sr)
    } else {
        Vec::new()
    };
    let mut prefix = Vec::with_capacity(energy.len() + 1);
    let mut total = 0.0f64;
    prefix.push(total);
    for e in &energy {
        total += e;
        prefix.push(total);
    }
    let momentary = window_mean_sq(&prefix, sr, MOMENTARY_SUBS);
    let short_term = window_mean_sq(&prefix, sr, SHORT_TERM_SUBS);

    out.momentary_max_lufs = max_lufs(&momentary);
    out.short_term_max_lufs = max_lufs(&short_term);

    // Gating blocks overlap by 75%: every 10th 10 ms position.
    let gated = gate(momentary.iter().copied().step_by(10), 10.0);
    if !gated.is_empty() {
        out.integrated_lufs = finite(lufs(mean(&gated)));
    }

    let mut st: Vec<f64> = gate(short_term.iter().copied().step_by(10), 20.0)
        .into_iter()
        .map(lufs)
        .collect();
    if st.len() >= 2 {
        st.sort_by(f64::total_cmp);
        // Percentile index as in libebur128.
        let at = |p: f64| st[((st.len() - 1) as f64 * p + 0.5) as usize];
        out.range_lu = finite(at(0.95) - at(0.10));
    }

    out.true_peak_dbtp = channels
        .outer_iter()
        // true_peak_db reports a -70 floor for silence; skip those rows instead.
        .filter(|row| row.iter().any(|v| v.abs() > SILENCE_PEAK))
        .map(true_peak_db)
        .fold(None, |best: Option<Float>, v| {
            Some(best.map_or(v, |b| b.max(v)))
        })
        .filter(|v| v.is_finite());
    out
}

/// Envelope frame length in samples: round(2048 * sr / 22050).
#[inline]
fn frame_len(sr: u32) -> usize {
    ((2048 * sr as u64 + 11025) / 22050) as usize
}

/// Centre of envelope frame `k`: round(k * 512 * sr / 22050). Rounded per
/// frame because the hop is fractional off 22050 Hz and a fixed one drifts.
#[inline]
fn frame_center(k: usize, sr: u32) -> usize {
    ((k as u64 * 512 * sr as u64 + 11025) / 22050) as usize
}

/// Peak, crest factor and peak envelope over the file's own channels (not a
/// mono mix). `channels` is (n_channels, n_samples) at sample rate `sr`.
pub fn native_dynamics(channels: ArrayView2<Float>, sr: u32) -> NativeDynamics {
    let mut out = NativeDynamics {
        sample_peak_dbfs: None,
        crest_factor_db: None,
        crest_factor_frame_mean_db: None,
        envelope: Vec::new(),
    };
    let (n_ch, n) = channels.dim();
    if sr == 0 || n_ch == 0 || n == 0 {
        return out;
    }

    let len = frame_len(sr);
    let half = len / 2;
    let n_frames = 1 + (n as u64 * 22050 / (512 * sr as u64)) as usize;
    let bounds: Vec<(usize, usize)> = (0..n_frames)
        .map(|k| {
            let center = frame_center(k, sr);
            (
                center.saturating_sub(half).min(n),
                (center + len - half).min(n),
            )
        })
        .collect();

    // Frames overlap 4x: cut the signal at every frame edge and read each
    // sample once; a frame is then a run of consecutive segments.
    let mut cuts: Vec<usize> = bounds.iter().flat_map(|&(a, b)| [a, b]).collect();
    cuts.extend([0, n]);
    cuts.sort_unstable();
    cuts.dedup();
    let segments: Vec<(Float, f64)> = cuts
        .windows(2)
        .map(|w| {
            let (mut seg_peak, mut seg_sq) = (0.0 as Float, 0.0f64);
            for row in channels.outer_iter() {
                for &x in row.slice(s![w[0]..w[1]]) {
                    seg_peak = seg_peak.max(x.abs());
                    seg_sq += x as f64 * x as f64;
                }
            }
            (seg_peak, seg_sq)
        })
        .collect();
    let fold = |segs: &[(Float, f64)]| {
        segs.iter()
            .fold((0.0 as Float, 0.0f64), |(p, s), &(sp, ss)| {
                (p.max(sp), s + ss)
            })
    };

    let (peak, sum_sq) = fold(&segments);
    let rms = (sum_sq / (n_ch * n) as f64).sqrt();
    if peak > SILENCE_PEAK && rms > 0.0 {
        out.sample_peak_dbfs = finite(20.0 * (peak as f64).log10());
        out.crest_factor_db = finite(20.0 * (peak as f64 / rms).log10());
    }

    let rms_gate = 10f64.powf(-70.0 / 20.0);
    let (mut crest_sum, mut crest_n) = (0.0f64, 0usize);
    out.envelope.reserve(n_frames);
    for (start, end) in bounds {
        // Every frame edge is a cut, so both searches hit.
        let at = |x: usize| cuts.binary_search(&x).unwrap_or_else(|i| i);
        let (frame_peak, frame_sq) = fold(&segments[at(start)..at(end)]);
        out.envelope.push(frame_peak);
        if end > start {
            let frame_rms = (frame_sq / (n_ch * (end - start)) as f64).sqrt();
            if frame_rms > rms_gate {
                crest_sum += 20.0 * (frame_peak as f64 / frame_rms).log10();
                crest_n += 1;
            }
        }
    }
    if crest_n > 0 {
        out.crest_factor_frame_mean_db = finite(crest_sum / crest_n as f64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// 1 kHz sine, identical in every channel; `segments` = (peak dBFS, seconds).
    fn tone(n_ch: usize, sr: u32, segments: &[(f64, f64)]) -> Array2<Float> {
        let mut amp = Vec::new();
        for &(dbfs, secs) in segments {
            let len = (secs * sr as f64).round() as usize;
            amp.extend(std::iter::repeat(10f64.powf(dbfs / 20.0)).take(len));
        }
        Array2::from_shape_fn((n_ch, amp.len()), |(_, i)| {
            let ph = 2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr as f64;
            (amp[i] * ph.sin()) as Float
        })
    }

    /// Six channels, the tone only in channel `ch`.
    fn only_channel(ch: usize) -> Array2<Float> {
        let src = tone(1, 48000, &[(-23.0, 2.0)]);
        let mut out = Array2::zeros((6, src.ncols()));
        out.row_mut(ch).assign(&src.row(0));
        out
    }

    fn close(got: Option<Float>, want: f64, tol: f64) -> bool {
        got.is_some_and(|g| (g as f64 - want).abs() <= tol)
    }

    #[test]
    fn k_weighting_matches_bs1770_table_at_48k() {
        let [shelf, hp] = k_weighting(48000.0);
        let want = [
            (shelf.b[0], 1.53512485958697),
            (shelf.b[1], -2.69169618940638),
            (shelf.b[2], 1.19839281085285),
            (shelf.a1, -1.69065929318241),
            (shelf.a2, 0.73248077421585),
            (hp.a1, -1.99004745483398),
            (hp.a2, 0.99007225036621),
        ];
        for (got, want) in want {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
        assert_eq!(hp.b, [1.0, -2.0, 1.0]);
    }

    #[test]
    fn sine_minus_23_dbfs_reads_minus_23_lufs() {
        for sr in [48000, 44100] {
            let r = ebur128(tone(2, sr, &[(-23.0, 20.0)]).view(), sr);
            assert!(close(r.integrated_lufs, -23.0, 0.1), "{sr}: {r:?}");
            assert!(close(r.momentary_max_lufs, -23.0, 0.1), "{sr}: {r:?}");
            assert!(close(r.short_term_max_lufs, -23.0, 0.1), "{sr}: {r:?}");
        }
        let mono = ebur128(tone(1, 48000, &[(-23.0, 20.0)]).view(), 48000);
        assert!(close(mono.integrated_lufs, -26.0, 0.1), "{mono:?}");
    }

    #[test]
    fn relative_gate_drops_quiet_sections() {
        let x = tone(2, 48000, &[(-36.0, 10.0), (-23.0, 60.0), (-36.0, 10.0)]);
        let r = ebur128(x.view(), 48000);
        assert!(close(r.integrated_lufs, -23.0, 0.1), "{r:?}");
    }

    #[test]
    fn loudness_range_of_two_levels() {
        let x = tone(2, 48000, &[(-20.0, 20.0), (-30.0, 20.0)]);
        let r = ebur128(x.view(), 48000);
        assert!(close(r.range_lu, 10.0, 1.0), "{r:?}");
    }

    #[test]
    fn channel_weights_by_layout() {
        let lfe = ebur128(only_channel(3).view(), 48000);
        assert_eq!(lfe.integrated_lufs, None);
        assert!(lfe.true_peak_dbtp.is_some());

        let front = ebur128(only_channel(0).view(), 48000).integrated_lufs;
        let surround = ebur128(only_channel(4).view(), 48000).integrated_lufs;
        let diff = (surround.unwrap() - front.unwrap()) as f64;
        assert!((diff - 10.0 * 1.41f64.log10()).abs() < 0.05, "{diff}");

        // Rear channels carry 1.41 in the quad and 5.0 layouts too.
        assert_eq!(channel_weights(4), [1.0, 1.0, 1.41, 1.41]);
        assert_eq!(channel_weights(5), [1.0, 1.0, 1.0, 1.41, 1.41]);

        // Unknown layout (> 5.1): no loudness, the peak needs no layout.
        let wide = tone(8, 48000, &[(-23.0, 5.0)]);
        let r = ebur128(wide.view(), 48000);
        assert_eq!(
            (r.integrated_lufs, r.momentary_max_lufs, r.range_lu),
            (None, None, None)
        );
        assert!(close(r.true_peak_dbtp, -23.0, 0.1), "{r:?}");
    }

    #[test]
    fn true_peak_sees_intersample_peaks() {
        // fs/4 with a 45 degree offset: every sample is +-0.7071, the peak is 1.0.
        let x = Array2::from_shape_fn((1, 48000), |(_, i)| {
            let ph = std::f64::consts::PI * (0.5 * i as f64 + 0.25);
            ph.sin() as Float
        });
        let tp = ebur128(x.view(), 48000).true_peak_dbtp.unwrap();
        let sp = native_dynamics(x.view(), 48000).sample_peak_dbfs.unwrap();
        assert!(tp >= sp);
        assert!(((tp - sp) as f64 - 3.01).abs() < 0.3, "tp {tp} sp {sp}");

        let plain = tone(2, 48000, &[(-6.0, 1.0)]);
        let tp = ebur128(plain.view(), 48000).true_peak_dbtp.unwrap();
        let sp = native_dynamics(plain.view(), 48000)
            .sample_peak_dbfs
            .unwrap();
        assert!(tp >= sp, "tp {tp} sp {sp}");
    }

    #[test]
    fn crest_factor_of_square_and_sine() {
        let square = Array2::from_shape_fn(
            (2, 48000),
            |(_, i)| if i / 24 % 2 == 0 { 0.5 } else { -0.5 },
        );
        let d = native_dynamics(square.view(), 48000);
        assert!(close(d.crest_factor_db, 0.0, 1e-4), "{d:?}");
        assert!(close(d.sample_peak_dbfs, -6.02, 0.01), "{d:?}");

        let d = native_dynamics(tone(2, 48000, &[(-3.0, 2.0)]).view(), 48000);
        assert!(close(d.crest_factor_db, 3.01, 0.02));
        assert!(close(d.crest_factor_frame_mean_db, 3.01, 0.05));
    }

    #[test]
    fn envelope_grid_follows_duration() {
        assert_eq!(frame_len(22050), 2048);
        for k in [0, 1, 7, 12345, 1_000_000] {
            assert_eq!(frame_center(k, 22050), 512 * k);
        }

        let sr = 48000u32;
        let n = 180 * sr as usize;
        let hit = 179 * sr as usize;
        let mut x = Array2::<Float>::zeros((1, n));
        x[[0, hit]] = 1.0;
        let d = native_dynamics(x.view(), sr);
        assert_eq!(
            d.envelope.len(),
            1 + (n as u64 * 22050 / (512 * sr as u64)) as usize
        );

        let want = (179.0 / ENVELOPE_HOP_SEC).round() as usize;
        let lit: Vec<usize> = (0..d.envelope.len())
            .filter(|&k| d.envelope[k] == 1.0)
            .collect();
        assert!(lit.contains(&want), "{lit:?} vs {want}");
        assert!(lit.iter().all(|&k| k.abs_diff(want) <= 2), "{lit:?}");
        // The frame length in seconds matches the main pass.
        assert!((frame_len(sr) as f64 / sr as f64 - ENVELOPE_FRAME_SEC).abs() < 1.0 / sr as f64);
    }

    #[test]
    fn silence_empty_and_short_signals() {
        let none = Ebur128 {
            integrated_lufs: None,
            momentary_max_lufs: None,
            short_term_max_lufs: None,
            range_lu: None,
            true_peak_dbtp: None,
        };

        let silence = Array2::<Float>::zeros((2, 48000 * 5));
        assert_eq!(ebur128(silence.view(), 48000), none);
        let d = native_dynamics(silence.view(), 48000);
        assert_eq!(d.sample_peak_dbfs, None);
        assert_eq!(d.crest_factor_db, None);
        assert_eq!(d.crest_factor_frame_mean_db, None);
        assert!(!d.envelope.is_empty() && d.envelope.iter().all(|&v| v == 0.0));

        for shape in [(0, 0), (2, 0), (0, 100)] {
            let empty = Array2::<Float>::zeros(shape);
            assert_eq!(ebur128(empty.view(), 48000), none);
            let d = native_dynamics(empty.view(), 48000);
            assert_eq!(d.sample_peak_dbfs, None);
            assert!(d.envelope.is_empty());
        }

        let x = tone(2, 48000, &[(-23.0, 0.3)]);
        assert_eq!(ebur128(x.view(), 0), none);
        assert!(native_dynamics(x.view(), 0).envelope.is_empty());

        // Shorter than one 400 ms block: only the peak is defined.
        let r = ebur128(x.view(), 48000);
        assert_eq!(r.integrated_lufs, None);
        assert_eq!(r.momentary_max_lufs, None);
        assert_eq!(r.short_term_max_lufs, None);
        assert_eq!(r.range_lu, None);
        assert!(close(r.true_peak_dbtp, -23.0, 0.1), "{r:?}");

        // Between 400 ms and 3 s: momentary exists, short-term does not.
        let r = ebur128(tone(2, 48000, &[(-23.0, 1.0)]).view(), 48000);
        assert!(close(r.integrated_lufs, -23.0, 0.1), "{r:?}");
        assert_eq!(r.short_term_max_lufs, None);
        assert_eq!(r.range_lu, None);
    }

    #[test]
    fn repeated_calls_are_bit_identical() {
        let x = tone(2, 44100, &[(-20.0, 4.0), (-35.0, 4.0)]);
        assert_eq!(ebur128(x.view(), 44100), ebur128(x.view(), 44100));
        assert_eq!(
            native_dynamics(x.view(), 44100),
            native_dynamics(x.view(), 44100)
        );
    }
}
