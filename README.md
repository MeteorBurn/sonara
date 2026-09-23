# sonara

**High-performance audio analysis library for Python, written in Rust.**

[Документация на русском: API, модули, алгоритмы и DSP](docs/ru/README.md)

Feature extraction, batch analysis, and built-in perceptual features (energy, danceability, valence, key, chords) for playlist generation and music discovery.

> *sonara* — from Latin *sonare*, "to sound, to resonate"

## Quick Start

```bash
python -m pip install https://github.com/MeteorBurn/sonara/releases/download/v0.3.7-meteorburn.1/sonara-0.3.7-cp310-abi3-win_amd64.whl
```

One call gets you 30+ features — tempo, key, chords, energy, mood, timbre — in ~4 ms per 10-second track:

```python
import sonara

r = sonara.analyze_file("track.mp3", mode="playlist")
r.print()
# TrackAnalysis  (3:42)
#
#   Rhythm
#     BPM            128.3
#     Beats          475
#     Onset density  3.21/sec
#
#   Tonal
#     Key                A minor (8A)  (conf 0.81)
#     Predominant chord  Am
#     Chord changes      1.42/sec
#     Dissonance         0.183
#
#   Perceptual
#     Energy         0.78
#     Danceability   0.71
#     Valence        0.42
#     Acousticness   0.12
#     Loudness       -9.2 LUFS
#     Dynamic range  12.4 dB
```

The result is a plain dict subclass — `r['bpm']`, `**r`, and `json.dumps(r)` all work as expected.

Scale to your whole library in parallel across all CPU cores:

```python
from pathlib import Path

files = [str(p) for p in Path("~/Music").expanduser().rglob("*.mp3")]
results = sonara.analyze_batch(files, mode="playlist")
```

This fork release provides a Windows x64 wheel. Requires Python 3.10+.

The package is based on SONARA 0.3.7 with Symphonia 0.6.1, patched Hound 3.5.1, WAV fallback, and reduced metadata probing when tags are not requested. 0.3.7 adds two opt-in rhythm features on top of 0.3.6: the [multiband onset timeline](#multiband-onset-strength-opt-in) (`onset_bands`) and [rhythmic regularity](#rhythmic-regularity-opt-in). Both are additive — `ANALYSIS_SCHEMA_VERSION` stays at `6` and every 0.3.6 field keeps its meaning.

The Quick Start above installs the `v0.3.7-meteorburn.1` release wheel (package version `0.3.7`) — [BUILD-METADATA.md](BUILD-METADATA.md) records that release, its exact source commit and its verification. On other platforms, build from a source checkout with `maturin build --release`.

Installing `sonara` from PyPI retrieves the upstream package. Upstream also provides Linux and macOS wheels.

## Analysis Pipeline

sonara includes a fused analysis pipeline that extracts all features in a single optimized pass. Three modes control the depth of analysis:

### Modes

| Mode | Features | Time (10s track) | Use case |
|------|----------|-------------------|----------|
| **`compact`** | 11 core features | ~1.2 ms | Fast scanning, metadata |
| **`playlist`** | 30+ features incl. tonal & perceptual | ~4 ms | Playlist generation, music discovery |
| **`full`** | All features incl. time signature | ~50 ms | Research, comprehensive analysis |

### Compact mode (default)

Core signal features, always computed:

```python
r = sonara.analyze_file("track.mp3", mode="compact")

r['bpm']                    # Tempo (BPM)
r['bpm_raw']                # Tempo before optional bpm_min/bpm_max alignment
r['bpm_confidence']         # Trust signal for bpm, 0-1 (>0.7 steady, <0.45 suspect)
r['bpm_candidates']         # Top tempo candidates as [bpm, score] pairs, best first
r['beats']                  # Beat frame positions
r['onset_frames']           # Onset positions
r['onset_density']          # Onsets per second
r['rms_mean']               # Average loudness (RMS)
r['rms_max']                # Peak loudness (RMS)
r['loudness_lufs']          # Integrated loudness (LUFS, ITU-R BS.1770-4)
r['dynamic_range_db']       # Loudness range (p95 - p5, dB)
r['spectral_centroid_mean'] # Brightness (Hz)
r['zero_crossing_rate']     # Percussiveness proxy
r['duration_sec']           # Track length
r['provenance']             # How the result was produced: {"schema_version",
                            # "sample_rate" (effective Hz), "hop_length",
                            # "mode", "requested_features" (when features=[...])}
                            # Frame indices convert to seconds as
                            # frame * hop_length / sample_rate.
```

### Playlist mode

Everything for playlist generation: spectral features, MFCCs (timbre fingerprint), chroma (harmony), tonal analysis (chords, dissonance), plus perceptual features:

```python
r = sonara.analyze_file("track.mp3", mode="playlist")

# Perceptual features (0.0 - 1.0)
r['energy']           # Perceived intensity (loudness + brightness + activity)
r['danceability']     # Beat regularity + tempo sweet spot + rhythm
r['valence']          # Mood (0 = sad/dark, 1 = happy/bright)
r['acousticness']     # Acoustic vs electronic character (electronic < 0.3, acoustic > 0.6)

# Note: acousticness and danceability absolute scales were recalibrated in 0.2.4
# (schema v3). Orderings are largely preserved, but consumers with 0.2.2/0.2.3-era
# numeric cutoffs on these fields must re-derive them.

# Musical key
r['key']              # e.g. "C major", "A minor"
r['key_camelot']      # Camelot wheel code for DJ harmonic mixing, e.g. "8B", "8A"
r['key_confidence']   # How confident the key detection is (0.0 - 1.0)

# Tonal analysis
r['chord_sequence']        # Beat-synchronous chord labels, e.g. ["Am", "F", "C", "G"]
r['chord_events']          # Time-spanned chords (merged runs of chord_sequence):
                           # [{"label": "Am", "start_sec": 0.0, "end_sec": 4.1}, ...]
                           # contiguous, covering the whole track
r['predominant_chord']     # Most frequent chord
r['chord_change_rate']     # Chord changes per second (harmonic complexity)
r['dissonance']            # Sensory dissonance (0 = consonant, 1 = rough)

# Spectral features
r['spectral_bandwidth_mean']   # Frequency spread
r['spectral_rolloff_mean']     # Frequency below which 85% of energy sits
r['spectral_flatness_mean']    # Tonal (0) vs noise-like (1)
r['spectral_contrast_mean']    # Peak-valley ratio per band (7 values)
r['mfcc_mean']                 # Timbre fingerprint (13 coefficients)
r['chroma_mean']               # Pitch class distribution (12 values)
```

### Full mode

Adds expensive rhythm analysis features on top of playlist mode:

```python
r = sonara.analyze_file("track.mp3", mode="full")

r['tempo_curve']                # Per-beat BPM values
r['tempo_variability']          # Coefficient of variation of tempo
r['time_signature']             # e.g. "4/4", "3/4"
r['time_signature_confidence']  # Detection confidence
```

### BPM range alignment

For host applications with a project-level tempo window — e.g. a DJ library
configured with a lowest/highest BPM — pass `bpm_min` and `bpm_max`. When both
are given, the estimated tempo is folded by octaves until it lands inside the
range: values below `bpm_min` are doubled, values above `bpm_max` are halved.
This corrects the half/double-tempo octave errors common on electronic music.

```python
# 79-192 BPM window (matching a typical electronic-music library)
r = sonara.analyze_file("track.mp3", mode="playlist", bpm_min=79.0, bpm_max=192.0)

r['bpm']              # Tempo folded into [79, 192]
r['bpm_raw']          # Tempo before alignment (what you'd get without the range)
r['bpm_confidence']   # How firmly the tempo is anchored, 0-1 (trust signal, not probability)
r['bpm_candidates']   # Ranked [bpm, score] candidates the estimate was chosen from
```

Both bounds must be provided together, be finite and positive with
`bpm_min < bpm_max`, and span at least one octave (`bpm_max >= 2 * bpm_min`).
Alignment is opt-in: without a range, `bpm` equals `bpm_raw`. The same
parameters are available on the lower-level beat tracker:

```python
tempo, beats = sonara.beat_track(y=y, sr=sr, bpm_min=79.0, bpm_max=192.0)
```

### Beat grid (opt-in)

For DJ-style beat matching, sonara can turn the raw beat list into a *grid*: where the first beat falls, which beats begin each bar (downbeats), and how rigidly the beats fit a constant-tempo lattice. This is **opt-in only** — it is never computed by the `compact`, `playlist`, or `full` modes and adds no cost to them. Request it with `features=["beatgrid"]`:

```python
r = sonara.analyze_file("track.mp3", features=["beatgrid"])

r['grid_offset_sec']   # Time (sec) of the first beat — the grid anchor
r['downbeats']         # Frame indices of bar-starting beats (subset of beats)
r['grid_stability']    # 0.0-1.0: how rigidly beats fit a constant grid
```

The three keys appear **only** when `beatgrid` is requested; they are absent otherwise. It reuses the beats and onset envelope already computed by the pipeline, so it is O(number of beats):

- **`grid_offset_sec`** — the time of the first tracked beat.
- **`downbeats`** — assuming 4/4 (or the detected `time_signature` when that is also requested), each of the possible bar phases is scored by onset-accent energy at the candidate downbeats; the highest-scoring phase wins. Kicks and other bar-anchoring accents typically land on beat one.
- **`grid_stability`** — `clamp(1 - MAD / median, 0, 1)`, where the inter-beat intervals have median `median` and median absolute deviation `MAD`. A perfectly regular grid scores `1.0`; jitter lowers the score monotonically. Useful both as a confidence measure and, in future, for tempo-octave disambiguation via grid regularity.

You can combine it with other features, e.g. `features=["bpm", "beatgrid"]` or `features=["beatgrid", "time_signature"]` (the latter lets the grid honour a detected non-4/4 meter).

### Loudness & gain (opt-in)

Broadcast-standard loudness and gain metrics — what players and mix software
consume for auto-gain and clip protection. These are **opt-in**: they are never
computed by any mode's defaults (for performance), only when you explicitly
request the `loudness` feature group. They extend the always-on
`loudness_lufs` (integrated LUFS, ITU-R BS.1770-4) and `dynamic_range_db`.

```python
r = sonara.analyze_file("track.mp3", features=["loudness"])

r['true_peak_db']                # True peak (dBTP), 4x oversampled per BS.1770-4
r['replaygain_db']               # Track gain to reach -18 LUFS: -18 - loudness_lufs
r['loudness_curve']              # Short-term LUFS per window (3 s window, 1 s hop)
r['loudness_momentary_max_db']   # Max momentary loudness (400 ms window), dB
r['loudness_range_lu']           # EBU R128 loudness range (LRA), LU
```

- **`true_peak_db`** — the highest inter-sample peak, computed on a 4x oversampled
  signal (windowed-sinc polyphase interpolation, per BS.1770-4 Annex 2). A value
  above `0.0` dBTP means the waveform overshoots full scale between samples and
  can clip a downstream reconstruction filter / DAC.
- **`replaygain_db`** — ReplayGain-style track gain to the -18 LUFS reference,
  `-18 - loudness_lufs`. Add it to the signal (or set the player's gain) for
  consistent perceived level across tracks.
- **`loudness_curve`** — the short-term loudness trajectory: one LUFS value per
  3-second window at a 1-second hop (empty for tracks under one window).
- **`loudness_range_lu`** — EBU R128 LRA: the gated 95th-10th percentile spread
  of the short-term distribution (with the -20 LU relative gate). This is the
  standardized counterpart to the approximate `dynamic_range_db`.

**Gain staging example** — normalize levels and guard against clipping:

```python
r = sonara.analyze_file("track.mp3", features=["loudness"])

gain = r['replaygain_db']            # dB to apply for a -18 LUFS target
# Applying `gain` shifts the true peak by the same amount; check for clipping:
projected_peak = r['true_peak_db'] + gain
if projected_peak > -1.0:            # keep ~1 dB of true-peak headroom
    gain -= (projected_peak + 1.0)   # back off so the peak lands at -1 dBTP

linear_gain = 10 ** (gain / 20.0)
y_out = y * linear_gain
```

### Custom feature selection

Cherry-pick specific features regardless of mode:

```python
r = sonara.analyze_file("track.mp3", features=["bpm", "energy", "key", "chords"])
```

Valid feature names: `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`, `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`, `chords`, `dissonance`, `energy`, `danceability`, `key`, `valence`, `acousticness`, `tempo_curve`, `time_signature` — plus the **opt-in-only** features `beatgrid`, `onset_bands`, `rhythmic_regularity`, `structure`, `embedding`, `aggression`, `fingerprint`, `loudness`, `silence`, `key_candidates`, `vocalness`, `mood`, `instrumentalness`, `tags`, which are never computed by any mode and must be requested explicitly (see their sections below).

### Multiband onset strength (opt-in)

Use `onset_bands` when a single broadband onset envelope is not enough and
low-, mid-, and high-frequency attacks must remain separate. The fused analyzer
reuses its existing log-mel spectrogram, so requesting this feature does not run
a second STFT:

```python
import numpy as np

signal = np.asarray(y, dtype=np.float32)
r = sonara.analyze_signal(signal, sr=22050, features=["onset_bands"])

r["onset_strength_bands"]  # list of n_bands lists, each n_frames long
r["onset_band_edges_hz"]   # boundaries, length n_bands + 1
```

The result is a **timeline**, not a summary: row `b` is the onset-strength
envelope of the band between `onset_band_edges_hz[b]` and
`onset_band_edges_hz[b + 1]`, and position `k` within a row is analysis frame
`k`. Frames share the pipeline's time base, so they convert to seconds with the
`provenance` fields every result carries:

```python
bands = r["onset_strength_bands"]        # plain lists, so this is JSON-round-trippable
hop   = r["provenance"]["hop_length"]
sr    = r["provenance"]["sample_rate"]   # effective Hz, after any resampling

times = [k * hop / sr for k in range(len(bands[0]))]
low   = bands[0]                         # kick/bass attacks over time
```

Note the type difference: in an `analyze_*` result the envelopes are plain
Python lists (so the whole result still survives `json.dumps`), while the
standalone `sonara.onset_strength_bands()` below returns a real
`float32` numpy array of shape `(n_bands, n_frames)`.

The band envelopes use the same frame-index time base as `onset_frames` and
`beats`. The `energy_curve` has a separate, coarser time step given by
`energy_curve_hop_sec`; do not align its samples with band frames by index.
The band envelopes are also an input to further rhythm analysis; see
[rhythmic regularity](#rhythmic-regularity-opt-in), which is measured from
exactly these envelopes.

The default boundaries start from `0, 200, 800, 3200, 8000, Nyquist` Hz.
At low sample rates, split points at or above Nyquist and split points that
would leave an adjacent band without one of the 128 mel-filter centers are
omitted. The returned boundaries therefore describe the rows exactly.

For standalone use, `onset_strength_bands` returns the envelopes and their
effective boundaries:

```python
import numpy as np

signal = np.asarray(y, dtype=np.float32)
bands, edges = sonara.onset_strength_bands(
    signal,
    sr=22050,
    hop_length=512,
    band_edges_hz=[0.0, 500.0, 4000.0, 11025.0],  # optional
)
```

Custom boundaries must contain at least two finite, strictly increasing values
within `0..=Nyquist`, and every resulting interval must contain at least one mel
center; invalid definitions raise `ValueError`.

### Rhythmic regularity (opt-in)

How *straight* the percussive pattern sits inside the metric grid: `1.0` is a
regular pattern locked to the pulse lattice (house, techno), `0.0` a syncopated,
polyrhythmic or breakbeat-driven one (garage, 2-step, breakbeat, drum & bass).
It is **opt-in only** — no mode computes it — and it is a deterministic DSP
measure: no model, no training data, no bundled artifact.

```python
r = sonara.analyze_file("track.mp3", features=["rhythmic_regularity"])

r["rhythmic_regularity"]             # 0.0-1.0, or None when it abstained
r["rhythmic_regularity_label"]       # "regular" | "irregular"  (>= 0.5 → regular)
r["rhythmic_regularity_confidence"]  # 0.0-1.0 quality of the rhythmic evidence
r["rhythmic_regularity_candidates"]  # [("regular", 0.83), ("irregular", 0.17)]
```

**It abstains rather than guess.** Silence, ambient and drumless passages carry
no rhythmic evidence, and calling them "irregular" would be wrong. When no
window clears the reliability gate the score, label and candidates are all
`None` and only the confidence is reported — so `None` means *measured, could
not tell*, never *not computed*:

```python
if r["rhythmic_regularity"] is None:
    print(f"no rhythmic evidence (confidence {r['rhythmic_regularity_confidence']:.2f})")
```

**This is not `1 - grid_stability`.** `grid_stability` describes temporal drift
*of* the beat grid; `rhythmic_regularity` takes the grid as given and describes
the distribution of drum hits *within* it. A track can have a rigid grid and a
heavily syncopated pattern on top of it.

It also does **not** distinguish the irregular idioms from one another — garage,
2-step, breakbeat and drum & bass are all simply `irregular`. It is a rhythm
descriptor, not a genre classifier.

**How it works.** Each inter-beat interval is split into 4 tatum cells by
interpolating between the *tracked* beats (which is what makes the measure
independent of tempo drift, and beat-relative rather than sample-rate
dependent). Every [multiband onset](#multiband-onset-strength-opt-in) band
samples its peak in each cell. Bars are then grouped into non-overlapping
4-bar **windows**, each measured on its own, so a fill, a breakdown or a
drumless intro cannot decide the track.

Within a window the cells are folded modulo the bar, phase aligned to the
detected downbeats, and averaged. Each band's profile is reduced to its
contrast above its own floor — a constant pedestal is pure DC and describes no
rhythm — and that contrast mass is the band's evidence weight, so a flat band
(steady 16th hats, or no percussion) contributes nothing.

A window is measured only if it passes a **reliability gate**: the profile's
between-slot variance is decomposed into real structure and the noise of
averaging finitely many bars, and the window counts only when structure
explains more of it than noise does. That is what makes abstention principled
rather than a tuned cutoff — a drumless passage has nothing above its own
noise floor.

Two bounded measures, pooled over all bands by evidence mass, are then
combined with equal weight:

- **metrical alignment** — the inner product of the profile with the
  Longuet-Higgins & Lee (1984) metrical-weight hierarchy, read against a
  meter-blind distribution (`0.0`) and an even beat lattice (`1.0`).
  Phase-sensitive: it catches a kick displaced onto the weak sixteenths.
- **sub-bar periodicity** — the share of the profile's AC energy on the DFT
  bins of periods shorter than the bar. Phase-insensitive: it separates a
  figure that recurs inside the bar from a chopped break that only closes over
  the whole bar, and its complement is the energy of **pulses competing with
  the metre** (three-against-four and friends).

The track value is the evidence-weighted **median** over valid windows —
robust, so a minority of deviant windows cannot move it.

Two further measures are computed and **reported as diagnostics but excluded
from the score** — `beat_concentration` (accent mass on the quarters versus
between them) and `beat_coverage` (how many strong beats are struck). Both ask
"is the kick on the quarters", and against manual labels neither answers this
target: AUC 0.469 and 0.603 against 0.829 and 0.882 for the two measures above.
The reasons are structural — the 93 ms window cannot resolve *which* sixteenth
a low-band attack occupies, and broken-beat house keeps a kick near most
quarters while displacing it *within* the beat, which is exactly the
displacement those measures cannot see. Including them only added a
near-constant offset that pushed the distribution off the boundary.

`confidence` is the geometric mean of the evidence factors: `bpm_confidence`,
`grid_stability`, `1 - 1/sqrt(n_valid_windows)`, the mean window reliability,
and how consistently the same accent pattern recurred across windows — plus
`time_signature_confidence` when that was also requested. Pattern stability
feeds *confidence only*, never the score: a stable breakbeat is still a
breakbeat, and no amount of grid rigidity makes a broken pattern straight. It
is a measure of evidence quality, **not** a class probability.

Requesting the feature internally computes the onset bands and beat grid it
reads, but does not add them to the result unless you ask for them too:

```python
r = sonara.analyze_file(
    "track.mp3", features=["rhythmic_regularity", "onset_bands", "beatgrid"]
)
```

**What it is measured against.** The score is validated on synthetic patterns
with a known metrical layout and on two real-music sets.

*Manual labels* — 145 hand-labelled tracks (100 straight four-on-the-floor, 45
broken-beat tech house at the same tempo, the hard case): ROC AUC **0.883**,
balanced accuracy **0.836** at the `0.5` boundary (straight 76%, broken 91%).
The boundary is the midpoint of the normalized scale, never fitted — its
equal-error point on these labels is `0.480`, so the principled midpoint is
already the right operating point.

15 of those broken-beat tracks were supplied *after* the measure was final and
share no track with the 30 used to choose its components. They scored 86.7%
correct against 93.3% on the selection set (means 0.420 and 0.407), so the
choice did not overfit. That holdout covers the irregular side only — the
straight side has no fresh held-out set yet, and is where the error sits.

*Genre folders* — an independent 148-track sample labelled only by library
folder: AUC **0.715**, balanced accuracy 0.62. That set is weaker, and known to
be contaminated: broken-beat tech house lives in the *House* folder, so folder
genre is not a rhythm label. Where the two disagree the manual labels decide.

Class separation is therefore sound, and the scale is calibrated to the point
where the natural midpoint works. It has **not** been validated on a fresh
held-out manual set.

Three known limitations, all measured:

- **It inherits beat phase from the beat tracker.** The two phase-sensitive
  components (`concentration`, `alignment`) collapse if the tracker locks half
  a beat off — on sparse material with a weak kick it can lock to the offbeat
  hats. `bpm_confidence` is the signal for that; the score degrades rather than
  flipping, because the other two components are phase-invariant.
- **Sixteenth resolution is at the edge of what the representation supports.**
  The 2048-sample STFT window is 93 ms against a 117 ms sixteenth at 128 BPM,
  so accents leak into the neighbouring cell and `beat_concentration` pins at
  zero on about 30% of real tracks. An eighth-note grid removes the leak but
  measures less and scored worse end to end (AUC 0.737).
- **It is a rhythm descriptor, not a genre detector.** A house track with a
  broken-beat section, or a liquid DnB roller with a four-on-the-floor kick,
  will read the way its *rhythm* reads. This is a feature, not a defect: on the
  manual set it correctly called 93% of broken-beat tech house irregular even
  though every one of those tracks is filed as house.

### Structure & energy (opt-in)

Where things happen in a track — a time-resolved energy curve, section
boundaries, intro/outro, and a coarse 1-10 energy level. This is **opt-in**:
it is *not* part of any mode (compact/playlist/full) and is only computed when
you explicitly request `features=["structure"]`, so the default pipelines pay
nothing for it.

```python
r = sonara.analyze_file("track.mp3", features=["structure"])

r['energy_level']          # Coarse intensity, 1-10 (spread across the range)
r['energy_curve']          # Per-window perceptual energy, 0-1
r['energy_curve_hop_sec']  # Seconds between curve samples (map index -> time)
r['intro_end_sec']         # End of the intro / pre-first-drop region
r['outro_start_sec']       # Start of the outro / final fade

for seg in r['segments']:  # Contiguous sections covering the whole track
    print(f"{seg['start_sec']:6.1f} - {seg['end_sec']:6.1f}s  "
          f"energy {seg['energy']:.2f}")

# Map the energy curve to a timeline:
hop = r['energy_curve_hop_sec']
times = [i * hop for i in range(len(r['energy_curve']))]
```

Example output for a 3:42 electronic track:

```
Energy level 8/10
Segments     6
Intro end    0:18
Outro start  3:19

  12.4 -  30.1s  energy 0.41   (build)
  30.1 -  78.6s  energy 0.79   (drop)
  78.6 - 110.2s  energy 0.55   (breakdown)
 110.2 - 158.9s  energy 0.81   (drop)
 ...
```

**How it works.** The energy curve reuses the per-frame RMS, spectral centroid,
and bandwidth already computed by the pipeline (1 s windows, ~0.5 s hop) fed
through the same 0-1 perceptual-energy model. Boundaries use classical
self-similarity novelty (Foote): a per-window timbral descriptor (mean MFCC)
builds a cosine self-similarity matrix, a Gaussian-tapered checkerboard kernel
is slid down its diagonal to produce a novelty curve, and adaptive peak-picking
(min 8 s spacing) yields the cuts. Intro/outro is an honest heuristic based on
where the energy curve crosses the midpoint between its 10th and 90th
percentiles, snapped to a nearby boundary. `energy_level` stretches the observed
0.25-0.60 mean-energy band (measured over a large commercial library) across 1-10 so real music spreads out instead of
clustering at 5-6.

### Opt-in extras

Three lightweight extras are **opt-in only** — they are never computed by any
mode (performance-first policy) and appear in the result dict only when you
request them explicitly via `features=[...]`. Request them alone or alongside a
mode's features.

#### Silence offsets — `features=["silence"]`

Leading/trailing silence duration in seconds, derived from the per-frame RMS the
pipeline already computes (pure arithmetic, effectively free — kept opt-in only
so default modes stay byte-for-byte unchanged).

```python
r = sonara.analyze_file("track.mp3", features=["silence"])
r['leading_silence_sec']    # e.g. 1.50  — silence at the start
r['trailing_silence_sec']   # e.g. 2.25  — silence at the end
```

A frame counts as silent when its RMS is below **-60 dBFS** relative to full
scale (amplitude `10^(-60/20) ≈ 0.001`). A small hysteresis rule (3 consecutive
frames) means an isolated loud click in an otherwise-silent lead-in does **not**
end the silence. Both values are clamped to `[0, duration_sec]`.

#### Key candidates — `features=["key_candidates"]`

Top-3 ranked key guesses as `(key, camelot, score)` tuples, mirroring the design
of `bpm_candidates`. Same 24-profile correlation as `key`; this exposes the
ranking instead of only the winner. The first entry always equals `key`.

```python
r = sonara.analyze_file("track.mp3", features=["key_candidates"])
r['key_candidates']
# [("A minor", "8A", 0.81), ("C major", "8B", 0.74), ("E minor", "9A", 0.52)]
```

`score` is the Pearson correlation against each key profile, clamped to `[0, 1]`
and in descending order; `camelot` is the Camelot-wheel code for harmonic mixing.

#### Vocal presence — `features=["vocalness"]`

A single **heuristic v2** score in `[0, 1]` for the prominence of vocal/broadband
energy — **changed semantics in 0.2.4** (the old v1 heuristic inverted on real
music, scoring distorted vocals low and solo instruments high). This is a rough
indicator, **not** a trained classifier.

```python
r = sonara.analyze_file("track.mp3", features=["vocalness"])
r['vocalness']   # e.g. 0.72
```

It measures how much energy fills the ~0.8–5.6 kHz **spectral valleys** (low
mid-band peak-to-valley contrast → high vocalness): voice and broadband/screamed
vocals fill the valleys, while clean solo pitched instruments leave them deep. It
therefore rises **harsh > clean > instrumental** (screamed metal ≈ 1.0, clean
singing mid, solo sax/flute/violin low). Known ambiguous cases: sparse
voice+piano ballads read mid-low, and a voice-mimicking solo violin reads
borderline. Treat it as a soft hint.

For higher precision, train a small model over the similarity embedding on your
own labeled library — the same **socket** used for [genre](#bring-your-own-genre-model)
works for a vocal/instrumental classifier.

#### Mood — `features=["mood"]`

Four rough mood affinities in `[0, 1]` — `mood_happy`, `mood_aggressive`,
`mood_relaxed`, `mood_sad` — all populated together. These are **heuristic v1,
not an ML classifier**: weighted blends of scalars the pipeline already computed
(musical mode, tempo, brightness, energy, rhythmic density, dissonance,
dynamics), in the same spirit as `valence`/`acousticness`. The four are
correlated (a happy track tends to be un-sad) but are **not** constrained to sum
to 1. Treat them as coarse tags, not ground truth.

```python
r = sonara.analyze_file("track.mp3", features=["mood"])
r['mood_happy'], r['mood_aggressive'], r['mood_relaxed'], r['mood_sad']
# e.g. (0.71, 0.30, 0.55, 0.12)
```

#### Instrumentalness — `features=["instrumentalness"]`

A single **heuristic** score in `[0, 1]`, the inverse of the vocalness heuristic
(`1 - vocalness`, clamped): higher means less vocal-like / more instrumental.
**Heuristic v2, not an ML classifier** — **changed semantics in 0.2.4** (now
contrast-based, tracking `vocalness`); a soft hint, not a trained speech/vocal
detector.

```python
r = sonara.analyze_file("track.mp3", features=["instrumentalness"])
r['instrumentalness']   # e.g. 0.28
```

#### File metadata tags — `features=["tags"]`

Pass through the container/stream metadata (ID3v2, Vorbis comments, …) already
embedded in the file. Available on `analyze_file`/`analyze_batch` only — a bare
signal (`analyze_signal`) has no container, so it never carries tags.

```python
r = sonara.analyze_file("track.mp3", features=["tags"])
r['tags']
# {"title": "...", "artist": "...", "album": "...",
#  "genre": "Electronic", "year": 2024, "original_year": 1969, "track_no": 3}
```

Keys appear only when the file actually carries that tag: `title`, `artist`,
`album`, `genre` (strings), `year`, `original_year` and `track_no` (ints).
`year` is derived from the leading 4 digits of the file's date tag (`Date`/
`ReleaseDate`); `original_year` from the original-release-date tags (ID3v2.4
`TDOR`, ID3v2.3 `TORY`, `TXXX:originalyear`, Vorbis `ORIGINALDATE`) — on a
reissue or compilation `year` is the reissue date and `original_year` the true
original release year, so prefer `original_year` for era reasoning when present.
`track_no` is the leading integer of a `"3/12"`-style value. Tags come from **symphonia**-decoded containers
(FLAC/Vorbis, MP3/AAC ID3v2, MP4, …); **WAV** goes through a separate fast path
that carries no tags, so `.wav` inputs yield no `tags` values. Note that
`tags['genre']` is the *file's* metadata genre and is unrelated to the reserved
top-level `genre` placeholder (a future computed field).

### Batch analysis

Analyze entire music libraries in parallel using all CPU cores:

```python
import sonara
from pathlib import Path

files = [str(p) for p in Path("~/Music").rglob("*.mp3")]

# Optional progress callback: called as progress(done, total) after each file
# finishes (success or failure). `done` counts completions in completion order
# (not input order); a raising callback never aborts the batch.
results = sonara.analyze_batch(
    files, mode="playlist",
    progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True),
)

for r in results:
    if r.failed:            # a file that could not be decoded/read
        print(f"SKIP [{r['error_kind']}] {r['path']}: {r['error']}")
        continue
    print(f"{r['bpm']:5.0f} BPM | {r['energy']:.2f} energy | "
          f"{r['key']:>10} | {r['predominant_chord']:>4} | "
          f"{r['dissonance']:.3f} diss | {r['valence']:.2f} valence")
```

**Per-file error handling.** `analyze_batch` never raises on a single bad file —
essential when scanning large libraries. It always returns exactly one entry per
input path, in input order. A file that fails to decode yields a failure entry
(`r.failed` is `True`) carrying `path`, `error` (human-readable, including the
container/codec and underlying cause) and `error_kind` — a short stable category:
`"io"`, `"decode"`, `"unsupported_format"`, `"invalid_audio"`, `"insufficient_data"`,
or `"compute"`. (`analyze_file` on a single path still raises as before.)

### Augmenting a cached analysis

`sonara.augment_analysis` recomputes named features onto a **copy** of a cached
result — a `TrackAnalysis`, or the same dict shape loaded back from JSON —
without re-running the whole pipeline. Features whose inputs are already on the
record (most scalar features, and the similarity embedding) are recomputed
**decode-free** from the cached fields alone; anything that genuinely needs the
waveform again is computed from `audio_path` in one targeted re-analysis at the
record's own `provenance["sample_rate"]`. The input is never mutated, fields you
did not ask about are never cleared, and unknown feature names raise
`ValueError` (no silent fallback).

```python
import json, sonara

cached = json.load(open("track.json"))            # a stored analyze_* result
r = sonara.augment_analysis(cached, ["mood"])     # decode-free from cached fields
r = sonara.augment_analysis(cached, ["aggression"], audio_path="track.mp3")

sonara.can_augment(cached, "mood")            # True → decode-free on this record
sonara.augment_blocker(cached, "aggression")  # e.g. "needs audio (...)" — or None
```

Whether a feature is decode-free is **per record** (it depends on which fields
that record actually carries), which is what `can_augment` / `augment_blocker`
answer. The static side is `sonara.feature_dependencies()` — the declared
per-feature dependency map, one dict per feature with its dependency `class`
(`"audio"`, `"frame_curves"`, `"scalars"`, `"embedding"`), the
`required_evidence` fields a decode-free recompute reads, and mode flags — so a
cache layer can plan what is refreshable without audio before touching any
record.

### Adding the 0.3.7 rhythm features to an existing library

**Keep your existing analysis.** Updating from 0.3.6 to 0.3.7 does not require
recomputing existing fields just to obtain these additions. The analysis schema
remains `6`. To get `onset_bands` and `rhythmic_regularity`, Sonara must read
each original audio file again, but you can request only those features and
keep the previously stored BPM, key, embeddings and other results.

If you saved a complete Sonara result dict (including `provenance`), add both
features in one call:

```python
import json
import sonara

with open("track.json", encoding="utf-8") as source:
    cached = json.load(source)  # original analyze_* result, not a custom DB row

updated = sonara.augment_analysis(
    cached,
    ["onset_bands", "rhythmic_regularity"],
    audio_path="track.mp3",
)
# Save updated through your application's storage layer.
# cached is unchanged; existing fields are preserved in updated.
```

If your database splits the result into renamed columns and a separate timeline,
that database row is **not** a Sonara result dict. The simplest option is a
targeted analysis, then saving only the new fields into your existing record:

```python
extra = sonara.analyze_file(
    "track.mp3",
    sr=22050,  # use the effective sample rate stored with your old timeline
    features=["onset_bands", "rhythmic_regularity"],
    # If the original analysis used a BPM range, pass the same bounds here:
    # bpm_min=79.0, bpm_max=192.0,
)

timeline_additions = {
    "onset_strength_bands": extra["onset_strength_bands"],
    "onset_band_edges_hz": extra["onset_band_edges_hz"],
}
rhythm_additions = {
    name: extra[name]
    for name in (
        "rhythmic_regularity",
        "rhythmic_regularity_label",
        "rhythmic_regularity_confidence",
        "rhythmic_regularity_candidates",
    )
}
```

Merge these additions rather than replacing the old result with `extra`.
Keep the new timeline's `extra["provenance"]["sample_rate"]` and `hop_length`;
verify they match the existing timeline before sharing its time-base metadata.
Sonara does not migrate or write your database automatically.

In 0.3.7, `can_augment(cached, "onset_bands")` and
`can_augment(cached, "rhythmic_regularity")` return `False`: this means
**audio is required**, not that augmentation is unsupported. Even cached onset
bands do not enable a decode-free regularity call through this API.

For a resumable backfill, mark regularity as processed when its confidence has
been stored, even if the score is `None` (insufficient evidence). Otherwise such
tracks would be retried forever. Store both onset-band fields together; the
envelopes can be large, so check storage growth on a small batch first.

### Duplicate detection (opt-in)

sonara can compute a compact acoustic **fingerprint** that identifies the *same
recording* across different encodings, bitrates and playback gains — the classic
"find duplicate files in my library" problem. It survives MP3/AAC re-encoding,
level normalization and a little extra leading silence, but is not meant to match
tempo- or pitch-shifted versions.

The fingerprint is **opt-in** (performance-first: no analysis mode computes it by
default). Request it with `features=["fingerprint"]`; the result then carries a
base64 `fingerprint` string and an integer `fingerprint_version`:

```python
import sonara

r = sonara.analyze_file("track.mp3", features=["fingerprint"])
r["fingerprint"]          # base64 string, ~8 sub-fingerprints/sec
r["fingerprint_version"]  # format version (int)
```

Compare any two fingerprints with `sonara.fingerprint_match`, which accepts either
the base64 strings or whole result dicts and returns a similarity in `[0, 1]`. A
score above **0.30** means "same recording" (duplicates typically score > 0.7,
unrelated tracks < 0.15):

```python
a = sonara.analyze_file("track.flac", features=["fingerprint"])
b = sonara.analyze_file("track_v0.mp3", features=["fingerprint"])
sonara.fingerprint_match(a, b)   # e.g. 0.98 → same recording
```

Find duplicates across a whole folder:

```python
import sonara
from pathlib import Path

files = [str(p) for p in Path("~/Music").expanduser().rglob("*.mp3")]
results = sonara.analyze_batch(files, features=["fingerprint"])

# Keep only successfully-analyzed tracks that have a fingerprint.
fps = [(f, r["fingerprint"]) for f, r in zip(files, results)
       if not r.failed and "fingerprint" in r]

seen, duplicates = [], []
for path, fp in fps:
    match = next((p for p, other in seen
                  if sonara.fingerprint_match(fp, other) > 0.30), None)
    if match is not None:
        duplicates.append((path, match))   # path is a duplicate of match
    else:
        seen.append((path, fp))

for dup, original in duplicates:
    print(f"DUPLICATE  {dup}\n     of     {original}")
```

The pairwise scan above is `O(n²)`; for very large libraries, bucket candidates
first (e.g. by rounded `duration_sec`) and only fingerprint-match within a bucket.

## Similarity & embeddings

sonara can produce a fixed-length **similarity vector** (a hand-crafted, 48-dimension embedding) for nearest-neighbor search over a music library — no ML dependency. It is assembled from features the pipeline already computes (MFCC timbre, chroma harmony, spectral shape, rhythm, dynamics, and tonal descriptors), each with **fixed, documented normalization** so vectors are comparable across tracks, machines, and library runs.

The vector is **opt-in** — it is never produced by a bare mode. Request it explicitly with `features=["embedding"]` (this also pulls in the playlist-level features it is built from):

```python
import sonara

r = sonara.analyze_file("track.mp3", features=["embedding"])
r["embedding"]          # list of 48 floats, each in [0, 1]
r["embedding_version"]  # layout version (int); compare only same-version vectors
```

Compare two tracks with `sonara.similarity(a, b)` — it returns a score in `0..1` (higher = more similar) and accepts either `TrackAnalysis` results or raw vectors:

```python
a = sonara.analyze_file("a.mp3", features=["embedding"])
b = sonara.analyze_file("b.mp3", features=["embedding"])
sonara.similarity(a, b)          # e.g. 0.65 for close neighbors; ~0.5 for unrelated tracks
sonara.similarity(a, a)          # 1.0 (identical)
sonara.similarity(a["embedding"], b["embedding"])  # raw vectors also work
```

`similarity` (and the lower-level `embedding_distance`) take a `profile=`
keyword selecting the comparison-time weight table: `"default"` is the
historical balanced metric, `"timbre"` makes spectral texture dominate and
demotes tempo/energy, so neighbors share *sonic style* rather than pace.
Profiles are applied at distance time and never change the stored vector — the
same persisted embeddings work with every profile. Unknown profile names raise
`ValueError`. `sonara.SIMILARITY_PROFILES` maps each profile name to its
weight-table version (the `"default"` profile's version aliases
`SIMILARITY_VERSION`; other profiles version independently):

```python
sonara.similarity(a, b, profile="timbre")  # sonic-texture neighbors
sonara.SIMILARITY_PROFILES                 # e.g. {"default": 2, "timbre": 1}
```

### Nearest-neighbor search over a library

```python
import sonara
from pathlib import Path

files = [str(p) for p in Path("~/Music").expanduser().rglob("*.mp3")]
library = sonara.analyze_batch(files, features=["embedding"])

def most_similar(query, library, k=5):
    scored = [
        (path, sonara.similarity(query, cand))
        for path, cand in zip(files, library)
        if cand is not query
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]

seed = library[0]
for path, score in most_similar(seed, library):
    print(f"{score:.3f}  {path}")
```

The metric is a **weighted, normalized Euclidean distance** (not cosine): all dimensions are non-negative and bounded to `[0, 1]`, where cosine is biased toward 1 — Euclidean stays discriminative, and per-dimension weights let timbre, harmony and tempo dominate over incidental dimensions like absolute loudness. Because loudness contributes little, the *same* track at a different gain still scores as highly similar. `sonara.similarity()` applies a calibrated stretch (measured on a large commercial library) so scores are interpretable: an unrelated pair lands near **0.5**, close neighbors **0.65+**, identical tracks **1.0**. The stretch is monotone in the raw distance, so nearest-neighbor rankings are unaffected. The hand-crafted vector sits behind `embedding_version`, so a learned (e.g. ONNX) embedding can later replace it behind the same field and API.

## Aggression model

sonara includes a bundled perceptual aggression ranker. It combines physical
force, harshness, tension, and rhythm evidence in the existing analysis pass,
then applies a compact FerricML linear rank plus a small tree ensemble. The
rank is in `[0, 1]`; it is not a probability.

```python
# Preferred: include the score in Sonara's single fused analysis pass.
r = sonara.analyze_file("track.mp3", features=["aggression"])
score = r["aggression_score"]
support = r["aggression_confidence"]
harshness = r["aggression_harshness"]

# The older embedding-only scorer remains available for stored v1 values:
score = sonara.aggression_score(
    stored_embedding, embedding_version=stored_embedding_version
)

# A convenience call is available when aggression is the only desired output:
aggression = sonara.analyze_aggression_file("track.mp3")
score = aggression["aggression_score"]  # None when evidence is insufficient

# Parallel fused library scan with one input-ordered result per path:
results = sonara.analyze_batch(["a.mp3", "b.flac"], features=["aggression"])
results[0]["aggression_score"]
```

The model artifacts are checksum- and schema-validated once. Feature extraction
reuses the pipeline's FFT, onset, rhythm, and perceptual work; internal
dependencies are not exposed unless separately requested. Rust users enable the
`aggression` Cargo feature and request `"aggression"`, or call
`sonara::aggression::{score, score_versioned, analyze_file, analyze_signal,
analyze_batch}`.
Model inference is allocation-free after one-time artifact loading. FerricML
and the rank model remain absent from Sonara's default Rust build.

## Bring your own genre model

sonara ships **no** genre model — genre is subjective and library-specific. Instead it exposes a **socket**: train a tiny classifier over the similarity embedding on *your* labeled library, save it as JSON, and pass its path to any `analyze_*` call. The result then carries `genre` and `genre_confidence`.

The full loop — scan for embeddings, label them, train, save, classify:

```python
import sonara
from sonara import genre

# 1. Scan a labeled library for embeddings (48-dim vectors).
files  = [...]                       # paths you already have genre labels for
labels = [...]                       # one label per file (e.g. "rock", "techno")
lib = sonara.analyze_batch(files, features=["embedding"])
X = [r["embedding"] for r in lib]    # (n, 48)

# 2. Train a classifier (pure numpy — no sklearn/torch). hidden=0 is softmax
#    regression; hidden=N adds one ReLU layer. Deterministic given `seed`.
model = genre.train(X, labels, hidden=0, epochs=300, lr=0.1, seed=0)
model.save("genre_model.json")

# 3. Classify new tracks by passing the model path.
r = sonara.analyze_file("track.mp3", genre_model="genre_model.json")
print(r["genre"], r["genre_confidence"])   # e.g. "techno" 0.87
```

`model.predict(x)` runs the same inference numpy-side (label + confidence) for parity with the Rust analysis. The embedding is computed internally whenever a model is set; the `embedding`/`embedding_version` fields are only added to the result if you *also* pass `features=["embedding"]`.

The model is a small feed-forward net stored as JSON — **hand-writable** as well as numpy-exportable:

```json
{
  "format_version": 1,
  "embedding_version": 2,
  "labels": ["rock", "electronic"],
  "layers": [
    {"weights": [[...], ...], "bias": [...], "activation": "relu"},
    {"weights": [[...]],       "bias": [...], "activation": "softmax"}
  ]
}
```

Each layer maps `x` → `activation(W·x + b)`, with `W` stored row-major (`out_dim` rows × `in_dim` cols). The first layer's `in_dim` must equal `EMBEDDING_DIM` (48); the last layer must be `softmax` with `out_dim == len(labels)`. Supported activations: `relu`, `softmax`, `identity`.

**Embedding-version note:** the model records the `embedding_version` it was trained against (defaulting to `sonara.SIMILARITY_VERSION`). Because the embedding layout is versioned, a model trained on an older layout would classify on incomparable vectors — so `analyze_*` **fails fast with a `ValueError`** when the model's `embedding_version` does not match the running build. Re-scan your library and re-train when the embedding version bumps.

## Tonal Analysis

Standalone tonal functions for detailed harmonic analysis:

```python
import sonara
import numpy as np

y, sr = sonara.load("track.mp3", sr=22050)
S = sonara.stft(y, n_fft=2048, hop_length=512)
power = np.abs(S) ** 2
freqs = sonara.fft_frequencies(sr=float(sr), n_fft=2048)

# HPCP — Harmonic Pitch Class Profile (Gomez 2006)
# More robust than energy-based chroma: uses spectral peaks + harmonic weighting
hpcp = sonara.hpcp(power, freqs)  # shape (12, n_frames)

# Chord detection from HPCP + beats
tempo, beats = sonara.beat_track(y=y, sr=sr)
chords = sonara.chords_from_beats(hpcp, list(beats))  # ["Am", "F", "C", "G", ...]
desc = sonara.chord_descriptors(chords, len(y) / sr)
print(f"Predominant: {desc['predominant_chord']}, "
      f"Changes: {desc['chord_change_rate']:.2f}/s, "
      f"Unique: {desc['n_unique']}")

# Dissonance — Sethares (1998) Plomp-Levelt model
diss = sonara.dissonance(power, freqs)  # mean dissonance (0-1)

# Or from specific peaks
d = sonara.dissonance_from_peaks([440.0, 466.16], [1.0, 1.0])  # minor 2nd
```

## Display

```python
import sonara
import sonara.display as display
import matplotlib.pyplot as plt

y, sr = sonara.load("track.mp3", sr=22050)
mel = sonara.melspectrogram(y=y, sr=22050.0)
mel_db = sonara.power_to_db(mel)

fig, ax = plt.subplots()
display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, ax=ax)
plt.show()
```

## Performance

All arithmetic uses f32 precision (matching native decoder format), with a parallelized fused FFT pipeline where all features (spectral, tonal, contrast) are computed in a single pass per frame — eliminating redundant FFT computation and keeping data in L1 cache.

### Analysis pipeline benchmarks (Apple Silicon)

| Mode | 10s track | 3-min track | Features |
|------|-----------|-------------|----------|
| `compact` | ~1.2 ms | ~39 ms | 11 core features |
| `playlist` | ~4 ms | ~80 ms | 30+ features |
| `full` | ~50 ms | ~510 ms | All features incl. time signature |

### Feature benchmarks (vs Python/librosa)

| Feature | Speedup |
|---------|---------|
| Mel spectrogram | ~3x |
| MFCC | ~3x |
| Beat tracking | ~4x |
| Onset detection | ~3x |
| Cold start (first call) | ~20-30x |
| **Batch analysis (parallel)** | **~5x** |

### Key optimizations

- **Fused single-pass pipeline** — one FFT per frame simultaneously produces mel, chroma, centroid, RMS, bandwidth, rolloff, flatness, spectral contrast, HPCP, and dissonance. No power spectrum matrix stored.
- **Pre-computed DCT matrix** — MFCCs use cached DCT-II coefficients (matrix multiply instead of per-element cos())
- **Sparse filterbanks** — both mel and chroma filterbanks skip zero entries (~97% sparsity for mel)
- **Partial sort for contrast** — uses O(n) selection instead of O(n log n) sort for percentile computation
- **Top-N peak detection** — spectral peaks sorted by magnitude for HPCP/dissonance, shared between both algorithms
- **f32 precision** — halves memory bandwidth vs f64, matches Symphonia's native decode format
- **Parallel FFT frames** — rayon parallelism across frames (for signals > 32 frames)
- **Fast 2:1 decimation** — half-band FIR filter for 44100-to-22050 Hz instead of full sinc resampling
- **Thread-local caches** — FFT plans, mel/chroma filterbanks, DCT matrix reused across calls

## Accuracy benchmarking

The benches in `sonara/benches/` guard *speed*. A separate two-layer harness guards *correctness* of tempo (and key) detection — catching octave errors (detecting ~0.5x/2x the true tempo) and near-miss drift that speed benchmarks can't see.

**Layer 1 — synthetic ground truth (runs in CI, no audio files):** `sonara/tests/bpm_accuracy.rs` synthesizes deterministic signals with exactly known tempo — click trains, kick patterns, and syncopated kick+offbeat-hat patterns (the classic half/double-tempo trap) — across a BPM spread covering the problem zones (60, 63, 70, 79, 85, 92, 100, 118, 126, 128.3, 140, 150, 160, 174, 192). It runs the pipeline's tempo detector and computes accuracy @ ±0.5 BPM, accuracy @ ±2%, octave-error rate, and median/p95 absolute error, asserting **zero octave errors** and a tight median error over the guarded subset.

```bash
cargo test -p sonara --test bpm_accuracy -- --nocapture   # full metrics report
cargo test -p sonara --test bpm_accuracy -- --ignored      # run known-failing cases
```

Tunable thresholds and the `KNOWN_FAILING` list (cases current `main` gets wrong, mirrored by `#[ignore]`d per-case tests) are grouped at the top of the file. The suite only measures — it never modifies detection logic — so it can be re-run to validate detector improvements.

**Layer 2 — external labeled corpus:** `sonara/examples/accuracy_eval.rs` evaluates the detector against a real labeled dataset (e.g. tracks tagged with Mixed In Key ground truth). It reads a CSV, analyzes every file in parallel, and prints the same metrics plus a worst-offenders table (top N by error, with 0.5x/2x octave flags) and key accuracy.

```bash
cargo run --release --example accuracy_eval -- labels.csv --mode playlist
```

CSV format (header optional, auto-detected; `key_ref` optional):

```text
path,bpm_ref,key_ref
/music/track01.mp3,128,A minor
/music/track02.wav,174,F# major
/music/track03.flac,90
```

Run `cargo run --example accuracy_eval -- --help` for all options (`--sr`, `--mode`, `--top`).

## API Reference

sonara provides 100+ audio analysis functions:

**Core Audio:** `load`, `stream`, `stft`, `istft`, `resample`, `to_mono`, `tone`, `chirp`, `clicks`, `autocorrelate`, `lpc`, `zero_crossings`, `mu_compress`, `mu_expand`

**Spectral Features:** `melspectrogram`, `mfcc`, `chroma_stft`, `tonnetz`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_contrast`, `rms`, `zero_crossing_rate`, `poly_features`

**Tonal Analysis:** `hpcp`, `chords_from_beats`, `chords_from_frames`, `chord_descriptors`, `dissonance`, `dissonance_from_peaks`

**Rhythm:** `beat_track`, `onset_detect`, `onset_strength`, `onset_strength_bands`, `onset_strength_multi`, `tempo`, `tempo_curve`, `tempo_variability`, `tempogram`, `fourier_tempogram`, `metrogram`, `detect_time_signature`, `plp`

**Pitch:** `yin`, `pyin`, `piptrack`, `estimate_tuning`, `pitch_tuning`, `salience`, `interp_harmonics`, `f0_harmonics`

**Transforms:** `cqt`, `vqt`, `icqt`, `hybrid_cqt`, `pseudo_cqt`, `griffinlim`, `griffinlim_cqt`, `phase_vocoder`, `iirt`, `reassigned_spectrogram`, `pcen`, `perceptual_weighting`

**Source Separation:** `hpss`, `harmonic`, `percussive`, `nn_filter`, `decompose_nmf`

**Effects:** `time_stretch`, `pitch_shift`, `trim`, `split`, `split_with_constraints`, `remix`, `melody_separate`, `preemphasis`, `deemphasis`

**Sequence Analysis:** `dtw`, `rqa`, `viterbi`, `viterbi_discriminative`, `viterbi_binary`, `recurrence_matrix`, `cross_similarity`, `path_enhance`

**Perceptual:** `loudness_lufs`, `energy`, `danceability`, `detect_key`, `valence`, `acousticness`, `aggression_score`

**Conversions (50+):** `hz_to_mel`, `mel_to_hz`, `hz_to_midi`, `midi_to_hz`, `note_to_hz`, `note_to_midi`, `hz_to_note`, `hz_to_octs`, `hz_to_svara_h`, `hz_to_svara_c`, `hz_to_fjs`, `fft_frequencies`, `mel_frequencies`, `cqt_frequencies`, `frames_to_time`, `time_to_frames`, frequency weighting (A/B/C/D/Z), notation helpers, and more

**Filters & DSP:** `mel` filterbank, `chroma` filterbank, `lfilter`, `filtfilt`, `sosfiltfilt`, window functions (Hann, Hamming, Blackman, Kaiser, Tukey, Gaussian)

**Pipeline:** `analyze_file`, `analyze_signal`, `analyze_batch`, `analyze_aggression_file`, `analyze_aggression_signal`, `analyze_aggression_batch`, `augment_analysis`, `can_augment`, `augment_blocker`, `feature_dependencies`

## Architecture

sonara is a two-crate Rust workspace:

- **`sonara`** — Pure Rust core library (~18,000 LOC)
- **`sonara-python`** — PyO3 bindings (~1,200 LOC)

```text
sonara/src/
  analyze.rs      — Fused analysis pipeline (compact/playlist/full modes)
  aggression.rs   — Versioned 39-feature fused aggression model
  perceptual.rs   — LUFS, energy, danceability, key detection, valence, acousticness
  loudness_ext.rs — True peak (dBTP), ReplayGain, short-term curve, momentary max, EBU R128 LRA
  tonal.rs        — HPCP, chord detection, dissonance (Sethares 1998)
  beat.rs         — Beat tracking (Ellis 2007 DP algorithm), tempo candidates, BPM range
  beatgrid.rs     — Beat grid: first-beat offset, downbeats, grid stability
  rhythmic_regularity.rs — Metrical alignment + sub-bar periodicity of the accent profile
  onset.rs        — Onset detection (spectral flux + peak picking)
  decompose.rs    — HPSS, NMF
  effects.rs      — Time stretch, pitch shift, trim, split
  segment.rs      — Recurrence matrix, cross-similarity, path enhancement
  structure.rs    — Energy curve + novelty segmentation (Foote), intro/outro
  similarity.rs   — 48-dim similarity embedding + calibrated distance
  fingerprint.rs  — Gain-invariant acoustic fingerprint for duplicate detection
  vocal.rs        — Legacy v1 pitched-melodic heuristic (vocalness now contrast-based in analyze.rs)
  sequence.rs     — DTW, RQA, Viterbi, transition matrices
  core/
    audio.rs      — Audio I/O, resampling, fast 2:1 decimation
    spectrum.rs   — STFT, CQT/VQT, phase vocoder, Griffin-Lim
    fft.rs        — FFT with thread-local plan caching
    pitch.rs      — YIN / pYIN pitch estimation
    harmonic.rs   — Harmonic salience, interpolation
    convert.rs    — Hz/mel/MIDI/note/SVara/FJS conversions, frequency weighting
  feature/
    spectral.rs   — Mel, MFCC, chroma, centroid, bandwidth, rolloff, flatness, contrast
    rhythm.rs     — Tempogram, metrogram, time signature detection
  dsp/
    windows.rs    — Window functions (Hann, Hamming, Blackman, Kaiser, Tukey, Gaussian)
    iir.rs        — IIR filters (lfilter, filtfilt, sosfiltfilt)
    extrema.rs    — Local maxima/minima detection
  filters.rs      — Mel/chroma filterbanks
```

## License

[MIT](LICENSE)
