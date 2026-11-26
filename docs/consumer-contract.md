# Consumer contract — sonara's core Rust API

**Owner:** sonara. This is the single durable copy.
**Status:** current, `ANALYSIS_SCHEMA_VERSION = 8` since 2026-11-26; last verified
against sonara 0.3.7 (MeteorBurn fork, `dev`) / schema 7 on 2026-10-03.
**Consumer of record:** sonagram (since 2026-07-18). Originally raised by
kglite on 2026-07-15.

sonara produces the API this contract is about, so sonara owns the obligation
and keeps the durable copy. Downstreams hold a pointer, not a second copy.

## Why this file is tracked

Between 2026-07-15 and 2026-07-29 this contract existed as two "durable copies"
in two *gitignored* `dev-docs/designs/` folders — one here, one in kglite —
each of which described itself as the durable one. They diverged, and both
still listed as **unmet** three API requirements sonara had already shipped in
0.2.1 on 2026-07-15. The closure was recorded only in a third repo's notes
(`sonagram:dev-docs/designs/upstream-contracts.md`). One tracked copy, in the
repo that owes the obligation, is the fix.

## Ownership split

Confirmed by kglite 2026-07-18 (kglite inbox
`2026-07-16-from-sonagram-ownership-boundary.md`, status footer), after
sonagram took the mapping seat kglite's original 2026-07-15 note had assumed
for itself.

| Concern | Owner |
|---|---|
| Analysis → typed `TrackAnalysis` | **sonara** |
| Music-graph schema, source hashing, mapping, library scan | **sonagram** |
| Graph storage, Cypher, MCP graph exposure, embedding *engine* | **kglite** |

sonagram may persist sonara's 48-dimensional similarity vector in kglite's
`EmbeddingStore` (as `model_id = "sonara-similarity-v1"`, re-keyed on sonara
version bumps). The stored vector is **unweighted** — the per-dimension weights
live inside `distance()` and are applied at query time, so a weighting change
(e.g. a new similarity profile) never alters a persisted vector. Ownership of
that music-domain vector and its provenance stays with sonagram; kglite owns
the generic storage and query machinery, not music-specific mapping or
lifecycle.

## The consumed surface

Typed Rust values directly from the `sonara` crate — `analyze_file` /
`analyze_signal` / `AnalysisConfig` → `TrackAnalysis`. **Not** Python dicts,
not JSON. `SonaraError` is a dependency-neutral structured error. sonara stays
graph-agnostic; it is never asked to know about graphs.

## Compatibility requests — all three CLOSED in 0.2.1 (2026-07-15)

The original asks and what shipped. Kept for lineage; do not re-raise them.

| Ask (2026-07-15) | Shipped |
|---|---|
| **Self-describing time.** `beats` / `onset_frames` / downbeats were frame indices; effective sample rate and `hop_length` were internal to `analyze.rs`. | `TrackAnalysis.provenance: AnalysisProvenance` (always present) carries effective post-resample `sample_rate` and `hop_length`, plus `frame_to_sec` / `beats_sec` / `onsets_sec` / `downbeats_sec` helpers. |
| **Typed event records with spans.** `chord_sequence: Option<Vec<String>>` had no temporal alignment; segments were tuples. | `chord_events: Option<Vec<ChordEvent>>` (`{ label, start_sec, end_sec }`, merged runs, contiguous, covering the track); `segments` is a named `SegmentEvent { start_sec, end_sec, energy }`. |
| **Persistable provenance** to detect stale analysis. | `AnalysisProvenance { schema_version, sample_rate, hop_length, mode, requested_features (sorted), vocalness_model_id, … }`, pinned to `ANALYSIS_SCHEMA_VERSION` (now 8), bumped whenever field meaning or units change. |

## Standing boundaries — what sonara undertakes to hold

Verified against sonara 0.3.5 on 2026-07-29 and re-verified against 0.3.7 on
2026-10-03 (`sonara/Cargo.toml`, `cargo tree -p sonara`):

- **The core crate stays PyO3-free.** PyO3 and NumPy are confined to
  `sonara-python`. A consumer's core dependency path must never acquire them
  transitively. *(Verified: no `pyo3` / `numpy` in the `sonara` tree.)*
- **`default = []`.** Platform acceleration (`accelerate` → `ndarray/blas` +
  `blas-src/accelerate`) is opt-in. *(Verified.)*
- **No ML/ONNX runtime in the default feature set.** The `aggression` feature
  is the only path that pulls a learning runtime (`ferricml`, `sha2`), and it
  is off by default. *(Verified.)*
  - Clarification added 2026-07-29: the **bundled/bring-your-own vocalness
    classifier** (`sonara::vocal_model`) is *in* the default build and does not
    breach this. It is a pure-Rust feed-forward evaluator over the existing
    hand-crafted embedding, using the same JSON MLP format as `genre`, with no
    runtime dependency. It extends the versioning discipline rather than
    weakening it: the model `id` is required and is carried into
    `AnalysisProvenance::vocalness_model_id` so downstream caches can
    invalidate scores produced by a different model or none.
- **Versioned vector layouts.** Fingerprint and similarity vectors carry
  explicit version constants and must remain distinguishable on persisted
  records. Never reinterpret an existing layout without a version bump; a
  future learned-vector layout gets a new explicit version/model identity
  rather than silently reusing the `embedding` field.
- **`analyze_signal` stays the low-level, dependency-light entry point.**
  Symphonia currently enables all codecs and formats unconditionally. Do not
  churn this speculatively — but if a signal-only or reduced-codec feature
  split is ever introduced, `analyze_signal` must stay buildable without the
  full decoder bundle.
- **No serde/JSON bridge is required** solely to serve a downstream. Direct
  typed access is preferable and avoids another dependency contract.

## Versioning

Downstreams pin a compatible pre-1.0 sonara release and key stored records on
`provenance.schema_version` (plus the per-subsystem `embedding_version` /
fingerprint version where those are consumed). sonagram's current floor is
`sonara >= 0.2.2`. Heuristic-semantics changes bump `ANALYSIS_SCHEMA_VERSION`.

### Schema 8: chords and the bar length of the beat grid (2026-11-26)

Every field keeps its meaning and units; these stored schema-7 values differ
from a fresh run:

- `chord_sequence`, `chord_events`, `chord_change_rate`, `predominant_chord`.
  The pitch-class profile behind the chords weights a spectral peak by its
  distance to the nearest of the 12 notes, the same way for all of them: a
  peak up to half a semitone below C counts toward C.
- `loudness_lufs` and `replaygain_db`. The loudness does not depend on the
  length of the signal; a stored schema-7 value is too low, the more so the
  longer the signal: about 0.05 dB at 5 minutes and 0.9 dB at one hour on a
  steady test signal.
- `spectral_flatness_mean`, `spectral_flatness_{std,iqr,p10,p90}`,
  `spectral_contrast_mean` and `spectral_contrast_std`, on a track with
  silence, a fade or a quiet tail. They cover the sounding frames only: a
  frame counts when its RMS is above zero and within 60 dB of the track's
  loudest frame (the non-silent rule of `effects.split`, librosa's default
  `top_db`); each is 0.0 when no frame counts. A stored schema-7 value counts
  every frame, where a silent frame enters as flatness 1.0 and contrast 0.
  `acousticness` and the built-in `vocalness` / `instrumentalness` are
  computed from the two means and differ with them. The aggression model
  keeps its own contrast input over every frame, so `aggression_score` is
  unchanged.
- `embedding`, dimensions 25-30 (contrast), 34 (flatness), 39 (loudness) and
  42 (the chord change rate) only,
  and what a supplied genre or vocalness model derives from the embedding.
  `SIMILARITY_VERSION` stays 2: the layout and the meaning of every dimension
  are unchanged.
- `downbeats` and `rhythmic_regularity*`, in a record whose request also
  computed `time_signature`. Both always use a 4-beat bar.

`augment_analysis` rejects a schema-7 record with `schema version mismatch`.

### Schema 7: beats and BPM from the beat grid (2026-10-02)

`bpm` and `bpm_raw` keep their meaning and units: the tempo in BPM, folded into
`bpm_min..bpm_max` for `bpm` and before folding for `bpm_raw`. They are now the
period of the tracked beats instead of the autocorrelation peak, so they are
more precise and stored schema-6 values differ slightly. The autocorrelation
estimate remains the fallback when fewer than 17 beats are tracked or the beat
period deviates from it by more than 5%. The period is measured on sub-frame
beat positions (the vertex of the tracker's local score at each beat). A
steady track, whose stronger beats stay on one constant grid over the whole
track and also fit the nearest integer tempo, reports that integer (`bpm_raw`
keeps the same octave relation); a track a few thousandths of a BPM off an
integer reports the fraction.
The tempo level is re-checked: the selected autocorrelation level and up to
five other distinct levels with a beat period of at least 0.28 s are tracked
again at their own tempo, and another level replaces the selected one only when
its beats agree with the onsets clearly better (beat-pair consensus plus the
tracker's local score). When the chosen level's beats keep a tempo more than
0.15 BPM off an integer and a level at 4/3, 3/2 or 5/4 of it (or the inverse)
with nearly the same evidence keeps an integer one, that level is reported
instead. `bpm_min`/`bpm_max` play no part in the choice and only fold the
reported value, so the tempo ignoring octave does not depend on the range. On a
switched track, `bpm`, `bpm_raw` and `beats` follow the new level;
`bpm_candidates` do not change.
The onset envelope is padded one frame less than librosa's
`lag + n_fft / (2 * hop_length)`, so `onset_frames`, `onset_strength_bands`,
`beats` and `downbeats` sit one frame (about 23 ms) earlier than in schema 6,
on the audible onset; every analysis path shares that padding. The public
`onset_strength_method` keeps librosa's padding. `downbeats` (beatgrid) now
follow the kick: the bar phase is the one whose beats carry the most low-band
(below 200 Hz) accent relative to the 200–800 Hz band, rather than the loudest
broadband accent, which is often the snare on beats two and four.
`beats` keep their meaning, but they
are now tracked on the onset envelope of the mel bins below 3.2 kHz, so they
follow the kick and snare rather than off-beat hi-hats; stored schema-6 beats,
and the fields derived from them, can differ. The tempo estimate and
`bpm_candidates` still come from the broadband envelope and are unchanged. The
`bpm_confidence` formula is unchanged; its value follows `bpm` and the beat
count.

### Per-feature freshness

Additive fields and APIs ship **without** an `ANALYSIS_SCHEMA_VERSION` bump —
the whole-record version moves only when existing field meaning or units
change. A consumer deciding whether a stored record needs refreshing should
therefore key **per feature**, not on the whole-record version: field presence
on the record, the per-feature model ids in provenance (`vocalness_model_id`,
`genre_model_id`, `aggression_model_id`), and the declared dependency map
`feature_dependencies()` —
which names each feature's dependency class and the record fields a decode-free
recompute reads (`augment_analysis` / `can_augment` consume the same map, so a
missing feature can often be filled in without re-decoding the audio).
Presence fails for one group: `tags` is an absent key, not an empty map, when the
container carries none (WAV always), so key it on
`provenance.requested_features` instead (a `full`-mode file record has no
feature list: there `provenance.mode` equal to `full` says tags were read). The Python module exports
`ANALYSIS_SCHEMA_VERSION` and `FINGERPRINT_VERSION` for these comparisons, and
`fingerprint_match` refuses a record whose `fingerprint_version` differs from
this build's.

An explicit `features=[...]` request returns only the named groups plus the core
signal scalars every run reports (2026-10-03, no schema bump: values are
unchanged, only presence differs). Before, any feature that runs the extended
pass (`energy`, `key`, `vocalness`, `chords`, `structure`, ...) also returned all
six spectral summaries: `mfcc_mean`, `chroma_mean` and
`spectral_{contrast,bandwidth,rolloff,flatness}_mean`. They are the decode-free
evidence for recomputing `key`, `valence`, `energy` and similar features, so a
consumer that relied on them from a narrow request must now name them (`mfcc`,
`chroma`, `contrast`, `bandwidth`, `rolloff`, `flatness`). Mode runs and requests
that include `embedding` are unaffected. One deliberate exception: `valence` also
returns the key it is derived from (`key`, `key_confidence`, `key_camelot`).
Since 2026-10-03 `augment_analysis` writes those key fields when it fills
`valence` too, as a fresh run does; before, it set only `valence`.

Frame statistics (2026-11-17, MeteorBurn fork, no schema bump: additive keys,
existing values unchanged). Each base feature now also returns the spread,
frame-to-frame change and quantiles of its per-frame values, over the same frames
as its `_mean`: `rms_{std,delta_mean,iqr,p10,p90}`, `zero_crossing_rate_std`,
`spectral_centroid_{std,delta_mean,iqr}`, `spectral_bandwidth_{std,delta_mean}`,
`spectral_rolloff_{std,p10,p90}`, `spectral_flatness_{std,iqr,p10,p90}`,
`spectral_contrast_std` (7), `mfcc_{std,delta_mean}` (13) and
`chroma_{std,delta_mean}` (12). They travel with their feature name (`rms`,
`mfcc`, ...): present exactly when it is, and `augment_analysis` refills them
with it. They need the audio (class `C`/`A`), are not embedding evidence, and
leave `embedding` and every model input unchanged.

Spectral shape (2026-11-17, MeteorBurn fork, no schema bump: additive keys,
existing values unchanged). The `spectral_shape` group (playlist/full, or
`features=["spectral_shape"]`) adds `spectral_skewness_mean`,
`spectral_kurtosis_excess_mean`, `spectral_slope_db_per_octave_mean` (dB/octave)
and `spectral_decrease_mean`. Each averages only
the frames where it is defined (all-zero frames skipped; 0.0 when none is). The four
keys travel with `spectral_shape`: present exactly when it is, refilled by
`augment_analysis` with it. They need the audio (class `C`) and are not embedding
evidence.

Spectral texture (2026-11-18, MeteorBurn fork, no schema bump: additive keys,
existing values unchanged). The `spectral_texture` group (playlist/full, or
`features=["spectral_texture"]`) adds `spectral_flux_mean`, `spectral_flux_std`,
`spectral_crest_mean`, `spectral_entropy_mean` and `spectral_irregularity_mean`,
all on the main pass's magnitude spectrum. Flux is the half-wave rectified L2
distance between consecutive L1-normalized magnitude spectra (Essentia
`Flux(halfRectify=true, norm="L2")` on a sum-normalized spectrum): range 0..1,
loudness-invariant; `_std` is the population std of the same values. Crest is
max/mean magnitude (≥ 1). Entropy is the Shannon entropy of the normalized
magnitudes divided by ln(n_bins), as MIRtoolbox `mirentropy` (0..1).
Irregularity is Jensen's Σ(a_{i+1} − a_i)² / Σa_i² over the fused pass's
spectral peaks (40–5000 Hz, top 50 by magnitude, sorted by frequency), so it
depends on that peak picking. Each value averages only the frames where it is
defined: silent frames are skipped (for flux, any pair with a silent frame),
irregularity needs at least two peaks; 0.0 when none is. The five keys travel
with `spectral_texture`: present exactly when it is, refilled by
`augment_analysis` with it. They need the audio (class `C`) and are not embedding
evidence.

Mel bands (2026-11-21, MeteorBurn fork, no schema bump: additive keys,
existing values unchanged). The `mel_bands` group (playlist/full, or
`features=["mel_bands"]`) adds `mel_bands_mean`, `mel_bands_std` (128 values
each, dB, low band first) and `mel_bands_floor_db` (dB). They summarize the main
pass's log-mel spectrogram, the very matrix the MFCCs and the onset envelopes
are computed from: 128 Slaney mel bands over 0..sr/2 (area-normalized triangles
applied to the power spectrum |X|², n_fft 2048, hop 512, Hann), converted with
10·log10 with a 1e-10 power floor, then floored at the track's loudest cell
minus 80 dB (librosa `power_to_db(ref=1.0, amin=1e-10, top_db=80)`).
`mel_bands_mean` is the per-band mean of the dB values over all main-pass frames
(silence included): the mean of dB, not the dB of the mean power (on music the
two differ by about 9 dB median). Averaging dB reads bands that span fewer FFT
bins lower: on white noise the narrow low bands sit 1–1.3 dB below the top
ones, so absolute levels of different bands compare only to about 1 dB.
`mfcc_mean` equals the first 13 coefficients
of the orthonormal DCT-II of this vector. `mel_bands_std` is the per-band
population std (ddof 0) over the same frames. `mel_bands_floor_db` is the lowest
value a cell can take: the loudest cell − 80 dB, or −100 dB (the power floor)
when that is higher, as on silence. A band whose mean is within about 1 dB of
it holds no content above the floor; exact equality is rare because edge frames
lift the mean slightly. A gain change shifts every mean and the floor by the
same dB and leaves the std unchanged. Band centre frequencies are
`sonara.mel_frequencies(n_mels=130, fmin=0, fmax=sr/2)[1:129]`. The three keys
travel with `mel_bands`: present exactly when it is, refilled by
`augment_analysis` with it. They need the audio (class `C`) and are not embedding
evidence.

Tuning (2026-11-21, MeteorBurn fork, no schema bump: additive keys, existing
values unchanged). The `tuning` group (playlist/full, or `features=["tuning"]`)
adds `tuning_offset_cents` (cents, [−50, 50)) and `tuning_confidence` (0..1):
the deviation of the recording's tuning from A4 = 440 Hz, by the weighted
circular mean of Dressler & Streich (ISMIR 2007). In every main-pass frame
(n_fft 2048, hop 512, Hann) the 10 strongest spectral peaks with bin centre in
150–5000 Hz are taken (peaks at or below −80 dB of the frame's strongest bin
are skipped); each peak frequency is refined by a parabola on the log
magnitude; its deviation from the nearest equal-tempered semitone (relative to
440 Hz) becomes an angle; the angles are averaged over all frames with weight
sqrt(magnitude). The angle of the mean vector gives `tuning_offset_cents`, its
length gives `tuning_confidence` (0 = the peaks agree on no grid, 1 = every peak
on the same grid): about 0.01–0.03 on noise of 5 s or more, 0.05–0.6 on mixed
music, 0.99 on clean tones. The group **abstains**: whenever the feature ran,
the result carries `tuning_confidence`, while `tuning_offset_cents` is `None`
when the confidence is below 0.08 (the Python result includes both keys). When
only n < 352 frames (about 8 s) carry a peak the bar is 1.5/√n instead: noise
reads about 0.4/√n, so on a short or mostly silent input it would otherwise
clear 0.08 by chance (20 % of 1 s noise clips). A stored `None` therefore
means "measured, could not tell", not "not yet computed" — key freshness on the
presence of the *confidence*, not of the offset. The value wraps: a track near
±50 cents lies between two semitones, so compare two tracks' offsets with a
circular difference. A known pitch shift of real tracks is recovered within
about 2.5 cents (median 0.2–0.4, at the default 22050 Hz analysis rate); clean
chords within 0.5 cents at 22050, 44100 and 48000 Hz; loud vibrato vocals lower
the confidence.
Chroma, HPCP, key and chord detection still assume 440 Hz and are not corrected
by this value, so a large offset means those are less reliable for that track.
The two keys travel with `tuning`: present exactly when it is, refilled by
`augment_analysis` with it. They need the audio (class `C`) and are not
embedding evidence.

HPSS percussive ratio (2026-11-22, MeteorBurn fork, no schema bump: additive
key, existing values unchanged). The opt-in `hpss` feature (`full` mode or
`features=["hpss"]`; `compact` and `playlist` do not compute it) adds
`hpss_percussive_ratio` (dimensionless, 0..1):
the share of percussive energy in the track by median-filter
harmonic/percussive separation. `0` means all energy is in sustained,
horizontal (tonal) structure of the spectrogram; `1` means all energy is in
broadband, vertical (transient) structure. Unlike the file-only `ebur128`
family it is computed in the main pass on the mono analysis signal, so
`analyze_file`, `analyze_batch` and `analyze_signal` all emit it. On the main
pass's magnitude spectrogram (n_fft 2048, hop 512, Hann, at the analysis sample
rate; all frames, silence included) a median along time gives the harmonic
estimate and a median along frequency gives the percussive estimate; the kernel
is 0.72 s by 334 Hz, i.e. 31 frames by 31 bins at 22050 Hz, scaled with the
sample rate (rounded up to an odd count of frames and of bins, the conversion
of the FMP reference code `libfmp`) and equal to the
default kernel of `librosa.decompose.hpss` at 22050 Hz, with mirrored edges. A
soft mask with exponent 2
splits each cell's magnitude into a harmonic part `H` and a percussive part `P`
(`H + P = |X|`), and the value is ΣP² / (ΣH² + ΣP²) over the whole track. This
equals `librosa.decompose.hpss` with this kernel, `power=2.0` and `margin=1.0`,
followed by that energy ratio. Reference readings: about 0 on a steady tone,
about 0.5 on white noise, about 1 on a click train; on the 150 songs of
MUSDB18-HQ, 0.76 on the drum stems (median; 0.38–0.94 for the middle 80 %),
0.14 on the same songs without drums and 0.23 on the full mixes. Known limits:
sustained low drums (kick and tom tails below 150 Hz) count as harmonic and
dense distorted guitars partly as percussive, so the value follows the energy
share of the drum stem closely in pop, rap and electronic music (rank
correlation 0.8–0.95) and hardly at all in heavy metal (about 0.2); each genre
is 8–12 songs of MUSDB18-HQ. The kernel
spans about the same time and frequency at every analysis sample rate
(`provenance.sample_rate`); that keeps the order of tracks (rank correlation
0.97 between 22050 and 44100 Hz) but not the values (about 0.07 higher at
44100 Hz), so compare values only between analyses at one sample rate. The
reference readings are for 22050 Hz. It is an energy ratio: loud passages dominate it, and the overall
level does not change it. The feature does **not abstain** and has no
confidence field: digital silence reads `0.0`, and the key is present exactly
when the feature ran (`None` in Rust when it was not requested) — key freshness
on the presence of `hpss_percussive_ratio` itself. It is not the Rust
`decompose::hpss` / `effects::hpss` separation, which handles the spectrogram
edges differently, and it returns no separated signals. The key travels with
`hpss`: present exactly when it is, refilled by `augment_analysis` with it. It
needs the audio (class `C`), is not embedding evidence and is not an input to
any model.

Rhythmic level ratios (2026-11-25, MeteorBurn fork, no schema bump: additive
keys, existing values unchanged). The `rhythm_levels` feature (part of the
`playlist` and `full` modes, or `features=["rhythm_levels"]`; `compact` does
not compute it) adds three
dimensionless keys, emitted by `analyze_file` and `analyze_signal` alike:
`rhythm_level_double_ratio` (≥ 0, no upper bound), the strength of the
eighth-note (double-tempo) level against the beat — `0` means nothing regular
lies between the beats, about `1` means the eighths are as strong as the
beats; `rhythm_level_one_and_half_ratio` (≥ 0, no upper bound), the strength of
the one-and-a-half level (triplets, shuffle, three against two) against the
beat, `0` on straight tracks; and `rhythm_level_confidence` (0..1), the mean
normalised autocorrelation of the onset curve at the beat period: how well the
onset curve repeats at the period set by `bpm`. Definition: `T` is the beat
period at the result's final `bpm`, one tempo for the whole track (a
simplification: Peeters uses the local tempo of each frame). The onset curve
is cut into frames of 8 s every 0.5 s (`round(8·sr/hop)` and
`round(0.5·sr/hop)` onset frames; Peeters 2011, sec. II); the mean of each
frame is removed (Tzanetakis & Cook 2002); the frame's autocorrelation is the
biased estimate with a rectangular window, divided by its value at lag zero
(Peeters 2011, sec. II-C); a value at a fractional lag is linearly
interpolated (as in `librosa.feature.tempogram_ratio`; Peeters does not
describe this step) and negative autocorrelation is set to zero (Tolonen &
Karjalainen 2000). The values at the lags `T/2`, `2T/3` and `T` are averaged
over the frames, and each ratio is a ratio of averages: the one at `T/2`, or
at `2T/3`, over the one at `T` (a level against the main one is the `RA`
feature of Tzanetakis & Cook 2002; the lags are points of Peeters's grid and
the factors 2 and 3/2 those of `librosa.feature.tempogram_ratio`). Lags, not
frequencies, are used because both levels are faster than the beat (Peeters
2011: the octave errors of the DFT and of the autocorrelation lie in opposite
domains; Klapuri 2006). No value was fitted to tracks or verification sets. The
feature **abstains**: whenever it ran, the result carries all three keys;
`rhythm_level_confidence` is always a number (`0` when nothing can be
measured), while each ratio is a number or `None` — `None` when there is no
tempo, the track is shorter than one 8 s frame, or the autocorrelation at the
beat period is not positive. A stored `None` therefore means "measured, could
not tell", not "not yet computed" — key freshness on the presence of the
*confidence*, not of the ratios. Reference readings: on synthetic drum
patterns at five tempos the double ratio is `0` on a pulse without eighths and
0.84–1.10 with eighths; the one-and-a-half ratio is 0.32–0.46 on a shuffle,
0.64–0.86 on triplets, 1.42–1.87 on three against two and `0` on even eighths
and sixteenths. On 346 drum recordings of Groove MIDI, against the same
formula applied to the MIDI hits, the rank correlation is +0.817 (95 %
interval +0.757 to +0.864) for the double ratio and +0.732 (+0.651 to +0.796)
for the one-and-a-half ratio (+0.832 among the 89 recordings where both values
are above zero); the shuffle and swing styles (17 recordings) against the
straight ones (301) give AUC 0.683 (0.545 to 0.813). On the 664 previews of
GiantSteps Tempo the double ratio has a median of 0.90 (quartiles 0.67 and
1.08) and the one-and-a-half ratio is exactly zero on 45 % of the tracks and
below 0.01 on 69 %; no link was
found between the double ratio and the share of listeners at the faster tempo
(67 tracks, −0.120, interval −0.357 to +0.139). Dependence on `bpm`: the
values are counted from `bpm`, so they inherit its errors. On the synthetic
patterns a 1 % tempo error changes them by up to 21–22 % and a 2 % error by up
to 44–49 %; on the Groove MIDI recordings where Sonara's tempo matched the
metronome the values differed from those at the metronome tempo by 0.0–0.1 %
(median). When `bpm` sits on another metrical level (32 % of the Groove MIDI
drum recordings), the numbers describe the levels relative to that `bpm`, not
to the felt tempo. A tempo range (`bpm_min`/`bpm_max`) folds `bpm` by octaves
and the ratios are computed from the folded value: on the GiantSteps previews
the range 79–192 moved the tempo of 27 tracks and 70–180 of 24, and on those
the double ratio changed by +0.63 and +0.79 (median) and the confidence by
−0.17 and −0.13; tracks whose tempo the range leaves alone are unchanged.
Values are therefore comparable only between analyses made with one range. Dependence on the analysis sample rate
(`provenance.sample_rate`): at a rate other than 22050 Hz the onset curve is
different, so compare values only between analyses at one sample rate; the
reference readings are for 22050 Hz. One synthetic beat with eighths at 120 BPM
read a double ratio of 1.21 at 22050 Hz, 0.76 at 44100 Hz and 0.81 at 48000 Hz
(confidence 0.61, 0.89 and 0.84). The confidence also depends on where the
beat period falls between onset frames: the same clean pulse read 0.71 to 0.85
at five tempos from 90 to 174 BPM. Other known limits: the one-and-a-half
ratio is zero on most straight tracks, and its check on styles passed with a
small margin on 17 recordings; at 174 BPM with even sixteenths the double
ratio falls to 0.32 on the synthetic pattern (the lag `T/2` is about 7 onset
frames); both ratios divide by the level at the beat period, so at a
confidence near zero they are a ratio of two small numbers (20 s of white
noise read a confidence of 0.05 and a double ratio of 1.28; no confidence
threshold was validated). The half-tempo level (half-time) is **not** measured by this
feature: on the full onset curve an ordinary beat (kick on every quarter,
snare on 2 and 4) reads the same as half-time, and on GiantSteps the link to
the share of listeners at the slower tempo was +0.097 (interval −0.025 to
+0.217, which includes zero). A reading of the onset curve below 200 Hz did
better there (+0.472, interval +0.373 to +0.557, 313 tracks) but has no second
set labelled by listeners to confirm it, so it is not shipped.

File-level loudness and dynamics (2026-11-18, MeteorBurn fork, no schema bump:
additive keys, existing values unchanged). Three opt-in features, computed by
`full` mode or an explicit request and never by `compact` or `playlist`,
measure the file's own channels at its own sample rate instead of the
mono mix at the analysis rate, so only `analyze_file` / `analyze_batch` emit
them; `analyze_signal` never does, even when they are requested or the mode is
`full`.
`features=["ebur128"]` adds `ebur128_integrated_lufs`,
`ebur128_momentary_max_lufs`, `ebur128_short_term_max_lufs` (LUFS),
`ebur128_range_lu` (LU) and `ebur128_true_peak_dbtp` (dBTP): ITU-R BS.1770-5 /
EBU Tech 3341 and 3342, with K-weighting for the file's rate, channel weights
and both gates (the true peak has the EBU Tech 3341 accuracy, +0.2 / −0.4 dB).
`features=["native_dynamics"]` adds `sample_peak_dbfs`, `crest_factor_db`
(track peak over the RMS of all samples of all channels) and
`crest_factor_frame_mean_db` (mean per-frame crest
factor over frames with RMS above −70 dBFS). `features=["amplitude_envelope"]`
adds `amplitude_envelope` (max |x| over the channels per frame, linear) with
`amplitude_envelope_hop_sec` (512/22050 s) and `amplitude_envelope_frame_sec`
(2048/22050 s): the grid is fixed in seconds, so it is the same at every file
rate, and frame `k` is centred at `k · hop`. A key is absent (`None` in Rust)
when its value is undefined — digital silence, a file shorter than the
400 ms / 3 s window, or (loudness only; the true peak is still reported) a file
with more than six channels, whose layout the channel count does not determine
(up to six are read in WAV order L R C LFE Ls Rs, LFE excluded) — so with the
feature listed in `provenance.requested_features` (or `provenance.mode` equal
to `full` on a record without a feature list) and `native_sample_rate`
present, a missing key means "measured, undefined", not "pending". Any of the
three features (and `stereo_field` below) sets `native_sample_rate` (Hz, int) when it measured a file, so
a record without it (an `analyze_signal` result) was never measured. The
envelope is always present for a measured file. The always-on `loudness_lufs` and the `loudness` group
(`true_peak_db`, ...) are a different measurement (mono mix, analysis rate,
ungated) and keep their values. The keys travel with their feature name and
`augment_analysis` refills them from the audio path (class `A`); they are not
embedding evidence.

Stereo field (2026-11-25, MeteorBurn fork, no schema bump: additive keys,
existing values unchanged). The opt-in `stereo_field` feature
(`full` mode or `features=["stereo_field"]`; `compact` and `playlist` do not
compute it) belongs to the
file-level family above: it is measured on the file's left and right channels
at the file's own sample rate, so only `analyze_file` / `analyze_batch` emit
it; `analyze_signal` never does, even when it is requested or the mode is
`full`. It adds four
dimensionless keys. `stereo_side_ratio` (0..1) is the share of the Side signal
in the energy of the whole file: with Mid `(L + R)/2` and Side `(L − R)/2` it
is `E_S / (E_M + E_S)` = `½·(1 − 2·cov(L,R) / (var L + var R))`. The variances
and the covariance have the means removed, so a constant offset in a channel
does not change the value. `0` means identical channels, `0.5` uncorrelated
channels of equal power or sound in one channel only, `1` antiphase
(`R = −L`); plain panning `R = a·L` reads `½·(1 − 2a/(1 + a²))`.
`stereo_side_ratio_low` (0..1) is the same after a 4th-order Butterworth
low-pass at 150 Hz applied to both channels. `stereo_correlation_mean`
(−1..+1) is the mean over frames of the Pearson correlation of L and R; the
frames are `2048/22050` s long, do not overlap, and an incomplete last frame
is dropped. `stereo_correlation_p05` (−1..+1) is the 5th percentile of the
same per-frame values, with the linear interpolation of the `_p10` fields.
The filter is set in Hz and the frame in seconds, so values are comparable
between files of different sample rates. Edge cases: a mono file is measured
as `L = R` (shares `0`, correlations `+1`: this is how mono plays on a stereo
system); a file with more than two channels gets none of the four keys,
because its channel layout is unknown; a share is undefined when
`(var L + var R)/2` in its band is below −70 dB; a frame in which the
variance of both channels is below −70 dB is left out of the correlation
statistics (a constant offset is not signal; each channel's mean is removed
before the low-pass as well); a frame
in which one channel has zero variance counts as `0`; the correlation is
undefined when no frame is left. Abstention follows the family: a key is
absent (`None` in Rust) when its value is undefined, and `stereo_field` sets
`native_sample_rate` like the three features above, so with the feature
listed in `provenance.requested_features` (or `provenance.mode` equal to
`full` on a record without a feature list) and `native_sample_rate` present, a
missing key means "measured, undefined", not "pending". Sources of the
parameters. The width measure is Tardieu, Deruty, Charbuillet and Peeters,
"Production Effect: Audio Features for Recording Techniques Description and
Decade Prediction" (DAFx 2011), eq. (3), `σ_LR = σ(L − R) / σ(L + R)`; the
field is `σ_LR² / (1 + σ_LR²)`, the same quantity in a bounded 0..1 form. A
related measure is the side/mid ratio of De Man et al. (ISMIR 2014). The
correlation follows Ziemer 2023 (arXiv:2302.01090): `np.corrcoef` per frame
of 2048 samples at 22050 Hz without overlap, averaged over the frames. Two
differences: the frame is set in seconds, and the whole file is read at its
own sample rate (the author takes 500 frames from the middle after
downsampling to 22050 Hz). The zero for a channel without variance is the
convention of hardware meters (RTW 1260C: with one input signal only the
meter shows 0) and of Essentia; the −70 dB frame threshold is the one of
Essentia's `FalseStereoDetector`, applied here to the variance instead of the
power. No standard defines a correlation meter or
a low-band edge. For the same purpose (mono compatibility of the low end)
100 Hz and 300 Hz (the Furnace record-pressing guide), 180 Hz (Ableton Hybrid
Reverb) and a 10–200 Hz scale (Maselec MTC-1X) are published; 150 Hz lies
inside that range and is a choice of this project, as are the Butterworth
filter of 4th order and the 5th percentile. No value was fitted to tracks or
verification sets. Known limits: the correlation is `+1` for identical
channels and for plain panning alike, so the width in that case shows only in
`stereo_side_ratio`; `ffmpeg aphasemeter` computes a different quantity (the
mean of `2LR / (L² + R²)` per sample) and matches the correlation fields only
at `+1` and `−1`; lossy coding (joint stereo) can change the side signal: on
10 MUSDB18-HQ mixtures MP3 at 128 kbit/s moved the four fields by at most
0.006 and at 320 kbit/s by at most 0.001. Readings on the 150 MUSDB18-HQ
mixtures (median, 10th–90th percentile): `stereo_side_ratio` 0.09
(0.02–0.19), `stereo_side_ratio_low` 0.04 (0.004–0.11),
`stereo_correlation_mean` 0.79 (0.55–0.93), `stereo_correlation_p05` 0.49
(0.02–0.82, below zero in 15 mixtures). The published reference agrees: De
Man et al. give a side/mid ratio of 0.101 ± 0.049 for finished mixes (averaged
over frames there), a share of about 0.09. The keys
travel with `stereo_field` and `augment_analysis` refills them from the audio
path (class `A`); they are not embedding evidence.

`mood_aggressive` no longer depends on the rest of the request (2026-10-03, no
schema bump): a `mood` request computes `dissonance` for its `0.20·dissonance`
term even when `dissonance` was not requested, and emits it only when it was.
Before, a request without `dissonance` scored that term as 0, so stored
`mood_aggressive` values from such requests are lower than a fresh run by
`0.20·dissonance` (about 0.002–0.005 on real music). Requests that included
`dissonance`, and `augment_analysis`, already used the full formula.

`rhythmic_regularity` is additive and therefore ships without a schema bump,
but it carries a shape consumers must handle explicitly: it **abstains**. When
a track has no measurable rhythmic evidence (silence, ambient, drumless), the
result carries `rhythmic_regularity_confidence` while `rhythmic_regularity`,
`rhythmic_regularity_label` and `rhythmic_regularity_candidates` are all
`None` together (the Python result includes all four keys when the feature is
computed).
A stored `None` therefore means "measured, could not
tell", not "not yet computed" — key freshness on the presence of the
*confidence*, not of the score. The confidence is evidence quality, never a
class probability, and the two candidate scores are complementary DSP
measurements (`r` and `1 - r`), not model posteriors. The feature is
`FrameCurves`-class, so `augment_analysis` always needs the audio again.

Similarity weighting profiles version independently of the vector layout:
`SIMILARITY_VERSION` (unchanged at 2) still identifies the stored vector and
the default metric, while each named profile (e.g. `timbre`) carries its own
weight-table version, surfaced through `SIMILARITY_PROFILES`. Profiles apply at
query time and never change stored vectors, so a profile bump forces no
re-keying. This shape was communicated to sonagram on 2026-08-17 (sonagram
inbox, `2026-08-17-from-sonara-augment-api-shape-genre-nogo-and-a-question.md`);
this section records the standing policy.

### Backfilling onset bands and regularity in 0.3.7

`augment_analysis` rejects a record of an earlier schema (schema 7 of fork
release `v0.3.7-bpm-precision-20261005`, schema 6 before it) with
`schema version mismatch`:
re-analyse such tracks, requesting these groups along with the rest. The
backfill below applies to records of the current schema.

These additions do not invalidate existing 0.3.6 schema-v6 results. Consumers
can preserve the old analysis and fill only the missing feature groups, but
must re-read the audio: both groups are `FrameCurves` dependencies. In 0.3.7,
even a record containing onset bands cannot be augmented with regularity
decode-free. `can_augment == false` describes that limitation, not a prohibition
on augmentation with an audio path.

- For a complete cached Sonara result, request both groups together through
  `augment_analysis` with the original audio path. The result is a copy with
  the requested groups merged in; unrelated fields are preserved.
- For a custom database mapping, either reconstruct the full Sonara input
  contract or call targeted `analyze_file` and persist only the new groups.
  Do not pass an arbitrary database row as a cached Sonara result, or replace
  a rich stored analysis with the narrower targeted result.
- Use the original effective sample rate and BPM-range configuration. Persist
  the new timeline's sample rate and hop length, checking compatibility before
  merging it under existing shared metadata. Band frame `k` has time
  `k * hop_length / sample_rate`; `energy_curve` uses its own hop in seconds.
- Store `onset_strength_bands` and `onset_band_edges_hz` together. Store all four
  regularity fields, allowing a null score, label and candidates. A non-null
  confidence (including zero) marks a completed measurement, not a pending job.
- Database migrations, merging, checkpoints and error handling belong to the
  consumer. Sonara never updates the database itself. Benchmark storage and
  throughput on representative files before scheduling a whole-library run.

Python call examples are in the README's
[0.3.7 backfill guide](../README.md#adding-the-037-rhythm-features-to-an-existing-library).

## Notification obligations

sonara notifies (via `notify`) **before** any change that:
- couples the core crate to PyO3 or NumPy,
- moves an ML/ONNX runtime into the default feature set,
- reinterprets a fingerprint or similarity layout without a version bump.

sonara replies if a requested boundary conflicts with its intended API shape.

**Recipient: sonagram. kglite is deliberately NOT on this list** (decided
2026-07-29). sonagram consumes the API and can act on the notice; kglite
cannot. The obligation was written on 2026-07-15 naming kglite, and stayed
that way through the 2026-07-18 handover to sonagram.

kglite was considered and ruled out on evidence, recorded here so it is not
re-added out of habit: it has **no cargo dependency on sonara**, no 48-dim
assumption anywhere in the engine, and `EmbeddingStore` is dimension-agnostic
(it carries `dimension` as data — `EmbeddingStore::new(src_store.dimension)`).
kglite stores vectors it is handed and never interprets them, so a change to
what they *mean* cannot break it. Notifying a repo that cannot act is the
noise that trains people to ignore the inbox.

## History

- 2026-07-15 — kglite raises the contract
  (`sonara:inbox/read/2026-07-15-from-kglite-future-core-consumer-contract.md`).
- 2026-07-15 — sonara ships all three asks in 0.2.1 and reports back
  (`kglite:inbox/read/2026-07-15-from-sonara-consumer-contract-shipped.md`).
- 2026-07-16 — sonagram claims the mapping seat
  (`kglite:inbox/read/2026-07-16-from-sonagram-ownership-boundary.md`).
- 2026-07-18 — kglite confirms the narrower split.
- 2026-07-29 — the two gitignored copies are collapsed into this tracked file;
  the notification-routing gap above is opened.
- 2026-08-17 — augment lane and similarity profiles ship additively (no schema
  bump); the "pre-weighted vector" mis-description is corrected (the stored
  vector is unweighted — weights apply inside `distance()` at query time) and
  the per-feature freshness policy above is recorded. sonagram notified the
  same day (API shape, profile versioning, genre NO-GO).
