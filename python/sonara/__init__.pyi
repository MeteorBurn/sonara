"""Type stubs for sonara — high-performance audio analysis."""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Type aliases
# Returned arrays are float32 (complex64 for complex spectra).
AudioArray = NDArray[np.float32]
SpectrogramArray = NDArray[np.float32]
ComplexArray = NDArray[np.complex64]
# Array arguments take any numeric array or sequence; sonara casts it to float32
# (complex64), borrowing a float32 array without a copy.
AudioInput = ArrayLike
SpectrogramInput = ArrayLike
ComplexInput = ArrayLike
# File arguments take a str or any os.PathLike (pathlib.Path); results echo paths as str.
PathArg = Union[str, "os.PathLike[str]"]

# ============================================================
# Core: Audio I/O
# ============================================================

def load(path: PathArg, *, sr: int = 22050, mono: bool = True, offset: float = 0.0, duration: float = 0.0) -> Tuple[AudioArray, int]: ...
def to_mono(y: ArrayLike) -> AudioArray: ...
def resample(y: AudioInput, *, orig_sr: int, target_sr: int) -> AudioArray: ...
def get_duration(path: PathArg) -> float: ...
def get_samplerate(path: PathArg) -> int: ...
def autocorrelate(y: AudioInput, *, max_size: Optional[int] = None) -> AudioArray: ...
def lpc(y: AudioInput, *, order: int) -> AudioArray: ...
def zero_crossings(y: AudioInput, *, threshold: float = 0.0) -> NDArray[np.bool_]: ...
def tone(frequency: float, *, sr: int = 22050, length: int = 22050) -> AudioArray: ...
def chirp(*, fmin: float, fmax: float, sr: int = 22050, length: int = 22050) -> AudioArray: ...
def clicks(*, times: List[float], sr: int = 22050, length: int = 22050, click_freq: float = 1000.0, click_duration: float = 0.1) -> AudioArray: ...
def mu_compress(x: AudioInput, *, mu: float = 255.0) -> AudioArray: ...
def mu_expand(y: AudioInput, *, mu: float = 255.0) -> AudioArray: ...
def stream_with_resample(path: PathArg, *, block_length: int = 64, frame_length: int = 2048, hop_length: int = 512, target_sr: int = 22050, mono: bool = True) -> List[AudioArray]: ...

# ============================================================
# Core: Spectrum
# ============================================================

def stft(y: AudioInput, *, n_fft: int = 2048, hop_length: Optional[int] = None, win_length: Optional[int] = None, window: str = "hann", center: bool = True, pad_mode: str = "constant") -> ComplexArray: ...
def istft(stft_matrix: ComplexInput, *, hop_length: Optional[int] = None, win_length: Optional[int] = None, window: str = "hann", center: bool = True, length: Optional[int] = None) -> AudioArray: ...
def magphase(d: ComplexInput, *, power: float = 1.0) -> Tuple[SpectrogramArray, ComplexArray]: ...
def phase_vocoder(d: ComplexInput, *, rate: float, hop_length: Optional[int] = None) -> ComplexArray: ...
def power_to_db(s: SpectrogramInput, *, ref_power: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> SpectrogramArray: ...
def db_to_power(s_db: SpectrogramInput, *, ref_power: float = 1.0) -> SpectrogramArray: ...
def amplitude_to_db(s: SpectrogramInput, *, ref_amplitude: float = 1.0, amin: float = 1e-5, top_db: float = 80.0) -> SpectrogramArray: ...
def db_to_amplitude(s_db: SpectrogramInput, *, ref_amplitude: float = 1.0) -> SpectrogramArray: ...
def griffinlim(s_mag: SpectrogramInput, *, n_iter: int = 32, hop_length: Optional[int] = None, win_length: Optional[int] = None, window: str = "hann") -> AudioArray: ...
def pcen(s: SpectrogramInput, *, sr: float = 22050.0, hop_length: int = 512, gain: float = 0.98, bias: float = 2.0, power: float = 0.5, time_constant: float = 0.06, eps: float = 1e-6) -> SpectrogramArray: ...
def perceptual_weighting(s: SpectrogramInput, frequencies: ArrayLike, *, kind: str = "A") -> SpectrogramArray: ...

# ============================================================
# Core: CQT
# ============================================================

def cqt(y: AudioInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, n_bins: int = 84, bins_per_octave: int = 12, filter_scale: float = 1.0) -> ComplexArray: ...
def vqt(y: AudioInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, n_bins: int = 84, bins_per_octave: int = 12, filter_scale: float = 1.0, gamma: float = 0.0) -> ComplexArray: ...
def hybrid_cqt(y: AudioInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, n_bins: int = 84, bins_per_octave: int = 12, filter_scale: float = 1.0) -> ComplexArray: ...
def pseudo_cqt(y: AudioInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, n_bins: int = 84, bins_per_octave: int = 12, filter_scale: float = 1.0) -> SpectrogramArray: ...
def icqt(cq: ComplexInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, bins_per_octave: int = 12, filter_scale: float = 1.0) -> AudioArray: ...
def griffinlim_cqt(cq_mag: SpectrogramInput, *, sr: int = 22050, hop_length: int = 512, fmin: Optional[float] = None, bins_per_octave: int = 12, n_iter: int = 32) -> AudioArray: ...

# ============================================================
# Core: Pitch
# ============================================================

def yin(y: AudioInput, *, fmin: float, fmax: float, sr: int = 22050, frame_length: int = 2048, hop_length: Optional[int] = None, trough_threshold: float = 0.1) -> NDArray[np.float32]: ...
def pyin(y: AudioInput, *, fmin: float, fmax: float, sr: int = 22050, frame_length: int = 2048, hop_length: Optional[int] = None) -> Tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]: ...
def estimate_tuning(*, y: Optional[AudioInput] = None, sr: int = 22050, n_fft: Optional[int] = None, resolution: Optional[float] = None, bins_per_octave: Optional[int] = None) -> float: ...
def pitch_tuning(pitches: List[float], *, resolution: Optional[float] = None, bins_per_octave: Optional[int] = None) -> float: ...
def piptrack(y: AudioInput, *, sr: int = 22050, n_fft: int = 2048, hop_length: Optional[int] = None) -> Tuple[SpectrogramArray, SpectrogramArray]: ...

# ============================================================
# Core: Harmonic
# ============================================================

def salience(s: SpectrogramInput, freqs: ArrayLike, *, harmonics: List[int], weights: Optional[List[float]] = None, fill_value: float = 0.0) -> SpectrogramArray: ...
def interp_harmonics(s: SpectrogramInput, freqs: ArrayLike, *, harmonics: List[float], fill_value: float = 0.0) -> SpectrogramArray: ...
def f0_harmonics(s: SpectrogramInput, freqs: ArrayLike, f0: ArrayLike, *, harmonics: List[float], fill_value: float = 0.0) -> SpectrogramArray: ...

# ============================================================
# Core: Conversions
# ============================================================

def hz_to_mel(freq: float, *, htk: bool = False) -> float: ...
def mel_to_hz(mel: float, *, htk: bool = False) -> float: ...
def hz_to_midi(freq: float, *, a4: Optional[float] = None) -> float: ...
def midi_to_hz(midi: float, *, a4: Optional[float] = None) -> float: ...
def note_to_hz(note: str, *, a4: Optional[float] = None) -> float: ...
def note_to_midi(note: str) -> float: ...
def midi_to_note(midi: float) -> str: ...
def hz_to_note(freq: float, *, a4: Optional[float] = None) -> str: ...
def hz_to_octs(freq: float, *, tuning: float = 0.0, bins_per_octave: int = 12, a4: Optional[float] = None) -> float: ...
def octs_to_hz(octs: float, *, tuning: float = 0.0, bins_per_octave: int = 12, a4: Optional[float] = None) -> float: ...
def A4_to_tuning(a4: float, *, bins_per_octave: int = 12) -> float: ...
def tuning_to_A4(tuning: float, *, bins_per_octave: int = 12) -> float: ...
def hz_to_svara_h(freq: float, *, sa: float = 261.63, abbr: bool = True) -> str: ...
def hz_to_svara_c(freq: float, *, sa: float = 261.63, abbr: bool = True) -> str: ...
def midi_to_svara_h(midi: float, *, sa: float = 60.0, abbr: bool = True) -> str: ...
def midi_to_svara_c(midi: float, *, sa: float = 60.0, abbr: bool = True) -> str: ...
def note_to_svara_h(note: str, *, sa: str = "C4", abbr: bool = True) -> str: ...
def note_to_svara_c(note: str, *, sa: str = "C4", abbr: bool = True) -> str: ...
def hz_to_fjs(freq: float, *, ref_freq: float = 261.63) -> str: ...
def fft_frequencies(*, sr: float = 22050.0, n_fft: int = 2048) -> NDArray[np.float32]: ...
def mel_frequencies(*, n_mels: int = 128, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False) -> NDArray[np.float32]: ...
def cqt_frequencies(n_bins: int, *, fmin: float = 32.7032, bins_per_octave: int = 12) -> NDArray[np.float32]: ...
def tempo_frequencies(n_bins: int, *, hop_length: int = 512, sr: float = 22050.0) -> NDArray[np.float32]: ...
def fourier_tempo_frequencies(*, sr: float = 22050.0, win_length: int = 384, hop_length: int = 512) -> NDArray[np.float32]: ...
def frames_to_samples(frames: List[int], *, hop_length: int = 512) -> List[int]: ...
def frames_to_time(frames: List[int], *, sr: float = 22050.0, hop_length: int = 512) -> List[float]: ...
def samples_to_frames(samples: List[int], *, hop_length: int = 512) -> List[int]: ...
def samples_to_time(samples: List[int], *, sr: float = 22050.0) -> List[float]: ...
def time_to_frames(times: List[float], *, sr: float = 22050.0, hop_length: int = 512) -> List[int]: ...
def time_to_samples(times: List[float], *, sr: float = 22050.0) -> List[int]: ...
def blocks_to_frames(blocks: List[int], *, block_length: int, hop_length: int = 512) -> List[int]: ...
def blocks_to_samples(blocks: List[int], *, block_length: int) -> List[int]: ...
def blocks_to_time(blocks: List[int], *, block_length: int, sr: float = 22050.0) -> List[float]: ...
def samples_like(n_frames: int, *, hop_length: int = 512) -> NDArray[np.uintp]: ...
def times_like(n_frames: int, *, sr: float = 22050.0, hop_length: int = 512) -> NDArray[np.float32]: ...

# ============================================================
# Weighting
# ============================================================

def A_weighting(frequencies: ArrayLike) -> NDArray[np.float32]: ...
def B_weighting(frequencies: ArrayLike) -> NDArray[np.float32]: ...
def C_weighting(frequencies: ArrayLike) -> NDArray[np.float32]: ...
def D_weighting(frequencies: ArrayLike) -> NDArray[np.float32]: ...
def Z_weighting(frequencies: ArrayLike) -> NDArray[np.float32]: ...
def frequency_weighting(frequencies: ArrayLike, *, kind: str = "A") -> NDArray[np.float32]: ...
def multi_frequency_weighting(frequencies: ArrayLike, kinds: List[str]) -> SpectrogramArray: ...

# ============================================================
# Notation
# ============================================================

def key_to_notes(key: str) -> List[str]: ...
def key_to_degrees(key: str) -> List[int]: ...
def mela_to_degrees(mela: int) -> List[int]: ...
def mela_to_svara(mela: int) -> List[str]: ...
def thaat_to_degrees(thaat: str) -> List[int]: ...
def list_mela() -> List[str]: ...
def list_thaat() -> List[str]: ...
def fifths_to_note(fifths: int) -> str: ...
def interval_to_fjs(interval: float) -> str: ...
def interval_frequencies(n_bins: int, *, fmin: float = 32.7, intervals: List[float], bins_per_octave: int = 12) -> NDArray[np.float32]: ...
def pythagorean_intervals(*, bins_per_octave: int = 12) -> NDArray[np.float32]: ...
def plimit_intervals(*, p: int = 5, bins_per_octave: int = 12) -> NDArray[np.float32]: ...

# ============================================================
# Features
# ============================================================

def melspectrogram(*, y: Optional[AudioInput] = None, S: Optional[SpectrogramInput] = None, sr: float = 22050.0, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, fmin: float = 0.0, fmax: float = 0.0, power: float = 2.0) -> SpectrogramArray: ...
def mfcc(*, y: Optional[AudioInput] = None, S: Optional[SpectrogramInput] = None, sr: float = 22050.0, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, fmin: float = 0.0, fmax: float = 0.0) -> SpectrogramArray: ...
def chroma_stft(*, y: Optional[AudioInput] = None, S: Optional[SpectrogramInput] = None, sr: float = 22050.0, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = 12, tuning: float = 0.0) -> SpectrogramArray: ...
def spectral_centroid(*, y: Optional[AudioInput] = None, S: Optional[SpectrogramInput] = None, sr: float = 22050.0, n_fft: int = 2048, hop_length: int = 512) -> SpectrogramArray: ...
def rms(*, y: Optional[AudioInput] = None, S: Optional[SpectrogramInput] = None, frame_length: int = 2048, hop_length: int = 512) -> SpectrogramArray: ...

# ============================================================
# Onset / Beat
# ============================================================

def onset_detect(*, y: Optional[AudioInput] = None, onset_envelope: Optional[ArrayLike] = None, sr: int = 22050, hop_length: int = 512, backtrack: bool = False, delta: float = 0.07, wait: int = 0) -> List[int]: ...
def onset_strength(y: AudioInput, *, sr: int = 22050, hop_length: int = 512) -> NDArray[np.float32]: ...
def onset_strength_method(y: AudioInput, *, sr: int = 22050, hop_length: int = 512, method: str = "spectral_flux") -> NDArray[np.float32]: ...
def onset_strength_bands(y: NDArray[np.float32], *, sr: int = 22050, hop_length: int = 512, band_edges_hz: Optional[List[float]] = None) -> Tuple[NDArray[np.float32], List[float]]: ...
def beat_track(*, y: Optional[AudioInput] = None, onset_envelope: Optional[ArrayLike] = None, sr: int = 22050, hop_length: int = 512, start_bpm: float = 120.0, tightness: float = 100.0, trim: bool = True, bpm_min: Optional[float] = None, bpm_max: Optional[float] = None) -> Tuple[float, List[int]]: ...
def tempo_curve(beat_frames: Sequence[int], *, sr: int = 22050, hop_length: int = 512, smooth: Optional[int] = None) -> List[float]: ...
def tempo_variability(tempo_curve: Sequence[float]) -> float: ...

# ============================================================
# Effects
# ============================================================

def trim(y: AudioInput, *, top_db: float = 60.0, frame_length: int = 2048, hop_length: int = 512) -> Tuple[AudioArray, Tuple[int, int]]: ...
def split(y: AudioInput, *, top_db: float = 60.0, frame_length: int = 2048, hop_length: int = 512) -> List[Tuple[int, int]]: ...
def split_with_constraints(y: AudioInput, *, sr: int = 22050, top_db: float = 60.0, frame_length: int = 2048, hop_length: int = 512, min_silence_duration: Optional[float] = None, min_signal_duration: Optional[float] = None) -> List[Tuple[int, int]]: ...
def melody_separate(y: AudioInput, *, sr: int = 22050, fmin: float = 65.0, fmax: float = 2100.0, n_harmonics: int = 10, n_fft: int = 2048, hop_length: int = 512) -> Tuple[AudioArray, AudioArray]: ...

# ============================================================
# Tonal
# ============================================================

def hpcp(power_spec: SpectrogramInput, freqs: ArrayLike, *, n_harmonics: int = 4, min_freq: float = 40.0, max_freq: float = 5000.0, peak_threshold: float = 0.0, max_peaks: int = 50) -> SpectrogramArray: ...
def chords_from_beats(hpcp: SpectrogramInput, beats: Sequence[int]) -> List[str]: ...
def chords_from_frames(hpcp: SpectrogramInput, *, segment_frames: int = 10) -> List[str]: ...
def chord_descriptors(chords: Sequence[str], duration_sec: float) -> Dict[str, Union[str, float, int]]: ...  # predominant_chord, chord_change_rate, n_unique
def dissonance(power_spec: SpectrogramInput, freqs: ArrayLike, *, peak_threshold: float = 0.0, max_peaks: int = 50) -> float: ...
def dissonance_from_peaks(peaks_freq: Sequence[float], peaks_mag: Sequence[float]) -> float: ...

# ============================================================
# Filters
# ============================================================

def mel(*, sr: float = 22050.0, n_fft: int = 2048, n_mels: int = 128, fmin: float = 0.0, fmax: float = 0.0, htk: bool = False, norm: str = "slaney") -> SpectrogramArray: ...

# ============================================================
# Fused Analysis (sonara-specific, high-performance)
# ============================================================

AnalysisResult = Dict[str, Union[float, int, str, List[int], List[float], List[List[float]], List[str], List[Tuple[float, float]]]]

# Versions to compare stored records against: provenance["schema_version"] and
# the "fingerprint_version" field.
ANALYSIS_SCHEMA_VERSION: int
FINGERPRINT_VERSION: int

class TrackAnalysis(Dict[str, Any]):
    """dict subclass returned by analyze_file / analyze_signal / analyze_batch / augment_analysis."""
    @property
    def failed(self) -> bool: ...  # True for a per-file analyze_batch failure (has "error")
    def print(self) -> None: ...

# Result dicts include string fields such as "key" ("A minor"), "key_camelot"
# (Camelot wheel code, e.g. "8A"), "time_signature", and "predominant_chord".
#
# Every result dict (all modes) carries "bpm_confidence": float in [0, 1] — a
# trust signal for the reported "bpm" (dominant tempo-peak strength + beat-rate
# agreement + onset density), high (>0.7) on steady percussive music, low
# (<0.45) on ambient/rubato/sparse material. A trust signal, not a probability.
# It also carries "bpm_raw" (tempo before range folding) and "n_beats" (len(beats)).
#
# Recalibrated in 0.2.4 (schema v3): "acousticness" (absolute scale; electronic
# < 0.3, acoustic > 0.6 on real music) and "danceability" (spread across [0, 1]).
# Orderings are largely preserved; consumers with 0.2.2/0.2.3-era numeric cutoffs
# on these fields must re-derive them.
#
# Every result dict carries a "provenance" sub-dict describing how it was
# produced (self-describing time + stale-record detection):
#   provenance: Dict — {
#     "schema_version":     int        # bumped when field meaning/units change
#     "sample_rate":        int        # effective Hz (post-resample); frame
#                                      # indices are valid against this rate
#     "hop_length":         int        # STFT hop of the main pass; seconds =
#                                      # frame * hop_length / sample_rate
#     "mode":               str        # "compact" | "playlist" | "full"
#     "requested_features": List[str]  # sorted; only when features=[...] given
#     "bpm_min":            float      # octave-folding tempo range in effect at
#     "bpm_max":            float      # analysis time; keys absent when unset.
#                                      # A changed range invalidates stored "bpm"
#                                      # (out-of-range raw tempos fold into it);
#                                      # "bpm_raw" stays comparable.
#     "genre_model_id":     str        # id of the genre_model used; absent otherwise
#     "vocalness_model_id": str        # id of the vocalness_model used; absent otherwise
#     "aggression_model_id": str       # with features=["aggression"] or full mode
#   }
#
# When chords are computed (playlist/full modes or features=["chords"]), the
# dict also carries typed time-spanned chord events alongside chord_sequence:
#   chord_events: List[Dict]  # {"label": str, "start_sec": float,
#                             #  "end_sec": float} — merged runs of
#                             # chord_sequence; contiguous, covering the track;
#                             # label "N" = no chord detected in that span
def analyze_file(path: PathArg, *, sr: int = 22050, mode: str = "compact", features: Optional[List[str]] = None, bpm_min: Optional[float] = None, bpm_max: Optional[float] = None, genre_model: Optional[PathArg] = None, vocalness_model: Optional[PathArg] = None) -> TrackAnalysis: ...
def analyze_signal(y: AudioInput, *, sr: int = 22050, mode: str = "compact", features: Optional[List[str]] = None, bpm_min: Optional[float] = None, bpm_max: Optional[float] = None, genre_model: Optional[PathArg] = None, vocalness_model: Optional[PathArg] = None) -> TrackAnalysis: ...
def analyze_batch(paths: Sequence[PathArg], *, sr: int = 22050, mode: str = "compact", features: Optional[List[str]] = None, bpm_min: Optional[float] = None, bpm_max: Optional[float] = None, progress: Optional[Callable[[int, int], None]] = None, genre_model: Optional[PathArg] = None, vocalness_model: Optional[PathArg] = None) -> List[TrackAnalysis]: ...
# analyze_batch never raises on a per-file decode/IO error. Each input path yields
# exactly one entry in input order, and every entry carries its input `path`.
# A failed file's entry has `path`, `error`, and `error_kind` (e.g. "decode",
# "io", "unsupported_format") instead of feature fields.
# `progress`, if given, is called as progress(done, total) after each completed
# file — `done` counts completions in COMPLETION order (not input order),
# `total == len(paths)`. Callback exceptions are ignored (never abort the batch).

# ============================================================
# Augment lane — recompute features onto a cached analysis dict
# ============================================================
# augment_analysis recomputes the named features onto a COPY of a cached
# analysis dict (as returned by analyze_* — or the same shape loaded back from
# JSON; tuples-as-lists are accepted). Decode-free where the record's evidence
# allows; otherwise pass audio_path to enable ONE re-analysis at the record's
# own provenance["sample_rate"] computing exactly the blocked features (an
# "aggression" request auto-routes through its dedicated 22.05 kHz lane).
# Feature names are case-insensitive; unknown names raise ValueError. Genre has
# no feature name — passing genre_model IS the request (features=[] is valid);
# vocalness/instrumentalness always update together. A schema_version mismatch
# raises ValueError (re-analyze instead). Fields not asked about are never
# cleared; provenance["requested_features"] becomes the union.
def augment_analysis(cached: Dict, features: Optional[List[str]] = None, *, audio_path: Optional[PathArg] = None, bpm_min: Optional[float] = None, bpm_max: Optional[float] = None, genre_model: Optional[PathArg] = None, vocalness_model: Optional[PathArg] = None) -> TrackAnalysis: ...
# can_augment: True iff `feature` is decode-free recomputable from THIS record
# (evidence is per-record; two same-version records can differ). False for
# unknown names.
def can_augment(cached: Dict, feature: str) -> bool: ...
# augment_blocker: the reason can_augment is False, as a stable descriptive
# string — one of: "unknown feature", "needs audio (<class>-class feature)",
# "schema version mismatch (record N, current M)", "embedding version mismatch
# (record N, current M)", "missing evidence: <field>, ..." — or None when the
# feature can be recomputed decode-free.
def augment_blocker(cached: Dict, feature: str) -> Optional[str]: ...
# feature_dependencies: the declared per-feature dependency map, one dict per
# public feature in canonical order:
#   name:              str        — canonical feature name
#   class:             str        — "audio" | "frame_curves" | "scalars" |
#                                 #  "embedding"; scalars/embedding are
#                                 #  decode-free recomputable from a record
#                                 #  carrying every required_evidence field
#   required_evidence: List[str]  — record fields a decode-free recompute reads
#                                 #  (non-empty exactly for scalars/embedding)
#   needs_extended:    bool       — fresh computation needs the extended pass
#   opt_in_only:       bool       — left out of the compact and playlist
#                                 #  defaults; computed by full mode or an
#                                 #  explicit features=[...]
#   full_only:         bool       — among mode defaults, Full mode only
def feature_dependencies() -> List[Dict[str, Union[str, bool, List[str]]]]: ...

# --- beat grid ---
# Opt-in via features=["beatgrid"] or full mode. When computed, the analyze_*
# result dict additionally contains:
#   grid_offset_sec: float        — time (sec) of the first beat (grid anchor)
#   downbeats:       List[int]    — frame indices of bar-starting beats
#   grid_stability:  float        — 0..1, how rigidly beats fit a constant grid

# --- multiband onset strength ---
# Opt-in via features=["onset_bands"] or full mode. When computed, the
# analyze_* result dict additionally contains:
#   onset_band_edges_hz: List[float]        — inclusive lower/upper band edges
#   onset_strength_bands: List[List[float]] — shape (n_bands, n_frames)

# --- rhythmic regularity ---
# Opt-in via features=["rhythmic_regularity"] or full mode. A deterministic DSP
# measure of how straight the percussive pattern sits inside the metric grid —
# not a model and not a genre classifier. When computed, all four keys are present:
#   rhythmic_regularity:            Optional[float] — 0..1; 1.0 a straight
#                                          #  house/techno pattern on the pulse
#                                          #  lattice, 0.0 a syncopated /
#                                          #  polyrhythmic / breakbeat one
#   rhythmic_regularity_label:      Optional[str]   — "regular" | "irregular"
#   rhythmic_regularity_confidence: float  — 0..1 quality of the rhythmic
#                                          #  evidence, NOT a class probability
#   rhythmic_regularity_candidates: Optional[List[Tuple[str, float]]]
#                                          # both labels with their DSP scores,
#                                          # ranked; complementary, not model
#                                          # probabilities
# ABSTENTION: on silence, ambient or drumless audio the measure reports its
# confidence and sets the score, label and candidates to None. `None` therefore
# means "measured, could not tell" — branch on the score being None, not on the
# key being absent.
# Requesting it internally computes the onset bands and beat grid it reads, but
# does not emit them unless they are separately requested.

# --- structure ---
# Opt-in structural segmentation & energy curve. Requested via
# `features=["structure"]` or computed by full mode; absent from compact and
# playlist. When present, the result dict additionally contains:
#   energy_curve:         List[float]  # per-window perceptual energy (0-1)
#   energy_curve_hop_sec: float        # seconds between curve samples
#   segments:             List[Dict[str, float]]
#                                      # {"start_sec", "end_sec", "energy"}, contiguous, covering
#   intro_end_sec:        float        # end of intro / pre-first-drop region
#   outro_start_sec:      float        # start of outro / final fade
#   energy_level:         int          # coarse 1-10 level
# Extended loudness / gain metrics are opt-in via features=["loudness"] or full
# mode (not computed by compact or playlist, for performance). When computed,
# the result dict also carries:
#   "true_peak_db"                — true peak in dBTP (4x oversampled, ITU-R BS.1770-4)
#   "replaygain_db"               — track gain to reach -18 LUFS (-18 - loudness_lufs)
#   "loudness_curve"              — List[float] short-term LUFS (3 s window, 1 s hop)
#   "loudness_momentary_max_db"   — max momentary (400 ms) loudness, dB
#   "loudness_range_lu"           — EBU R128 loudness range (LRA), LU

# --- ebur128 / native_dynamics / amplitude_envelope ---
# File-only, opt-in via features=[...] or full mode (compact and playlist do
# not compute them): measured by
# analyze_file / analyze_batch on the file's own channels at its own sample
# rate, not on the mono mix at the analysis rate. analyze_signal never emits
# them, in full mode either.
# A key is absent when its value is undefined (digital silence, file
# shorter than the measurement window). The older loudness_lufs / true_peak_db
# (mono, analysis rate, ungated) are unchanged.
# features=["ebur128"] — ITU-R BS.1770-5 / EBU Tech 3341, 3342, all float.
# Channels are read in WAV order (L R C LFE Ls Rs, LFE excluded); a file with
# more than 6 channels gets no loudness keys (unknown layout), only the peak:
#   "ebur128_integrated_lufs"      — gated programme loudness, LUFS
#   "ebur128_momentary_max_lufs"   — max momentary (400 ms) loudness, LUFS
#   "ebur128_short_term_max_lufs"  — max short-term (3 s) loudness, LUFS
#   "ebur128_range_lu"             — loudness range (LRA), LU
#   "ebur128_true_peak_dbtp"       — true peak over the channels, dBTP (4x);
#                                    EBU Tech 3341 accuracy, +0.2 / -0.4 dB
# Any of the three features (and stereo_field below) also sets:
#   "native_sample_rate"           — int, sample rate of the file, Hz. Present
#                                    means "measured from a file", so a missing
#                                    level is undefined, not unmeasured
# features=["native_dynamics"]:
#   "sample_peak_dbfs"             — max |x| over the file's channels, dBFS
#   "crest_factor_db"              — track peak / RMS of all samples, dB
#   "crest_factor_frame_mean_db"   — mean per-frame crest factor, dB, over
#                                    envelope frames with RMS above -70 dBFS
# features=["amplitude_envelope"]:
#   "amplitude_envelope"           — List[float], max |x| over the channels in
#                                    each frame (linear amplitude; above 1
#                                    for float files with overs); frame k is
#                                    centred at k * amplitude_envelope_hop_sec
#   "amplitude_envelope_hop_sec"   — hop, seconds (512/22050)
#   "amplitude_envelope_frame_sec" — frame length, seconds (2048/22050)

# --- stereo_field --- features=["stereo_field"] (opt-in: full mode or an explicit request)
# File-only like the family above: measured by analyze_file / analyze_batch on
# the file's left and right channels at its own sample rate; analyze_signal
# never emits it. All float, dimensionless. A key is absent when its value is
# undefined; "native_sample_rate" (above) is set as well and marks the record
# as measured from a file. A mono file is measured as L = R (shares 0,
# correlations +1); a file with more than two channels gets none of the four
# keys (unknown layout).
#   "stereo_side_ratio"        — 0..1, share of the Side signal in the energy
#                                of the whole file:
#                                0.5 * (1 - 2*cov(L,R) / (var L + var R)),
#                                means removed (a constant offset in a channel
#                                does not change it). 0 = identical channels,
#                                0.5 = uncorrelated channels of equal power or
#                                sound in one channel only, 1 = antiphase
#                                (R = -L). Tardieu et al., DAFx 2011, eq. (3):
#                                s^2 / (1 + s^2) with s = std(L-R) / std(L+R).
#                                Absent when (var L + var R) / 2 is below -70 dB
#   "stereo_side_ratio_low"    — the same after a 4th-order Butterworth
#                                low-pass at 150 Hz on both channels; absent
#                                when the band holds less than -70 dB
#   "stereo_correlation_mean"  — -1..+1, mean over frames of the Pearson
#                                correlation of L and R; frames of 2048/22050 s
#                                without overlap, an incomplete last frame is
#                                dropped (Ziemer 2023, arXiv:2302.01090).
#                                Frames with both variances below -70 dB are
#                                skipped; a frame where one channel has zero
#                                variance counts as 0. Absent when no frame is
#                                left. +1 for identical channels and for plain
#                                panning (R = a*L) alike
#   "stereo_correlation_p05"   — 5th percentile of the same frames (linear
#                                interpolation, as the _p10 keys)
# The filter is set in Hz and the frame in seconds, so values are comparable
# between files of different sample rates. The 150 Hz edge, the filter order
# and the 5th percentile are project choices, not taken from a standard.

# --- frame statistics ---
# Spread, frame-to-frame change and quantiles of the per-frame values behind the
# `_mean` fields, over the same frames (all main-pass frames, silence included).
# Each key is present exactly when its base feature is:
#   rms      → rms_std, rms_delta_mean, rms_iqr, rms_p10, rms_p90          (float)
#   zcr      → zero_crossing_rate_std                                      (float)
#   centroid → spectral_centroid_std, spectral_centroid_delta_mean,
#              spectral_centroid_iqr                                       (float, Hz)
#   bandwidth → spectral_bandwidth_std, spectral_bandwidth_delta_mean      (float, Hz)
#   rolloff  → spectral_rolloff_std, spectral_rolloff_p10, spectral_rolloff_p90 (float, Hz)
#   flatness → spectral_flatness_std, spectral_flatness_iqr,
#              spectral_flatness_p10, spectral_flatness_p90                (float)
#   contrast → spectral_contrast_std                       List[float] (7)
#   mfcc     → mfcc_std, mfcc_delta_mean                   List[float] (13)
#   chroma   → chroma_std, chroma_delta_mean               List[float] (12)
# _std = population standard deviation; _delta_mean = mean absolute difference
# between consecutive frames (base unit per hop); quantiles interpolate linearly
# (NumPy default); _iqr = p75 - p25.

# --- spectral shape --- features=["spectral_shape"] (also playlist/full modes)
# Track means of per-frame spectral-shape descriptors (MIR_FEATURES group 3.1),
# each averaged over only the frames where it is defined (silent frames are
# skipped; 0.0 when no frame is). All float:
#   spectral_skewness_mean             — magnitude-weighted skewness about the
#                                        centroid, standardized by bandwidth
#   spectral_kurtosis_excess_mean      — same moment, 4th order, minus 3
#   spectral_slope_db_per_octave_mean  — dB/octave, least-squares slope of 8
#                                        octave-band power levels (62.5..8000 Hz
#                                        centres); pink noise ≈ -3, white ≈ 0,
#                                        music with bass ≈ -3..-7; near-empty
#                                        low octaves (no bass) bias it upward
#   spectral_decrease_mean             — Peeters decrease, DC-bin reference: in
#                                        practice 1/k-weighted low-frequency
#                                        emphasis; at most 1, usually positive,
#                                        negative when the DC bin is strong
#                                        (DC offset or lowest-bass leakage)

# --- spectral texture --- features=["spectral_texture"] (also playlist/full modes)
# Per-frame spectral-texture descriptors (MIR_FEATURES group 3.2) on the
# magnitude spectrum |X_k| (all bins), each over only the frames where it is
# defined (silent frames are skipped; 0.0 when no frame is). All float:
#   spectral_flux_mean / _std   — half-wave rectified L2 distance between
#                                 consecutive L1-normalized magnitude spectra
#                                 (Essentia Flux(halfRectify, L2) on a
#                                 sum-normalized spectrum); 0..1,
#                                 loudness-invariant; pairs with a silent frame
#                                 skipped; _std is the population std (ddof 0)
#   spectral_crest_mean         — max|X| / mean|X|, >= 1
#   spectral_entropy_mean       — -sum p ln p / ln(n_bins), p = |X| / sum|X|;
#                                 0..1 (MIRtoolbox mirentropy normalization)
#   spectral_irregularity_mean  — Jensen: sum (a[i+1]-a[i])^2 / sum a[i]^2 over
#                                 the fused pass's spectral peaks (40..5000 Hz,
#                                 top 50, sorted by frequency); depends on that
#                                 peak picking; frames with < 2 peaks skipped

# --- mel bands --- features=["mel_bands"] (also playlist/full modes)
# Track summary of the main pass's log-mel spectrogram (MIR_FEATURES group 4.1),
# the matrix the MFCCs and the onset envelopes are computed from: 128 Slaney mel
# bands over 0..sr/2 (area-normalized triangles on |X|^2, n_fft 2048, hop 512,
# Hann), 10*log10 with a 1e-10 power floor, then floored at the track's loudest
# cell - 80 dB (librosa power_to_db(ref=1.0, amin=1e-10, top_db=80)). Statistics
# run over every main-pass frame, silence included. Band centres:
# mel_frequencies(n_mels=130, fmin=0, fmax=sr/2)[1:129].
#   mel_bands_mean      List[float] (128), dB, low band first — per-band mean of
#                       the dB values, not the dB of the mean power; mfcc_mean
#                       is the first 13 coefficients of its orthonormal DCT-II
#   mel_bands_std       List[float] (128), dB — per-band population std (ddof 0)
#                       over the same frames; unchanged by a gain change
#   mel_bands_floor_db  float, dB — the lowest value a cell can take: the loudest
#                       cell - 80 dB, or -100 dB (the power floor) when that
#                       is higher, as on silence; a band whose mean is within
#                       about 1 dB of it holds no content above the floor; a
#                       gain change shifts it and every mean by the same dB

# --- tuning --- features=["tuning"] (also playlist/full modes)
# Deviation of the recording's tuning from A4 = 440 Hz (MIR_FEATURES group 5.2),
# by the weighted circular mean of Dressler & Streich (ISMIR 2007). In every
# main-pass frame (n_fft 2048, hop 512, Hann) the 10 strongest spectral peaks
# with bin centre in 150..5000 Hz are taken (peaks at or below -80 dB of the
# frame's strongest bin are skipped), each peak frequency is refined by a
# parabola on the log magnitude, and its deviation from the nearest
# equal-tempered semitone (relative to 440 Hz) becomes an angle; the angles are
# averaged over all frames with weight sqrt(magnitude). Chroma, HPCP, key and
# chord detection still assume 440 Hz and are not corrected by this value, so a
# large offset means those are less reliable for that track.
#   tuning_offset_cents  Optional[float], cents, [-50, 50) — angle of the mean
#                        vector; None (key present) when tuning_confidence is
#                        below 0.08: measured, could not tell. When only
#                        n < 352 frames (about 8 s) carry a peak the bar is
#                        1.5/sqrt(n) instead, since noise alone would clear
#                        0.08 on short or mostly silent input. Wraps: a track
#                        near +-50 lies between two semitones, so compare two
#                        tracks with a circular difference. A known pitch
#                        shift of real tracks is recovered within about 2.5
#                        cents (median 0.2-0.4, at the default 22050 Hz
#                        analysis rate); clean chords within 0.5 cents at
#                        22050, 44100 and 48000 Hz
#   tuning_confidence    float, 0..1 — length of the mean vector: 0 = the peaks
#                        agree on no grid, 1 = every peak on the same grid;
#                        about 0.01-0.03 on noise of 5 s or more, 0.05-0.6 on
#                        mixed music, 0.99 on clean tones; loud vibrato
#                        vocals lower it. Present
#                        whenever the feature ran, also when the offset is
#                        None: key freshness on it, not on the offset

# --- hpss --- features=["hpss"] (opt-in: full mode or an explicit request)
# Share of percussive energy in the track by median-filter harmonic/percussive
# separation (MIR_FEATURES group 6.2). Computed in the main pass on the mono
# analysis signal, so analyze_file, analyze_batch and analyze_signal all emit
# it. On the main-pass magnitude spectrogram (n_fft 2048, hop 512, Hann, at the
# analysis sample rate; all frames, silence included) a median along time gives
# the harmonic estimate and a median along frequency the percussive estimate;
# the kernel is 0.72 s by 334 Hz, i.e. 31 frames by 31 bins at 22050 Hz, scaled
# with the sample rate and equal to the default kernel of
# librosa.decompose.hpss at 22050 Hz, edges mirrored. A soft mask with exponent
# 2 splits each cell's magnitude into a harmonic part H and a percussive part P
# (H + P = |X|): the split of librosa.decompose.hpss with this kernel,
# power=2.0 and margin=1.0. It is not the Rust decompose::hpss / effects::hpss
# separation, which handles the edges differently.
#   hpss_percussive_ratio  float, 0..1 — sum(P^2) / (sum(H^2) + sum(P^2)) over
#                          the whole track: 0 = all energy in sustained,
#                          horizontal (tonal) structure, 1 = all energy in
#                          broadband, vertical (transient) structure. About 0
#                          on a steady tone, about 0.5 on white noise, about 1
#                          on a click train; on the 150 songs of MUSDB18-HQ
#                          0.76 on the drum stems (median), 0.14 on the same
#                          songs without drums, 0.23 on the full mixes.
#                          Sustained low drums (kick and tom tails) count as
#                          harmonic and dense distorted guitars partly as
#                          percussive: the value follows the drum share in
#                          pop, rap and electronic music and hardly at all in
#                          heavy metal. An energy ratio: loud passages
#                          dominate and the overall level does not change it.
#                          The kernel spans about the same time and frequency
#                          at every analysis sample rate (rounded up to odd
#                          counts of frames and bins); that keeps the order
#                          of tracks but not the values (about 0.07 higher at
#                          44100 Hz), so compare values only between analyses
#                          at one sample rate. These numbers are for 22050 Hz.
#                          Digital silence reads 0.0: there is no
#                          abstention and no confidence, so key freshness on
#                          the presence of this key

# --- rhythm_levels --- features=["rhythm_levels"] (also playlist/full modes)
# Strength of the metrical levels next to the beat, from the autocorrelation
# of the onset envelope the tempo is estimated from. Computed in the main pass,
# so analyze_file, analyze_batch and analyze_signal all emit it. The envelope
# (one value per hop of 512 samples) is cut into frames of 8 s every 0.5 s;
# each frame has its mean removed and is autocorrelated, the result is divided
# by its value at lag 0, read at a fractional lag by linear interpolation and
# set to zero where negative. With T the beat period of the reported bpm, each
# level is the mean over frames at its lag. The three keys are present
# together whenever the feature ran.
#   rhythm_level_double_ratio        Optional[float], >= 0 — level at T/2
#                                    (eighth notes) divided by the level at T:
#                                    0 = nothing repeats at half a beat
#   rhythm_level_one_and_half_ratio  Optional[float], >= 0 — level at 2T/3
#                                    (triplets, shuffle, three against two)
#                                    divided by the level at T
#   rhythm_level_confidence          float, 0..1 — the level at T itself: mean
#                                    normalized autocorrelation of the onset
#                                    envelope at the beat period
# Both ratios are None (keys present) when the measure abstained: the track
# holds no full 8 s frame, no frame has any variation, the beat period does not
# fit a frame, or nothing repeats at the beat period (confidence 0.0). Key
# freshness on rhythm_level_confidence, not on the ratios. The ratios are
# relative to the reported bpm: a tempo read at double or half speed moves
# every level by one step.

# Opt-in analysis keys (present only in full mode or when requested via
# `features=[...]`; never enabled by compact or playlist). Values are plain
# Python types in the returned dict:
# --- silence --- features=["silence"]
#   "leading_silence_sec": float   # seconds of leading silence (>= 0, <= duration)
#   "trailing_silence_sec": float  # seconds of trailing silence (>= 0, <= duration)
# --- key candidates --- features=["key_candidates"]
#   "key_candidates": List[Tuple[str, str, float]]  # top-3 (key, camelot, score); [0] == "key"
# --- vocalness --- features=["vocalness"]
#   "vocalness": float             # heuristic v2 in [0, 1] (rough, not a classifier); changed semantics in 0.2.4.
#                                  # Prominence of vocal/broadband energy filling the ~0.8-5.6 kHz spectral
#                                  # valleys; rises harsh > clean > instrumental.
# --- aggression --- features=["aggression"]
#   "aggression_score": Optional[float] # perceptual rank; None means abstain
#   "aggression_confidence": float # evidence support, not score certainty
#   "aggression_forcefulness": float
#   "aggression_harshness": float
#   "aggression_tension": float
#   "aggression_rhythm": float
#   provenance["aggression_model_id"]: str
#                                  # dependencies stay internal unless separately requested.
# --- mood --- features=["mood"]
#   "mood_happy": float            # heuristic v1, not an ML classifier — in [0, 1]
#   "mood_aggressive": float       # heuristic v1, not an ML classifier — in [0, 1]
#   "mood_relaxed": float          # heuristic v1, not an ML classifier — in [0, 1]
#   "mood_sad": float              # heuristic v1, not an ML classifier — in [0, 1]
#                                  # All four populate together; correlated but not summing to 1.
# --- instrumentalness --- features=["instrumentalness"]
#   "instrumentalness": float      # heuristic v2, not an ML classifier — 1 - vocalness, in [0, 1]; changed semantics in 0.2.4.
# --- tags --- features=["tags"]  (analyze_file / analyze_batch only)
#   "tags": Dict  # container/stream metadata read from the file; each key present
#                 # only when that tag exists. Keys: "title", "artist", "album",
#                 # "genre" (str), "year", "original_year", "track_no" (int). Read
#                 # from symphonia containers (FLAC/Vorbis, MP3/ID3v2, MP4, ...);
#                 # WAV (hound fast path) carries no tags. Never set by
#                 # analyze_signal.
#                 # "year" is the file/edition date; "original_year" the original
#                 # release year (ID3 TDOR/TORY/TXXX:originalyear, Vorbis
#                 # ORIGINALDATE) — prefer it for era reasoning on reissues.
#                 # NOTE: tags["genre"] is the file's metadata genre, distinct
#                 # from the reserved top-level "genre" placeholder.
# ============================================================
# Duplicate detection (acoustic fingerprint) — fingerprint
# ============================================================

# Opt-in: pass features=["fingerprint"] to any analyze_* call to add two
# fields to the result dict — `fingerprint` (a base64 string) and
# `fingerprint_version` (int). Full mode computes them too; compact and
# playlist do not.
def fingerprint_match(a: Union[str, Dict], b: Union[str, Dict]) -> float: ...
# Similarity in [0, 1] between two fingerprints, robust to gain / re-encoding /
# small leading-silence differences. Accepts base64 `fingerprint` strings or
# analysis dicts containing one. A score above ~0.30 means "same recording".
# ============================================================
# Bring-your-own genre model — genre_model=<path>
# ============================================================
# sonara ships NO genre model. Pass `genre_model=<path to a JSON model>` to any
# analyze_* call to classify each track over sonara's similarity embedding; the
# result dict then carries:
#   "genre":            str    — predicted label (argmax class)
#   "genre_confidence": float  — winning softmax probability, in (0, 1]
# The embedding is computed internally when a model is set, but the "embedding"/
# "embedding_version" fields are only emitted if you ALSO request
# features=["embedding"] or analyze in full mode.
#
# Model JSON format (hand-writable and numpy-exportable):
#   {
#     "format_version": 1,           # schema version this build understands
#     "embedding_version": 2,        # MUST match sonara.SIMILARITY_VERSION at use
#                                    # time, else analyze_* raises ValueError
#     "labels": ["rock", "electronic", ...],
#     "layers": [                    # feed-forward chain; last activation softmax
#       {"weights": [[...], ...],    # row-major out_dim x in_dim (W . x)
#        "bias": [...],              # length out_dim
#        "activation": "relu"},      # "relu" | "softmax" | "identity"
#       {"weights": [[...]], "bias": [...], "activation": "softmax"}
#     ]
#   }
# The first layer's in_dim must equal EMBEDDING_DIM (48); the last layer must be
# softmax with out_dim == len(labels).
#
# Train one with the pure-numpy `sonara.genre` module:
#   from sonara import genre
#   model = genre.train(X, y, hidden=0, epochs=300, lr=0.1, seed=0)  # X: (n, 48)
#   model.save("genre_model.json"); genre.load("genre_model.json")
#   model.predict(x)  # (label, confidence) — numpy parity with the Rust inference

# ============================================================
# Similarity & embeddings
# ============================================================
# The similarity vector is OPT-IN: analyze with features=["embedding"] to add
# "embedding" (List[float], EMBEDDING_DIM long) and "embedding_version" (int) to
# the result. Full mode produces it too; compact and playlist do not.

SIMILARITY_VERSION: int
EMBEDDING_DIM: int
# Selectable distance-time weighting profiles for similarity/embedding_distance:
# profile name -> weight-table version. Profiles are applied at comparison time
# and never change the stored vector; the "default" profile's version aliases
# SIMILARITY_VERSION, while other profiles (e.g. "timbre") version their weight
# tables independently.
SIMILARITY_PROFILES: Dict[str, int]
AGGRESSION_MODEL_VERSION: int
AGGRESSION_SAMPLE_RATE: int
AGGRESSION_EMBEDDING_VERSION: int
AGGRESSION_MODEL_ID: str
LEGACY_AGGRESSION_MODEL_ID: str

class AggressionAnalysis(TypedDict):
    aggression_score: Optional[float]
    aggression_confidence: float
    aggression_forcefulness: float
    aggression_harshness: float
    aggression_tension: float
    aggression_rhythm: float
    aggression_model_id: str

def similarity(a: Union[Dict, List[float], NDArray[np.float32]], b: Union[Dict, List[float], NDArray[np.float32]], *, profile: str = "default") -> float: ...
def embedding_distance(a: List[float], b: List[float], *, profile: str = "default") -> float: ...
# Retained legacy-v1 scorer for stored 48D similarity embeddings.
def aggression_score(embedding: List[float], *, embedding_version: int = AGGRESSION_EMBEDDING_VERSION) -> float: ...
def analyze_aggression_file(path: PathArg, *, sr: int = 22050) -> AggressionAnalysis: ...
def analyze_aggression_signal(y: AudioInput, *, sr: int = 22050) -> AggressionAnalysis: ...
def analyze_aggression_batch(paths: Sequence[PathArg], *, sr: int = 22050) -> List[Dict[str, Union[str, float, None]]]: ...
