#!/usr/bin/env python3
"""Tests for the file-only loudness / dynamics features.

Covers `features=["ebur128", "native_dynamics", "amplitude_envelope"]`: EBU R128
loudness, sample peak, crest factor and the peak envelope, measured by
`analyze_file` / `analyze_batch` on the file's own channels and sample rate.
Writes a synthetic stereo WAV — no audio fixtures required.

Runs standalone (`python test_native_dynamics.py`) or under pytest.
"""

import atexit
import os
import shutil
import sys
import tempfile
import wave

import numpy as np
import sonara

passed = 0
failed = 0
errors = []


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


FILE_SR = 48000
SECONDS = 5
N = FILE_SR * SECONDS
FEATURES = ["ebur128", "native_dynamics", "amplitude_envelope"]
KEYS = {
    "ebur128": [
        "ebur128_integrated_lufs",
        "ebur128_momentary_max_lufs",
        "ebur128_short_term_max_lufs",
        "ebur128_range_lu",
        "ebur128_true_peak_dbtp",
    ],
    "native_dynamics": [
        "native_sample_rate",
        "sample_peak_dbfs",
        "crest_factor_db",
        "crest_factor_frame_mean_db",
    ],
    "amplitude_envelope": [
        "amplitude_envelope",
        "amplitude_envelope_hop_sec",
        "amplitude_envelope_frame_sec",
    ],
}
ALL_KEYS = [key for name in FEATURES for key in KEYS[name]]


def _write_wav(path, amplitude):
    """16-bit stereo 48 kHz, 1 kHz sine in phase on both channels."""
    tone = amplitude * np.sin(2 * np.pi * 1000.0 * np.arange(N) / FILE_SR)
    pcm = np.round(tone * 32768.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FILE_SR)
        w.writeframes(np.column_stack([pcm, pcm]).tobytes())


_dir = tempfile.mkdtemp(prefix="sonara_native_dynamics_")
atexit.register(shutil.rmtree, _dir, True)
WAV = os.path.join(_dir, "tone.wav")
SILENT_WAV = os.path.join(_dir, "silence.wav")
# EBU Tech 3341 case: a 1 kHz stereo sine peaking at -23 dBFS reads -23.0 LUFS.
_write_wav(WAV, 10.0 ** (-23.0 / 20.0))
_write_wav(SILENT_WAV, 0.0)


def test_absent_by_default():
    """Opt-in: compact and playlist leave these keys out, full measures a file."""
    for mode in ("compact", "playlist"):
        r = sonara.analyze_file(WAV, mode=mode)
        for key in ALL_KEYS:
            assert key not in r, f"{key} should be absent in mode={mode} (opt-in)"
    full = sonara.analyze_file(WAV, mode="full")
    missing = [key for key in ALL_KEYS + ["native_sample_rate"] if key not in full]
    assert not missing, f"mode=full misses {missing}"


def test_reference_tone_values():
    r = sonara.analyze_file(WAV, features=FEATURES)
    for key in ALL_KEYS:
        assert key in r, f"{key} missing"
    assert abs(r["ebur128_integrated_lufs"] + 23.0) < 0.1, r["ebur128_integrated_lufs"]
    for key in KEYS["ebur128"]:
        assert isinstance(r[key], float) and np.isfinite(r[key]), (key, r[key])
    assert r["ebur128_range_lu"] >= 0.0
    assert r["native_sample_rate"] == FILE_SR
    assert isinstance(r["native_sample_rate"], int)
    assert abs(r["sample_peak_dbfs"] + 23.0) < 0.05, r["sample_peak_dbfs"]
    assert abs(r["crest_factor_db"] - 3.01) < 0.05, r["crest_factor_db"]
    assert abs(r["crest_factor_frame_mean_db"] - 3.01) < 0.1, r["crest_factor_frame_mean_db"]


def test_envelope_grid():
    r = sonara.analyze_file(WAV, features=["amplitude_envelope"])
    envelope = r["amplitude_envelope"]
    assert isinstance(envelope, list) and isinstance(envelope[0], float)
    # One frame per 512/22050 s of the file, whatever its sample rate.
    assert len(envelope) == 1 + (N * 22050) // (512 * FILE_SR), len(envelope)
    assert max(abs(v - 10.0 ** (-23.0 / 20.0)) for v in envelope) < 1e-3
    assert abs(r["amplitude_envelope_hop_sec"] - 512 / 22050) < 1e-7
    assert abs(r["amplitude_envelope_frame_sec"] - 2048 / 22050) < 1e-7


def test_each_feature_emits_only_its_keys():
    for name in FEATURES:
        r = sonara.analyze_file(WAV, features=[name])
        present = {key for key in ALL_KEYS if key in r}
        # Any of the three marks the record as measured from a file.
        assert present == set(KEYS[name]) | {"native_sample_rate"}, (name, present)


def test_other_keys_unchanged():
    """Requesting the file-only features must not move any other value."""
    base = sonara.analyze_file(WAV, features=["rms", "loudness"])
    with_native = sonara.analyze_file(WAV, features=["rms", "loudness"] + FEATURES)
    skip = set(ALL_KEYS) | {"provenance"}
    assert set(with_native) - set(ALL_KEYS) == set(base)
    for key in set(base) - skip:
        assert np.all(with_native[key] == base[key]), key


def test_signal_never_has_keys():
    """A bare signal has no file channels or native rate."""
    t = np.arange(2 * 22050) / 22050
    y = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    r = sonara.analyze_signal(y, sr=22050, features=FEATURES)
    for key in ALL_KEYS:
        assert key not in r, f"{key} must be absent for analyze_signal"


def test_silence_omits_undefined_levels():
    r = sonara.analyze_file(SILENT_WAV, features=FEATURES)
    for key in KEYS["ebur128"] + KEYS["native_dynamics"][1:]:
        assert key not in r, f"{key} is undefined for digital silence"
    assert r["native_sample_rate"] == FILE_SR
    assert r["amplitude_envelope"] and not any(r["amplitude_envelope"])


def test_batch_matches_file():
    direct = sonara.analyze_file(WAV, features=FEATURES)
    (batch,) = sonara.analyze_batch([WAV], features=FEATURES)
    for key in ALL_KEYS:
        assert batch[key] == direct[key], key


def test_augment_adds_keys_from_audio():
    rec = sonara.analyze_file(WAV, mode="compact")
    assert not [key for key in ALL_KEYS if key in rec]
    for name in FEATURES:
        assert sonara.augment_blocker(rec, name) is not None, name
    aug = sonara.augment_analysis(rec, FEATURES, audio_path=WAV)
    direct = sonara.analyze_file(WAV, features=FEATURES)
    for key in ALL_KEYS:
        assert aug[key] == direct[key], key
    # Untouched compact fields keep their record values.
    assert aug["bpm"] == rec["bpm"]
    # The patched record round-trips through the cached-dict reader.
    again = sonara.augment_analysis(aug, [])
    for key in ALL_KEYS:
        assert again[key] == aug[key], key


TESTS = [
    ("file-only keys absent by default", test_absent_by_default),
    ("-23 dBFS stereo tone: LUFS, peak, crest", test_reference_tone_values),
    ("amplitude envelope grid and values", test_envelope_grid),
    ("each feature emits only its keys", test_each_feature_emits_only_its_keys),
    ("other keys unchanged by the request", test_other_keys_unchanged),
    ("analyze_signal never has the keys", test_signal_never_has_keys),
    ("silence omits undefined levels", test_silence_omits_undefined_levels),
    ("analyze_batch matches analyze_file", test_batch_matches_file),
    ("augment adds the keys from the audio", test_augment_adds_keys_from_audio),
]


if __name__ == "__main__":
    print("File-only loudness / dynamics")
    for name, fn in TESTS:
        check(name, fn)
    print(f"\n  RESULTS: {passed} PASSED, {failed} FAILED")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    sys.exit(1 if failed > 0 else 0)
