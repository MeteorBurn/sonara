#!/usr/bin/env python3
"""Tests for the loudness / dynamics features measured on the channels.

Covers `features=["loudness", "native_dynamics", "amplitude_envelope"]`: EBU R128
loudness, sample peak, crest factor and the peak envelope, measured on the
input's own channels and sample rate, for a file and for a signal alike.
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
FEATURES = ["loudness", "native_dynamics", "amplitude_envelope"]
KEYS = {
    "loudness": [
        "true_peak_db",
        "replaygain_db",
        "loudness_momentary_max_db",
        "loudness_short_term_max_db",
        "loudness_range_lu",
        "loudness_curve",
    ],
    "native_dynamics": [
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


def _pcm(amplitude):
    """16-bit 1 kHz sine at 48 kHz, one channel."""
    tone = amplitude * np.sin(2 * np.pi * 1000.0 * np.arange(N) / FILE_SR)
    return np.round(tone * 32768.0).astype("<i2")


def _write_wav(path, amplitude):
    """16-bit stereo 48 kHz, the sine in phase on both channels."""
    pcm = _pcm(amplitude)
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
AMPLITUDE = 10.0 ** (-23.0 / 20.0)
_write_wav(WAV, AMPLITUDE)
_write_wav(SILENT_WAV, 0.0)


def test_absent_by_default():
    """Opt-in: compact and playlist leave these keys out, full measures them."""
    for mode in ("compact", "playlist"):
        r = sonara.analyze_file(WAV, mode=mode)
        for key in ALL_KEYS:
            assert key not in r, f"{key} should be absent in mode={mode} (opt-in)"
        # What the channel features were measured on is recorded in every mode.
        assert (r["native_sample_rate"], r["native_channels"]) == (FILE_SR, 2)
    full = sonara.analyze_file(WAV, mode="full")
    missing = [key for key in ALL_KEYS if key not in full]
    assert not missing, f"mode=full misses {missing}"


def test_reference_tone_values():
    r = sonara.analyze_file(WAV, features=FEATURES)
    for key in ALL_KEYS:
        assert key in r, f"{key} missing"
    assert abs(r["loudness_lufs"] + 23.0) < 0.1, r["loudness_lufs"]
    for key in KEYS["loudness"][:-1]:
        assert isinstance(r[key], float) and np.isfinite(r[key]), (key, r[key])
    assert r["loudness_range_lu"] >= 0.0
    # 3 s windows at a 1 s hop over 5 s.
    assert len(r["loudness_curve"]) == SECONDS - 2, r["loudness_curve"]
    assert r["native_sample_rate"] == FILE_SR
    assert isinstance(r["native_sample_rate"], int)
    assert r["native_channels"] == 2 and isinstance(r["native_channels"], int)
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
        assert present == set(KEYS[name]), (name, present)


def test_other_keys_unchanged():
    """Requesting the channel features must not move any other value."""
    base = sonara.analyze_file(WAV, features=["rms"])
    with_native = sonara.analyze_file(WAV, features=["rms"] + FEATURES)
    skip = set(ALL_KEYS) | {"provenance"}
    assert set(with_native) - set(ALL_KEYS) == set(base)
    for key in set(base) - skip:
        assert np.all(with_native[key] == base[key]), key


def _stereo_signal():
    """The samples of `WAV` as a float32 array shaped (channels, samples)."""
    channel = (_pcm(AMPLITUDE) / 32768.0).astype(np.float32)
    return np.stack([channel, channel])


def test_stereo_signal_is_measured_on_its_channels():
    """A (channels, samples) signal gives what the file with those channels gives."""
    features = FEATURES + ["stereo_field"]
    r = sonara.analyze_signal(_stereo_signal(), sr=FILE_SR, target_sr=22050, features=features)
    assert r["native_channels"] == 2
    assert r["native_sample_rate"] == FILE_SR
    assert r["provenance"]["sample_rate"] == 22050
    # Two channels read -23 LUFS; their mono mix alone would read -26.
    assert abs(r["loudness_lufs"] + 23.0) < 0.1, r["loudness_lufs"]
    from_file = sonara.analyze_file(WAV, sr=22050, features=features)
    for key in ALL_KEYS + ["loudness_lufs", "stereo_side_ratio", "stereo_correlation_mean"]:
        assert r[key] == from_file[key], key


def test_samples_by_channels_signal_is_rejected():
    """(samples, channels), the layout of other libraries, is an error, not garbage."""
    try:
        sonara.analyze_signal(_stereo_signal().T, sr=FILE_SR)
    except ValueError as e:
        assert "(channels, samples)" in str(e), e
    else:
        raise AssertionError("a (samples, channels) array was accepted")
    # An empty signal is reported as empty, not as one to transpose.
    for empty in (np.zeros(0, dtype=np.float32), np.zeros((2, 0), dtype=np.float32)):
        try:
            sonara.analyze_signal(empty, sr=FILE_SR)
        except ValueError as e:
            assert "at least one sample" in str(e), e
        else:
            raise AssertionError("an empty signal was accepted")


def test_silence_omits_undefined_levels():
    r = sonara.analyze_file(SILENT_WAV, features=FEATURES)
    for key in KEYS["loudness"][:-1] + KEYS["native_dynamics"]:
        assert key not in r, f"{key} is undefined for digital silence"
    # The always-on loudness and the curve sit at the absolute gate.
    assert r["loudness_lufs"] == -70.0
    assert r["loudness_curve"] == [-70.0] * (SECONDS - 2)
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
    ("channel groups absent by default", test_absent_by_default),
    ("-23 dBFS stereo tone: LUFS, peak, crest", test_reference_tone_values),
    ("amplitude envelope grid and values", test_envelope_grid),
    ("each feature emits only its keys", test_each_feature_emits_only_its_keys),
    ("other keys unchanged by the request", test_other_keys_unchanged),
    ("stereo signal is measured on its channels", test_stereo_signal_is_measured_on_its_channels),
    ("(samples, channels) signal is rejected", test_samples_by_channels_signal_is_rejected),
    ("silence omits undefined levels", test_silence_omits_undefined_levels),
    ("analyze_batch matches analyze_file", test_batch_matches_file),
    ("augment adds the keys from the audio", test_augment_adds_keys_from_audio),
]


if __name__ == "__main__":
    print("Loudness / dynamics on the channels")
    for name, fn in TESTS:
        check(name, fn)
    print(f"\n  RESULTS: {passed} PASSED, {failed} FAILED")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    sys.exit(1 if failed > 0 else 0)
