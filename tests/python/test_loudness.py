#!/usr/bin/env python3
"""Tests for the opt-in loudness / gain metrics.

Covers the `features=["loudness"]` group: true peak (dBTP), ReplayGain-style
track gain, short-term loudness curve, momentary max, and EBU R128 loudness
range (LRA). Uses only synthetic signals — no audio files required.

Runs standalone (`python test_loudness.py`) or under pytest.
"""

import sys
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


sr = 22050
# A 10 s multi-tone signal comfortably below full scale (never clips).
t = np.arange(10 * sr) / sr
y = (
    0.3 * np.sin(2 * np.pi * 440.0 * t)
    + 0.2 * np.sin(2 * np.pi * 554.37 * t)
    + 0.15 * np.sin(2 * np.pi * 220.0 * t)
).astype(np.float32)

LOUDNESS_FIELDS = [
    "true_peak_db",
    "replaygain_db",
    "loudness_curve",
    "loudness_momentary_max_db",
    "loudness_range_lu",
]


def test_absent_by_default():
    """Opt-in: loudness fields must NOT appear in any default mode."""
    for mode in ("compact", "playlist", "full"):
        r = sonara.analyze_signal(y, sr=sr, mode=mode)
        for f in LOUDNESS_FIELDS:
            assert f not in r, f"{f} should be absent in mode={mode} (opt-in)"


def test_present_when_requested():
    """features=['loudness'] populates the whole group."""
    r = sonara.analyze_signal(y, sr=sr, features=["loudness"])
    for f in LOUDNESS_FIELDS:
        assert f in r, f"{f} missing when features=['loudness']"


def test_replaygain_formula():
    r = sonara.analyze_signal(y, sr=sr, features=["loudness"])
    assert abs(r["replaygain_db"] - (-18.0 - r["loudness_lufs"])) < 1e-3


def test_true_peak_ge_sample_peak():
    """True peak (dBTP) must be >= the raw sample peak in dB."""
    r = sonara.analyze_signal(y, sr=sr, features=["loudness"])
    sample_peak = float(np.max(np.abs(y)))
    sample_peak_db = 20.0 * np.log10(sample_peak)
    assert r["true_peak_db"] >= sample_peak_db - 1e-3, (
        f"true_peak {r['true_peak_db']} < sample peak {sample_peak_db}"
    )


def test_curve_floats_and_plausible():
    r = sonara.analyze_signal(y, sr=sr, features=["loudness"])
    curve = r["loudness_curve"]
    assert isinstance(curve, list) and len(curve) > 0
    # floor((10 - 3) / 1) + 1 = 8 windows.
    assert len(curve) == 8, f"expected 8 windows, got {len(curve)}"
    for v in curve:
        assert isinstance(v, float)
        assert np.isfinite(v)
        # Non-clipping signal well below 0 dBFS -> short-term loudness < 0 LUFS.
        assert v < 0.0, f"curve value {v} unexpectedly >= 0"


def test_range_and_momentary_finite():
    r = sonara.analyze_signal(y, sr=sr, features=["loudness"])
    assert np.isfinite(r["loudness_range_lu"])
    assert r["loudness_range_lu"] >= 0.0
    assert np.isfinite(r["loudness_momentary_max_db"])
    # Steady signal: momentary max close to integrated loudness.
    assert r["loudness_momentary_max_db"] < 0.0


def test_loudness_lufs_unchanged():
    """Requesting loudness must not change the always-on integrated value."""
    base = sonara.analyze_signal(y, sr=sr, mode="compact")["loudness_lufs"]
    withl = sonara.analyze_signal(y, sr=sr, features=["loudness"])["loudness_lufs"]
    assert abs(base - withl) < 1e-6


TESTS = [
    ("loudness fields absent by default", test_absent_by_default),
    ("loudness fields present when requested", test_present_when_requested),
    ("replaygain_db == -18 - loudness_lufs", test_replaygain_formula),
    ("true_peak_db >= sample peak (dB)", test_true_peak_ge_sample_peak),
    ("loudness_curve floats & plausible", test_curve_floats_and_plausible),
    ("LRA & momentary finite", test_range_and_momentary_finite),
    ("loudness_lufs unchanged by opt-in", test_loudness_lufs_unchanged),
]


if __name__ == "__main__":
    print("Loudness / gain metrics")
    for name, fn in TESTS:
        check(name, fn)
    print(f"\n  RESULTS: {passed} PASSED, {failed} FAILED")
    if errors:
        for name, err in errors:
            print(f"  - {name}: {err}")
    sys.exit(1 if failed > 0 else 0)
