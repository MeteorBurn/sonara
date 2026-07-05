#!/usr/bin/env python3
"""Tests for the opt-in acoustic fingerprint (duplicate detection).

Verifies the Python surface: the `fingerprint` / `fingerprint_version` dict
fields (present only when requested), base64 decodability, and that
`sonara.fingerprint_match` recognises same-recording variants while rejecting
unrelated audio — accepting both dicts and raw base64 strings.
"""

import base64
import sys

import numpy as np

import sonara

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


SR = 22050
DUR = 14.0  # long enough that unrelated recordings separate cleanly


def _hashf(a, b):
    h = (a + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h = (h + (b * 0x94D049BB133111EB)) & 0xFFFFFFFFFFFFFFFF
    return ((h >> 40) & 0xFFFFFF) / float(1 << 24)


def melody(dur, seed):
    """Broadband, time-varying test 'recording': 64 partials densely covering the
    300-2000 Hz fingerprint band, each driven by two slow amplitude LFOs. Models
    real music; distinct ``seed``s give genuinely uncorrelated recordings."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    n_part = 64
    y = np.zeros(n, dtype=np.float32)
    for p in range(n_part):
        f = 300.0 * (1950.0 / 300.0) ** (p / (n_part - 1))
        f *= 0.97 + 0.06 * _hashf(seed, p + 2000)
        w = 0.4 + 1.2 * _hashf(seed, p + 3000)
        r1 = 0.6 + 2.4 * _hashf(seed, p)
        r2 = 0.6 + 2.4 * _hashf(seed, p + 5000)
        ph1 = 2 * np.pi * _hashf(seed, p + 1000)
        ph2 = 2 * np.pi * _hashf(seed, p + 6000)
        amp = np.clip(
            0.35 + 0.35 * np.sin(2 * np.pi * r1 * t + ph1)
            + 0.3 * np.sin(2 * np.pi * r2 * t + ph2),
            0, None,
        )
        y += w * amp * np.sin(2 * np.pi * f * t + ph1)
    return (y / n_part).astype(np.float32)


def analyze_fp(y):
    return sonara.analyze_signal(y, sr=SR, features=["fingerprint"])


# ------------------------------------------------------------
print("--- Fingerprint field presence & opt-in ---")

_base = melody(DUR, 1)


def _field_present():
    r = analyze_fp(_base)
    assert "fingerprint" in r, "fingerprint field missing when requested"
    assert "fingerprint_version" in r, "fingerprint_version missing"
    assert isinstance(r["fingerprint"], str) and r["fingerprint"]
    assert isinstance(r["fingerprint_version"], int)


test("fingerprint present when requested", _field_present)


def _absent_by_default():
    # No mode may compute it by default — strictly opt-in.
    for mode in ("compact", "playlist", "full"):
        r = sonara.analyze_signal(_base, sr=SR, mode=mode)
        assert "fingerprint" not in r, f"fingerprint leaked into {mode} mode"
        assert "fingerprint_version" not in r


test("fingerprint absent unless requested (all modes)", _absent_by_default)


def _base64_decodes():
    r = analyze_fp(_base)
    raw = base64.b64decode(r["fingerprint"])
    assert len(raw) % 4 == 0, "fingerprint bytes must be a whole number of u32s"
    assert len(raw) // 4 > 20, "expected a non-trivial fingerprint"


test("fingerprint base64 decodes", _base64_decodes)

# ------------------------------------------------------------
print("\n--- Matching semantics ---")


def _self_match():
    r = analyze_fp(_base)
    assert abs(sonara.fingerprint_match(r, r) - 1.0) < 1e-6, "self-match must be 1.0"


test("self-match == 1.0", _self_match)


def _gain_match():
    a = analyze_fp(_base)
    b = analyze_fp((_base * 0.5).astype(np.float32))
    assert sonara.fingerprint_match(a, b) > 0.95, "0.5x gain should match high"


test("gain-variant match high", _gain_match)


def _different_low():
    a = analyze_fp(_base)
    b = analyze_fp(melody(DUR, 2))
    assert sonara.fingerprint_match(a, b) < 0.30, "different signals should score low"


test("different-signal match low", _different_low)


def _accepts_dicts_and_strings():
    a = analyze_fp(_base)
    b = analyze_fp((_base * 0.5).astype(np.float32))
    s_dicts = sonara.fingerprint_match(a, b)
    s_strs = sonara.fingerprint_match(a["fingerprint"], b["fingerprint"])
    s_mixed = sonara.fingerprint_match(a, b["fingerprint"])
    assert abs(s_dicts - s_strs) < 1e-6, "dict and string forms must agree"
    assert abs(s_dicts - s_mixed) < 1e-6, "mixed dict/string form must agree"


test("fingerprint_match accepts dicts and strings", _accepts_dicts_and_strings)

# ------------------------------------------------------------
print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")
if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")
sys.exit(1 if failed > 0 else 0)
