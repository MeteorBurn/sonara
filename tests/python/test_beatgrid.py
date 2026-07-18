#!/usr/bin/env python3
"""Tests for the opt-in beat-grid feature.

Verifies that:
  - the three beat-grid keys are ABSENT in the default modes (opt-in),
  - they are PRESENT when requested via features=["beatgrid"],
  - downbeats are a subset of beats, roughly one per bar,
  - grid_offset_sec is non-negative and grid_stability is in [0, 1].
"""

import sys
import numpy as np

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


import sonara

# ============================================================
# Synthetic kick-accented 4/4 click train (120 BPM)
# ============================================================
sr = 22050
dur_s = 8
bpm = 120.0
interval = int(round(60.0 / bpm * sr))  # samples per beat
y = np.zeros(dur_s * sr, dtype=np.float32)
beat_idx = 0
pos = 0
while pos < len(y):
    # Loud kick (low sine) on beat 1 of each bar, quieter click elsewhere.
    if beat_idx % 4 == 0:
        amp, freq = 1.0, 80.0
    else:
        amp, freq = 0.3, 1500.0
    n = min(200, len(y) - pos)
    y[pos:pos + n] += (amp * np.sin(2 * np.pi * freq * np.arange(n) / sr)).astype(np.float32)
    pos += interval
    beat_idx += 1

GRID_KEYS = ("grid_offset_sec", "downbeats", "grid_stability")

print("=" * 70)
print("  Testing sonara beat grid (opt-in)")
print("=" * 70)

# ============================================================
# Opt-in: absent by default in every mode
# ============================================================
print("\n--- Default modes must NOT contain beat-grid keys ---")

for mode in ("compact", "playlist", "full"):
    r = sonara.analyze_signal(y, sr=sr, mode=mode)
    test(
        f"{mode}: no beat-grid keys",
        lambda r=r: (
            None
            if all(k not in r for k in GRID_KEYS)
            else (_ for _ in ()).throw(AssertionError(
                f"unexpected keys: {[k for k in GRID_KEYS if k in r]}"))
        ),
    )

# ============================================================
# Present when requested
# ============================================================
print("\n--- features=['beatgrid'] populates the keys ---")

r = sonara.analyze_signal(y, sr=sr, mode="playlist", features=["beatgrid"])

for k in GRID_KEYS:
    test(f"key present: {k}", lambda k=k: (
        None if k in r else (_ for _ in ()).throw(AssertionError(f"{k} missing"))
    ))

# Also works without a mode (pure feature request).
r_compact = sonara.analyze_signal(y, sr=sr, features=["beatgrid"])
test("feature-only request populates keys",
     lambda: (None if all(k in r_compact for k in GRID_KEYS)
              else (_ for _ in ()).throw(AssertionError("keys missing for feature-only request"))))

# ============================================================
# Value / relationship checks
# ============================================================
print("\n--- Value checks ---")

beats = r["beats"]
downbeats = r["downbeats"]

test("grid_offset_sec >= 0",
     lambda: (None if r["grid_offset_sec"] >= 0.0
              else (_ for _ in ()).throw(AssertionError(f"offset {r['grid_offset_sec']} < 0"))))

test("grid_stability in [0, 1]",
     lambda: (None if 0.0 <= r["grid_stability"] <= 1.0
              else (_ for _ in ()).throw(AssertionError(f"stability {r['grid_stability']} out of range"))))

test("regular click train has high stability",
     lambda: (None if r["grid_stability"] > 0.8
              else (_ for _ in ()).throw(AssertionError(f"stability {r['grid_stability']} too low for a steady grid"))))

test("downbeats are a subset of beats",
     lambda: (None if set(downbeats).issubset(set(beats))
              else (_ for _ in ()).throw(AssertionError("downbeats not a subset of beats"))))

test("downbeats non-empty",
     lambda: (None if len(downbeats) > 0
              else (_ for _ in ()).throw(AssertionError("no downbeats found"))))


def _check_downbeat_count():
    expected = len(beats) / 4.0
    # ~one downbeat per bar; allow generous tolerance for edge bars.
    if abs(len(downbeats) - expected) > max(1.0, 0.5 * expected):
        raise AssertionError(
            f"len(downbeats)={len(downbeats)} not ~ len(beats)/4={expected:.1f}")


test("len(downbeats) ~= len(beats)/4", _check_downbeat_count)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'='*70}")

if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

sys.exit(1 if failed > 0 else 0)
