#!/usr/bin/env python3
"""Tests for opt-in structural segmentation & energy curve.

Structure is opt-in: it must be ABSENT from every default mode and PRESENT only
when requested via features=["structure"].
"""

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
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


sr = 22050
STRUCTURE_KEYS = [
    "energy_curve",
    "energy_curve_hop_sec",
    "segments",
    "intro_end_sec",
    "outro_start_sec",
    "energy_level",
]


def _seg(dur, loud):
    n = int(dur * sr)
    t = np.arange(n) / sr
    if loud:
        y = 0.5 * (np.sin(2 * np.pi * 200 * t)
                   + np.sin(2 * np.pi * 1500 * t)
                   + np.sin(2 * np.pi * 4000 * t))
    else:
        y = 0.04 * np.sin(2 * np.pi * 200 * t)
    return y.astype(np.float32)


# 30s quiet -> 60s loud/broadband -> 30s quiet: known low-high-low structure.
y_struct = np.concatenate([_seg(30, False), _seg(60, True), _seg(30, False)]).astype(np.float32)
duration = len(y_struct) / sr

print("=" * 70)
print("  Testing sonara structure (opt-in)")
print("=" * 70)

# ---- Opt-in / absence ----
print("\n--- Absent in default modes ---")

for mode in ("compact", "playlist", "full"):
    r = sonara.analyze_signal(y_struct[: 15 * sr], sr=sr, mode=mode)
    test(f"structure absent in {mode} mode",
         lambda r=r, mode=mode: (
             None if all(k not in r for k in STRUCTURE_KEYS)
             else (_ for _ in ()).throw(AssertionError(f"structure keys leaked into {mode}"))
         ))

# ---- Present when requested ----
print("\n--- Present with features=['structure'] ---")

r = sonara.analyze_signal(y_struct, sr=sr, features=["structure"])

test("all structure keys present",
     lambda: None if all(k in r for k in STRUCTURE_KEYS)
     else (_ for _ in ()).throw(AssertionError(f"missing: {[k for k in STRUCTURE_KEYS if k not in r]}")))

test("energy_curve is a non-empty list of floats in [0,1]",
     lambda: None if (len(r["energy_curve"]) > 0
                      and all(0.0 <= v <= 1.0 for v in r["energy_curve"]))
     else (_ for _ in ()).throw(AssertionError("bad energy_curve")))

test("energy_curve_hop_sec > 0",
     lambda: None if r["energy_curve_hop_sec"] > 0
     else (_ for _ in ()).throw(AssertionError("hop must be > 0")))

test("energy_level in 1..=10",
     lambda: None if (isinstance(r["energy_level"], int) and 1 <= r["energy_level"] <= 10)
     else (_ for _ in ()).throw(AssertionError(f"energy_level={r['energy_level']}")))

# ---- Segment invariants ----
print("\n--- Segment invariants ---")

segs = r["segments"]

test("segments is a list of dicts with the right keys",
     lambda: None if (isinstance(segs, list) and len(segs) >= 1
                      and all(set(s) >= {"start_sec", "end_sec", "energy"} for s in segs))
     else (_ for _ in ()).throw(AssertionError("bad segment dicts")))

test("first segment starts at ~0",
     lambda: None if abs(segs[0]["start_sec"]) < 0.5
     else (_ for _ in ()).throw(AssertionError(f"first start={segs[0]['start_sec']}")))

test("last segment ends at ~duration",
     lambda: None if abs(segs[-1]["end_sec"] - duration) < 1.0
     else (_ for _ in ()).throw(AssertionError(f"last end={segs[-1]['end_sec']} vs {duration}")))

test("segments ordered, non-overlapping, contiguous",
     lambda: None if all(
         segs[i]["end_sec"] > segs[i]["start_sec"]
         and abs(segs[i]["end_sec"] - segs[i + 1]["start_sec"]) < 0.05
         for i in range(len(segs) - 1)
     ) and segs[-1]["end_sec"] > segs[-1]["start_sec"]
     else (_ for _ in ()).throw(AssertionError("segments not contiguous/ordered")))

test("sane segment count (2..=12 for this structured signal)",
     lambda: None if 2 <= len(segs) <= 12
     else (_ for _ in ()).throw(AssertionError(f"segment count={len(segs)}")))

test("each segment energy in [0,1]",
     lambda: None if all(0.0 <= s["energy"] <= 1.0 for s in segs)
     else (_ for _ in ()).throw(AssertionError("segment energy out of range")))

# ---- Known structure recovered ----
print("\n--- Known structure recovered ---")

interior = [s["start_sec"] for s in segs[1:]]

test("boundary near 30s",
     lambda: None if any(abs(b - 30.0) < 8.0 for b in interior)
     else (_ for _ in ()).throw(AssertionError(f"no boundary near 30s: {interior}")))

test("boundary near 90s",
     lambda: None if any(abs(b - 90.0) < 8.0 for b in interior)
     else (_ for _ in ()).throw(AssertionError(f"no boundary near 90s: {interior}")))

test("intro_end in the first (quiet) region",
     lambda: None if r["intro_end_sec"] < 45.0
     else (_ for _ in ()).throw(AssertionError(f"intro_end={r['intro_end_sec']}")))

test("outro_start in the last (quiet) region",
     lambda: None if r["outro_start_sec"] > 80.0
     else (_ for _ in ()).throw(AssertionError(f"outro_start={r['outro_start_sec']}")))

test(".print() runs without error",
     lambda: r.print())

# ---- Summary ----
print(f"\n{'=' * 70}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {passed + failed} tests")
print(f"{'=' * 70}")
if errors:
    print("\nFailed tests:")
    for name, err in errors:
        print(f"  - {name}: {err}")

sys.exit(1 if failed > 0 else 0)
