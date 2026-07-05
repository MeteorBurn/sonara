#!/usr/bin/env python3
"""Test Camelot wheel key notation in analysis results.

Verifies that playlist/full analysis populates `key_camelot` alongside `key`,
and that the code is consistent with the detected key string.
"""

import sys

import numpy as np

import sonara

# Camelot wheel: (tonic_pitch_class, mode) -> code.
# Minor keys are the "A" ring, major keys the "B" ring.
_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}
_CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]
_CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]


def expected_camelot(key_str):
    tonic, mode = key_str.rsplit(" ", 1)
    pc = _PC[tonic]
    return _CAMELOT_MINOR[pc] if mode == "minor" else _CAMELOT_MAJOR[pc]


def make_signal():
    # A minor triad (A, C, E) so key detection has real tonal content.
    sr = 22050
    t = np.arange(3 * sr) / sr
    y = np.zeros_like(t, dtype=np.float32)
    for f in (220.0, 261.63, 329.63):  # A3, C4, E4
        y += np.sin(2 * np.pi * f * t).astype(np.float32)
    return (y / 3.0).astype(np.float32), sr


def main():
    y, sr = make_signal()

    failures = []

    def check(cond, msg):
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    for mode in ("playlist", "full"):
        r = sonara.analyze_signal(y, sr=sr, mode=mode)
        check("key" in r, f"[{mode}] 'key' present")
        check("key_camelot" in r, f"[{mode}] 'key_camelot' present")
        if "key" in r and "key_camelot" in r:
            exp = expected_camelot(r["key"])
            check(
                r["key_camelot"] == exp,
                f"[{mode}] key_camelot {r['key_camelot']!r} consistent with key "
                f"{r['key']!r} (expected {exp!r})",
            )
            code = r["key_camelot"]
            check(
                code[-1] in ("A", "B") and code[:-1].isdigit() and 1 <= int(code[:-1]) <= 12,
                f"[{mode}] key_camelot {code!r} is a valid wheel code",
            )

    # Compact mode does not compute key; key_camelot must be absent too.
    r_compact = sonara.analyze_signal(y, sr=sr, mode="compact")
    check("key_camelot" not in r_compact, "[compact] 'key_camelot' absent when key not computed")

    print(f"\n{'PASS' if not failures else 'FAIL'}: "
          f"{'all Camelot checks passed' if not failures else 'Camelot checks FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
