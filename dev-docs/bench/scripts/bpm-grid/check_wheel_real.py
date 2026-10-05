"""Check the Python binding end to end on the real tracks behind the rhythm fixtures.

    python check_wheel_real.py SPEC.tsv [--sonara-src]

SPEC.tsv is the make_fixtures.py spec (name, envelopes, audio, kind, label_bpm,
tolerance, note); its `audio` column holds the machine-local source files. Each track
goes through `sonara.analyze_file(path, features=["bpm"])` without a range and must
meet the same expectation as sonara/tests/bpm_accuracy.rs checks on its envelopes:
`integer` exactly round(label) octave-free, `fractional` not an integer and within
tolerance octave-free, `tempo` within tolerance in its own octave. `--sonara-src`
imports the package from the repository's python/ (an in-place `maturin develop`
build) instead of the installed one. Exit status 1 on any failure.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def fold(value: float, target: float) -> float:
    return value * 2.0 ** round(math.log2(target / value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--sonara-src", action="store_true")
    args = ap.parse_args()
    if args.sonara_src:
        sys.path.insert(0, str(REPO / "python"))
    import sonara

    with args.spec.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    failures = 0
    print(f"sonara {getattr(sonara, '__version__', '?')} from {Path(sonara.__file__).parent}")
    for row in rows:
        label, tolerance = float(row["label_bpm"]), float(row["tolerance"])
        result = sonara.analyze_file(row["audio"], features=["bpm"])
        tempo = float(result["bpm"])
        folded = fold(tempo, label)
        is_integer = abs(folded - round(folded)) < 1e-4
        ok = {
            "integer": is_integer and round(folded) == round(label),
            "fractional": not is_integer and abs(folded - label) <= tolerance,
            "not-integer": not is_integer,
            "tempo": abs(tempo - label) <= tolerance,
        }[row["kind"]]
        failures += not ok
        print(f"{row['name']:>16} {row['kind']:>11} label {label:>9.4f} bpm {tempo:>10.4f} "
              f"{'ok' if ok else 'FAIL'}")
    print(f"{len(rows) - failures} of {len(rows)} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
