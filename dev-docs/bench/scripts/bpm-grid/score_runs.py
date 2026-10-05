"""Score bpm-grid runner TSVs against the bench references (bpm-precision metrics).

    python score_runs.py --run broken=A.tsv --run straight=B.tsv
                         [--before broken=C.tsv --before straight=D.tsv]
                         [--beats] [--json OUT.json] [--lists N] [--parity even|odd]

Read-only. A run is the runner's default TSV (idx, status, bpm, bpm_raw, conf, cands,
beats, downbeats; frames at hop 512, sr 22050). References come from the newest
data/bpm_comparison_2000*.json (MIK, Rekordbox, Beat This! grid, consensus and class
targets do not depend on the Sonara revision); --beats also reads the Beat This! beats
from data/4_beat_this/beat_this-<set>.sqlite. Uses Python with numpy (beat_this_python).

Metrics per set and for both sets (`fold(v, t)` = v times the power of 2 closest to t):
  precision     consensus tracks: |fold(bpm, M)/M - 1| <= 0.1%; abs error |fold - M| as
                MedAE, p90, p99 (M = MIK, the consensus target)
  integer       integer refs: MIK within 0.005 BPM of an integer and Rekordbox equal to
                it; exact = fold(bpm, MIK) is that integer (|.| < 1e-4: the lock acts at
                the tracked level, before the octave fold); fractional refs: MIK more
                than 0.005 off an integer; false lock = fold(bpm, MIK) an exact integer
  2 decimals    round(fold(bpm, X), 2) == round(X, 2) for X = MIK and X = Rekordbox
  class         class target C: x = fold(bpm, C)/C: correct |x - 1| <= 3%, else x3/4
                (= x3/2), x4/3 (= x2/3), x5/4 or x4/5 within 3%, else major
  octave        class-correct consensus tracks: k = round(log2(bpm/MIK)), and the same
                against Rekordbox; bpm < 60 and bpm > 200 over all tracks
  beats         (--beats) F-measure +-70 ms against Beat This! beats (first 5 s
                skipped), and the beat phase class where bpm is at Beat This!'s level
Before/after (--before): rows with identical bpm, beats and downbeats; bpm moved by more
than 0.1% octave-free; class, precision and integer-lock changes with their idx.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpm_grid_paths as P  # noqa: E402

FRAME_SEC = 512 / 22050
PREC_REL = 0.001
CLASS_TOL = 0.03
INT_REF_TOL = 0.005
EXACT_INT = 1e-4
CLASS_MULTIPLES = ((0.75, "x3/4"), (4 / 3, "x4/3"), (1.25, "x5/4"), (0.8, "x4/5"))
SETS = ("broken", "straight")


def fold(value: float, target: float) -> float:
    return value * 2.0 ** round(math.log2(target / value))


def tempo_class(value: float, ref: float) -> str:
    ratio = fold(value, ref) / ref
    if abs(ratio - 1) <= CLASS_TOL:
        return "correct"
    for multiple, label in CLASS_MULTIPLES:
        if abs(ratio / multiple - 1) <= CLASS_TOL:
            return label
    return "major"


def is_int(value: float) -> bool:
    return abs(value - round(value)) < EXACT_INT


def read_run(path: Path) -> dict[int, dict]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(r["idx"]): r for r in csv.DictReader(handle, delimiter="\t")}


def references() -> dict[tuple[str, int], dict]:
    newest = max(P.DATA.glob("bpm_comparison_2000*.json"), key=lambda p: p.stat().st_mtime)
    rows = json.loads(newest.read_text(encoding="utf-8"))["rows"]
    return {(r["set"], r["idx"]): r for r in rows}


def beat_this(set_name: str) -> dict[int, list[float]]:
    import compare_all as C  # numpy; imported only for --beats

    rows = C.db_rows(P.BEAT_THIS_DIR / f"beat_this-{set_name}.sqlite", ("status",), ("beats_sec",))
    return {idx: r.get("beats_sec") for idx, r in rows.items()}


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def score_set(set_name: str, run: dict[int, dict], refs, bt_beats=None, parity=None) -> tuple[Counter, dict, list]:
    c: Counter = Counter()
    per_track: dict[int, dict] = {}
    errors: list[float] = []
    if bt_beats is not None:
        import numpy as np
        import compare_all as C
    for idx, row in sorted(run.items()):
        if parity is not None and idx % 2 != parity:
            continue
        ref = refs[(set_name, idx)]
        c["tracks"] += 1
        if row["status"] != "ok":
            c["decode_errors"] += 1
            continue
        bpm = float(row["bpm"])
        mik, rb = ref["mik_bpm"], ref["rb_bpm"]
        # integer-ness octave-free: the lock acts at the tracked level, before the octave fold
        t: dict = {"bpm": bpm, "int": is_int(fold(bpm, mik) if mik else bpm)}
        c["bpm_lt_60"] += bpm < 60
        c["bpm_gt_200"] += bpm > 200
        if ref["class_ref_bpm"]:
            t["class"] = tempo_class(bpm, ref["class_ref_bpm"])
            c["class_n"] += 1
            c[f"class_{t['class']}"] += 1
        if ref["consensus"]:
            target = ref["target_bpm"]
            folded = fold(bpm, target)
            t["prec"] = abs(folded / target - 1) <= PREC_REL + 1e-12
            c["prec_n"] += 1
            c["prec_ok"] += t["prec"]
            errors.append(abs(folded - target))
            c["dec2_mik"] += round(fold(bpm, mik), 2) == round(mik, 2)
            c["dec2_rb"] += round(fold(bpm, rb), 2) == round(rb, 2)
            off = abs(mik - round(mik))
            if off <= INT_REF_TOL and abs(rb - round(mik)) < 1e-6:
                c["int_ref"] += 1
                exact = abs(fold(bpm, mik) - round(mik)) < EXACT_INT
                c["int_exact"] += exact
                t["int_ref"] = "exact" if exact else "missed"
            elif off > INT_REF_TOL:
                c["frac_ref"] += 1
                c["false_lock"] += t["int"]
                t["int_ref"] = "false_lock" if t["int"] else "frac_ok"
            if t.get("class") == "correct":
                c["oct_n"] += 1
                k_mik = round(math.log2(bpm / mik))
                k_rb = round(math.log2(bpm / rb))
                t["oct_mik"] = k_mik
                c[f"oct_mik_{k_mik:+d}"] += 1
                c["oct_rb_ok"] += k_rb == 0
        if bt_beats is not None:
            ref_beats = bt_beats.get(idx)
            if row["beats"] and ref_beats is not None and len(ref_beats):
                est = np.array([int(x) for x in row["beats"].split(",")], dtype=float) * FRAME_SEC
                f, _ = C.f_measure(est, np.asarray(ref_beats))
                if f is not None:
                    c["beat_f_n"] += 1
                    c["beat_f_sum"] += f
                bt_bpm = 60.0 / float(np.median(np.diff(ref_beats))) if len(ref_beats) > 8 else None
                if bt_bpm and abs(bpm / bt_bpm - 1) <= CLASS_TOL:
                    phase = C.phase_class(C.beat_phase(est, np.asarray(ref_beats)))
                    c[f"phase_{phase}"] += 1
        per_track[idx] = t
    return c, per_track, errors


def summary(c: Counter, errors: list[float]) -> dict:
    pct = lambda a, b: round(100.0 * a / b, 2) if b else None  # noqa: E731
    return {
        "tracks": c["tracks"], "decode_errors": c["decode_errors"],
        "prec_pct": pct(c["prec_ok"], c["prec_n"]), "prec_n": c["prec_n"],
        "medae": quantile(errors, 0.5), "p90": quantile(errors, 0.9), "p99": quantile(errors, 0.99),
        "int_ref": c["int_ref"], "int_exact_pct": pct(c["int_exact"], c["int_ref"]),
        "int_missed": c["int_ref"] - c["int_exact"],
        "frac_ref": c["frac_ref"], "false_lock": c["false_lock"],
        "dec2_mik_pct": pct(c["dec2_mik"], c["prec_n"]), "dec2_rb_pct": pct(c["dec2_rb"], c["prec_n"]),
        "class_n": c["class_n"],
        "class": {k[6:]: v for k, v in c.items() if k.startswith("class_") and k != "class_n"},
        "oct_n": c["oct_n"], "oct_mik_ok_pct": pct(c["oct_mik_+0"], c["oct_n"]),
        "oct_mik": {k[8:]: v for k, v in sorted(c.items()) if k.startswith("oct_mik_")},
        "oct_rb_ok_pct": pct(c["oct_rb_ok"], c["oct_n"]),
        "bpm_lt_60": c["bpm_lt_60"], "bpm_gt_200": c["bpm_gt_200"],
        "beat_f": round(c["beat_f_sum"] / c["beat_f_n"], 4) if c["beat_f_n"] else None,
        "phase": {k[6:]: v for k, v in c.items() if k.startswith("phase_")} or None,
    }


def compare(set_name, before: dict[int, dict], after: dict[int, dict], tb, ta) -> dict:
    same = moved = 0
    changes: dict[str, list[int]] = {k: [] for k in (
        "class_fixed", "class_broken", "prec_fixed", "prec_broken", "int_locked", "int_unlocked",
        "moved_0.1pct", "beats_changed")}
    for idx, a in after.items():
        if idx not in ta:
            continue
        b = before.get(idx)
        if not b or a["status"] != "ok" or b["status"] != "ok":
            continue
        if (a["bpm"], a["beats"], a["downbeats"]) == (b["bpm"], b["beats"], b["downbeats"]):
            same += 1
        if a["beats"] != b["beats"]:
            changes["beats_changed"].append(idx)
        ba, bb = float(a["bpm"]), float(b["bpm"])
        if abs(fold(ba, bb) / bb - 1) > PREC_REL:
            moved += 1
            changes["moved_0.1pct"].append(idx)
        x, y = tb.get(idx, {}), ta.get(idx, {})
        if "class" in x and "class" in y and (x["class"] == "correct") != (y["class"] == "correct"):
            changes["class_fixed" if y["class"] == "correct" else "class_broken"].append(idx)
        if "prec" in x and "prec" in y and x["prec"] != y["prec"]:
            changes["prec_fixed" if y["prec"] else "prec_broken"].append(idx)
        if x.get("int") != y.get("int"):
            changes["int_locked" if y.get("int") else "int_unlocked"].append(idx)
    return {"identical_rows": same, "moved": moved, **{k: v for k, v in changes.items()}}


def parse_pairs(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values or []:
        name, _, path = value.partition("=")
        if name not in SETS or not path:
            raise SystemExit(f"expected <broken|straight>=<tsv>, got {value!r}")
        out[name] = Path(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="<set>=<runner tsv>")
    ap.add_argument("--before", action="append", help="<set>=<runner tsv> of the base revision")
    ap.add_argument("--beats", action="store_true", help="beat F-measure and phase against Beat This!")
    ap.add_argument("--json", type=Path, help="write the full result as JSON")
    ap.add_argument("--lists", type=int, default=25, help="idx shown per change list")
    ap.add_argument("--parity", choices=("even", "odd"), help="score only even or odd idx (tune / check split)")
    args = ap.parse_args()
    runs, befores = parse_pairs(args.run), parse_pairs(args.before)
    parity = None if args.parity is None else int(args.parity == "odd")
    refs = references()
    result: dict = {"sets": {}, "before_after": {}}
    totals: Counter = Counter()
    all_errors: list[float] = []
    for set_name, path in runs.items():
        bt = beat_this(set_name) if args.beats else None
        run = read_run(path)
        c, per_track, errors = score_set(set_name, run, refs, bt, parity)
        totals.update(c)
        all_errors += errors
        result["sets"][set_name] = summary(c, errors)
        if set_name in befores:
            before = read_run(befores[set_name])
            _, per_before, _ = score_set(set_name, before, refs, parity=parity)
            result["before_after"][set_name] = compare(set_name, before, run, per_before, per_track)
    if len(runs) > 1:
        result["sets"]["all"] = summary(totals, all_errors)
    for name, s in result["sets"].items():
        print(f"== {name}: {s['tracks']} tracks, {s['decode_errors']} decode errors")
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"   0.1%: {s['prec_pct']}% of {s['prec_n']} | MedAE {fmt(s['medae'])} p90 {fmt(s['p90'])} "
              f"p99 {fmt(s['p99'])} BPM")
        print(f"   integer refs {s['int_ref']}: exact {s['int_exact_pct']}% (missed {s['int_missed']}) | "
              f"fractional refs {s['frac_ref']}: false locks {s['false_lock']} | "
              f"2 decimals MIK {s['dec2_mik_pct']}% RB {s['dec2_rb_pct']}%")
        print(f"   class ({s['class_n']}): {s['class']}")
        print(f"   octave (correct class, {s['oct_n']}): =MIK {s['oct_mik_ok_pct']}% {s['oct_mik']} =RB "
              f"{s['oct_rb_ok_pct']}% | bpm<60 {s['bpm_lt_60']} bpm>200 {s['bpm_gt_200']}")
        if s["beat_f"] is not None:
            print(f"   beats vs Beat This!: F {s['beat_f']} phase {s['phase']}")
    for name, d in result["before_after"].items():
        print(f"-- before/after {name}: identical rows {d['identical_rows']}, bpm moved >0.1% {d['moved']}")
        for key in ("class_fixed", "class_broken", "prec_fixed", "prec_broken", "int_locked",
                    "int_unlocked", "moved_0.1pct", "beats_changed"):
            if d[key]:
                print(f"   {key} {len(d[key])}: {d[key][:args.lists]}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
