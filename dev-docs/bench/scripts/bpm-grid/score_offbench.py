"""Score runner TSVs on the off-bench set (offbench.py) against Beat This! and tag BPM.

    python score_offbench.py --run A.tsv [--before B.tsv] [--json OUT.json] [--list]

Read-only; needs numpy (beat_this_python). The set has no MIK / Rekordbox values, so
the judges are Beat This! (beats, and the tempo of the constant grid that best fits
them) and the library's tag BPM (weak: often MIK's, often an octave off for fast
tracks). Per stratum of selection.csv:
  class         x = fold(bpm, R)/R against R = Beat This! level tempo (60 / median beat
                interval) and against the tag: correct within 3%, else x3/4 (= x3/2),
                x4/3 (= x2/3), x5/4, x4/5 or major (octave-free)
  octave        tracks of correct class: k = round(log2(bpm / R)), R = Beat This! level
  precision     tracks whose Beat This! beats fit one grid (R(T) >= 0.8): |fold(bpm, G)/G
                - 1| <= 0.1%, G = the grid tempo; integer: G within 0.005 of an integer
                and fit >= 0.9 -> Sonara exact integer there; fractional G (> 0.02 off)
                with Sonara an exact integer -> false lock
  beats         F-measure +-70 ms against Beat This! (first 5 s skipped), per track and
                per quarter of the track (drift: does the phase hold in every quarter)
Before/after: identical rows and the idx whose class, octave or integer lock changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_all as C  # noqa: E402
import offbench as O  # noqa: E402

FRAME_SEC = 512 / 22050
CLASS_TOL = 0.03
FIT_STEADY = 0.8
FIT_INTEGER = 0.9


def fold(value: float, target: float) -> float:
    return value * 2.0 ** round(math.log2(target / value))


def tempo_class(value: float, ref: float) -> str:
    ratio = fold(value, ref) / ref
    if abs(ratio - 1) <= CLASS_TOL:
        return "correct"
    for multiple, label in ((0.75, "x3/4"), (4 / 3, "x4/3"), (1.25, "x5/4"), (0.8, "x4/5")):
        if abs(ratio / multiple - 1) <= CLASS_TOL:
            return label
    return "major"


def read_tsv(path: Path) -> dict[int, dict]:
    csv.field_size_limit(10**9)
    with path.open(encoding="utf-8", newline="") as handle:
        return {int(r["idx"]): r for r in csv.DictReader(handle, delimiter="\t")}


def references() -> dict[int, dict]:
    with (O.DATASET / "selection.csv").open(encoding="utf-8", newline="") as handle:
        sel = {int(r["idx"]): r for r in csv.DictReader(handle)}
    bt_dir = O.DATASET / "beat_this"
    refs = {}
    for idx, row in sel.items():
        files = list(bt_dir.glob(f"{idx:04d} - *.json"))
        bt = json.loads(files[0].read_text(encoding="utf-8")) if files else {}
        beats = np.asarray(bt.get("beats_sec") or [], dtype=float)
        level = bt.get("bpm_raw")
        grid, fit = C.grid_bpm(beats, level) if level and len(beats) else (None, None)
        refs[idx] = {"stratum": row["stratum"], "label": row["label"], "path": row["path"],
                     "tag": float(row["tag_bpm"]) if row["tag_bpm"] else None,
                     "bt_level": level, "bt_grid": grid, "bt_fit": fit, "bt_beats": beats,
                     "duration": bt.get("duration_sec")}
    return refs


def bt_reference(ref: dict) -> float | None:
    """Beat This!'s tempo: its grid tempo where its beats fit one grid (R >= 0.5),
    else its level tempo (60 / median beat interval, 20 ms steps)."""
    if ref["bt_grid"] and ref["bt_fit"] is not None and ref["bt_fit"] >= 0.5:
        return ref["bt_grid"]
    return ref["bt_level"]


def verdict(t: dict) -> str:
    """Who disagrees on the tempo class and octave: a hint for listening, not a label."""
    bt_ok = t.get("class_bt") == "correct" and t.get("oct_bt") == 0
    tag_ok = t.get("class_tag") == "correct" and t.get("oct_tag") == 0
    if "class_tag" not in t:
        return "agrees with Beat This!" if bt_ok else "against Beat This! (no tag)"
    if bt_ok and tag_ok:
        return "agrees with both"
    if t.get("judges_agree"):
        return "agrees with neither; judges agree on class" if not (bt_ok or tag_ok) else (
            "octave differs from one judge" if t.get("class_bt") == "correct" else "class against both")
    if bt_ok:
        return "agrees with Beat This!, not the tag"
    if tag_ok:
        return "agrees with the tag, not Beat This!"
    return "judges disagree, Sonara with neither"


def quarters_f(est: np.ndarray, ref: np.ndarray, duration: float) -> list[float | None]:
    out = []
    for q in range(4):
        lo, hi = duration * q / 4, duration * (q + 1) / 4
        e, r = est[(est >= lo) & (est < hi)], ref[(ref >= lo) & (ref < hi)]
        f, _ = C.f_measure(e + C.SKIP, r + C.SKIP) if len(e) and len(r) else (None, None)
        out.append(f)
    return out


def score(run: dict[int, dict], refs: dict[int, dict]) -> tuple[dict, dict[int, dict]]:
    by: dict[str, Counter] = defaultdict(Counter)
    per: dict[int, dict] = {}
    for idx, ref in refs.items():
        row = run.get(idx)
        groups = (ref["stratum"], "all")
        if not row or row["status"] != "ok":
            for g in groups:
                by[g]["errors"] += 1
            continue
        bpm = float(row["bpm"])
        t: dict = {"bpm": bpm}
        for g in groups:
            by[g]["tracks"] += 1
            by[g]["bpm_lt_60"] += bpm < 60
            by[g]["bpm_gt_200"] += bpm > 200
        bt_tempo = bt_reference(ref)
        if bt_tempo:
            t["class_bt"] = tempo_class(bpm, bt_tempo)
            if t["class_bt"] == "correct":
                t["oct_bt"] = round(math.log2(bpm / bt_tempo))
        if ref["tag"]:
            t["class_tag"] = tempo_class(bpm, ref["tag"])
            if t["class_tag"] == "correct":
                t["oct_tag"] = round(math.log2(bpm / ref["tag"]))
        if bt_tempo and ref["tag"]:
            t["judges_agree"] = tempo_class(ref["tag"], bt_tempo) == "correct"
        grid, fit = ref["bt_grid"], ref["bt_fit"]
        if grid and fit is not None and fit >= FIT_STEADY and t.get("class_bt") == "correct":
            t["prec"] = abs(fold(bpm, grid) / grid - 1) <= 0.001 + 1e-12
            folded = fold(bpm, grid)
            locked = abs(folded - round(folded)) < 1e-4
            if fit >= FIT_INTEGER and abs(grid - round(grid)) <= 0.005:
                t["int"] = "exact" if locked and round(folded) == round(grid) else "missed"
            elif abs(grid - round(grid)) > 0.02:
                t["int"] = "false_lock" if locked else "frac_ok"
        if row["beats"] and len(ref["bt_beats"]):
            est = np.array([int(x) for x in row["beats"].split(",")], dtype=float) * FRAME_SEC
            f, _ = C.f_measure(est, ref["bt_beats"])
            t["beat_f"] = f
            if ref["duration"]:
                t["quarters"] = quarters_f(est, ref["bt_beats"], ref["duration"])
        per[idx] = t
        for g in groups:
            c = by[g]
            for key in ("class_bt", "class_tag"):
                if key in t:
                    c[f"{key}_n"] += 1
                    c[f"{key}_{t[key]}"] += 1
            if "oct_bt" in t:
                c["oct_n"] += 1
                c[f"oct_{t['oct_bt']:+d}"] += 1
            c[f"verdict_{verdict(t)}"] += 1
            if "prec" in t:
                c["prec_n"] += 1
                c["prec_ok"] += t["prec"]
            if "int" in t:
                c[f"int_{t['int']}"] += 1
            if t.get("beat_f") is not None:
                c["beat_f_n"] += 1
                c["beat_f_sum"] += t["beat_f"]
                qs = [q for q in t.get("quarters", []) if q is not None]
                if qs:
                    c["q_n"] += 1
                    c["q_min_sum"] += min(qs)
                    c["q_hold"] += min(qs) >= 0.9
    summary = {}
    for g, c in by.items():
        pct = lambda a, b: round(100.0 * a / b, 1) if b else None  # noqa: E731
        summary[g] = {
            "tracks": c["tracks"], "errors": c["errors"],
            "class_bt_correct_pct": pct(c["class_bt_correct"], c["class_bt_n"]),
            "class_bt": {k[9:]: v for k, v in c.items() if k.startswith("class_bt_") and not k.endswith("_n")},
            "class_tag_correct_pct": pct(c["class_tag_correct"], c["class_tag_n"]),
            "octave_bt": {k[4:]: v for k, v in sorted(c.items()) if k.startswith("oct_") and k != "oct_n"},
            "prec_pct": pct(c["prec_ok"], c["prec_n"]), "prec_n": c["prec_n"],
            "integer": {k[4:]: v for k, v in c.items() if k.startswith("int_")},
            "beat_f": round(c["beat_f_sum"] / c["beat_f_n"], 3) if c["beat_f_n"] else None,
            "quarter_min_f": round(c["q_min_sum"] / c["q_n"], 3) if c["q_n"] else None,
            "phase_holds_pct": pct(c["q_hold"], c["q_n"]),
            "bpm_lt_60": c["bpm_lt_60"], "bpm_gt_200": c["bpm_gt_200"],
            "verdicts": {k[8:]: v for k, v in sorted(c.items()) if k.startswith("verdict_")},
        }
    return summary, per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--before", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--list", action="store_true", help="print every track's judgement")
    ap.add_argument("--listening", type=Path,
                    help="CSV of tracks to settle by ear: tempo class disputed by Beat This! or the tag, "
                         "or octave against Beat This!")
    args = ap.parse_args()
    refs = references()
    run = read_tsv(args.run)
    summary, per = score(run, refs)
    for g in ("slow", "fast", "broken", "straight", "live", "all"):
        if g not in summary:
            continue
        s = summary[g]
        print(f"== {g}: {s['tracks']} tracks ({s['errors']} errors) | class vs BT {s['class_bt_correct_pct']}% "
              f"{s['class_bt']} | vs tag {s['class_tag_correct_pct']}% | octave vs BT {s['octave_bt']}")
        print(f"   0.1% vs BT grid {s['prec_pct']}% of {s['prec_n']} | integer {s['integer']} | beat F {s['beat_f']} "
              f"quarter-min F {s['quarter_min_f']} phase holds {s['phase_holds_pct']}% | bpm<60 {s['bpm_lt_60']} "
              f">200 {s['bpm_gt_200']}")
        print(f"   verdicts {s['verdicts']}")
    result = {"summary": summary}
    if args.before:
        before = read_tsv(args.before)
        _, per_before = score(before, refs)
        same = sum(1 for i, r in run.items() if i in before and
                   (r["bpm"], r["beats"], r["downbeats"]) == (before[i]["bpm"], before[i]["beats"], before[i]["downbeats"]))
        changes: dict[str, list[int]] = defaultdict(list)
        for idx in sorted(per):
            a, b = per[idx], per_before.get(idx, {})
            for key in ("class_bt", "oct_bt", "int"):
                if a.get(key) != b.get(key):
                    changes[f"{key}: {b.get(key)} -> {a.get(key)}"].append(idx)
        print(f"-- before/after: identical rows {same}")
        for k, v in sorted(changes.items()):
            print(f"   {k}: {len(v)} {v[:30]}")
        result["changes"] = changes
    if args.list:
        for idx in sorted(per):
            r, t = refs[idx], per[idx]
            print(f"{idx:4d} {r['stratum']:8s} {r['label']:24s} sonara {t['bpm']:8.3f} BT {r['bt_level'] or 0:7.2f} "
                  f"grid {r['bt_grid'] or 0:8.3f} fit {r['bt_fit'] or 0:.2f} tag {r['tag'] or 0:6.1f} "
                  f"class {t.get('class_bt', '-')}/{t.get('class_tag', '-')} F {t.get('beat_f') or 0:.2f} "
                  f"{Path(r['path']).name[:60]}")
    if args.listening:
        with args.listening.open("w", encoding="utf-8", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["idx", "stratum", "label", "verdict", "reasons", "sonara_bpm", "bt_level_bpm", "bt_grid_bpm",
                        "bt_fit", "tag_bpm", "class_vs_bt", "class_vs_tag", "beat_f", "quarter_f", "path"])
            for idx in sorted(per, key=lambda i: (verdict(per[i]), i)):
                r, t = refs[idx], per[idx]
                reasons = [name for name, hit in (
                    ("class vs Beat This!", t.get("class_bt") not in (None, "correct")),
                    ("class vs tag", t.get("class_tag") not in (None, "correct")),
                    ("octave vs Beat This!", t.get("oct_bt") not in (None, 0)),
                    ("octave vs tag", t.get("oct_tag") not in (None, 0)),
                ) if hit]
                if reasons:
                    w.writerow([idx, r["stratum"], r["label"], verdict(t), "; ".join(reasons), f"{t['bpm']:.3f}",
                                r["bt_level"], None if r["bt_grid"] is None else f"{r['bt_grid']:.3f}",
                                None if r["bt_fit"] is None else f"{r['bt_fit']:.2f}", r["tag"],
                                t.get("class_bt"), t.get("class_tag"),
                                None if t.get("beat_f") is None else f"{t['beat_f']:.2f}",
                                "/".join("-" if q is None else f"{q:.2f}" for q in t.get("quarters", [])), r["path"]])
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
