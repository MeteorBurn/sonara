"""Sonara vs Beat This! vs Mixed In Key on both sets (broken + straight, 2000 tracks).

Read-only. Writes data/bpm_comparison_2000.xlsx and data/bpm_comparison_2000.json.

Reference BPM = MIK tempo on tracks where Beat This! basic BPM (60 / median
beat interval, folded to 79-192) is within ±3% of MIK, i.e. both chose the same
metrical level. Grid / downbeat reference = Beat This! beats and downbeats.
Categories vs reference: exact <=0.05 | fine <=0.5 | minor <=2 |
x2 x0.5 x1.5 x2/3 x4/3 x3/4 x5/4 x4/5 | major ; critical = any multiple or major.
Beat phase (tracks where Sonara and Beat This! share the tempo level): circular
median position of Sonara beats inside Beat This! beat intervals ->
in-phase |pos| < 0.125, half-beat |pos| > 0.375, other in between.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from beat_this_bpm import fold_to_range  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

DATA = P.DATA
SETS = {
    "broken": {"sonara": DATA / "broken_1000.sqlite", "bt": DATA / "beat_this", "mik": DATA / "mik"},
    "straight": {"sonara": DATA / "straight_1000.sqlite", "bt": DATA / "beat_this_straight",
                 "mik": DATA / "mik_straight"},
}
AGREE_REL = 0.03
MULTIPLES = [(2.0, "x2"), (0.5, "x0.5"), (1.5, "x1.5"), (2 / 3, "x2/3"), (4 / 3, "x4/3"),
             (0.75, "x3/4"), (1.25, "x5/4"), (0.8, "x4/5")]
OK_CATS = ("exact", "fine", "minor")
ORDER = ["exact", "fine", "minor"] + [m[1] for m in MULTIPLES] + ["major", "n/a"]
TOL = 0.07   # beat match window, s
SKIP = 5.0   # beats in the first SKIP seconds are ignored


def f_measure(est, ref):
    """Greedy one-to-one beat matching: (F-measure, median offset est-ref in ms)."""
    est = est[est >= SKIP]
    ref = ref[ref >= SKIP]
    if len(est) == 0 or len(ref) == 0:
        return None, None
    used = np.zeros(len(est), dtype=bool)
    offsets = []
    for r in ref:
        i = int(np.searchsorted(est, r))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(est) and not used[j] and abs(est[j] - r) <= TOL:
                if best is None or abs(est[j] - r) < abs(est[best] - r):
                    best = j
        if best is not None:
            used[best] = True
            offsets.append(est[best] - r)
    hits = len(offsets)
    p, rc = hits / len(est), hits / len(ref)
    f = 0.0 if hits == 0 else 2 * p * rc / (p + rc)
    return f, (float(np.median(offsets)) * 1000.0 if offsets else None)


def load_json_dir(folder: Path) -> dict[int, dict]:
    out = {}
    for f in folder.glob("[0-9][0-9][0-9][0-9] - *.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[int(d["idx"])] = d
    return out


def sonara_rows(db: Path) -> dict[int, dict]:
    # immutable: a read-only open of a WAL database would leave -wal/-shm files
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    out = {}
    for path, pidx, status, error, sj in con.execute(
            "SELECT path, playlist_idx, status, error, scalars_json FROM tracks"):
        row = {"path": path, "status": status, "error": error}
        if sj:
            s = json.loads(sj)
            prov = s.get("provenance", {})
            hop, sr = prov.get("hop_length", 512), prov.get("sample_rate", 22050)
            row.update(bpm=s.get("bpm"), bpm_raw=s.get("bpm_raw"),
                       bpm_conf=s.get("bpm_confidence"),
                       reg=s.get("rhythmic_regularity"),
                       reg_conf=s.get("rhythmic_regularity_confidence"),
                       grid_stability=s.get("grid_stability"),
                       duration=s.get("duration_sec"))
            arrays = {n: np.frombuffer(zlib.decompress(d), dtype=dt).reshape(json.loads(sh))
                      for n, dt, sh, d in con.execute(
                          "SELECT name, dtype, shape, data FROM arrays WHERE path = ? "
                          "AND name IN ('beats', 'downbeats')", (path,))}
            for n in ("beats", "downbeats"):
                frames = arrays.get(n)
                if frames is None and isinstance(s.get(n), list):
                    frames = np.asarray(s[n])
                row[n] = None if frames is None else frames.astype(float) * hop / sr
        out[pidx + 1] = row
    con.close()
    return out


def category(value, ref):
    if value is None or ref is None:
        return "n/a"
    d = abs(value - ref)
    if d <= 0.05:
        return "exact"
    if d <= 0.5:
        return "fine"
    if d <= 2.0:
        return "minor"
    for m, label in MULTIPLES:
        if abs(value / ref / m - 1.0) <= 0.02:
            return label
    return "major"


def same_level(a, b):
    return a is not None and b is not None and abs(a / b - 1.0) <= AGREE_REL


def beat_phase(est, ref):
    """Circular median position of est beats inside ref beat intervals, in [-0.5, 0.5)."""
    est, ref = np.asarray(est), np.asarray(ref)
    if len(est) < 8 or len(ref) < 8:
        return None
    i = np.searchsorted(ref, est) - 1
    ok = (i >= 0) & (i < len(ref) - 1)
    if ok.sum() < 8:
        return None
    pos = (est[ok] - ref[i[ok]]) / (ref[i[ok] + 1] - ref[i[ok]])
    ang = np.angle(np.mean(np.exp(2j * np.pi * pos)))
    return float(ang / (2 * np.pi))


def phase_class(p):
    if p is None:
        return "n/a"
    a = abs(p)
    return "in-phase" if a < 0.125 else ("half-beat" if a > 0.375 else "other")


def bar_phase_ok(son_beats, son_down, bt_down):
    """Does Sonara's downbeat residue class coincide with Beat This! downbeats?"""
    if son_down is None or len(son_down) < 4 or len(bt_down) < 4 or len(son_beats) < 16:
        return None
    hits = [np.mean([np.min(np.abs(bt_down - t)) <= 0.07 for t in son_beats[p::4]]) for p in range(4)]
    best = int(np.argmax(hits))
    if hits[best] < 0.5:
        return None                       # Beat This! bar not consistent with Sonara beats
    chosen = int(np.argmin([np.min(np.abs(son_beats[p::4][:, None] - son_down[None, :]), axis=1).mean()
                            for p in range(4)]))
    return chosen == best


def build_rows():
    rows = []
    for set_name, src in SETS.items():
        son = sonara_rows(src["sonara"])
        bt = load_json_dir(src["bt"])
        mik = load_json_dir(src["mik"])
        for idx in range(1, 1001):
            s, b, m = son.get(idx, {}), bt.get(idx, {}), mik.get(idx, {})
            path = m.get("path") or b.get("path") or s.get("path")
            s_bpm = s.get("bpm")
            ibi = b.get("ibi_median_sec")
            b_bpm = fold_to_range(60.0 / ibi, 79.0, 192.0) if ibi else None
            m_bpm = m.get("tempo_raw") if m.get("bpm") is not None else None
            agree = same_level(b_bpm, m_bpm)
            ref = m_bpm if agree else None
            s_cat = category(s_bpm, ref) if agree else "n/a"
            level_with_bt = same_level(s_bpm, b_bpm)
            beat_f = phase_ms = down_f = phase = None
            sb, sd = s.get("beats"), s.get("downbeats")
            bb = np.asarray(b.get("beats_sec") or [])
            bd = np.asarray(b.get("downbeats_sec") or [])
            if sb is not None and len(bb):
                beat_f, phase_ms = f_measure(np.asarray(sb), bb)
                if level_with_bt:
                    phase = beat_phase(sb, bb)
            if sd is not None and len(bd):
                down_f, _ = f_measure(np.asarray(sd), bd)
            bar_ok = (bar_phase_ok(np.asarray(sb), np.asarray(sd), bd)
                      if level_with_bt and phase_class(phase) == "in-phase" and sb is not None else None)
            rows.append({
                "set": set_name, "idx": idx,
                "file": Path(path).name if path else None, "path": path,
                "sonara_status": s.get("status"), "sonara_error": s.get("error"),
                "sonara_bpm": s_bpm, "beat_this_bpm": b_bpm, "mik_bpm": m_bpm,
                "refs": "agree" if agree else ("n/a" if b_bpm is None or m_bpm is None else "disagree"),
                "bt_vs_mik": category(b_bpm, m_bpm),
                "ref_bpm": ref,
                "sonara_minus_ref": None if ref is None or s_bpm is None else s_bpm - ref,
                "sonara_cat": s_cat, "sonara_critical": s_cat not in OK_CATS + ("n/a",),
                "sonara_vs_mik": category(s_bpm, m_bpm),
                "sonara_level_with_bt": level_with_bt,
                "sonara_bpm_conf": s.get("bpm_conf"),
                "beat_f": beat_f, "phase_ms": phase_ms,
                "beat_phase": phase, "phase_class": phase_class(phase),
                "downbeat_f": down_f, "bar_phase_ok": bar_ok,
                "grid_stability": s.get("grid_stability"),
                "regularity": s.get("reg"), "regularity_conf": s.get("reg_conf"),
                "duration_sec": s.get("duration"),
            })
    return rows


def pct(n, d):
    return None if not d else round(100.0 * n / d, 1)


def set_stats(rows):
    st = {"tracks": len(rows)}
    st["with_bpm"] = {k: sum(r[c] is not None for r in rows)
                      for k, c in (("sonara", "sonara_bpm"), ("beat_this", "beat_this_bpm"), ("mik", "mik_bpm"))}
    st["refs"] = {k: sum(r["refs"] == k for r in rows) for k in ("agree", "disagree", "n/a")}
    st["bt_vs_mik"] = {k: v for k in ORDER if (v := sum(r["bt_vs_mik"] == k for r in rows))}
    ref = [r for r in rows if r["refs"] == "agree" and r["sonara_bpm"] is not None]
    st["sonara_vs_ref"] = {k: v for k in ORDER if (v := sum(r["sonara_cat"] == k for r in ref))}
    st["sonara_vs_ref_rates_%"] = {
        "exact<=0.05": pct(sum(r["sonara_cat"] == "exact" for r in ref), len(ref)),
        "<=0.5": pct(sum(r["sonara_cat"] in ("exact", "fine") for r in ref), len(ref)),
        "critical": pct(sum(r["sonara_critical"] for r in ref), len(ref)),
    }
    ok = [abs(r["sonara_minus_ref"]) for r in ref if r["sonara_cat"] in OK_CATS]
    if ok:
        st["abs_err_correct_level_bpm"] = {"p50": round(float(np.median(ok)), 3),
                                           "p90": round(float(np.percentile(ok, 90)), 3)}
    lv = [r for r in rows if r["sonara_level_with_bt"]]
    st["phase_class_same_level_as_bt"] = {k: sum(r["phase_class"] == k for r in lv)
                                          for k in ("in-phase", "half-beat", "other", "n/a")}
    st["phase_class_rates_%"] = {k: pct(v, len(lv)) for k, v in st["phase_class_same_level_as_bt"].items()}
    inph = [r for r in lv if r["phase_class"] == "in-phase"]
    if inph:
        st["in_phase_tracks"] = {
            "latency_ms_median": round(float(np.median([r["phase_ms"] for r in inph if r["phase_ms"] is not None])), 1),
            "beat_f70_median": round(float(np.median([r["beat_f"] for r in inph])), 3),
            "downbeat_f_median": round(float(np.median([r["downbeat_f"] for r in inph if r["downbeat_f"] is not None])), 3),
        }
        bars = [r["bar_phase_ok"] for r in inph if r["bar_phase_ok"] is not None]
        st["in_phase_tracks"]["bar_phase_correct_%"] = pct(sum(bars), len(bars))
        st["in_phase_tracks"]["bar_phase_checked"] = len(bars)
    return st


def main() -> int:
    rows = build_rows()
    stats = {s: set_stats([r for r in rows if r["set"] == s]) for s in SETS}
    stats["all"] = set_stats(rows)
    (DATA / "bpm_comparison_2000.json").write_text(
        json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    bold = Font(bold=True)
    for line in (
        "Sonara 0.3.7 vs Beat This! vs Mixed In Key — 1000 broken + 1000 straight tracks",
        "Reference BPM: MIK where Beat This! basic BPM is within ±3% of MIK (same metrical level). Grid/downbeat reference: Beat This!",
        "Categories: exact <=0.05 | fine <=0.5 | minor <=2 | x2 x0.5 x1.5 x2/3 x4/3 x3/4 x5/4 x4/5 | major; critical = multiple or major",
        "Beat phase on tracks where Sonara and Beat This! share the tempo level: in-phase |pos|<0.125, half-beat |pos|>0.375 (beats on the offbeat)",
        "",
    ):
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13)
    keys = list(SETS) + ["all"]
    ws.append(["Metric"] + keys)
    for c in ws[ws.max_row]:
        c.font = bold

    def flat(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(f"{prefix}{k} / ", v)
            else:
                yield f"{prefix}{k}", v

    metric_names = []
    for k in keys:
        for name, _ in flat("", stats[k]):
            if name not in metric_names:
                metric_names.append(name)
    flat_stats = {k: dict(flat("", stats[k])) for k in keys}
    for name in metric_names:
        ws.append([name] + [flat_stats[k].get(name) for k in keys])
    ws.column_dimensions["A"].width = 62
    for col in "BCD":
        ws.column_dimensions[col].width = 12

    cols = [
        ("set", "Set", 9, None), ("idx", "#", 6, "0"), ("file", "File", 46, None),
        ("sonara_bpm", "Sonara BPM", 11, "0.00"), ("beat_this_bpm", "Beat This! BPM", 12, "0.00"),
        ("mik_bpm", "MIK BPM", 10, "0.00"), ("ref_bpm", "Reference BPM", 12, "0.00"),
        ("sonara_minus_ref", "Sonara - ref", 11, "+0.00;-0.00;0.00"),
        ("sonara_cat", "Sonara vs ref", 12, None), ("sonara_critical", "Critical", 8, None),
        ("refs", "Refs", 9, None), ("bt_vs_mik", "BT vs MIK", 10, None),
        ("sonara_vs_mik", "Sonara vs MIK", 12, None),
        ("phase_class", "Beat phase", 11, None), ("beat_phase", "Phase (beats)", 11, "+0.00;-0.00;0.00"),
        ("phase_ms", "Phase ms", 9, "+0;-0;0"), ("beat_f", "Beat F ±70ms", 10, "0.00"),
        ("downbeat_f", "Downbeat F", 10, "0.00"), ("bar_phase_ok", "Bar phase ok", 10, None),
        ("sonara_bpm_conf", "BPM conf", 9, "0.00"), ("grid_stability", "Grid stab.", 9, "0.00"),
        ("regularity", "Regularity", 9, "0.00"), ("duration_sec", "Duration s", 9, "0"),
        ("sonara_status", "Sonara status", 10, None), ("sonara_error", "Sonara error", 30, None),
        ("path", "Path", 80, None),
    ]
    red = PatternFill("solid", fgColor="F4C7C3")
    amber = PatternFill("solid", fgColor="FCE8B2")
    green = PatternFill("solid", fgColor="D9EAD3")
    keys_order = [c[0] for c in cols]

    def sheet(title, subset, sort_key=None):
        w = wb.create_sheet(title)
        w.append([c[1] for c in cols])
        for cell in w[1]:
            cell.font = bold
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for r in (sorted(subset, key=sort_key) if sort_key else subset):
            w.append([r[c[0]] for c in cols])
        for i, (_, _, width, fmt) in enumerate(cols, 1):
            letter = get_column_letter(i)
            w.column_dimensions[letter].width = width
            if fmt:
                for cell in w[letter][1:]:
                    cell.number_format = fmt
        last = w.max_row
        if last > 1:
            for key in ("sonara_cat", "bt_vs_mik", "sonara_vs_mik"):
                L = get_column_letter(keys_order.index(key) + 1)
                rng = f"{L}2:{L}{last}"
                w.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"exact"'], fill=green))
                w.conditional_formatting.add(rng, FormulaRule(formula=[f'OR({L}2="fine",{L}2="minor")'], fill=amber))
                w.conditional_formatting.add(rng, FormulaRule(
                    formula=[f'AND({L}2<>"exact",{L}2<>"fine",{L}2<>"minor",{L}2<>"n/a",{L}2<>"")'], fill=red))
            L = get_column_letter(keys_order.index("phase_class") + 1)
            rng = f"{L}2:{L}{last}"
            w.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"in-phase"'], fill=green))
            w.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"half-beat"'], fill=red))
            w.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"other"'], fill=amber))
            for key in ("beat_f", "downbeat_f"):
                L = get_column_letter(keys_order.index(key) + 1)
                rng = f"{L}2:{L}{last}"
                w.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0.5"], fill=red))
                w.conditional_formatting.add(rng, CellIsRule(operator="between", formula=["0.5", "0.9"], fill=amber))
        w.freeze_panes = "D2"
        w.auto_filter.ref = w.dimensions

    sheet("All tracks", rows)
    sheet("Sonara critical", [r for r in rows if r["sonara_critical"]],
          lambda r: (r["set"], r["sonara_cat"], r["idx"]))
    sheet("Beats on offbeat", [r for r in rows if r["phase_class"] == "half-beat"],
          lambda r: (r["set"], r["idx"]))
    sheet("Refs disagree", [r for r in rows if r["refs"] == "disagree"],
          lambda r: (r["set"], r["bt_vs_mik"], r["idx"]))
    sheet("Not exact", [r for r in rows if r["refs"] == "agree" and r["sonara_cat"] in ("fine", "minor")],
          lambda r: -abs(r["sonara_minus_ref"] or 0))
    sheet("Sonara errors", [r for r in rows if r["sonara_status"] != "ok"], lambda r: (r["set"], r["idx"]))
    out = DATA / "bpm_comparison_2000.xlsx"
    wb.save(out)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
