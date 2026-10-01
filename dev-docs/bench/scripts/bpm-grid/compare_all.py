"""Sonara vs Beat This! vs Mixed In Key vs Rekordbox on both sets (broken + straight, 2000 tracks).

Read-only. Reads the four program databases (data/<n>_<program>/<program>-<set>.sqlite,
built by build_tables.py) and writes data/bpm_comparison_2000.xlsx and .json.

MIK BPM is an authoritative value, not ground truth. Reference BPM combines
Beat This! and MIK, MIK weighted higher: on tracks where Beat This! basic BPM
(60 / median beat interval, folded to 79-192) is within ±3% of MIK (both chose
the same metrical level), Reference = 0.75 * MIK + 0.25 * BT grid BPM, where BT
grid BPM is the tempo of the constant grid that best fits the Beat This! beats
(fit R = |mean exp(2*pi*i*t*bpm/60)|). If R < 0.8 the Beat This! beats are too
irregular to time a grid and Reference = MIK. Grid / downbeat reference =
Beat This! beats and downbeats.
Categories vs reference: exact <=0.05 | fine <=0.5 | minor <=2 |
x2 x0.5 x1.5 x2/3 x4/3 x3/4 x5/4 x4/5 | major ; critical = any multiple or major.
Beat phase (tracks where Sonara and Beat This! share the tempo level): circular
median position of Sonara beats inside Beat This! beat intervals ->
in-phase |pos| < 0.125, half-beat |pos| > 0.375, other in between.
Rekordbox (third validator, added beside the reference, never part of it): its
octave follows its range preset (70-180 shows (90, 180]), so Rekordbox BPM is
compared octave-free: folded by powers of 2 to the octave of the other value.
Phase vs Rekordbox: circular mean position of Sonara beats on the Rekordbox grid
(first TEMPO marker + 60/Bpm), on constant-tempo tracks where Sonara shares the
Rekordbox tempo level.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule, FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpm_grid_paths as P  # noqa: E402

DATA = P.DATA
SETS = {s: {"sonara": P.SONARA_DIR / f"sonara-{s}.sqlite", "bt": P.BEAT_THIS_DIR / f"beat_this-{s}.sqlite",
            "mik": P.MIK_DIR / f"mik-{s}.sqlite", "rb": P.REKORDBOX_DIR / f"rekordbox-{s}.sqlite"}
        for s in ("broken", "straight")}
AGREE_REL = 0.03
W_MIK = 0.75          # MIK weight in Reference BPM; Beat This! grid BPM gets the rest
GRID_FIT_MIN = 0.8    # below this Beat This! beats do not time a grid; Reference = MIK
GRID_MIN_BEATS = 32
MULTIPLES = [(2.0, "x2"), (0.5, "x0.5"), (1.5, "x1.5"), (2 / 3, "x2/3"), (4 / 3, "x4/3"),
             (0.75, "x3/4"), (1.25, "x5/4"), (0.8, "x4/5")]
OK_CATS = ("exact", "fine", "minor")
ORDER = ["exact", "fine", "minor"] + [m[1] for m in MULTIPLES] + ["major", "n/a"]
TOL = 0.07   # beat match window, s
SKIP = 5.0   # beats in the first SKIP seconds are ignored
WIDTH_CAP = 120                      # widest auto-fitted column, characters
WIDTH_CAPS = {"file": 75, "path": 140}   # File sits in the frozen pane


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


def db_rows(db: Path, cols: tuple[str, ...], arrays: tuple[str, ...] = ()) -> dict[int, dict]:
    """Per-track columns and arrays of a program database (tables tracks, arrays), keyed by idx."""
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    out = {r[0]: dict(zip(cols, r[1:])) for r in con.execute(f"SELECT idx, {', '.join(cols)} FROM tracks")}
    if arrays:
        for idx, name, dt, sh, d in con.execute(
                f"SELECT idx, name, dtype, shape, data FROM arrays WHERE name IN ({','.join('?' * len(arrays))})",
                arrays):
            out[idx][name] = np.frombuffer(zlib.decompress(d), dtype=dt).reshape(json.loads(sh))
    con.close()
    return out


def to_octave_of(value, target):
    """`value` times the power of 2 that brings it closest to `target`."""
    if value is None or target is None:
        return None
    return value * 2.0 ** round(np.log2(target / value))


def octave_label(value, target):
    if value is None or target is None:
        return "n/a"
    k = round(float(np.log2(value / target)))
    return "same" if k == 0 else f"x{2 ** k:g}"


def sonara_rows(db: Path) -> dict[int, dict]:
    """Sonara tracks from data/1_sonara/sonara-<set>.sqlite (tables run, tracks, arrays)."""
    # immutable: a read-only open of a WAL database would leave -wal/-shm files
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    run = dict(con.execute("SELECT key, value FROM run"))
    hop, sr = int(run["hop_length"]), int(run["sample_rate"])
    out = {}
    # bpm_raw = Sonara's bpm at full precision; bpm_unfolded = Sonara's own bpm_raw (before folding)
    for idx, path, status, bpm, bpm_raw, conf, reg, reg_conf, stability, duration in con.execute(
            "SELECT idx, path, status, bpm_raw, bpm_unfolded, bpm_confidence, rhythmic_regularity, "
            "rhythmic_regularity_confidence, grid_stability, duration_sec FROM tracks"):
        row = {"path": path, "status": status, "error": None, "bpm": bpm, "bpm_raw": bpm_raw,
               "bpm_conf": conf, "reg": reg, "reg_conf": reg_conf, "grid_stability": stability,
               "duration": duration}
        arrays = {n: np.frombuffer(zlib.decompress(d), dtype=dt).reshape(json.loads(sh))
                  for n, dt, sh, d in con.execute(
                      "SELECT name, dtype, shape, data FROM arrays WHERE idx = ? "
                      "AND name IN ('beats', 'downbeats')", (idx,))}
        for n in ("beats", "downbeats"):
            frames = arrays.get(n)
            row[n] = None if frames is None else frames.astype(float) * hop / sr
        out[idx] = row
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


def grid_bpm(beats_sec, center):
    """Tempo of the constant grid that best fits the beats, within ±AGREE_REL of `center`.

    Returns (bpm, fit); fit = |mean exp(2*pi*i*t*bpm/60)| is 1 when every beat
    sits on the grid. Missing or extra beats lower the fit but do not shift it.
    """
    t = np.asarray(beats_sec, dtype=float)
    t = t[t >= SKIP]
    if center is None or len(t) < GRID_MIN_BEATS:
        return None, None

    def scan(lo, hi, step):
        grid = np.arange(lo, hi + step / 2, step)
        fit = np.abs(np.exp(2j * np.pi * np.outer(grid / 60.0, t)).mean(axis=1))
        i = int(fit.argmax())
        return float(grid[i]), float(fit[i])

    coarse, _ = scan(center * (1 - AGREE_REL), center * (1 + AGREE_REL), 0.01)
    return scan(coarse - 0.01, coarse + 0.01, 0.0005)


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


def rb_grid_phase(son_beats, rb):
    """(phase in beats, phase in ms) of Sonara beats on a constant Rekordbox grid."""
    t0, bpm = rb.get("grid_start_sec"), rb.get("tempo_min")
    if son_beats is None or len(son_beats) < 8 or t0 is None or not bpm:
        return None, None
    period = 60.0 / bpm
    ang = np.angle(np.mean(np.exp(2j * np.pi * (np.asarray(son_beats) - t0) / period)))
    phase = float(ang / (2 * np.pi))
    return phase, phase * period * 1000.0


def build_rows():
    rows = []
    for set_name, src in SETS.items():
        son = sonara_rows(src["sonara"])
        bt = db_rows(src["bt"], ("path", "status", "bpm_raw"), ("beats_sec", "downbeats_sec"))
        mik = db_rows(src["mik"], ("path", "status", "bpm_raw"))
        rkb = db_rows(src["rb"], ("path", "status", "bpm_raw", "tempo_min", "tempo_max", "grid_start_sec"))
        for idx in range(1, 1001):
            s, b, m, rb = son.get(idx, {}), bt.get(idx, {}), mik.get(idx, {}), rkb.get(idx, {})
            path = m.get("path") or b.get("path") or s.get("path")
            s_bpm = s.get("bpm")
            b_bpm = b.get("bpm_raw")      # 60 / median beat interval folded into 79-192
            m_bpm = m.get("bpm_raw") if m.get("status") == "ok" else None
            agree = same_level(b_bpm, m_bpm)
            b_beats = b.get("beats_sec")
            bt_grid, bt_fit = grid_bpm(b_beats if b_beats is not None else [], m_bpm) if agree else (None, None)
            if not agree:
                ref = None
            elif bt_fit is not None and bt_fit >= GRID_FIT_MIN:
                ref = W_MIK * m_bpm + (1.0 - W_MIK) * bt_grid
            else:
                ref = m_bpm
            s_cat = category(s_bpm, ref) if agree else "n/a"
            level_with_bt = same_level(s_bpm, b_bpm)
            beat_f = phase_ms = down_f = phase = None
            sb, sd = s.get("beats"), s.get("downbeats")
            bb = np.asarray(b_beats if b_beats is not None else [])
            bd = np.asarray(b.get("downbeats_sec") if b.get("downbeats_sec") is not None else [])
            if sb is not None and len(bb):
                beat_f, phase_ms = f_measure(np.asarray(sb), bb)
                if level_with_bt:
                    phase = beat_phase(sb, bb)
            if sd is not None and len(bd):
                down_f, _ = f_measure(np.asarray(sd), bd)
            bar_ok = (bar_phase_ok(np.asarray(sb), np.asarray(sd), bd)
                      if level_with_bt and phase_class(phase) == "in-phase" and sb is not None else None)
            r_bpm = rb.get("bpm_raw") if rb.get("status") == "ok" else None
            r_dynamic = None if r_bpm is None else rb.get("tempo_max") != rb.get("tempo_min")
            rb_phase = rb_ms = None
            if r_dynamic is False and same_level(s_bpm, r_bpm):
                rb_phase, rb_ms = rb_grid_phase(sb, rb)
            rows.append({
                "set": set_name, "idx": idx,
                "file": Path(path).name if path else None, "path": path,
                "sonara_status": s.get("status"), "sonara_error": s.get("error"),
                "sonara_bpm": s_bpm, "beat_this_bpm": b_bpm,
                "bt_grid_bpm": bt_grid, "bt_grid_fit": bt_fit, "mik_bpm": m_bpm,
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
                "rb_bpm": r_bpm, "rb_dynamic": r_dynamic,
                "rb_octave_vs_mik": octave_label(r_bpm, m_bpm),
                "rb_vs_mik": category(to_octave_of(r_bpm, m_bpm), m_bpm),
                "sonara_vs_rb": category(to_octave_of(s_bpm, r_bpm), r_bpm),
                "rb_phase": rb_phase, "rb_phase_ms": rb_ms, "rb_phase_class": phase_class(rb_phase),
            })
    return rows


def pct(n, d):
    return None if not d else round(100.0 * n / d, 1)


def shown_len(value, fmt=None):
    """Approximate character count Excel displays for a value under a number format."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 5
    if isinstance(value, float):
        if fmt:
            first = fmt.split(";")[0]
            decimals = len(first.partition(".")[2])
            return len(f"{value:{'+' if first.startswith('+') else ''}.{decimals}f}")
        return len(f"{value:g}")
    return len(str(value))


def fit_width(header, values, fmt=None, cap=WIDTH_CAP):
    """Column width that shows the header (plus filter arrow) and every value."""
    longest = max((shown_len(v, fmt) for v in values), default=0)
    return min(max(len(str(header or "")) + 3, longest + 2, 6), cap)


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
    # Rekordbox: third validator, reported beside the reference
    st["with_bpm"]["rekordbox"] = sum(r["rb_bpm"] is not None for r in rows)
    st["rekordbox_dynamic_tempo"] = sum(r["rb_dynamic"] is True for r in rows)
    st["rekordbox_octave_vs_mik"] = {k: v for k in ("same", "x2", "x0.5", "x4", "x0.25", "n/a")
                                     if (v := sum(r["rb_octave_vs_mik"] == k for r in rows))}
    st["rekordbox_vs_mik_octave_free"] = {k: v for k in ORDER if (v := sum(r["rb_vs_mik"] == k for r in rows))}
    both = [r for r in rows if r["sonara_vs_rb"] != "n/a"]
    st["sonara_vs_rekordbox_octave_free"] = {k: v for k in ORDER if (v := sum(r["sonara_vs_rb"] == k for r in both))}
    st["sonara_vs_rekordbox_rates_%"] = {
        "exact<=0.05": pct(sum(r["sonara_vs_rb"] == "exact" for r in both), len(both)),
        "<=0.5": pct(sum(r["sonara_vs_rb"] in ("exact", "fine") for r in both), len(both)),
        "critical": pct(sum(r["sonara_vs_rb"] not in OK_CATS for r in both), len(both)),
    }
    lvr = [r for r in rows if r["rb_phase_class"] != "n/a"]
    st["phase_vs_rekordbox_grid"] = {k: sum(r["rb_phase_class"] == k for r in lvr)
                                     for k in ("in-phase", "half-beat", "other")}
    st["phase_vs_rekordbox_grid_rates_%"] = {k: pct(v, len(lvr)) for k, v in st["phase_vs_rekordbox_grid"].items()}
    inrb = [r["rb_phase_ms"] for r in lvr if r["rb_phase_class"] == "in-phase"]
    if inrb:
        st["phase_vs_rekordbox_grid_rates_%"]["latency_ms_median"] = round(float(np.median(inrb)), 1)
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
    ws.sheet_properties.tabColor = "1F4E78"
    bold = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    band_fill = PatternFill("solid", fgColor="EEF3F8")
    notes = (
        "Sonara 0.3.7 vs Beat This! vs Mixed In Key vs Rekordbox — 1000 broken + 1000 straight tracks",
        "MIK BPM is authoritative, not ground truth. Reference BPM (only where Beat This! basic BPM is within ±3% of MIK): 0.75·MIK + 0.25·BT grid BPM; if BT grid fit < 0.8, Reference = MIK",
        "BT grid BPM: tempo of the constant grid that best fits the Beat This! beats (fit 1 = every beat on the grid). Grid/downbeat reference: Beat This!",
        "Categories: exact <=0.05 | fine <=0.5 | minor <=2 | x2 x0.5 x1.5 x2/3 x4/3 x3/4 x5/4 x4/5 | major; critical = multiple or major",
        "Beat phase on tracks where Sonara and Beat This! share the tempo level: in-phase |pos|<0.125, half-beat |pos|>0.375 (beats on the offbeat)",
        "Rekordbox (7.2.19, range 70-180, Auto) is a third validator beside the reference, not part of it: BPM compared octave-free (its octave is a range preset); phase = Sonara beats on the Rekordbox grid, constant-tempo tracks at the same level",
        "",
    )
    for line in notes:
        ws.append([line])
    ws["A1"].font = Font(bold=True, size=13, color="1F4E78")
    for row in range(2, len(notes)):
        ws.cell(row, 1).font = Font(italic=True, color="595959")
    keys = list(SETS) + ["all"]
    ws.append(["Metric"] + keys)
    header_row = ws.max_row
    for c in ws[header_row]:
        c.font = header_font
        c.fill = header_fill
    # the baseline columns of results.csv
    headline = {
        "sonara_vs_ref_rates_% / exact<=0.05", "sonara_vs_ref_rates_% / <=0.5",
        "sonara_vs_ref_rates_% / critical", "abs_err_correct_level_bpm / p50",
        "abs_err_correct_level_bpm / p90", "phase_class_rates_% / half-beat",
        "in_phase_tracks / latency_ms_median", "in_phase_tracks / beat_f70_median",
        "in_phase_tracks / bar_phase_correct_%",
    }

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
    banded, group = False, None
    for name in metric_names:
        ws.append([name] + [flat_stats[k].get(name) for k in keys])
        if name.split(" / ")[0] != group:
            group, banded = name.split(" / ")[0], not banded
        for c in ws[ws.max_row]:
            if banded:
                c.fill = band_fill
            if name in headline:
                c.font = bold
    ws.column_dimensions["A"].width = fit_width("Metric", metric_names)
    for i, k in enumerate(keys, 2):
        ws.column_dimensions[get_column_letter(i)].width = fit_width(
            k, [flat_stats[k].get(name) for name in metric_names])
    ws.freeze_panes = f"B{header_row + 1}"

    cols = [
        ("set", "Set", None), ("idx", "#", "0"), ("file", "File", None),
        ("sonara_bpm", "Sonara BPM", "0.00"), ("beat_this_bpm", "Beat This! BPM", "0.00"),
        ("bt_grid_bpm", "BT grid BPM", "0.000"), ("bt_grid_fit", "BT grid fit", "0.00"),
        ("mik_bpm", "MIK BPM", "0.000"), ("rb_bpm", "Rekordbox BPM", "0.00"),
        ("ref_bpm", "Reference BPM", "0.000"),
        ("sonara_minus_ref", "Sonara - ref", "+0.00;-0.00;0.00"),
        ("sonara_cat", "Sonara vs ref", None), ("sonara_critical", "Critical", None),
        ("refs", "Refs", None), ("bt_vs_mik", "BT vs MIK", None),
        ("sonara_vs_mik", "Sonara vs MIK", None),
        ("rb_vs_mik", "RB vs MIK", None), ("rb_octave_vs_mik", "RB octave", None),
        ("sonara_vs_rb", "Sonara vs RB", None), ("rb_dynamic", "RB dynamic", None),
        ("phase_class", "Beat phase", None), ("beat_phase", "Phase (beats)", "+0.00;-0.00;0.00"),
        ("phase_ms", "Phase ms", "+0;-0;0"), ("beat_f", "Beat F ±70ms", "0.00"),
        ("downbeat_f", "Downbeat F", "0.00"), ("bar_phase_ok", "Bar phase ok", None),
        ("rb_phase_class", "Phase vs RB", None), ("rb_phase", "Phase RB (beats)", "+0.00;-0.00;0.00"),
        ("rb_phase_ms", "Phase RB ms", "+0;-0;0"),
        ("sonara_bpm_conf", "BPM conf", "0.00"), ("grid_stability", "Grid stab.", "0.00"),
        ("regularity", "Regularity", "0.00"), ("duration_sec", "Duration s", "0"),
        ("sonara_status", "Sonara status", None), ("sonara_error", "Sonara error", None),
        ("path", "Path", None),
    ]
    red = PatternFill("solid", fgColor="F4C7C3")
    amber = PatternFill("solid", fgColor="FCE8B2")
    green = PatternFill("solid", fgColor="D9EAD3")
    grey = PatternFill("solid", fgColor="EDEDED")
    set_fills = {"broken": PatternFill("solid", fgColor="DDEBF7"),
                 "straight": PatternFill("solid", fgColor="E4DFEC")}
    keys_order = [c[0] for c in cols]
    # 0..1 scores: red -> yellow -> green; signed timing: blue (early) -> white -> orange (late)
    score_scale = dict(start_type="num", start_value=0, start_color="F8696B",
                       mid_type="num", mid_value=0.5, mid_color="FFEB84",
                       end_type="num", end_value=1, end_color="63BE7B")

    def signed_scale(limit):
        return ColorScaleRule(start_type="num", start_value=-limit, start_color="9BC2E6",
                              mid_type="num", mid_value=0, mid_color="FFFFFF",
                              end_type="num", end_value=limit, end_color="F4B084")

    def sheet(title, subset, sort_key=None, tab=None):
        w = wb.create_sheet(title)
        if tab:
            w.sheet_properties.tabColor = tab
        w.append([c[1] for c in cols])
        for cell in w[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="top")
        ordered = sorted(subset, key=sort_key) if sort_key else subset
        for r in ordered:
            w.append([r[c[0]] for c in cols])
            if r["set"] in set_fills:
                w.cell(w.max_row, 1).fill = set_fills[r["set"]]
        for i, (key, header, fmt) in enumerate(cols, 1):
            letter = get_column_letter(i)
            w.column_dimensions[letter].width = fit_width(
                header, [r[key] for r in ordered], fmt, WIDTH_CAPS.get(key, WIDTH_CAP))
            if fmt:
                for cell in w[letter][1:]:
                    cell.number_format = fmt
        last = w.max_row
        if last > 1:
            def col(key):
                L = get_column_letter(keys_order.index(key) + 1)
                return L, f"{L}2:{L}{last}"

            cf = w.conditional_formatting
            for key in ("sonara_cat", "bt_vs_mik", "sonara_vs_mik", "rb_vs_mik", "sonara_vs_rb"):
                L, rng = col(key)
                cf.add(rng, CellIsRule(operator="equal", formula=['"exact"'], fill=green))
                cf.add(rng, FormulaRule(formula=[f'OR({L}2="fine",{L}2="minor")'], fill=amber))
                cf.add(rng, FormulaRule(
                    formula=[f'AND({L}2<>"exact",{L}2<>"fine",{L}2<>"minor",{L}2<>"n/a",{L}2<>"")'], fill=red))
                cf.add(rng, CellIsRule(operator="equal", formula=['"n/a"'], fill=grey))
            L, rng = col("sonara_minus_ref")
            cf.add(rng, FormulaRule(formula=[f"AND(ISNUMBER({L}2),ABS({L}2)<=0.05)"], fill=green))
            cf.add(rng, FormulaRule(formula=[f"AND(ISNUMBER({L}2),ABS({L}2)>0.05,ABS({L}2)<=2)"], fill=amber))
            cf.add(rng, FormulaRule(formula=[f"AND(ISNUMBER({L}2),ABS({L}2)>2)"], fill=red))
            _, rng = col("sonara_critical")
            cf.add(rng, CellIsRule(operator="equal", formula=["TRUE"], fill=red))
            _, rng = col("refs")
            cf.add(rng, CellIsRule(operator="equal", formula=['"agree"'], fill=green))
            cf.add(rng, CellIsRule(operator="equal", formula=['"disagree"'], fill=red))
            cf.add(rng, CellIsRule(operator="equal", formula=['"n/a"'], fill=grey))
            for key in ("phase_class", "rb_phase_class"):
                _, rng = col(key)
                cf.add(rng, CellIsRule(operator="equal", formula=['"in-phase"'], fill=green))
                cf.add(rng, CellIsRule(operator="equal", formula=['"half-beat"'], fill=red))
                cf.add(rng, CellIsRule(operator="equal", formula=['"other"'], fill=amber))
                cf.add(rng, CellIsRule(operator="equal", formula=['"n/a"'], fill=grey))
            cf.add(col("beat_phase")[1], signed_scale(0.5))
            cf.add(col("phase_ms")[1], signed_scale(50))
            cf.add(col("rb_phase")[1], signed_scale(0.5))
            cf.add(col("rb_phase_ms")[1], signed_scale(50))
            L, rng = col("rb_octave_vs_mik")
            cf.add(rng, FormulaRule(formula=[f'AND({L}2<>"same",{L}2<>"n/a",{L}2<>"")'], fill=amber))
            _, rng = col("rb_dynamic")
            cf.add(rng, CellIsRule(operator="equal", formula=["TRUE"], fill=amber))
            for key in ("beat_f", "downbeat_f"):
                _, rng = col(key)
                cf.add(rng, CellIsRule(operator="lessThan", formula=["0.5"], fill=red))
                cf.add(rng, CellIsRule(operator="between", formula=["0.5", "0.9"], fill=amber))
                cf.add(rng, CellIsRule(operator="greaterThan", formula=["0.9"], fill=green))
            _, rng = col("bar_phase_ok")
            cf.add(rng, CellIsRule(operator="equal", formula=["TRUE"], fill=green))
            cf.add(rng, CellIsRule(operator="equal", formula=["FALSE"], fill=red))
            for key in ("sonara_bpm_conf", "grid_stability", "regularity", "bt_grid_fit"):
                cf.add(col(key)[1], ColorScaleRule(**score_scale))
            L, rng = col("sonara_status")
            cf.add(rng, CellIsRule(operator="equal", formula=['"ok"'], fill=green))
            cf.add(rng, FormulaRule(formula=[f'AND({L}2<>"",{L}2<>"ok")'], fill=red))
        w.freeze_panes = "D2"
        w.auto_filter.ref = w.dimensions

    sheet("All tracks", rows, tab="5B9BD5")
    sheet("Sonara critical", [r for r in rows if r["sonara_critical"]],
          lambda r: (r["set"], r["sonara_cat"], r["idx"]), tab="C00000")
    sheet("Beats on offbeat", [r for r in rows if r["phase_class"] == "half-beat"],
          lambda r: (r["set"], r["idx"]), tab="ED7D31")
    sheet("Refs disagree", [r for r in rows if r["refs"] == "disagree"],
          lambda r: (r["set"], r["bt_vs_mik"], r["idx"]), tab="7030A0")
    sheet("Not exact", [r for r in rows if r["refs"] == "agree" and r["sonara_cat"] in ("fine", "minor")],
          lambda r: -abs(r["sonara_minus_ref"] or 0), tab="FFC000")
    sheet("Sonara errors", [r for r in rows if r["sonara_status"] != "ok"], lambda r: (r["set"], r["idx"]),
          tab="7F7F7F")
    sheet("MIK vs RB disagree", [r for r in rows if r["rb_vs_mik"] not in ("exact", "fine", "n/a")],
          lambda r: (r["set"], r["rb_vs_mik"], r["idx"]), tab="548235")
    sheet("RB dynamic tempo", [r for r in rows if r["rb_dynamic"]], lambda r: (r["set"], r["idx"]), tab="BF8F00")
    out = DATA / "bpm_comparison_2000.xlsx"
    wb.save(out)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
