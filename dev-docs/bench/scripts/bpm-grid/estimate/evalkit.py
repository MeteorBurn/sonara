"""Per-track metrics for one pipeline variant and split-wise aggregation.

A variant output per track is a dict: bpm (float|None), beats (s), downbeats (s),
optionally onsets (s). References (evaluation only, never used for fitting on
reported data): MIK tempo_raw on the 941 'agree' tracks; Beat This! beats and
downbeats; MIK beat grid (tag `beatgrid`, else constant grid through MIK cue
points at MIK tempo); MIK cue points (phrase starts = downbeats).
"""

from __future__ import annotations

import numpy as np

import est_common as ec

TOLS = (0.07, 0.02, 0.01)


def mik_grid(t):
    """MIK beat grid in seconds or None. Tag grid if constant, else from cues."""
    tag, tr, dur = t["tag"], t["mik"]["tempo_raw"], t.get("duration") or 0
    g = tag.get("grid")
    if g and len(g) > 8:
        g = np.asarray(g, dtype=float)
        ibi = np.diff(g)
        if ibi.std() < 1e-3:
            return g, "tag"
    cues = sorted(c for c in tag.get("cues", []) if c >= 5.0)
    if tr and len(cues) >= 2 and dur:
        per = 60.0 / tr
        c = np.asarray(cues)
        k = np.round((c - c[0]) / per)
        resid = c - c[0] - k * per
        if np.max(np.abs(resid)) < 0.005:
            phase = np.mean(c - k * per)          # anchor at k = 0 (first cue)
            k0 = -int(np.floor(phase / per))
            kk = np.arange(k0, int((dur - phase) / per) + 1)
            grid = phase + kk * per
            return grid[(grid >= 0) & (grid <= dur)], "cues"
    return None, None


def track_metrics(t, out):
    c = t["comp"]
    m = {"idx": t["idx"], "dev": ec.is_dev(t["idx"]), "agree": c["refs"] == "agree",
         "ref_bpm": c["ref_bpm"]}
    bpm = out.get("bpm")
    if m["agree"]:
        cat = ec.category(bpm, c["ref_bpm"])
        m.update(cat=cat, critical=ec.is_critical(cat),
                 abs_err=None if bpm is None else abs(bpm - c["ref_bpm"]))
    beats = out.get("beats")
    bt = t["bt"]
    if beats is not None and len(bt["beats"]):
        for tol in TOLS:
            m[f"F{int(tol * 1000)}"] = ec.f_measure(beats, bt["beats"], tol)
        m["phase_ms"] = ec.phase_ms(beats, bt["beats"])
    downs = out.get("downbeats")
    if downs is not None and len(bt["downbeats"]):
        m["DF70"] = ec.f_measure(downs, bt["downbeats"], 0.07)
    g, src = t.get("_mik_grid", (None, None))
    if g is not None and beats is not None:
        m["mik_src"] = src
        for tol in TOLS:
            m[f"MF{int(tol * 1000)}"] = ec.f_measure(beats, g, tol)
        m["mik_phase_ms"] = ec.phase_ms(beats, g)
    cues = t["tag"].get("cues")
    if cues and downs is not None:
        h, n = ec.cue_hits(downs, cues)
        m["cue_hits"], m["cue_n"] = h, n
    return m


def prepare(D):
    for t in D.values():
        t["_mik_grid"] = mik_grid(t) if t["status"] == "ok" else (None, None)


def subset(rows, split):
    if split == "dev":
        return [r for r in rows if r["dev"]]
    if split == "holdout":
        return [r for r in rows if not r["dev"]]
    return list(rows)


def aggregate(rows, split, boot=True, B=2000):
    rs = subset(rows, split)
    out = {"split": split, "tracks": len(rs)}
    ag = [r for r in rs if r["agree"]]
    out["agree_tracks"] = len(ag)
    cats = {}
    for r in ag:
        cats[r["cat"]] = cats.get(r["cat"], 0) + 1
    out["bpm_cats"] = {k: cats[k] for k in ec.CAT_ORDER if k in cats}
    exact = [r["cat"] == "exact" for r in ag]
    le05 = [r["cat"] in ("exact", "fine") for r in ag]
    crit = [r["critical"] for r in ag]
    errs = [r["abs_err"] for r in ag if r["abs_err"] is not None and not r["critical"]]
    if boot:
        out["exact_rate"] = ec.boot_rate(exact, B)
        out["le_0.5_rate"] = ec.boot_rate(le05, B)
        out["critical_rate"] = ec.boot_rate(crit, B)
    else:
        out["exact_rate"] = {"mean": float(np.mean(exact))}
        out["le_0.5_rate"] = {"mean": float(np.mean(le05))}
        out["critical_rate"] = {"mean": float(np.mean(crit))}
    out["counts"] = {"exact": int(sum(exact)), "fine": int(sum(le05) - sum(exact)),
                     "minor": int(sum(r["cat"] == "minor" for r in ag)),
                     "critical": int(sum(crit))}
    out["abs_err_noncrit_p50"] = float(np.median(errs)) if errs else None
    out["abs_err_noncrit_p90"] = float(np.percentile(errs, 90)) if errs else None
    # grid vs Beat This! on agree tracks (fixed set: every agree track with BT beats)
    for key in ("F70", "F20", "F10", "DF70"):
        v = [r[key] for r in ag if r.get(key) is not None]
        out[key + "_mean"] = (ec.boot_rate(v, B) if boot else {"mean": float(np.mean(v))}) if v else None
    ph = [r["phase_ms"] for r in ag if r.get("phase_ms") is not None]
    out["phase_ms_median"] = float(np.median(ph)) if ph else None
    out["phase_ms_p10_p90"] = [float(np.percentile(ph, 10)), float(np.percentile(ph, 90))] if ph else None
    # vs MIK grid
    mk = [r for r in ag if r.get("MF70") is not None]
    out["mik_grid_tracks"] = len(mk)
    for key in ("MF70", "MF20", "MF10"):
        v = [r[key] for r in mk]
        out[key + "_mean"] = float(np.mean(v)) if v else None
    mph = [r["mik_phase_ms"] for r in mk if r.get("mik_phase_ms") is not None]
    out["mik_phase_ms_median"] = float(np.median(mph)) if mph else None
    # MIK cues on downbeats
    cu = [r for r in ag if r.get("cue_n")]
    out["cue_tracks"] = len(cu)
    out["cue_hit_rate"] = (sum(r["cue_hits"] for r in cu) / max(1, sum(r["cue_n"] for r in cu))) if cu else None
    out["cue_track_majority"] = (sum(r["cue_hits"] * 2 > r["cue_n"] for r in cu) / len(cu)) if cu else None
    return out
