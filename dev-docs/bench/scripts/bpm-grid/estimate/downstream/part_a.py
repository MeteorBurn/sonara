"""Part A - straight-rhythm control set (100 straight four-on-the-floor tech
house), with the 45 labelled broken tracks and the broken-1000 references
under identical definitions for comparison.

References: Beat This! final0 beats/downbeats. BPM level: Beat This! basic BPM
(60 / median IBI, folded 79-192) within +-3%. Fine BPM: consensus line fit over
the Beat This! beats (+-20 ms inliers) - on the broken-1000 set this fit equals
MIK within 0.05 BPM on 836/846 tracks (../results/fix1.json).

Beat phase classes (tracks whose Sonara BPM level agrees with Beat This!):
  F_on  = F(+-70 ms) of Sonara beats vs Beat This! beats,
  F_mid = F(+-70 ms) of the midpoints between Sonara beats vs Beat This! beats;
  in-phase  F_on >= 0.5 and F_on >= F_mid
  half-beat F_mid >= 0.5 and F_mid > F_on      ("offbeat lock")
  other     neither >= 0.5
Times use the fix-2 frame shift (-1 frame) so the +24 ms latency does not bias
the classes; the raw baseline is reported too. Also the circular phase of the
Sonara beats inside the Beat This! beat (median, in beat fractions).
Downbeat phase reference p* = argmax_p F(beats[p::4] vs Beat This! downbeats)
when that F >= 0.5 (as ../fix3_downbeat.py).
Writes results/part_a.json and results/part_a_tracks.json.
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import ds_data as dd
import est_common as ec
from fix1_bpm import consensus, unwrap

FRAME = ec.HOP_SEC


def bt_fit(bt_beats):
    if len(bt_beats) < 16:
        return None
    k = unwrap(bt_beats).astype(float)
    return ec.fold(60.0 / consensus(k, bt_beats, 0.02)[0])


def circ_phase(est, ref):
    """Median position of est beats inside the ref beat interval, in [-0.5, 0.5)."""
    ref = np.asarray(ref)
    est = np.asarray(est)
    est = est[(est > ref[0]) & (est < ref[-1])]
    if len(est) < 8:
        return None
    i = np.searchsorted(ref, est) - 1
    ph = (est - ref[i]) / (ref[i + 1] - ref[i])
    ang = np.angle(np.mean(np.exp(2j * np.pi * ph)))
    return float(ang / (2 * np.pi))


def level(sonara, ref):
    if sonara is None or ref is None:
        return "n/a"
    r = sonara / ref
    for m, lab in ((1.0, "same"), (2.0, "x2"), (0.5, "x1/2"), (4 / 3, "x4/3"), (0.75, "x3/4"),
                   (1.5, "x3/2"), (2 / 3, "x2/3"), (1.25, "x5/4"), (0.8, "x4/5")):
        if abs(r / m - 1) <= 0.03:
            return lab
    return "other"


def track_row(t, V):
    bt = t["bt"]
    bb, bd = bt["beats"], bt["downbeats"]
    row = {"idx": t["idx"], "set": t["set"], "source": t.get("source"), "label": t.get("label"),
           "dev": ec.is_dev(t["idx"]) if t["set"] == "B" else None}
    if bt.get("status") != "ok" or len(bb) < 16:
        row["bt_ok"] = False
        return row
    row["bt_ok"] = True
    row["bt_basic"] = bt["bpm"]
    row["bt_fit"] = bt_fit(bb)
    base, f2 = V["base"], V["f2"]
    row["bpm_base"] = base["bpm"]
    row["level_base"] = level(base["bpm"], bt["bpm"])
    row["level_f4"] = level(V["f4"]["bpm"], bt["bpm"])
    row["level_f1234"] = level(V["f1234"]["bpm"], bt["bpm"])
    row["f4_switched"] = V["f4"]["switched"]
    for name in ("base", "f1", "f1234"):
        b = V[name]["bpm"]
        row[f"err_fit_{name}"] = None if row["bt_fit"] is None else b - row["bt_fit"]
    raw = ec.frames_to_sec(base["beats"])
    sh = ec.frames_to_sec(f2["beats"])
    mid = 0.5 * (sh[:-1] + sh[1:])
    row["F_on_raw"] = ec.f_measure(raw, bb) or 0.0
    row["F_on"] = ec.f_measure(sh, bb) or 0.0
    row["F_mid"] = ec.f_measure(mid, bb) or 0.0
    if row["F_on"] >= 0.5 and row["F_on"] >= row["F_mid"]:
        row["phase_class"] = "in-phase"
    elif row["F_mid"] >= 0.5 and row["F_mid"] > row["F_on"]:
        row["phase_class"] = "half-beat"
    else:
        row["phase_class"] = "other"
    row["circ_phase"] = circ_phase(sh, bb)
    for name in ("base", "f2", "f2x", "f1234"):
        s = ec.frames_to_sec(V[name]["beats"])
        row[f"F70_{name}"] = ec.f_measure(s, bb) or 0.0
        row[f"F20_{name}"] = ec.f_measure(s, bb, 0.02) or 0.0
        row[f"phase_ms_{name}"] = ec.phase_ms(s, bb)
    # downbeat phase vs Beat This!
    if len(bd) >= 4 and len(base["beats"]) >= 8:
        fs = [ec.f_measure(sh[p::4], bd) or 0.0 for p in range(4)]
        p_star = int(np.argmax(fs)) if max(fs) >= 0.5 else None
        b0 = list(base["beats"])

        def phase_of(downs, beats):
            d = int(downs[0]) if len(downs) else None
            return (list(beats).index(d) % 4) if d is not None and d in list(beats) else 0
        row["p_star"] = p_star
        row["p_base"] = phase_of(base["downs"], b0)
        row["p_f3"] = phase_of(V["f3"]["downs"], b0)
        row["DF70_base"] = ec.f_measure(ec.frames_to_sec(base["downs"]) - FRAME, bd) or 0.0
        row["DF70_f3"] = ec.f_measure(ec.frames_to_sec(V["f3"]["downs"]) - FRAME, bd) or 0.0
        row["DF70_f1234"] = ec.f_measure(ec.frames_to_sec(V["f1234"]["downs"]), bd) or 0.0
    return row


def boot(x, B=4000, seed=0):
    return ec.boot_rate(np.asarray(x, dtype=float), B, seed)


def summarize(rows):
    ok = [r for r in rows if r.get("bt_ok")]
    out = {"tracks": len(rows), "with_bt": len(ok)}
    lv = {}
    for r in ok:
        lv[r["level_base"]] = lv.get(r["level_base"], 0) + 1
    out["bpm_level_base_vs_bt_basic"] = lv
    out["level_agree_base"] = boot([r["level_base"] == "same" for r in ok])
    out["level_agree_f4"] = boot([r["level_f4"] == "same" for r in ok])
    out["level_agree_f1234"] = boot([r["level_f1234"] == "same" for r in ok])
    sw = [r for r in ok if r["f4_switched"]]
    out["fix4_switched"] = len(sw)
    out["fix4_switch_supported_by_bt"] = sum(r["level_f4"] == "same" for r in sw)
    out["fix4_switch_regressions"] = sum(r["level_base"] == "same" and r["level_f4"] != "same" for r in sw)
    same = [r for r in ok if r["level_base"] == "same"]
    out["same_level_tracks"] = len(same)
    pc = {}
    for r in same:
        pc[r["phase_class"]] = pc.get(r["phase_class"], 0) + 1
    out["phase_classes_same_level"] = pc
    out["half_beat_rate"] = boot([r["phase_class"] == "half-beat" for r in same])
    out["half_beat_rate_strict_Fmid_gt_Fon_plus_0.3"] = boot([r["F_mid"] > r["F_on"] + 0.3 for r in same])
    out["in_phase_rate"] = boot([r["phase_class"] == "in-phase" for r in same])
    cp = [abs(r["circ_phase"]) for r in same if r["circ_phase"] is not None]
    out["circ_phase_abs_hist"] = {f"{lo:.3f}-{hi:.3f}": int(sum(lo <= c < hi for c in cp))
                                  for lo, hi in ((0, 0.125), (0.125, 0.375), (0.375, 0.5001))}
    inph = [r for r in same if r["phase_class"] == "in-phase"]
    for name in ("base", "f2", "f2x", "f1234"):
        ph = [r[f"phase_ms_{name}"] for r in inph if r[f"phase_ms_{name}"] is not None]
        out[f"latency_ms_inphase_{name}"] = ec.boot_median(ph) if ph else None
        out[f"F70_{name}_same_level"] = boot([r[f"F70_{name}"] for r in same])
        out[f"F20_{name}_same_level"] = boot([r[f"F20_{name}"] for r in same])
    # fix 1 vs Beat This! fit
    fit = [r for r in same if r["bt_fit"] is not None]
    for name in ("base", "f1", "f1234"):
        e = np.array([abs(r[f"err_fit_{name}"]) for r in fit])
        out[f"bpm_exact_0.05_vs_bt_fit_{name}"] = boot(e <= 0.05)
        out[f"bpm_abs_err_vs_bt_fit_p50_{name}"] = float(np.median(e)) if len(e) else None
        out[f"bpm_abs_err_vs_bt_fit_p90_{name}"] = float(np.percentile(e, 90)) if len(e) else None
    # downbeats
    dn = [r for r in same if r.get("p_star") is not None]
    out["downbeat_ref_tracks"] = len(dn)
    for name in ("base", "f3"):
        out[f"bar_phase_acc_{name}"] = boot([r[f"p_{name}"] == r["p_star"] for r in dn])
        rel = {}
        for r in dn:
            d = (r[f"p_{name}"] - r["p_star"]) % 4
            rel[d] = rel.get(d, 0) + 1
        out[f"bar_phase_choice_minus_ref_{name}"] = {int(k): v for k, v in sorted(rel.items())}
    for name in ("base", "f3", "f1234"):
        v = [r[f"DF70_{name}"] for r in same if f"DF70_{name}" in r]
        out[f"downbeat_F70_{name}_same_level"] = boot(v) if v else None
    return out


def main() -> int:
    res, all_rows = {}, []
    for which in ("L", "B"):
        D = dd.load_labelled() if which == "L" else dd.load_broken()
        V = pickle.loads((dd.CACHE / f"variants_{which}.pkl").read_bytes())
        rows = [track_row(D[i], V[i]) for i in sorted(V)]
        all_rows += rows
        if which == "L":
            for src in ("straight", "lomanka", "lomanka_holdout"):
                res[src] = summarize([r for r in rows if r["source"] == src])
            res["lomanka_all45"] = summarize([r for r in rows if r["source"] != "straight"])
        else:
            res["broken1000_holdout"] = summarize([r for r in rows if r["dev"] is False])
            res["broken1000_all"] = summarize(rows)
    (dd.RESULTS / "part_a.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (dd.RESULTS / "part_a_tracks.json").write_text(json.dumps(all_rows, default=float), encoding="utf-8")
    for k, v in res.items():
        print("\n==", k)
        for kk, vv in v.items():
            if isinstance(vv, dict) and "mean" in vv:
                vv = f"{vv['mean']:.3f} [{vv['lo']:.3f}, {vv['hi']:.3f}] n={vv['n']}"
            elif isinstance(vv, dict) and "median" in vv:
                vv = f"{vv['median']:+.2f} [{vv['lo']:+.2f}, {vv['hi']:+.2f}] n={vv['n']}"
            print(f"  {kk}: {vv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
