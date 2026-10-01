"""Port fidelity: recompute stored Sonara 0.3.7 outputs from stored inputs.

rhythmic_regularity (4/4 main pass and the 2/4 meter pass), bpm_confidence,
grid_stability, danceability, tempo_curve/variability and aggression_rhythm
(= mean of onset-density term, danceability, grid regularity). On the 145
labelled tracks also the regularity components against eval/*_v3.tsv.
Writes results/port_fidelity_downstream.json.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

import ds_data as dd
import rr_port as rr

EVAL = dd.P.REGULARITY_EVAL  # paths.json regularity_eval


def check(D, tag):
    rows = []
    for idx, t in sorted(D.items()):
        if t["status"] != "ok" or "onset_strength_bands" not in t:
            continue
        S = t["S"]
        beats = np.asarray(t["beats"], dtype=np.int64)
        gs = rr.grid_stability(beats)
        bc = rr.bpm_confidence(S["bpm"], len(beats), S["duration_sec"], S["onset_density"],
                               S["bpm_candidates"][0][1] if S.get("bpm_candidates") else 0.0)
        res = rr.rr_analyze(t["onset_strength_bands"], beats, t["downbeats"], 4, S["bpm_confidence"],
                            S["grid_stability"], None)
        res_m = rr.rr_analyze(t["onset_strength_bands"], beats, t.get("meter_downbeats", []), 2,
                              S["bpm_confidence"], S.get("meter_grid_stability", S["grid_stability"]),
                              S.get("time_signature_confidence"))
        dance = rr.danceability(S["bpm"], beats, S["onset_density"])
        greg = rr.grid_regularity(beats)
        arhy = (min(1.0, max(0.0, S["onset_density"] / 12.0)) + dance + greg) / 3.0
        tc = rr.tempo_curve(beats)
        tv = rr.tempo_variability(tc)
        r = {"idx": idx,
             "gs": abs(gs - S["grid_stability"]),
             "bc": abs(bc - S["bpm_confidence"]),
             "dance": abs(dance - S["danceability"]),
             "arhy": abs(arhy - S["aggression_rhythm"]) if S.get("aggression_rhythm") is not None else None,
             "tv": abs(tv - S["tempo_variability"]),
             "tc": float(np.max(np.abs(tc - t["tempo_curve"]))) if t.get("tempo_curve") is not None
             and len(tc) == len(t["tempo_curve"]) else None,
             "reg_stored": S.get("rhythmic_regularity"), "reg_port": res["regularity"],
             "lab_eq": res["label"] == S.get("rhythmic_regularity_label"),
             "conf": abs(res["confidence"] - S.get("rhythmic_regularity_confidence", 0.0)),
             "mreg_stored": S.get("meter_rhythmic_regularity"), "mreg_port": res_m["regularity"],
             "mconf": abs(res_m["confidence"] - S.get("meter_rhythmic_regularity_confidence", 0.0)),
             "comp": res, "path": t["path"]}
        r["reg"] = (abs(r["reg_port"] - r["reg_stored"]) if (r["reg_port"] is not None and r["reg_stored"] is not None)
                    else (0.0 if r["reg_port"] is None and r["reg_stored"] is None else 9.0))
        r["mreg"] = (abs(r["mreg_port"] - r["mreg_stored"]) if (r["mreg_port"] is not None and r["mreg_stored"] is not None)
                     else (0.0 if r["mreg_port"] is None and r["mreg_stored"] is None else 9.0))
        rows.append(r)
    n = len(rows)

    def st(key, tol):
        v = [r[key] for r in rows if r[key] is not None]
        return {"n": len(v), "max_abs": float(max(v)) if v else None, "within_tol": int(sum(x <= tol for x in v)),
                "tol": tol}
    summ = {"set": tag, "tracks": n,
            "regularity": st("reg", 1e-5), "regularity_conf": st("conf", 1e-5),
            "label_equal": int(sum(r["lab_eq"] for r in rows)),
            "abstain_stored": int(sum(r["reg_stored"] is None for r in rows)),
            "meter_regularity": st("mreg", 1e-5), "meter_conf": st("mconf", 1e-5),
            "bpm_confidence": st("bc", 1e-5), "grid_stability": st("gs", 1e-6),
            "danceability": st("dance", 1e-5), "aggression_rhythm": st("arhy", 1e-5),
            "tempo_variability": st("tv", 1e-5), "tempo_curve": st("tc", 1e-2)}
    bad = [{k: r[k] for k in ("idx", "reg", "conf", "mreg", "bc", "dance", "arhy", "tv")}
           for r in rows if r["reg"] > 1e-5 or r["conf"] > 1e-5 or r["bc"] > 1e-5 or r["dance"] > 1e-5]
    return summ, bad, rows


def components_vs_v3(rows):
    """Labelled set: compare port components with eval/*_v3.tsv (0.3.6/0.3.7 eval output)."""
    ref = {}
    for name in ("straight_v3.tsv", "lomanka_v3.tsv", "lomanka_holdout.tsv"):
        p = EVAL / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ref[r["path"]] = r
    keys = [("score", "regularity"), ("confidence", "confidence"), ("concentration", "concentration"),
            ("coverage", "coverage"), ("alignment", "alignment"), ("periodicity", "periodicity"),
            ("stability", "stability"), ("windows_valid", "windows_valid"), ("windows_total", "windows_total")]
    out = {k: [] for k, _ in keys}
    n = 0
    for r in rows:
        v = ref.get(r["path"])
        if not v:
            continue
        n += 1
        for k_ref, k_port in keys:
            a, b = v.get(k_ref), r["comp"].get(k_port)
            if a in (None, "") or b is None:
                continue
            out[k_ref].append(abs(float(a) - float(b)))
    return {"matched": n, **{k: {"max_abs": float(max(v)) if v else None, "within_1e-3": int(sum(x <= 1e-3 for x in v)),
                                 "n": len(v)} for k, v in out.items()}}


def main() -> int:
    which = sys.argv[1:] or ["B", "L"]
    out = {}
    if "B" in which:
        s, bad, _ = check(dd.load_broken(), "broken_1000")
        out["broken_1000"] = {"summary": s, "mismatch_examples": bad[:20]}
        print(json.dumps(s, indent=1))
    if "L" in which:
        s, bad, rows = check(dd.load_labelled(), "labelled_145")
        out["labelled_145"] = {"summary": s, "mismatch_examples": bad[:20],
                               "components_vs_v3_tsv": components_vs_v3(rows)}
        print(json.dumps(s, indent=1))
        print(json.dumps(out["labelled_145"]["components_vs_v3_tsv"], indent=1))
    p = dd.RESULTS / "port_fidelity_downstream.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
