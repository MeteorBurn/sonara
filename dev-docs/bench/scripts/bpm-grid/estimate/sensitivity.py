"""Sensitivity checks (reported for context, never used to choose parameters):
1. fix 2 latency constant sweep, incl. the implementable one-frame shift (pad 3->2);
2. consensus inlier fraction (fix 1 by-product) as a BPM-error flag, per split,
   against the current bpm_confidence. Writes results/sensitivity.json."""

from __future__ import annotations

import json
import sys

import numpy as np

import est_common as ec
import evalkit as ek
import pipeline as pl


def auc(score, pos):
    s, y = np.asarray(score, float), np.asarray(pos, bool)
    p, n = s[y], s[~y]
    if len(p) == 0 or len(n) == 0:
        return None
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def main() -> int:
    D = ec.load_data()
    ek.prepare(D)
    out = {"latency_sweep": {}}
    consts = [0.0, 20.0, ec.HOP_SEC * 1000, pl.LATENCY_S * 1000, 26.0, 28.0, 30.0, 33.0]
    for split in ("holdout", "all"):
        rows = [t for i, t in D.items() if t["status"] == "ok" and t["comp"]["refs"] == "agree"
                and (split == "all" or not ec.is_dev(i))]
        res = {}
        for c in consts:
            f = {"F20": [], "F10": [], "MF20": [], "MF10": []}
            for t in rows:
                b = ec.frames_to_sec(t["beats"]) - c / 1000.0
                if len(t["bt"]["beats"]):
                    f["F20"].append(ec.f_measure(b, t["bt"]["beats"], 0.02) or 0.0)
                    f["F10"].append(ec.f_measure(b, t["bt"]["beats"], 0.01) or 0.0)
                g = t["_mik_grid"][0]
                if g is not None:
                    f["MF20"].append(ec.f_measure(b, g, 0.02) or 0.0)
                    f["MF10"].append(ec.f_measure(b, g, 0.01) or 0.0)
            res[f"{c:.2f}"] = {k: float(np.mean(v)) for k, v in f.items()}
        out["latency_sweep"][split] = res
        print(split, json.dumps(res, indent=0))
    # inlier fraction as a critical-error flag
    rows = json.loads((ec.RESULTS / "fix1_tracks.json").read_text(encoding="utf-8"))["rows"]
    rows = {r["idx"]: r for r in rows}
    out["inlier_fraction_auc"] = {}
    for split in ("dev", "holdout", "all"):
        q, conf, crit = [], [], []
        for i, r in rows.items():
            t = D[i]
            if t["comp"]["refs"] != "agree" or r["inlier_frac"] is None or (
                    split != "all" and ec.is_dev(i) != (split == "dev")):
                continue
            q.append(r["inlier_frac"])
            conf.append(t["bpm_conf"])
            crit.append(ec.is_critical(ec.category(t["bpm"], t["comp"]["ref_bpm"])))
        crit = np.asarray(crit)
        out["inlier_fraction_auc"][split] = {
            "n": len(q), "critical": int(crit.sum()),
            "auc_inlier_frac_noncritical_vs_critical": auc(q, ~crit),
            "auc_bpm_confidence_noncritical_vs_critical": auc(conf, ~crit)}
    print(json.dumps(out["inlier_fraction_auc"], indent=1))
    (ec.RESULTS / "sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
