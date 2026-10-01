"""EXPLORATORY (not one of the four requested fixes): beat-phase (antiphase) correction.

121/843 correct-tempo agree tracks have Sonara beats on the offbeat vs Beat This!;
synthetic straight 4/4 with offbeat open hats shows the same. Candidate rule: if
the low band (0-200 Hz, kick) is stronger half-way between Sonara's beats than on
them, move every beat to the midpoint. Variant/margin chosen on dev by mean
F +-70 ms vs Beat This!; holdout reported. Runs on the 1+2+3+4 beats (fix 4
re-tracking) so it measures what would remain. Writes results/fix5.json.
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np

import est_common as ec
import pipeline as pl
from fix3_downbeat import peak_at


def scores(t, beats):
    bands = t["onset_strength_bands"].astype(float)
    norm = bands / np.maximum(bands.mean(axis=1, keepdims=True), 1e-9)
    b = np.asarray(beats)
    mid = np.round((b[:-1] + b[1:]) / 2).astype(int)
    on = {"low": peak_at(norm[0], b[:-1]).mean(), "lowhi": (peak_at(norm[0], b[:-1])
                                                           - 0.5 * (peak_at(norm[3], b[:-1]) + peak_at(norm[4], b[:-1]))).mean()}
    off = {"low": peak_at(norm[0], mid).mean(), "lowhi": (peak_at(norm[0], mid)
                                                         - 0.5 * (peak_at(norm[3], mid) + peak_at(norm[4], mid))).mean()}
    return on, off, mid


def main() -> int:
    D = ec.load_data()
    C = pickle.loads((ec.RESULTS / "fix4_cands.pkl").read_bytes())
    recs = []
    for idx, t in sorted(D.items()):
        c = t["comp"]
        if t["status"] != "ok" or c["refs"] != "agree" or len(t["bt"]["beats"]) < 16:
            continue
        tempo, nb, sw = pl.fix4_retrack(t, C[idx])
        beats = np.asarray(nb if sw else t["beats"])
        if len(beats) < 16 or ec.is_critical(ec.category(tempo if sw else t["bpm"], c["ref_bpm"])):
            continue
        on, off, mid = scores(t, beats)
        bs = ec.frames_to_sec(beats) - pl.LATENCY_S
        ms = ec.frames_to_sec(mid) - pl.LATENCY_S
        recs.append({"idx": idx, "dev": ec.is_dev(idx), "on": on, "off": off,
                     "F_on": ec.f_measure(bs, t["bt"]["beats"]) or 0.0,
                     "F_mid": ec.f_measure(ms, t["bt"]["beats"]) or 0.0})
    res, best = {}, None
    for var in ("low", "lowhi"):
        for margin in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            for split in ("dev", "holdout", "all"):
                rs = [r for r in recs if split == "all" or r["dev"] == (split == "dev")]
                if var == "low":
                    flip = [r["off"][var] > r["on"][var] * (1 + margin) for r in rs]
                else:
                    flip = [r["off"][var] > r["on"][var] + margin for r in rs]
                f_before = np.mean([r["F_on"] for r in rs])
                f_after = np.mean([r["F_mid"] if fl else r["F_on"] for r, fl in zip(rs, flip)])
                good = sum(fl and r["F_mid"] > r["F_on"] + 0.3 for r, fl in zip(rs, flip))
                bad = sum(fl and r["F_on"] > r["F_mid"] + 0.3 for r, fl in zip(rs, flip))
                anti = sum(r["F_mid"] > r["F_on"] + 0.3 for r in rs)
                res[f"{var}/{margin}/{split}"] = {"n": len(rs), "F70_before": f_before, "F70_after": f_after,
                                                  "flipped": int(sum(flip)), "corrected": good, "broken": bad,
                                                  "antiphase_tracks": anti}
            d = res[f"{var}/{margin}/dev"]
            if best is None or d["F70_after"] > res[best]["F70_after"]:
                best = f"{var}/{margin}/dev"
    var, margin, _ = best.split("/")
    out = {"chosen": {"variant": var, "margin": float(margin)}, "dev": res[best],
           "holdout": res[f"{var}/{margin}/holdout"], "all": res[f"{var}/{margin}/all"], "grid": res}
    (ec.RESULTS / "fix5.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k in ("chosen", "dev", "holdout", "all"):
        print(k, out[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
