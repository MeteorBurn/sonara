"""Fidelity of the numpy port: re-run Sonara's tempo + DP + downbeat picker on the
broadband envelope reconstructed from the stored onset bands and compare with
the stored Sonara 0.3.7 outputs. Writes results/port_fidelity.json."""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import est_common as ec


def main() -> int:
    D = ec.load_data()
    rows, t0 = [], time.perf_counter()
    for idx, t in sorted(D.items()):
        if t["status"] != "ok" or "onset_strength_bands" not in t:
            continue
        oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
        est, beats, downs, fpb = ec.sonara_pipeline(oenv)
        sb, sd = np.asarray(t["beats"]), np.asarray(t["downbeats"])
        top_stored = [round(c[0], 3) for c in t["cands"]]
        top_port = [round(c[0], 3) for c in est["top5"]]
        score_rel = (max(abs(a[1] - b[1]) / max(abs(a[1]), 1e-9)
                         for a, b in zip(t["cands"], est["top5"]))
                     if len(t["cands"]) == len(est["top5"]) and t["cands"] else None)
        same_len = len(beats) == len(sb)
        rows.append({
            "idx": idx,
            "bpm_stored": t["bpm"], "bpm_port": est["tempo"],
            "bpm_absdiff": abs(est["tempo"] - t["bpm"]),
            "cands_equal": top_stored == top_port, "cand_score_maxrel": score_rel,
            "beats_equal": bool(same_len and np.array_equal(beats, sb)),
            "beats_frac_common": len(np.intersect1d(beats, sb)) / max(len(sb), 1),
            "n_beats_stored": int(len(sb)), "n_beats_port": int(len(beats)),
            "downbeats_equal": bool(len(downs) == len(sd) and np.array_equal(downs, sd)),
        })
    n = len(rows)
    summ = {
        "tracks": n,
        "bpm_absdiff_max": max(r["bpm_absdiff"] for r in rows),
        "bpm_within_1e-3": sum(r["bpm_absdiff"] < 1e-3 for r in rows),
        "cands_equal": sum(r["cands_equal"] for r in rows),
        "cand_score_maxrel_p99": float(np.percentile([r["cand_score_maxrel"] for r in rows
                                                     if r["cand_score_maxrel"] is not None], 99)),
        "beats_identical": sum(r["beats_equal"] for r in rows),
        "beats_frac_common_p1": float(np.percentile([r["beats_frac_common"] for r in rows], 1)),
        "beats_frac_common_min": min(r["beats_frac_common"] for r in rows),
        "downbeats_identical": sum(r["downbeats_equal"] for r in rows),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    bad = [r for r in rows if not (r["beats_equal"] and r["downbeats_equal"]
                                   and r["bpm_absdiff"] < 1e-3)]
    ec.RESULTS.mkdir(exist_ok=True)
    (ec.RESULTS / "port_fidelity.json").write_text(
        json.dumps({"summary": summ, "mismatches": bad}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    for r in bad[:15]:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
