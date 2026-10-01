"""Before/after table: baseline, each fix alone, cumulative 1 -> 4 (and an extra
grid-snap row) on dev / holdout / all, with bootstrap 95% CIs and paired
holdout differences vs baseline. Writes results/combined.json."""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

import est_common as ec
import evalkit as ek
import pipeline as pl

VARIANTS = [("baseline", set(), False), ("fix1", {1}, False), ("fix2", {2}, False),
            ("fix3", {3}, False), ("fix4", {4}, False), ("1+2", {1, 2}, False),
            ("1+2+3", {1, 2, 3}, False), ("1+2+3+4", {1, 2, 3, 4}, False),
            ("1+2+3+4+snap", {1, 2, 3, 4}, True)]
KEYS = ["exact", "le05", "critical", "F70", "F20", "F10", "DF70", "MF70", "MF20", "MF10"]


def per_track_vec(rows, key):
    out = []
    for r in rows:
        if not r["agree"]:
            continue
        if key == "exact":
            out.append(r["cat"] == "exact")
        elif key == "le05":
            out.append(r["cat"] in ("exact", "fine"))
        elif key == "critical":
            out.append(r["critical"])
        else:
            out.append(r.get(key))
    return out


def main() -> int:
    D = ec.load_data()
    ek.prepare(D)
    C = pickle.loads((ec.RESULTS / "fix4_cands.pkl").read_bytes())
    all_rows, t0 = {}, time.perf_counter()
    switched = {}
    for name, fx, snap in VARIANTS:
        rows = []
        for idx, t in sorted(D.items()):
            if t["status"] != "ok":
                c = t["comp"]
                rows.append({"idx": idx, "dev": ec.is_dev(idx), "agree": c["refs"] == "agree",
                             "cat": "n/a", "critical": False, "abs_err": None, "ref_bpm": c["ref_bpm"]})
                continue
            out = pl.run(t, fx, cands=C.get(idx), snap=snap)
            m = ek.track_metrics(t, out)
            m["switched"] = out["switched"]
            m["inlier_frac"] = out["inlier_frac"]
            m["phase_choice"] = out["phase"]
            rows.append(m)
        all_rows[name] = rows
        switched[name] = sum(bool(r.get("switched")) for r in rows)
        print(f"{name:14s} done {time.perf_counter() - t0:.0f}s", flush=True)
    table = {name: {s: ek.aggregate(rows, s) for s in ("dev", "holdout", "all")}
             for name, rows in all_rows.items()}
    # paired differences on holdout (and all) vs baseline, per key metric
    diffs = {}
    for name, rows in all_rows.items():
        if name == "baseline":
            continue
        diffs[name] = {}
        for split in ("holdout", "all"):
            base = ek.subset(all_rows["baseline"], split)
            cand = ek.subset(rows, split)
            diffs[name][split] = {}
            for key in KEYS:
                a = per_track_vec(base, key)
                b = per_track_vec(cand, key)
                pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                if not pairs:
                    continue
                aa, bb = zip(*pairs)
                diffs[name][split][key] = ec.boot_diff(np.asarray(aa, float), np.asarray(bb, float))
    # transitions for fix 4 (critical before -> after) on holdout/all
    trans = {}
    for name in ("fix4", "1+2+3+4"):
        for split in ("dev", "holdout", "all"):
            b = {r["idx"]: r for r in ek.subset(all_rows["baseline"], split) if r["agree"]}
            a = {r["idx"]: r for r in ek.subset(all_rows[name], split) if r["agree"]}
            trans[f"{name}/{split}"] = {
                "fixed": sum(b[i]["critical"] and not a[i]["critical"] for i in b),
                "regressed": sum((not b[i]["critical"]) and a[i]["critical"] for i in b),
                "switched": sum(bool(a[i].get("switched")) for i in b)}
    out = {"table": table, "paired_diff_vs_baseline": diffs, "fix4_transitions": trans,
           "switched": switched, "latency_s": pl.LATENCY_S,
           "params": {"fix1": pl.FIX1, "fix3": pl.FIX3, "fix4": pl.FIX4}}
    (ec.RESULTS / "combined.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    with (ec.RESULTS / "combined_rows.pkl").open("wb") as f:
        pickle.dump(all_rows, f)
    for split in ("holdout", "all"):
        print(f"\n=== {split}")
        for name in all_rows:
            a = table[name][split]
            f = lambda k: (a[k]["mean"] if isinstance(a.get(k), dict) and a[k] else float("nan"))
            print(f"{name:14s} exact {f('exact_rate'):.3f} le0.5 {f('le_0.5_rate'):.3f} crit {f('critical_rate'):.3f} "
                  f"p50 {a['abs_err_noncrit_p50']:.3f} p90 {a['abs_err_noncrit_p90']:.3f} | F70 {f('F70_mean'):.3f} "
                  f"F20 {f('F20_mean'):.3f} F10 {f('F10_mean'):.3f} ph {a['phase_ms_median']:+.1f} | "
                  f"MF70 {a['MF70_mean']:.3f} MF20 {a['MF20_mean']:.3f} MF10 {a['MF10_mean']:.3f} "
                  f"mph {a['mik_phase_ms_median']:+.1f} | DF70 {f('DF70_mean'):.3f} cue {a['cue_hit_rate']:.3f}")
    print("\nfix4 transitions:", json.dumps(trans))
    return 0


if __name__ == "__main__":
    sys.exit(main())
