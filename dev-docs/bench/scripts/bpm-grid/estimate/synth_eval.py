"""Apply baseline / each fix / cumulative fixes to the synthetic Sonara outputs and
score them against exact ground truth. Writes results/synthetic.json."""

from __future__ import annotations

import json
import sys

import numpy as np

import est_common as ec
import pipeline as pl

D = ec.SYNTHETIC
VARIANTS = {"baseline": set(), "fix1": {1}, "fix2": {2}, "fix3": {3}, "fix4": {4},
            "1+2": {1, 2}, "1+2+3": {1, 2, 3}, "1+2+3+4": {1, 2, 3, 4}}


def offsets(est, ref, tol):
    est, ref = np.asarray(est), np.asarray(ref)
    out = []
    for e in est:
        i = int(np.argmin(np.abs(ref - e)))
        if abs(ref[i] - e) <= tol:
            out.append(e - ref[i])
    return np.asarray(out)


def main() -> int:
    truth = json.loads((D / "truth.json").read_text(encoding="utf-8"))
    son = json.loads((D / "sonara_out.json").read_text(encoding="utf-8"))
    res = {}
    port_ok = 0
    for name, tr in truth.items():
        s = son[name]
        t = {"bpm": s["bpm"], "cands": s["cands"], "beats": np.asarray(s["beats"]),
             "downbeats": np.asarray(s["downbeats"]), "onset_frames": np.asarray(s["onset_frames"]),
             "onset_strength_bands": np.asarray(s["onset_strength_bands"], dtype=np.float32),
             "band_edges": s["band_edges"]}
        oenv = ec.broadband(t["onset_strength_bands"], t["band_edges"])
        _, pb, pd, _ = ec.sonara_pipeline(oenv)
        port_ok += bool(np.array_equal(pb, t["beats"]) and np.array_equal(pd, t["downbeats"]))
        tb, td = np.asarray(tr["beats"]), np.asarray(tr["downbeats"])
        grid16 = np.sort(np.concatenate([tb + q * 60.0 / tr["bpm"] for q in (0, 0.25, 0.5, 0.75)]))
        res[name] = {"pattern": tr["pattern"], "true_bpm": tr["bpm"], "sr": tr["sr"]}
        for vname, fx in VARIANTS.items():
            out = pl.run(t, fx)
            bo = offsets(out["beats"], tb, 0.07)
            oo = offsets(out["onsets"], grid16, 0.06)
            dpos = [int(np.argmin(np.abs(tb - d))) % 4 for d in out["downbeats"]
                    if np.min(np.abs(tb - d)) <= 0.07]
            res[name][vname] = {
                "bpm": out["bpm"], "bpm_err": out["bpm"] - tr["bpm"],
                "cat": ec.category(out["bpm"], tr["bpm"]),
                "F70": ec.f_measure(out["beats"], tb, 0.07, skip=0.0),
                "F20": ec.f_measure(out["beats"], tb, 0.02, skip=0.0),
                "F10": ec.f_measure(out["beats"], tb, 0.01, skip=0.0),
                "beat_offset_ms": float(np.median(bo) * 1000) if len(bo) else None,
                "DF70": ec.f_measure(out["downbeats"], td, 0.07, skip=0.0),
                "downbeat_pos": max(set(dpos), key=dpos.count) if dpos else None,
                "onset_offset_ms": float(np.median(oo) * 1000) if len(oo) else None,
                "switched": out["switched"]}
    (ec.RESULTS / "synthetic.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"port reproduces Sonara beats+downbeats on {port_ok}/{len(truth)} synthetic files")
    hdr = f"{'file':24s}" + "".join(f"{v:>26s}" for v in ("baseline", "1+2+3", "1+2+3+4"))
    print(hdr)
    for name, r in res.items():
        line = f"{name[:-4]:24s}"
        for v in ("baseline", "1+2+3", "1+2+3+4"):
            x = r[v]
            line += (f"  {x['bpm_err']:+7.3f} {x['beat_offset_ms'] or 0:+5.1f}ms F10 {x['F10'] or 0:.2f}"
                     f" d{x['downbeat_pos']}")
        print(line)
    # per pattern summary
    summ = {}
    for pat in ("breaks", "four", "four_bass", "dnb"):
        rs = [r for r in res.values() if r["pattern"] == pat]
        summ[pat] = {}
        for v in VARIANTS:
            xs = [r[v] for r in rs]
            summ[pat][v] = {
                "n": len(xs), "exact": sum(abs(x["bpm_err"]) <= 0.05 for x in xs),
                "critical": sum(ec.is_critical(x["cat"]) for x in xs),
                "max_abs_err_noncrit": max([abs(x["bpm_err"]) for x in xs if not ec.is_critical(x["cat"])],
                                           default=None),
                "beat_offset_ms_median": float(np.median([x["beat_offset_ms"] for x in xs
                                                          if x["beat_offset_ms"] is not None])),
                "F20_mean": float(np.mean([x["F20"] or 0 for x in xs])),
                "F10_mean": float(np.mean([x["F10"] or 0 for x in xs])),
                "downbeat_correct": sum(x["downbeat_pos"] == 0 for x in xs),
                "downbeat_positions": [x["downbeat_pos"] for x in xs],
                "onset_offset_ms_median": float(np.median([x["onset_offset_ms"] for x in xs
                                                           if x["onset_offset_ms"] is not None])),
                "switched": sum(bool(x["switched"]) for x in xs)}
    (ec.RESULTS / "synthetic_summary.json").write_text(json.dumps(summ, indent=1, default=float),
                                                       encoding="utf-8")
    for pat, d in summ.items():
        print(f"\n== {pat}")
        for v, x in d.items():
            print(f"  {v:8s} exact {x['exact']}/{x['n']} crit {x['critical']} maxerr {x['max_abs_err_noncrit']} "
                  f"off {x['beat_offset_ms_median']:+.1f}ms F20 {x['F20_mean']:.2f} F10 {x['F10_mean']:.2f} "
                  f"downbeat ok {x['downbeat_correct']}/{x['n']} pos {x['downbeat_positions']} "
                  f"onset {x['onset_offset_ms_median']:+.1f}ms sw {x['switched']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
