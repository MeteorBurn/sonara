"""Fix 2 constant: Sonara's beat / onset latency against synthetic ground truth only.

Reads ../data/synthetic/{truth,sonara}.json (stored outputs; read-only). Writes
results/latency.json. The constant is the median over the 15 files of each
file's median matched offset (Sonara beat - true beat, +-70 ms).
"""

from __future__ import annotations

import json
import sys

import numpy as np

import est_common as ec

SYN = ec.P.DATA / "synthetic"


def offsets(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    out = []
    for e in est:
        i = int(np.argmin(np.abs(ref - e)))
        if abs(ref[i] - e) <= tol:
            out.append(e - ref[i])
    return np.asarray(out)


def main() -> int:
    truth = json.loads((SYN / "truth.json").read_text(encoding="utf-8"))
    son = json.loads((SYN / "sonara.json").read_text(encoding="utf-8"))
    per_file = {}
    for name, t in truth.items():
        tb = np.asarray(t["beats"])
        bo = offsets(son[name]["beats"], tb)
        ev = np.sort(np.concatenate([tb, tb + 30.0 / t["bpm"]]))  # beats + offbeat hats
        oo = offsets(son[name]["onsets"], ev, tol=0.06)
        # frame-grid residual: true beat time mod one hop (sub-frame position)
        per_file[name] = {"bpm": t["bpm"], "sr": t["sr"],
                          "beat_offset_ms_median": float(np.median(bo) * 1000),
                          "beat_offset_ms_mean": float(np.mean(bo) * 1000),
                          "beat_offset_ms_p10": float(np.percentile(bo, 10) * 1000),
                          "beat_offset_ms_p90": float(np.percentile(bo, 90) * 1000),
                          "beats_matched": int(len(bo)), "beats_true": int(len(tb)),
                          "onset_offset_ms_median": float(np.median(oo) * 1000)}
    med = np.array([v["beat_offset_ms_median"] for v in per_file.values()])
    mean = np.array([v["beat_offset_ms_mean"] for v in per_file.values()])
    on = np.array([v["onset_offset_ms_median"] for v in per_file.values()])
    out = {
        "per_file": per_file,
        "beat_latency_ms": float(np.median(med)),
        "beat_latency_ms_mean_of_means": float(np.mean(mean)),
        "beat_latency_ms_range": [float(med.min()), float(med.max())],
        "onset_latency_ms": float(np.median(on)),
        "onset_latency_ms_range": [float(on.min()), float(on.max())],
        "hop_ms": ec.HOP_SEC * 1000,
        "note": "constant derived from synthetic ground truth only; never from MIK/Beat This!",
    }
    ec.RESULTS.mkdir(exist_ok=True)
    (ec.RESULTS / "latency.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in per_file.items():
        print(f"{k:28s} beat {v['beat_offset_ms_median']:+6.1f} ms (mean {v['beat_offset_ms_mean']:+6.1f}, "
              f"p10 {v['beat_offset_ms_p10']:+6.1f} p90 {v['beat_offset_ms_p90']:+6.1f})  "
              f"onset {v['onset_offset_ms_median']:+6.1f} ms  matched {v['beats_matched']}/{v['beats_true']}")
    print({k: v for k, v in out.items() if k != "per_file"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
