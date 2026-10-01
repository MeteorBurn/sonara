"""Measure Sonara (and optionally Beat This!) against synthetic ground truth.

Usage:
  <sonara python> synthetic_measure.py sonara     -> data/synthetic/sonara.json
  <beat-this python> synthetic_measure.py beatthis -> data/synthetic/beatthis.json
  <any python with numpy> synthetic_measure.py report
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpm_grid_paths as P  # noqa: E402

D = P.DATA / "synthetic"
truth = json.loads((D / "truth.json").read_text(encoding="utf-8"))


def run_sonara():
    import sonara
    out = {}
    for name in truth:
        r = sonara.analyze_file(str(D / name), features=["bpm", "beats", "beatgrid", "onsets"],
                                bpm_min=79.0, bpm_max=192.0)
        hop, sr = r["provenance"]["hop_length"], r["provenance"]["sample_rate"]
        out[name] = {"bpm": r["bpm"],
                     "beats": [f * hop / sr for f in r["beats"]],
                     "downbeats": [f * hop / sr for f in r["downbeats"]],
                     "onsets": [f * hop / sr for f in r["onset_frames"]]}
    (D / "sonara.json").write_text(json.dumps(out), encoding="utf-8")


def run_beatthis():
    sys.path.insert(0, str(D.parent.parent))
    import beat_this_bpm as b
    from beat_this.inference import Audio2Beats
    a2b = Audio2Beats(checkpoint_path="final0", device="cuda", float16=False, dbn=False)
    out = {}
    for name in truth:
        beats, downbeats = a2b(b.decode(str(D / name)), b.SR)
        est = b.bpm_estimates(np.asarray(beats))
        out[name] = {"bpm": est["bpm_median"], "beats": list(map(float, beats)),
                     "downbeats": list(map(float, downbeats))}
    (D / "beatthis.json").write_text(json.dumps(out), encoding="utf-8")


def offsets(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    d = []
    for e in est:
        i = int(np.argmin(np.abs(ref - e)))
        if abs(ref[i] - e) <= tol:
            d.append(e - ref[i])
    return np.asarray(d)


def report():
    for tool in ("sonara", "beatthis"):
        f = D / f"{tool}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        print(f"\n== {tool}")
        print(f"{'file':28s} {'bpm':>8s} {'err':>7s} {'beat off ms':>11s} {'matched':>8s} {'beat-1 pos':>10s}"
              + ("  onset off ms" if tool == "sonara" else ""))
        for name, t in truth.items():
            r = res[name]
            tb = np.asarray(t["beats"])
            off = offsets(r["beats"], tb)
            # which true beat (0..3 within bar) does the tool call beat 1?
            pos = []
            for d in r["downbeats"]:
                i = int(np.argmin(np.abs(tb - d)))
                if abs(tb[i] - d) <= 0.07:
                    pos.append(i % 4)
            mode = max(set(pos), key=pos.count) if pos else None
            line = (f"{name:28s} {r['bpm']:8.3f} {r['bpm'] - t['bpm']:+7.3f} "
                    f"{np.median(off) * 1000 if len(off) else float('nan'):11.1f} "
                    f"{len(off):4d}/{len(tb):<4d} {str(mode):>10s}")
            if tool == "sonara":
                # onsets vs true kick/snare/hat event times (beats and offbeats)
                events = np.sort(np.concatenate([tb, tb + 30.0 / t["bpm"]]))
                oo = offsets(r["onsets"], events, tol=0.06)
                line += f"  {np.median(oo) * 1000:12.1f}"
            print(line)


if __name__ == "__main__":
    {"sonara": run_sonara, "beatthis": run_beatthis, "report": report}[sys.argv[1]]()
