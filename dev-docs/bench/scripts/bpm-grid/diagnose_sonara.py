"""Read-only diagnostics of Sonara's BPM / grid errors against the references."""

import base64
import json
import sys
from pathlib import Path

import mutagen
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_bpm as cb  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

rows = json.loads((P.DATA / "bpm_comparison.json").read_text(encoding="utf-8"))["rows"]
son = cb.sonara_rows(P.DATA / "broken_1000.sqlite")
bt = cb.load_json_dir(P.DATA / "beat_this")
HOP_SEC = 512 / 22050

# ---- 1. BPM precision: is the error tied to the autocorrelation lag grid? ----
ok = [r for r in rows if r["refs"] == "agree" and r["sonara_cat"] in ("exact", "fine", "minor")]
err = np.array([r["sonara_bpm"] - r["ref_bpm"] for r in ok])
ref = np.array([r["ref_bpm"] for r in ok])
lag = 60.0 / (ref * HOP_SEC)                       # true period in frames
frac = lag - np.round(lag)                         # position between integer lags
print("1. BPM precision on", len(ok), "tracks with correct metrical level")
print("   signed err median %.3f  mean %.3f  |err| p50 %.3f p90 %.3f" % (
    np.median(err), err.mean(), np.median(abs(err)), np.percentile(abs(err), 90)))
for lo, hi in ((-0.5, -0.25), (-0.25, -0.05), (-0.05, 0.05), (0.05, 0.25), (0.25, 0.5)):
    m = (frac >= lo) & (frac < hi)
    if m.any():
        print(f"   lag fraction [{lo:+.2f},{hi:+.2f}): n={m.sum():3d}  signed err median {np.median(err[m]):+.3f}"
              f"  |err| median {np.median(abs(err[m])):.3f}")
# Sonara's own beats: does the beat sequence carry a better tempo than the reported bpm?
sb = np.array([r["sonara_beats_fit_bpm"] - r["ref_bpm"] for r in ok if r["sonara_beats_fit_bpm"]])
print("   Sonara beats median-interval BPM minus ref: |err| p50 %.3f (frame-quantized)" % np.median(abs(sb)))
# slope of Sonara beats vs index on clean tracks: exact period carried by the beats
fits = []
for r in ok:
    b = son.get(r["idx"], {}).get("beats")
    if b is None or len(b) < 50:
        continue
    ibi = np.diff(b)
    if np.std(ibi) / np.mean(ibi) > 0.05:          # only very regular beat lists
        continue
    slope = np.polyfit(np.arange(len(b)), b, 1)[0]
    fits.append(cb.fold_to_range(60 / slope, 79, 192) - r["ref_bpm"])
fits = np.array(fits)
print("   Sonara beats line fit (regular lists only, n=%d): |err| p50 %.3f p90 %.3f" % (
    len(fits), np.median(abs(fits)), np.percentile(abs(fits), 90)))
print("   share with |sonara bpm - beats fit| > 0.1:",
      np.round(np.mean([abs(f) > 0.1 for f in fits]), 2) if len(fits) else None)

# ---- 2. Grid phase: who is offset? Use MIK beat grids found in file tags ----
print("\n2. Grid phase against MIK beatgrid tags")
def mik_grid(path):
    f = mutagen.File(path)
    t = f.tags if f else None
    if t is None:
        return None
    if hasattr(t, "getall"):
        fr = [x for x in t.getall("GEOB") if x.desc.lower() == "beatgrid"]
        raw = fr[0].data if fr else None
    else:
        v = t.get("beatgrid")
        raw = v[0].encode() if v else None
    if not raw:
        return None
    try:
        g = json.loads(raw)
    except Exception:
        g = json.loads(base64.b64decode(raw))
    return np.asarray(g["beats"])

def offset(est, refb):
    _, ph = cb.f_measure(np.asarray(est), np.asarray(refb))
    return ph

res = []
for r in rows:
    s, b = son.get(r["idx"], {}), bt.get(r["idx"], {})
    if r["refs"] != "agree" or s.get("beats") is None or not b.get("beats_sec"):
        continue
    g = mik_grid(r["path"])
    if g is None:
        continue
    res.append((r["idx"], offset(s["beats"], g), offset(b["beats_sec"], g), r["sonara_cat"]))
for idx, so, bo, cat in res:
    print(f"   #{idx:4d}  Sonara-MIK {so if so is None else round(so,1)} ms   BeatThis-MIK "
          f"{bo if bo is None else round(bo,1)} ms   sonara bpm {cat}")
so = [x[1] for x in res if x[1] is not None]
bo = [x[2] for x in res if x[2] is not None]
print("   median Sonara-MIK %.1f ms (n=%d), median BeatThis-MIK %.1f ms (n=%d)" % (
    np.median(so), len(so), np.median(bo), len(bo)))

# ---- 3. Downbeats: which beat of Beat This!'s bar does Sonara call beat 1? ----
print("\n3. Downbeat phase (Sonara downbeat position within Beat This! bar)")
counts = {}
for r in rows:
    if r["refs"] != "agree" or r["sonara_cat"] not in ("exact", "fine") or (r["beat_f"] or 0) < 0.9:
        continue
    s, b = son[r["idx"]], bt[r["idx"]]
    bb, bd = np.asarray(b["beats_sec"]), np.asarray(b["downbeats_sec"])
    if len(bd) < 4 or s.get("downbeats") is None:
        continue
    # beat number (0..3) of each BT beat inside its bar
    num = np.searchsorted(bd, bb + 1e-3, side="right") - 1
    pos = np.array([np.sum((bb > bd[k]) & (bb < bb[i] + 1e-3)) if k >= 0 else -1
                    for i, k in enumerate(num)])
    hits = []
    for d in s["downbeats"]:
        i = int(np.argmin(abs(bb - d)))
        if abs(bb[i] - d) <= 0.07 and pos[i] >= 0:
            hits.append(int(pos[i]) % 4)
    if hits:
        mode = max(set(hits), key=hits.count)
        counts[mode] = counts.get(mode, 0) + 1
print("   tracks by Sonara beat-1 position in Beat This! bar (0 = same downbeat):", dict(sorted(counts.items())))

# ---- 4. Multiple errors: tempo range and confidence ----
print("\n4. Critical Sonara errors on reference tracks")
crit = [r for r in rows if r["refs"] == "agree" and r["sonara_critical"]]
for cat in ("x2/3", "x4/3", "major"):
    sub = [r for r in crit if r["sonara_cat"] == cat]
    if not sub:
        continue
    refs = np.array([r["ref_bpm"] for r in sub]); conf = np.array([r["sonara_bpm_conf"] or 0 for r in sub])
    print(f"   {cat:5s} n={len(sub):3d}  ref BPM median {np.median(refs):6.1f} [{refs.min():.0f}-{refs.max():.0f}]"
          f"  sonara conf median {np.median(conf):.2f}")
good = [r["sonara_bpm_conf"] or 0 for r in rows if r["refs"] == "agree" and r["sonara_cat"] in ("exact", "fine")]
print("   sonara conf median on correct tracks %.2f" % np.median(good))
maj = [r for r in crit if r["sonara_cat"] == "major"]
ratios = np.array([r["sonara_bpm"] / r["ref_bpm"] for r in maj])
print("   major ratios:", np.round(np.sort(ratios), 3).tolist())
