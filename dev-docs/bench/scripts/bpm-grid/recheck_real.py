"""Re-check the diagnosis on real tracks (read-only)."""

import base64
import collections
import json
import sys
from pathlib import Path

import mutagen
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_bpm as cb  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

rows = {r["idx"]: r for r in json.loads(
    (P.DATA / "bpm_comparison.json").read_text(encoding="utf-8"))["rows"]}
son = cb.sonara_rows(P.DATA / "broken_1000.sqlite")
bt = cb.load_json_dir(P.DATA / "beat_this")
mik = cb.load_json_dir(P.DATA / "mik")
syn_truth = json.loads((P.DATA / "synthetic" / "truth.json").read_text(encoding="utf-8"))
syn_son = json.loads((P.DATA / "synthetic" / "sonara.json").read_text(encoding="utf-8"))


def line_fit_bpm(beats):
    b = np.asarray(beats)
    slope = np.polyfit(np.arange(len(b)), b, 1)[0]
    return 60.0 / slope


# ---- A. Does a line through Sonara's own beats give the exact tempo? ----
print("A. Line fit through Sonara's own beats")
print("   synthetic (truth known):")
for name, t in syn_truth.items():
    if name.endswith("_22050.wav"):
        print(f"     {t['bpm']:6.1f} BPM: reported {syn_son[name]['bpm']:8.3f}   "
              f"beats line fit {line_fit_bpm(syn_son[name]['beats']):8.3f}")
ok = [r for r in rows.values() if r["refs"] == "agree" and r["sonara_cat"] in ("exact", "fine", "minor")]
rep, fit, contiguous = [], [], []
for r in ok:
    b = son[r["idx"]].get("beats")
    if b is None or len(b) < 50:
        continue
    ibi = np.diff(b)
    # contiguous = no gap longer than 1.5 beats (line fit by index is then valid)
    contiguous.append(bool(np.max(ibi) < 1.5 * np.median(ibi)))
    rep.append(abs(r["sonara_bpm"] - r["ref_bpm"]))
    fit.append(abs(cb.fold_to_range(line_fit_bpm(b), 79, 192) - r["ref_bpm"]))
rep, fit, contiguous = map(np.asarray, (rep, fit, contiguous))
print(f"   real, {len(fit)} tracks at the correct metrical level (reference = MIK):")
for label, m in (("all", np.ones(len(fit), bool)), ("contiguous beat lists", contiguous),
                 ("lists with gaps", ~contiguous)):
    if m.any():
        print(f"     {label:22s} n={m.sum():3d}  reported |err| p50 {np.median(rep[m]):.3f} "
              f"p90 {np.percentile(rep[m], 90):.3f} exact(<=0.05) {np.mean(rep[m] <= 0.05):.0%}   |   "
              f"beats line fit |err| p50 {np.median(fit[m]):.3f} p90 {np.percentile(fit[m], 90):.3f} "
              f"exact {np.mean(fit[m] <= 0.05):.0%}")

# ---- B. Phase offset vs Beat This! by source sample rate ----
print("\nB. Sonara - Beat This! beat offset by source sample rate (real tracks, correct tempo)")
by_sr = collections.defaultdict(list)
for r in ok:
    if r["phase_ms"] is not None:
        by_sr[mik.get(r["idx"], {}).get("sample_rate")].append(r["phase_ms"])
for sr, v in sorted(by_sr.items(), key=lambda kv: str(kv[0])):
    print(f"   {str(sr):>6s} Hz  n={len(v):3d}  median {np.median(v):+.1f} ms  p10 {np.percentile(v, 10):+.1f}"
          f"  p90 {np.percentile(v, 90):+.1f}")

# ---- C. Downbeats: MIK cue points as an independent witness ----
print("\nC. MIK cue points vs downbeats (cues are placed on phrase starts)")


def tag(path, name):
    f = mutagen.File(path)
    t = f.tags if f else None
    if t is None:
        return None
    if hasattr(t, "getall"):
        fr = [x for x in t.getall("GEOB") if x.desc.lower() == name]
        raw = fr[0].data if fr else None
    else:
        v = t.get(name)
        raw = v[0].encode() if v else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(base64.b64decode(raw))


votes = collections.Counter()
per_track = []
for idx, r in rows.items():
    if r["refs"] != "agree" or r["path"] is None:
        continue
    cues = tag(r["path"], "cuepoints")
    if not cues:
        continue
    b = bt[idx]
    bb, bd = np.asarray(b["beats_sec"]), np.asarray(b["downbeats_sec"])
    sd = son[idx].get("downbeats")
    if len(bd) < 4 or sd is None:
        continue
    hit_bt = hit_son = n = 0
    for c in cues["cues"]:
        t = c["time"] / 1000.0
        if t < 5.0:                                    # skip the track start
            continue
        n += 1
        hit_bt += bool(np.min(np.abs(bd - t)) <= 0.07)
        hit_son += bool(np.min(np.abs(np.asarray(sd) - t)) <= 0.07)
    if n:
        per_track.append((idx, n, hit_bt, hit_son))
        votes["cue on Beat This! downbeat"] += hit_bt
        votes["cue on Sonara downbeat"] += hit_son
        votes["cues"] += n
print(f"   tracks with MIK cues: {len(per_track)}; {dict(votes)}")

# ---- D. The 0.80 (4:5) group ----
print("\nD. Sonara/ref ratio ~0.80 group")
grp = [r for r in rows.values() if r["refs"] == "agree" and r["sonara_bpm"]
       and abs(r["sonara_bpm"] / r["ref_bpm"] - 0.8) < 0.01]
con = __import__("sqlite3").connect(f"file:{P.DATA / 'broken_1000.sqlite'}?mode=ro", uri=True)
in_cand = 0
for r in grp[:12]:
    s = json.loads(con.execute("SELECT scalars_json FROM tracks WHERE playlist_idx = ?",
                               (r["idx"] - 1,)).fetchone()[0])
    cands = s.get("bpm_candidates") or []
    print(f"   #{r['idx']:4d} ref {r['ref_bpm']:7.2f}  sonara {r['sonara_bpm']:7.2f} raw {s.get('bpm_raw'):7.2f}"
          f"  beats-fit {cb.fold_to_range(line_fit_bpm(son[r['idx']]['beats']), 79, 192):7.2f}"
          f"  candidates {[round(c[0], 1) for c in cands]}")
for r in grp:
    s = json.loads(con.execute("SELECT scalars_json FROM tracks WHERE playlist_idx = ?",
                               (r["idx"] - 1,)).fetchone()[0])
    cands = [c[0] for c in (s.get("bpm_candidates") or [])]
    in_cand += any(abs(cb.fold_to_range(c, 79, 192) / r["ref_bpm"] - 1) < 0.02 for c in cands)
vals = np.array([r["sonara_bpm"] for r in grp])
print(f"   n={len(grp)}; correct tempo among Sonara candidates: {in_cand}; "
      f"Sonara BPM values: median {np.median(vals):.2f} range {vals.min():.2f}-{vals.max():.2f}")
