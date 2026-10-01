"""Independent cross-check of the ml-engineer estimate (own simple methods, read-only).

Fix 1: BPM = 60 / median slope of beat pairs 64 beats apart (no refits, no tuning).
Fix 3: bar phase = argmax over p of mean low-band (0-200 Hz) onset peak at beats
       with index = p (mod 4); accuracy = chosen phase matches Beat This! downbeats.
"""

import json
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_bpm as cb  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

rows = json.loads((P.DATA / "bpm_comparison.json").read_text(encoding="utf-8"))["rows"]
son = cb.sonara_rows(P.DATA / "broken_1000.sqlite")
bt = cb.load_json_dir(P.DATA / "beat_this")
con = sqlite3.connect(f"file:{P.DATA / 'broken_1000.sqlite'}?mode=ro", uri=True)
FR = 512 / 22050

ref_rows = [r for r in rows if r["refs"] == "agree" and r["sonara_bpm"]]
correct_level = [r for r in ref_rows if r["sonara_cat"] in ("exact", "fine", "minor")]

# ---- Fix 1 ----
cur, new = [], []
for r in correct_level:
    b = son[r["idx"]]["beats"]
    k = 64 if len(b) > 96 else max(4, len(b) // 2)
    slopes = (b[k:] - b[:-k]) / k
    bpm = cb.fold_to_range(60.0 / np.median(slopes), 79, 192)
    cur.append(abs(r["sonara_bpm"] - r["ref_bpm"]))
    new.append(abs(bpm - r["ref_bpm"]))
cur, new = np.array(cur), np.array(new)
print(f"Fix 1 (own median-of-64-beat-slopes), {len(new)} correct-level tracks:")
print(f"  exact<=0.05: current {np.mean(cur <= 0.05):.1%} -> {np.mean(new <= 0.05):.1%};"
      f"  |err| p50 {np.median(cur):.3f} -> {np.median(new):.4f}, p90 {np.percentile(cur, 90):.3f}"
      f" -> {np.percentile(new, 90):.3f}")

# ---- Fix 3 ----
def bands_of(path):
    name, dt, sh, data = con.execute(
        "SELECT name, dtype, shape, data FROM arrays WHERE path = ? AND name = 'onset_strength_bands'",
        (path,)).fetchone()
    return np.frombuffer(zlib.decompress(data), dtype=dt).reshape(json.loads(sh))

ok_cur = ok_new = n = 0
for r in ref_rows:
    if r["sonara_cat"] not in ("exact", "fine") or (r["beat_f"] or 0) < 0.9:
        continue
    s, b = son[r["idx"]], bt[r["idx"]]
    bd = np.asarray(b["downbeats_sec"])
    beats = s["beats"]
    if len(bd) < 8 or len(beats) < 16:
        continue
    low = bands_of(s["path"])[0]
    frames = np.round(beats / FR).astype(int)
    peak = np.array([low[max(0, f - 1):f + 2].max() if f < len(low) else 0.0 for f in frames])
    scores = [peak[p::4].mean() for p in range(4)]
    p_new = int(np.argmax(scores))
    # ground-truth phase: which residue class of Sonara beats sits on BT downbeats
    hits = [np.mean([np.min(np.abs(bd - t)) <= 0.07 for t in beats[p::4]]) for p in range(4)]
    p_true = int(np.argmax(hits))
    if hits[p_true] < 0.5:                       # BT downbeats not consistent with Sonara beats
        continue
    sd = s["downbeats"]
    p_cur = int(np.argmin([np.min(np.abs(beats[p::4][:, None] - sd[None, :]), axis=1).mean()
                           for p in range(4)]))
    n += 1
    ok_cur += p_cur == p_true
    ok_new += p_new == p_true
print(f"Fix 3 (own low-band mean rule), {n} tracks with correct tempo, F>=0.9 and a consistent BT bar:")
print(f"  bar-phase accuracy: current {ok_cur / n:.1%} -> low band {ok_new / n:.1%}")
