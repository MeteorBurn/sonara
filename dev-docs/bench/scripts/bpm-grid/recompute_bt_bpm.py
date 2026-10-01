"""Recompute Beat This! BPM fields from the stored beat times (no model run).

Applies the BPM definition currently in beat_this_bpm.py (basic: 60 / median
inter-beat interval, folded into the MIK range 79-192) to every
data/beat_this/NNNN - *.json, drops fields of earlier definitions, and rewrites
_bpm.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beat_this_bpm as b  # noqa: E402
import bpm_grid_paths as P  # noqa: E402

STALE = ("bpm_fit_exact", "bpm_median", "phase_coherence", "fit_beats_used",
         "fit_residual_ms")
out_dir = P.DATA / "beat_this"
changed = 0
for f in sorted(out_dir.glob("[0-9][0-9][0-9][0-9] - *.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    if d.get("status") != "ok":
        continue
    est = b.bpm_estimates(np.asarray(d["beats_sec"], dtype=np.float64))
    raw = est["bpm_median"]
    old = d.get("bpm")
    for key in STALE:
        d.pop(key, None)
    d.update({
        "bpm": None if raw is None else round(b.fold_to_range(raw, *b.MIK_RANGE), 2),
        "bpm_raw": None if raw is None else round(raw, 2),
        "bpm_method": "60 / median inter-beat interval",
        "ibi_median_sec": est["ibi_median_sec"],
        "ibi_std_sec": est["ibi_std_sec"],
        "ibi_cv": est["ibi_cv"],
    })
    changed += old != d["bpm"]
    b.write_json(f, d)
paths = b.read_playlist(P.PLAYLIST_BROKEN)
n, n_ok = b.write_bpm_summary(out_dir, paths)
print(f"recomputed; BPM changed on {changed} tracks; _bpm.json: {n} tracks, {n_ok} with BPM")
