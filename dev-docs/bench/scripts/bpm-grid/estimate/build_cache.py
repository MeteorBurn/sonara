"""Build a read-only snapshot of everything the estimate needs, in one pickle.

Reads (read-only): ../data/broken_1000.sqlite (URI mode=ro), ../data/beat_this,
../data/mik, ../data/bpm_comparison.json. Writes estimate/cache/data.pkl.
"""

from __future__ import annotations

import json
import pickle
import re
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
sys.path.insert(0, str(BUILD))
import bpm_grid_paths as P  # noqa: E402

DATA = P.DATA
OUT = P.ESTIMATE / "cache" / "data.pkl"
ARRAYS = ("beats", "downbeats", "onset_frames", "meter_downbeats", "onset_strength_bands")


def ts(s: str) -> float:
    """'HH:MM:SS.fffffff' -> seconds."""
    m = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", s)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def load_json_dir(folder: Path) -> dict[int, dict]:
    out = {}
    for f in folder.glob("[0-9][0-9][0-9][0-9] - *.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[int(d["idx"])] = d
    return out


def main() -> int:
    con = sqlite3.connect(f"file:{DATA / 'broken_1000.sqlite'}?mode=ro", uri=True)
    bt = load_json_dir(DATA / "beat_this")
    mik = load_json_dir(DATA / "mik")
    comp = {r["idx"]: r for r in json.loads(
        (DATA / "bpm_comparison.json").read_text(encoding="utf-8"))["rows"]}
    tracks = {}
    for path, pidx, status, error, sj in con.execute(
            "SELECT path, playlist_idx, status, error, scalars_json FROM tracks"):
        idx = pidx + 1
        t = {"idx": idx, "path": path, "status": status, "error": error}
        if sj:
            s = json.loads(sj)
            prov = s.get("provenance", {})
            t.update(sr=prov.get("sample_rate", 22050), hop=prov.get("hop_length", 512),
                     bpm=s.get("bpm"), bpm_raw=s.get("bpm_raw"),
                     bpm_conf=s.get("bpm_confidence"), cands=s.get("bpm_candidates") or [],
                     duration=s.get("duration_sec"), grid_offset=s.get("grid_offset_sec"),
                     grid_stability=s.get("grid_stability"), n_beats=s.get("n_beats"),
                     band_edges=s.get("onset_band_edges_hz"),
                     time_signature=s.get("time_signature"),
                     tags=s.get("tags"))
            for name, dt, sh, blob in con.execute(
                    "SELECT name, dtype, shape, data FROM arrays WHERE path = ?", (path,)):
                if name in ARRAYS:
                    t[name] = np.frombuffer(zlib.decompress(blob), dtype=dt).reshape(
                        json.loads(sh)).copy()
            for name in ("beats", "downbeats", "onset_frames", "meter_downbeats"):
                if name not in t and isinstance(s.get(name), list):
                    t[name] = np.asarray(s[name], dtype=np.int64)
        b = bt.get(idx, {})
        t["bt"] = {"status": b.get("status"), "bpm": b.get("bpm"), "bpm_raw": b.get("bpm_raw"),
                   "ibi_median": b.get("ibi_median_sec"),
                   "beats": np.asarray(b.get("beats_sec") or [], dtype=float),
                   "downbeats": np.asarray(b.get("downbeats_sec") or [], dtype=float)}
        m = mik.get(idx, {})
        st = m.get("structure") or {}
        t["mik"] = {"status": m.get("status"), "bpm": m.get("bpm"), "tempo_raw": m.get("tempo_raw"),
                    "sample_rate": m.get("sample_rate"),
                    "seg_starts": [ts(e["StartTime"]) for e in st.get("EnergySegments", [])],
                    "aseg_starts": [ts(e["StartTime"]) for e in st.get("AnalysisEnergySegments", [])]}
        c = comp.get(idx, {})
        t["comp"] = {k: c.get(k) for k in ("refs", "ref_bpm", "sonara_cat", "beat_this_bpm",
                                           "mik_bpm", "bt_vs_mik", "beat_f", "phase_ms",
                                           "downbeat_f")}
        tracks[idx] = t
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as f:
        pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)
    ok = sum(1 for t in tracks.values() if t["status"] == "ok")
    print(f"tracks {len(tracks)}  ok {ok}  bands {sum('onset_strength_bands' in t for t in tracks.values())}"
          f"  -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
