"""Data loading for the downstream estimate (read-only on every input).

Two sets, one dict layout (the one ../est_common.load_data() uses):
  B  1000 broken-rhythm references: ../cache/data.pkl (+ the full stored
     scalar record from ../../data/broken_1000.sqlite as t["S"]);
  L  145 manually labelled tracks (100 straight / 30 lomanka / 15 lomanka
     holdout): data/labelled_145.sqlite (Sonara 0.3.7, same preset) and
     data/beat_this/*.json (Beat This! final0).
Each track gets t["set"] in {"B", "L"}, t["label"] in {"regular",
"irregular", None}, t["source"] and t["S"] (all stored Sonara scalars).
Cached in downstream/cache/*.pkl.
"""

from __future__ import annotations

import csv
import json
import pickle
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EST = HERE.parent
BUILD = EST.parent
sys.path.insert(0, str(EST))
import est_common as ec  # noqa: E402
from est_common import P  # noqa: E402

CACHE = P.DOWNSTREAM / "cache"
RESULTS = P.DOWNSTREAM / "results"
ARRAYS = ("beats", "downbeats", "onset_frames", "meter_downbeats", "onset_strength_bands", "tempo_curve")


def _read_sqlite(db: Path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for path, pidx, status, error, sj in con.execute(
            "SELECT path, playlist_idx, status, error, scalars_json FROM tracks"):
        rec = {"path": path, "pidx": pidx, "status": status, "error": error,
               "S": json.loads(sj) if sj else {}}
        arr = {}
        for name, dt, sh, blob in con.execute(
                "SELECT name, dtype, shape, data FROM arrays WHERE path = ?", (path,)):
            if name in ARRAYS:
                arr[name] = np.frombuffer(zlib.decompress(blob), dtype=dt).reshape(json.loads(sh)).copy()
        rec["A"] = arr
        out[pidx + 1] = rec
    con.close()
    return out


def load_broken():
    f = CACHE / "broken.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    D = ec.load_data()
    raw = _read_sqlite(P.DATA / "broken_1000.sqlite")
    for idx, t in D.items():
        r = raw.get(idx, {})
        S = r.get("S", {})
        for k in ("embedding", "chord_sequence", "chord_events", "segments", "fingerprint",
                  "key_candidates", "tags"):
            pass
        t["S"] = S
        t["tempo_curve"] = r.get("A", {}).get("tempo_curve")
        t["set"], t["label"], t["source"] = "B", None, "broken_1000"
        if "onset_frames" not in t and isinstance(S.get("onset_frames"), list):
            t["onset_frames"] = np.asarray(S["onset_frames"])
    CACHE.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(D, protocol=pickle.HIGHEST_PROTOCOL))
    return D


def _bt_json(folder: Path):
    out = {}
    for p in folder.glob("[0-9][0-9][0-9][0-9] - *.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[int(d["idx"])] = d
    return out


def load_labelled():
    f = CACHE / "labelled.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    raw = _read_sqlite(P.DOWNSTREAM / "data" / "labelled_145.sqlite")
    bt = _bt_json(P.DOWNSTREAM / "data" / "beat_this")
    labels = {}
    with (P.DOWNSTREAM / "labelled_145.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            labels[int(r["idx"])] = r
    D = {}
    for idx, r in sorted(raw.items()):
        S = r["S"]
        lab = labels[idx]
        assert lab["path"] == r["path"], (idx, lab["path"], r["path"])
        t = {"idx": idx, "path": r["path"], "status": r["status"], "error": r["error"], "S": S,
             "set": "L", "label": lab["expected"], "source": lab["source"], "tag": {}}
        if r["status"] == "ok":
            prov = S.get("provenance", {})
            t.update(sr=prov.get("sample_rate", 22050), hop=prov.get("hop_length", 512),
                     bpm=S.get("bpm"), bpm_raw=S.get("bpm_raw"), bpm_conf=S.get("bpm_confidence"),
                     cands=S.get("bpm_candidates") or [], duration=S.get("duration_sec"),
                     grid_offset=S.get("grid_offset_sec"), grid_stability=S.get("grid_stability"),
                     n_beats=S.get("n_beats"), band_edges=S.get("onset_band_edges_hz"),
                     time_signature=S.get("time_signature"))
            for name, a in r["A"].items():
                t[name] = a
            for name in ("beats", "downbeats", "onset_frames", "meter_downbeats", "tempo_curve"):
                if name not in t and isinstance(S.get(name), list):
                    t[name] = np.asarray(S[name])
        b = bt.get(idx, {})
        t["bt"] = {"status": b.get("status"), "bpm": b.get("bpm"), "bpm_raw": b.get("bpm_raw"),
                   "ibi_median": b.get("ibi_median_sec"),
                   "beats": np.asarray(b.get("beats_sec") or [], dtype=float),
                   "downbeats": np.asarray(b.get("downbeats_sec") or [], dtype=float)}
        t["comp"] = {"refs": "bt_only", "ref_bpm": None}
        t["mik"] = {}
        D[idx] = t
    CACHE.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(D, protocol=pickle.HIGHEST_PROTOCOL))
    return D


def is_dev_B(idx):
    return ec.is_dev(idx)
