"""Extract SONARA rhythm data for a reference playlist into SQLite + JSON.

Temporary tool for tuning `rhythmic_regularity`. For every track it stores
every feature the Python bindings extract (Mixed In Key BPM range 79-192),
including the raw inputs the measure is computed from (multiband onset
envelopes, beats, downbeats), so the algorithm can later be re-run offline
without decoding audio again.

Writes into data/1_sonara/: run-<name>.sqlite (the full run: every scalar and
array; it is what makes the run resumable) and json_<name>/ (one JSON per
track). <name> is the set, plus a revision suffix for a new Sonara build
(broken_<sha>). The 0.3.7 baseline lives in json_broken/ and json_straight/
without its run database, so those names are refused.

Run (resumable: already analysed tracks are skipped):

    & <README machine paths: sonara_python> `
        "<repo>\\dev-docs\\bench\\scripts\\bpm-grid\\extract_reference.py"

Options: --playlist, --name, --workers, --threads, --limit, --retry-failed,
--export-json (only rebuild the JSON from the database), --revision (source
revision of the build, recorded as sonara_repo_head), --wheel (wheel whose
SHA-256 is recorded; default: README machine paths sonara_wheel). The path and
SHA-256 of the loaded native module are always recorded, whatever the flags say.
--features (comma list) replaces the main pass's PRESET features and
--no-meter-pass skips the separate time_signature pass; the run database
records the preset actually used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpm_grid_paths as P  # noqa: E402

WHEEL = P.SONARA_WHEEL  # README machine paths: sonara_wheel

# ---------------------------------------------------------------------------
# PRESET: every analysis parameter lives here and is recorded in the database.
# ---------------------------------------------------------------------------
PRESET = {
    "name": "all-features-mik-v1",
    "sr": 22050,
    # Mixed In Key BPM range, as DJTS SONARA_BPM_PRESETS["mixed-in-key"].
    "bpm_min": 79.0,
    "bpm_max": 192.0,
    "vocalness_model": "bundled",
    # Every feature the Python bindings expose, except `genre` (SONARA ships
    # no genre model) and `time_signature` (see METER_PASS).
    "features": [
        # full-mode features
        "bpm", "beats", "onsets", "rms", "dynamic_range", "centroid", "zcr",
        "onset_density", "bandwidth", "rolloff", "flatness", "contrast",
        "mfcc", "chroma", "chords", "dissonance", "energy", "danceability",
        "key", "valence", "acousticness", "tempo_curve",
        # opt-in-only features
        "beatgrid", "onset_bands", "rhythmic_regularity", "structure",
        "embedding", "aggression", "fingerprint", "loudness", "silence",
        "key_candidates", "vocalness", "mood", "instrumentalness", "tags",
    ],
    # Second pass. Requesting `time_signature` makes rhythmic_regularity and
    # beatgrid use the detected meter (synthetic 4/4 read as 2/4 halves the
    # bars and zeroes the confidence), so it runs separately and its outputs
    # are stored with the `meter_` prefix, beside the 4/4 main-pass values.
    "meter_pass": ["time_signature", "beatgrid", "rhythmic_regularity"],
}

# Lists longer than this are stored as compressed binary arrays, not JSON.
ARRAY_MIN_LEN = 64
# Stored in SQLite only (too large for the JSON file).
DB_ONLY_ARRAYS = {"onset_strength_bands"}


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
def _encode_array(value):
    """Return (dtype, shape, zlib bytes) for a numeric list or list of lists."""
    import numpy as np

    arr = np.asarray(value)
    if arr.dtype.kind in "iu":
        arr = arr.astype("<i4")
    else:
        arr = arr.astype("<f4")
    return arr.dtype.str, list(arr.shape), zlib.compress(arr.tobytes(), 6)


def _is_numeric_list(value) -> bool:
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if isinstance(first, list):
        return bool(first) and isinstance(first[0], (int, float))
    return isinstance(first, (int, float)) and not isinstance(first, bool)


def analyze_one(path: str, preset: dict):
    """Analyse one file; return a picklable dict (never raises)."""
    started = time.perf_counter()
    if not os.path.exists(path):
        return {"path": path, "status": "missing", "error": "file not found",
                "elapsed": 0.0}
    try:
        import sonara

        common = dict(sr=preset["sr"], bpm_min=preset["bpm_min"],
                      bpm_max=preset["bpm_max"])
        result = dict(sonara.analyze_file(
            path, features=preset["features"],
            vocalness_model=preset["vocalness_model"], **common))
        meter = (sonara.analyze_file(path, features=preset["meter_pass"], **common)
                 if preset["meter_pass"] else {})
        for key, value in meter.items():
            if key in ("time_signature", "time_signature_confidence"):
                result[key] = value
            elif key in ("downbeats", "grid_offset_sec", "grid_stability") \
                    or key.startswith("rhythmic_regularity"):
                result[f"meter_{key}"] = value
        scalars, arrays = {}, []
        for key, value in result.items():
            big = _is_numeric_list(value) and (
                key in DB_ONLY_ARRAYS or len(value) > ARRAY_MIN_LEN
                or isinstance(value[0], list))
            if big:
                dtype, shape, blob = _encode_array(value)
                arrays.append((key, dtype, shape, blob))
            else:
                scalars[key] = value
        return {"path": path, "status": "ok", "scalars": scalars,
                "arrays": arrays, "elapsed": time.perf_counter() - started}
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return {"path": path, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed": time.perf_counter() - started}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS run_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tracks (
    path         TEXT PRIMARY KEY,
    playlist_idx INTEGER NOT NULL,
    status       TEXT NOT NULL,          -- ok | error | missing
    error        TEXT,
    elapsed_sec  REAL,
    analyzed_at  TEXT,
    -- convenience columns, duplicated from scalars_json
    bpm                  REAL,
    bpm_confidence       REAL,
    duration_sec         REAL,
    grid_stability       REAL,
    regularity           REAL,
    regularity_label     TEXT,
    regularity_confidence REAL,
    scalars_json TEXT
);
CREATE TABLE IF NOT EXISTS arrays (
    path  TEXT NOT NULL REFERENCES tracks(path) ON DELETE CASCADE,
    name  TEXT NOT NULL,
    dtype TEXT NOT NULL,                 -- numpy dtype string, little endian
    shape TEXT NOT NULL,                 -- JSON list
    data  BLOB NOT NULL,                 -- zlib(raw array bytes)
    PRIMARY KEY (path, name)
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def store(con: sqlite3.Connection, idx: int, res: dict) -> None:
    s = res.get("scalars", {})
    with con:
        con.execute("DELETE FROM arrays WHERE path = ?", (res["path"],))
        con.execute(
            """INSERT OR REPLACE INTO tracks VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (res["path"], idx, res["status"], res.get("error"),
             round(res["elapsed"], 3),
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             s.get("bpm"), s.get("bpm_confidence"), s.get("duration_sec"),
             s.get("grid_stability"), s.get("rhythmic_regularity"),
             s.get("rhythmic_regularity_label"),
             s.get("rhythmic_regularity_confidence"),
             json.dumps(s, ensure_ascii=False) if s else None))
        con.executemany(
            "INSERT INTO arrays VALUES (?,?,?,?,?)",
            [(res["path"], n, d, json.dumps(sh), b)
             for n, d, sh, b in res.get("arrays", [])])


_BAD_NAME_CHARS = str.maketrans({c: "_" for c in '<>:"/\\|?*'})


def track_json_path(json_dir: Path, idx: int, path: str) -> Path:
    """`0001 - <file stem>.json`, numbered by playlist position (1-based)."""
    stem = Path(path).stem.translate(_BAD_NAME_CHARS).strip(" .")[:120]
    return json_dir / f"{idx + 1:04d} - {stem}.json"


def write_track_json(con: sqlite3.Connection, json_dir: Path, path: str) -> None:
    """Write one track's database content (scalars + arrays) to its own JSON."""
    import numpy as np

    row = con.execute(
        "SELECT playlist_idx, status, error, elapsed_sec, scalars_json "
        "FROM tracks WHERE path = ?", (path,)).fetchone()
    if row is None:
        return
    idx, status, error, elapsed, scalars_json = row
    entry = {"idx": idx, "path": path, "status": status}
    if error:
        entry["error"] = error
    entry["elapsed_sec"] = elapsed
    if scalars_json:
        entry.update(json.loads(scalars_json))
    for name, dtype, shape, data in con.execute(
            "SELECT name, dtype, shape, data FROM arrays WHERE path = ?",
            (path,)):
        if name in DB_ONLY_ARRAYS:
            entry[name] = {"stored_in": "sqlite", "shape": json.loads(shape)}
            continue
        arr = np.frombuffer(zlib.decompress(data), dtype=dtype)
        entry[name] = arr.reshape(json.loads(shape)).tolist()
    target = track_json_path(json_dir, idx, path)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(target)


def write_meta_json(con: sqlite3.Connection, json_dir: Path) -> None:
    meta = {k: _maybe_json(v) for k, v in con.execute("SELECT key, value FROM run_meta")}
    (json_dir / "_run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def export_json(con: sqlite3.Connection, json_dir: Path) -> int:
    """Rebuild every per-track JSON file from the database."""
    json_dir.mkdir(parents=True, exist_ok=True)
    write_meta_json(con, json_dir)
    paths = [p for (p,) in con.execute("SELECT path FROM tracks ORDER BY playlist_idx")]
    for path in paths:
        write_track_json(con, json_dir, path)
    return len(paths)


def _maybe_json(value: str):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def read_playlist(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    out, seen = [], set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        norm = str(Path(line))
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist", default=str(P.PLAYLIST_BROKEN))
    ap.add_argument("--name", required=True,
                    help="run name: the set plus a revision suffix, e.g. broken_<sha>; "
                         "writes run-<name>.sqlite and json_<name>/")
    ap.add_argument("--out-dir", default=str(P.SONARA_DIR))
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel processes (default 4 = ssd-balanced profile)")
    ap.add_argument("--threads", type=int, default=4,
                    help="RAYON_NUM_THREADS per process (HDD: try 1)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-analyse tracks stored as error/missing")
    ap.add_argument("--export-json", action="store_true",
                    help="only rebuild the per-track JSON files from the existing database")
    ap.add_argument("--revision",
                    help="source revision the analysing build was made from, recorded as "
                         "sonara_repo_head (default: recorded as unknown, with a warning)")
    ap.add_argument("--wheel", default=str(WHEEL) if WHEEL else None,
                    help="wheel the analysing environment was installed from; its SHA-256 "
                         "is recorded (default: README machine paths sonara_wheel)")
    ap.add_argument("--features",
                    help="comma-separated features of the main pass, instead of the PRESET list")
    ap.add_argument("--no-meter-pass", action="store_true",
                    help="skip the separate time_signature (meter_) pass")
    args = ap.parse_args()

    # The preset actually used: PRESET unless --features / --no-meter-pass change it.
    preset = dict(PRESET)
    if args.features:
        preset["features"] = [f.strip() for f in args.features.split(",") if f.strip()]
    if args.no_meter_pass:
        preset["meter_pass"] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"run-{args.name}.sqlite"
    json_dir = out_dir / f"json_{args.name}"
    if not db_path.exists():
        if args.export_json:
            print(f"ERROR: no run database {db_path}; nothing to export.")
            return 2
        if any(json_dir.glob("[0-9][0-9][0-9][0-9] - *.json")):
            print(f"ERROR: {json_dir} holds results of a run without a database "
                  "(e.g. the 0.3.7 baseline); use another --name.")
            return 2
    con = open_db(db_path)

    if args.export_json:
        n = export_json(con, json_dir)
        print(f"JSON rebuilt: {json_dir} ({n} files)")
        return 0

    # Children inherit this; it must be set before sonara (rayon) loads.
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import sonara
    import sonara._sonara as sonara_native

    # Provenance: the native module that actually runs is pinned by its hash.
    native_path = Path(sonara_native.__file__)
    wheel = Path(args.wheel) if args.wheel else None
    revision = args.revision
    if not revision:
        revision = "unknown (no --revision)"
        print(f"WARNING: no --revision given; sonara_repo_head is recorded as '{revision}'.")

    playlist_path = Path(args.playlist)
    paths = read_playlist(playlist_path)
    if args.limit:
        paths = paths[: args.limit]

    stored_preset = con.execute(
        "SELECT value FROM run_meta WHERE key = 'preset'").fetchone()
    if stored_preset and json.loads(stored_preset[0]) != preset:
        print("ERROR: this database was filled with a different PRESET.\n"
              f"       Use another --name or delete {db_path}.")
        return 2

    meta = {
        "preset": preset,
        "sonara_version": sonara.__version__,
        "sonara_module": sonara.__file__,
        "sonara_native_module": str(native_path),
        "sonara_module_sha256": sha256(native_path),
        "wheel": str(wheel) if wheel else None,
        "wheel_sha256": sha256(wheel) if wheel else None,
        "sonara_repo_head": revision,
        "playlist": str(playlist_path),
        "playlist_sha256": sha256(playlist_path),
        "playlist_tracks": len(paths),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "workers": args.workers,
        "rayon_threads": args.threads,
        "last_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with con:
        con.executemany("INSERT OR REPLACE INTO run_meta VALUES (?, ?)",
                        [(k, json.dumps(v, ensure_ascii=False)) for k, v in meta.items()])
    json_dir.mkdir(parents=True, exist_ok=True)
    write_meta_json(con, json_dir)

    skip_status = ("ok",) if args.retry_failed else ("ok", "error", "missing")
    done = {p for (p,) in con.execute(
        f"SELECT path FROM tracks WHERE status IN ({','.join('?' * len(skip_status))})",
        skip_status)}
    index = {p: i for i, p in enumerate(paths)}
    todo = [p for p in paths if p not in done]

    print(f"SONARA {sonara.__version__}  preset '{preset['name']}'  "
          f"sr={preset['sr']}  bpm={preset['bpm_min']}..{preset['bpm_max']}")
    if preset != PRESET:
        print(f"features : {','.join(preset['features'])}  meter pass: "
              f"{','.join(preset['meter_pass']) or 'skipped'}")
    print(f"revision : {revision}")
    print(f"module   : {native_path}  (sha256 {meta['sonara_module_sha256']})")
    print(f"wheel    : {meta['wheel']}  (sha256 {meta['wheel_sha256']})")
    print(f"playlist : {playlist_path}  ({len(paths)} tracks)")
    print(f"database : {db_path}")
    print(f"json     : {json_dir}  (one file per track)")
    print(f"workers  : {args.workers} x RAYON_NUM_THREADS={args.threads}")
    print(f"to do    : {len(todo)}  (already stored: {len(paths) - len(todo)})\n")

    counts = {"ok": 0, "error": 0, "missing": 0}
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyze_one, p, preset): p for p in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                store(con, index[res["path"]], res)
                write_track_json(con, json_dir, res["path"])
                counts[res["status"]] += 1

                wall = time.perf_counter() - started
                rate = n / wall if wall else 0.0
                eta = (len(todo) - n) / rate if rate else 0.0
                name = Path(res["path"]).name
                if res["status"] == "ok":
                    s = res["scalars"]
                    reg = s.get("rhythmic_regularity")
                    reg_txt = ("abstain  " if reg is None
                               else f"{reg:.3f} {s.get('rhythmic_regularity_label', ''):<9}")
                    info = (f"reg {reg_txt} conf {s.get('rhythmic_regularity_confidence', 0):.2f}"
                            f"  bpm {s.get('bpm', 0):6.1f}")
                else:
                    info = f"{res['status'].upper()}: {res.get('error', '')}"[:60]
                print(f"[{n:4d}/{len(todo)}] {n / len(todo):6.1%}  {res['elapsed']:5.1f}s  "
                      f"{info}  | {rate:4.2f} tr/s  ETA {fmt_eta(eta)} | {name[:60]}",
                      flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Stored results are kept; rerun to continue.")
        return 130
    finally:
        total = time.perf_counter() - started
        print(f"\nThis run: ok {counts['ok']}, error {counts['error']}, "
              f"missing {counts['missing']}  in {fmt_eta(total)}")
        summary = con.execute(
            "SELECT status, COUNT(*) FROM tracks GROUP BY status").fetchall()
        print(f"Database totals: {dict(summary)}")
        n_json = sum(1 for f in json_dir.glob("*.json") if f.name != "_run_meta.json")
        print(f"JSON files: {json_dir} ({n_json} tracks)")
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
