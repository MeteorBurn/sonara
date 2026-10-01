"""Read Mixed In Key tags (cue points, beat grid) from the playlist files, read-only.

Writes estimate/cache/mik_tags.json: {idx: {"cues": [sec...], "grid": [sec...]}}.
Audio files are opened by mutagen for reading only; nothing is written to them.
Tag formats (as in ../diagnose_sonara.py and ../recheck_real.py): FLAC/Vorbis
comment `cuepoints` / `beatgrid`, or ID3 GEOB frames `CuePoints` / `BeatGrid`;
payload is JSON or base64 JSON; cue times are in milliseconds, grid beats in s.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import mutagen

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
sys.path.insert(0, str(BUILD))
import bpm_grid_paths as P  # noqa: E402

OUT = P.ESTIMATE / "cache" / "mik_tags.json"


def decode(raw):
    if not raw:
        return None
    if isinstance(raw, str):
        raw = raw.encode()
    for f in (lambda r: json.loads(r), lambda r: json.loads(base64.b64decode(r))):
        try:
            return f(raw)
        except Exception:  # noqa: BLE001
            continue
    return None


def read(path: str) -> dict:
    f = mutagen.File(path)
    t = f.tags if f else None
    out = {}
    if t is None:
        return out
    for name in ("cuepoints", "beatgrid"):
        if hasattr(t, "getall"):
            fr = [x for x in t.getall("GEOB") if x.desc.lower() == name]
            raw = fr[0].data if fr else None
        else:
            v = t.get(name)
            raw = v[0] if v else None
        out[name] = decode(raw)
    return out


def main() -> int:
    rows = json.loads((P.DATA / "bpm_comparison.json").read_text(encoding="utf-8"))["rows"]
    res, t0 = {}, time.perf_counter()
    for n, r in enumerate(rows, 1):
        if not r.get("path"):
            continue
        entry = {}
        try:
            tags = read(r["path"])
            cues = tags.get("cuepoints")
            grid = tags.get("beatgrid")
            if cues and isinstance(cues, dict) and cues.get("cues"):
                entry["cues"] = [c["time"] / 1000.0 for c in cues["cues"] if "time" in c]
                entry["cues_raw"] = cues
            if grid and isinstance(grid, dict) and grid.get("beats"):
                entry["grid"] = [float(b) for b in grid["beats"]]
                entry["grid_keys"] = sorted(grid.keys())
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        res[r["idx"]] = entry
        if n % 100 == 0:
            print(f"{n}/{len(rows)}  {time.perf_counter() - t0:.1f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res), encoding="utf-8")
    print("cues:", sum("cues" in v for v in res.values()),
          "grids:", sum("grid" in v for v in res.values()),
          "errors:", sum("error" in v for v in res.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
