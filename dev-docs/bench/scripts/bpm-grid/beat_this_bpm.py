"""Beat This! beat/downbeat tracking and exact BPM for a playlist.

Writes, under <dataset>/data/4_beat_this/json_broken/ (default; --out-dir for the
straight set; see bpm_grid_paths.py):
  - one JSON per track: `0001 - <file stem>.json` (beats, downbeats, BPM
    estimates, interval statistics), numbered by playlist position;
  - `_bpm.json`: only the BPM of every track, two decimals;
  - `_run_meta.json`: model, versions, preset.

Audio is decoded by the ffmpeg CLI straight to mono float32 at 22050 Hz (the
model's own rate), so no torchaudio/torchcodec backend is involved. Decoding
runs in background threads while the GPU processes the previous track.

Run (resumable: tracks that already have a JSON are skipped):

    & "<repo>\\dev-docs\\envs\\beat-this-venv\\Scripts\\python.exe" `
        "<repo>\\dev-docs\\bench\\scripts\\bpm-grid\\beat_this_bpm.py"

BPM definition (basic, no fitting)
  bpm_raw    60 / median inter-beat interval of the Beat This! beats.
  bpm        bpm_raw folded by x2 / x0.5 into the Mixed In Key range
             79-192 BPM, the same preset SONARA is run with. This is the
             value written to _bpm.json.
  Beat This! places beats on its 50 fps frame grid, so the median interval
  moves in 20 ms steps and this BPM is correspondingly coarse.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpm_grid_paths as P  # noqa: E402

FFMPEG = P.FFMPEG
SR = 22050
MIK_RANGE = (79.0, 192.0)
_BAD_NAME_CHARS = str.maketrans({c: "_" for c in '<>:"/\\|?*'})


# ---------------------------------------------------------------------------
# Audio and BPM
# ---------------------------------------------------------------------------
def decode(path: str) -> np.ndarray:
    """Decode any format to mono float32 at 22050 Hz via the ffmpeg CLI."""
    proc = subprocess.run(
        [str(FFMPEG), "-nostdin", "-v", "error", "-i", path,
         "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg: {msg[-1] if msg else 'no audio decoded'}")
    return np.frombuffer(proc.stdout, dtype="<f4")


def fold_to_range(bpm: float, lo: float, hi: float) -> float:
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


def bpm_estimates(beats: np.ndarray) -> dict:
    """Basic BPM from beat times: 60 / median inter-beat interval."""
    out = {"bpm_median": None, "ibi_median_sec": None, "ibi_std_sec": None,
           "ibi_cv": None}
    if len(beats) < 3:
        return out
    ibi = np.diff(beats)
    med = float(np.median(ibi))
    if med <= 0:
        return out
    out["bpm_median"] = 60.0 / med
    out["ibi_median_sec"] = med
    out["ibi_std_sec"] = float(np.std(ibi))
    out["ibi_cv"] = float(np.std(ibi) / np.mean(ibi))
    return out


def beats_per_bar(beats: np.ndarray, downbeats: np.ndarray) -> dict:
    """Histogram of beats between consecutive downbeats (meter evidence)."""
    if len(downbeats) < 2:
        return {}
    counts = np.searchsorted(beats, downbeats[1:] - 1e-3) - np.searchsorted(
        beats, downbeats[:-1] - 1e-3)
    values, freq = np.unique(counts, return_counts=True)
    return {str(int(v)): int(f) for v, f in zip(values, freq)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def read_playlist(path: Path) -> list[str]:
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            norm = str(Path(line))
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def track_json_path(out_dir: Path, idx: int, path: str) -> Path:
    stem = Path(path).stem.translate(_BAD_NAME_CHARS).strip(" .")[:120]
    return out_dir / f"{idx + 1:04d} - {stem}.json"


def write_json(target: Path, payload: dict) -> None:
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(target)


def write_bpm_summary(out_dir: Path, paths: list[str]) -> tuple[int, int]:
    """_bpm.json: idx, path and BPM (two decimals) of every analysed track."""
    lines, n_ok = [], 0
    for idx, path in enumerate(paths):
        f = track_json_path(out_dir, idx, path)
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        bpm = d.get("bpm")
        bpm_txt = "null" if bpm is None else f"{bpm:.2f}"
        n_ok += bpm is not None
        lines.append(f'  {{"idx": {idx + 1}, "path": {json.dumps(path, ensure_ascii=False)}, '
                     f'"bpm": {bpm_txt}}}')
    text = "[\n" + ",\n".join(lines) + "\n]\n"
    tmp = out_dir / "_bpm.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_dir / "_bpm.json")
    return len(lines), n_ok


def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist", default=str(P.PLAYLIST_BROKEN))
    ap.add_argument("--out-dir", default=str(P.BEAT_THIS_DIR / "json_broken"))
    ap.add_argument("--model", default="final0", help="Beat This! checkpoint")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--float16", action="store_true", help="half precision on GPU")
    ap.add_argument("--decoders", type=int, default=3,
                    help="background ffmpeg decode threads")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tracks")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-analyse tracks whose JSON records an error")
    ap.add_argument("--summary-only", action="store_true",
                    help="only rebuild _bpm.json from the per-track files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = Path(args.playlist)
    paths = read_playlist(playlist_path)
    if args.limit:
        paths = paths[: args.limit]

    if args.summary_only:
        n, n_ok = write_bpm_summary(out_dir, paths)
        print(f"_bpm.json rebuilt: {n} tracks, {n_ok} with BPM")
        return 0

    import torch
    from beat_this.inference import Audio2Beats

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    todo = []
    for idx, path in enumerate(paths):
        f = track_json_path(out_dir, idx, path)
        if f.exists():
            if not args.retry_failed:
                continue
            if json.loads(f.read_text(encoding="utf-8")).get("status") == "ok":
                continue
        todo.append((idx, path))

    meta = {
        "tool": "Beat This! (CPJKU/beat_this)",
        "beat_this_version": md.version("beat-this"),
        "model": args.model,
        "postprocessing": "minimal (no DBN)",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "float16": args.float16,
        "torch": torch.__version__,
        "decoder": f"{FFMPEG} -> mono f32le {SR} Hz",
        "bpm_range_mik": list(MIK_RANGE),
        "playlist": str(playlist_path),
        "playlist_tracks": len(paths),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "last_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(out_dir / "_run_meta.json", meta)

    print(f"Beat This! {meta['beat_this_version']}  model {args.model}  "
          f"device {device}{' (' + meta['gpu'] + ')' if meta['gpu'] else ''}"
          f"{'  float16' if args.float16 else ''}")
    print(f"playlist : {playlist_path}  ({len(paths)} tracks)")
    print(f"output   : {out_dir}")
    print(f"to do    : {len(todo)}  (already done: {len(paths) - len(todo)})")
    print("loading model (first run downloads the checkpoint)...", flush=True)
    audio2beats = Audio2Beats(checkpoint_path=args.model, device=device,
                              float16=args.float16, dbn=False)
    print()

    def load(item):
        idx, path = item
        t0 = time.perf_counter()
        try:
            return idx, path, decode(path), None, time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            return idx, path, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - t0

    counts = {"ok": 0, "error": 0}
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.decoders)) as pool:
            # map() keeps playlist order and decodes ahead in the background.
            for n, (idx, path, signal, error, t_decode) in enumerate(
                    pool.map(load, todo), 1):
                entry = {"idx": idx + 1, "path": path}
                t0 = time.perf_counter()
                if error is None:
                    try:
                        beats, downbeats = audio2beats(signal, SR)
                        beats = np.asarray(beats, dtype=np.float64)
                        downbeats = np.asarray(downbeats, dtype=np.float64)
                        est = bpm_estimates(beats)
                        raw = est["bpm_median"]
                        bpm = None if raw is None else fold_to_range(raw, *MIK_RANGE)
                        entry.update({
                            "status": "ok",
                            "bpm": None if bpm is None else round(bpm, 2),
                            "bpm_raw": None if raw is None else round(raw, 2),
                            "bpm_method": "60 / median inter-beat interval",
                            "duration_sec": round(len(signal) / SR, 3),
                            "n_beats": int(len(beats)),
                            "n_downbeats": int(len(downbeats)),
                            "beats_per_bar": beats_per_bar(beats, downbeats),
                            "ibi_median_sec": est["ibi_median_sec"],
                            "ibi_std_sec": est["ibi_std_sec"],
                            "ibi_cv": est["ibi_cv"],
                            "beats_sec": [round(float(b), 4) for b in beats],
                            "downbeats_sec": [round(float(b), 4) for b in downbeats],
                        })
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                if error is not None:
                    entry.update({"status": "error", "error": error, "bpm": None})
                entry["decode_sec"] = round(t_decode, 3)
                entry["infer_sec"] = round(time.perf_counter() - t0, 3)
                write_json(track_json_path(out_dir, idx, path), entry)
                counts[entry["status"]] += 1

                wall = time.perf_counter() - started
                rate = n / wall if wall else 0.0
                eta = (len(todo) - n) / rate if rate else 0.0
                if entry["status"] == "ok":
                    bpm_txt = "  none " if entry["bpm"] is None else f"{entry['bpm']:7.2f}"
                    raw = entry["bpm_raw"]
                    folded = "" if raw is None or abs(raw - (entry["bpm"] or 0)) < 0.01 \
                        else f" (raw {raw:.2f})"
                    info = f"BPM {bpm_txt}{folded:<13} beats {entry['n_beats']:4d}"
                else:
                    info = f"ERROR: {entry['error']}"[:48]
                print(f"[{n:4d}/{len(todo)}] {n / len(todo):6.1%}  "
                      f"dec {t_decode:4.1f}s gpu {entry['infer_sec']:4.1f}s  {info}  "
                      f"| {rate:4.2f} tr/s ETA {fmt_eta(eta)} | {Path(path).name[:55]}",
                      flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished tracks are kept; rerun to continue.")
    finally:
        n, n_ok = write_bpm_summary(out_dir, paths)
        print(f"\nThis run: ok {counts['ok']}, error {counts['error']}  "
              f"in {fmt_eta(time.perf_counter() - started)}")
        print(f"_bpm.json: {out_dir / '_bpm.json'}  ({n} tracks, {n_ok} with BPM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
