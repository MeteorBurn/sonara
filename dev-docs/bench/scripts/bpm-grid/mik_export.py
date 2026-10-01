"""Export Mixed In Key analysis of a playlist, read-only, to <dataset>/data/mik/.

Writes one JSON per track (`0001 - <file stem>.json`, numbered by playlist
position; same naming as data/beat_this/) and `_bpm.json` with only the BPM of
every track, two decimals. Tracks are matched by FilePathHash, the key MIK
itself uses: upper(md5(lower("<volume serial>:<path without drive letter>"))),
with backslashes; the forward-slash form is tried second for rows inserted by
mik_add_playlist.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bpm_grid_paths as P  # noqa: E402

MIK_DB = P.MIK_DB
DEFAULT_PLAYLIST = str(P.PLAYLIST_BROKEN)
DEFAULT_OUT = P.DATA / "mik"
_BAD = str.maketrans({c: "_" for c in '<>:"/\\|?*'})


def volume_serial(drive: str) -> str:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{drive}'\").VolumeSerialNumber"],
        capture_output=True, text=True, check=True).stdout.strip()


def read_playlist(path: Path) -> list[str]:
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", default=DEFAULT_PLAYLIST,
                    help="playlist with the ORIGINAL paths MIK analysed (FilePathHash key)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    PLAYLIST, OUT = Path(args.playlist), Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{MIK_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    paths = read_playlist(PLAYLIST)
    serials: dict[str, str] = {}
    summary, n_ok = [], 0
    for idx, path in enumerate(paths):
        drive = path[:2].upper()
        if drive not in serials:
            serials[drive] = volume_serial(drive)
        # MIK hashes the backslash form; rows inserted by mik_add_playlist.py
        # were hashed from the playlist's forward-slash form.
        row = None
        for rel in dict.fromkeys((path[2:].replace("/", "\\"), path[2:])):
            fph = hashlib.md5(f"{serials[drive]}:{rel}".lower().encode()).hexdigest().upper()
            row = con.execute("SELECT * FROM Song WHERE FilePathHash = ?", (fph,)).fetchone()
            if row is not None:
                break
        entry = {"idx": idx + 1, "path": path}
        if row is None:
            entry.update(status="not_in_mik", bpm=None)
        else:
            tempo = row["Tempo"]
            analyzed = bool(row["IsAnalyzed"]) and tempo is not None and tempo > 0
            entry.update(
                status="ok" if analyzed else "not_analyzed",
                bpm=round(tempo, 2) if analyzed else None,
                tempo_raw=tempo,
                key=row["MainKey"], key_confidence=row["MainKeyConfidence"],
                second_key=row["SecondKey"], key_summary=row["KeyResultSummary"],
                energy=row["OverallEnergy"], lufs=row["OverallVolumeLUFS"],
                last_analyzed_utc=row["LastAnalyzedUtc"],
                artist=row["ArtistName"], title=row["SongName"],
                sample_rate=row["SampleRate"], bitrate=row["Bitrate"],
                mik_song_id=row["Id"], file_path_hash=fph,
            )
            st = con.execute("SELECT Data FROM SerializedSongStructure WHERE SongId = ?",
                             (row["Id"],)).fetchone()
            if st:
                entry["structure"] = json.loads(st[0])
        n_ok += entry["status"] == "ok"
        stem = Path(path).stem.translate(_BAD).strip(" .")[:120]
        target = OUT / f"{idx + 1:04d} - {stem}.json"
        target.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        bpm_txt = "null" if entry["bpm"] is None else f"{entry['bpm']:.2f}"
        summary.append(f'  {{"idx": {idx + 1}, "path": {json.dumps(path, ensure_ascii=False)}, '
                       f'"bpm": {bpm_txt}}}')
    (OUT / "_bpm.json").write_text("[\n" + ",\n".join(summary) + "\n]\n", encoding="utf-8")
    meta = {"source": str(MIK_DB), "playlist": str(PLAYLIST), "tracks": len(paths),
            "with_bpm": n_ok, "volume_serials": serials,
            "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (OUT / "_run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"exported {len(paths)} tracks to {OUT}; {n_ok} with BPM")
    statuses: dict[str, int] = {}
    for line in summary:
        statuses["null" if '"bpm": null' in line else "bpm"] = statuses.get(
            "null" if '"bpm": null' in line else "bpm", 0) + 1
    print(statuses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
