"""Add playlist tracks to the Mixed In Key 11 library as unanalysed songs.

Rows mirror what MIK itself stores for a not-yet-analysed track, and every
track is also put into one MIK collection named after the playlist. MIK must be
closed; the database is backed up first. Tracks already in the library (same
FilePathHash) are only added to the collection.

FilePathHash = upper(md5(lower("<volume serial>:<path without drive letter>")))
(verified on all 44,497 rows of the previous MIK database, both slash styles).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import bpm_grid_paths as P

MIK_DB = P.MIK_DB
BACKUP_DIR = P.MIK_BACKUPS


def volume_info(drive: str) -> tuple[str, str, int]:
    """(serial, label, is_removable) of a drive such as 'M:'."""
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         f"$d = Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='{drive}'\"; "
         "\"$($d.VolumeSerialNumber)|$($d.VolumeName)|$($d.DriveType)\""],
        capture_output=True, text=True, check=True).stdout.strip()
    serial, label, dtype = out.split("|")
    if not serial:
        raise RuntimeError(f"no volume serial for {drive}")
    return serial, label, int(dtype == "2")


def file_path_hash(path: str, serial: str) -> str:
    return hashlib.md5(f"{serial}:{path[2:]}".lower().encode("utf-8")).hexdigest().upper()


def utc(ts: float | None = None) -> str:
    moment = (dt.datetime.now(dt.timezone.utc) if ts is None
              else dt.datetime.fromtimestamp(ts, dt.timezone.utc))
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def read_playlist(path: Path) -> list[str]:
    out, seen = [], set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and line not in seen:
            seen.add(line)
            out.append(line)          # keep MIK's own form: "M:/Volumes/..."
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", default=str(P.PLAYLIST_BROKEN))
    ap.add_argument("--collection", default=None, help="default: playlist file stem")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    running = subprocess.run(["tasklist", "/FI", "IMAGENAME eq MixedInKey.exe", "/NH"],
                             capture_output=True, text=True).stdout
    if "MixedInKey.exe" in running:
        print("ERROR: Mixed In Key is running; close it first.")
        return 2

    playlist = Path(args.playlist)
    name = args.collection or playlist.stem
    paths = read_playlist(playlist)
    volumes: dict[str, tuple[str, str, int]] = {}

    con = sqlite3.connect(MIK_DB)
    con.execute("PRAGMA foreign_keys=ON")
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        print("ERROR: MIK database integrity check failed; nothing changed.")
        return 2
    existing = dict(con.execute("SELECT FilePathHash, Id FROM Song"))

    songs, members, missing = [], [], []
    added_at = utc()
    coll = con.execute("SELECT Id FROM Collection WHERE Name = ? AND IsFolder = 0 "
                       "AND LibraryTypeId = 1 AND IsLibrary = 0", (name,)).fetchone()
    coll_id = coll[0] if coll else str(uuid.uuid4())
    already_member = {s for (s,) in con.execute(
        "SELECT SongId FROM SongCollectionMembership WHERE CollectionId = ?", (coll_id,))}
    for seq, path in enumerate(paths):
        p = Path(path)
        if not p.is_file():
            missing.append(path)
            continue
        drive = path[:2].upper()
        if drive not in volumes:
            volumes[drive] = volume_info(drive)
        serial, label, removable = volumes[drive]
        fph = file_path_hash(path, serial)
        song_id = existing.get(fph)
        if song_id is None:
            song_id = str(uuid.uuid4())
            existing[fph] = song_id
            artist, sep, title = p.stem.partition(" - ")
            if not sep:
                artist, title = "", p.stem
            st = p.stat()
            songs.append({
                "Id": song_id, "File": path, "FilePathHash": fph,
                "ArtistName": artist, "SongName": title,
                "OverallVolume": 0.0, "EnergySegmentsCount": 0, "StandardPitch": 0.0,
                "KeyResultSummary": "", "DateAdded": added_at, "ClippedPeaksCount": 0,
                "LastAnalyzedUtc": "0001-01-01 00:00:00", "MainKey": "-1A",
                "MainKeyConfidence": 0.0, "SecondKey": "-1A", "SecondKeyConfidence": 0.0,
                "IsAnalyzed": 0, "HasPNTag": 0, "PNTagIsProcessed": 0,
                "PNTagAppliedClipRepair": 0, "PNTagVolumeAnalysisVersion": 0,
                "PNTagVolumeUnits": "", "PNTagOutputVolume": 0.0,
                "LastModifiedUtc": utc(st.st_mtime), "DiskIsRemovable": removable,
                "DiskLabel": label, "DiskSerialNumber": serial,
                "FileType": p.suffix.lower(), "FileSize": st.st_size,
            })
        if song_id not in already_member:
            already_member.add(song_id)
            members.append((str(uuid.uuid4()), song_id, coll_id, seq))

    print(f"playlist   : {playlist} ({len(paths)} tracks)")
    print(f"collection : {name} ({'existing' if coll else 'new'})")
    print(f"new songs  : {len(songs)}   already in library: {len(paths) - len(missing) - len(songs)}")
    print(f"memberships: {len(members)}   missing files: {len(missing)}")
    for m in missing[:10]:
        print("  missing:", m)
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / f"MIKStore.db.before-{playlist.stem}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    con.execute(f"VACUUM INTO '{backup}'")
    print(f"backup     : {backup}")

    with con:
        if not coll:
            seq = con.execute("SELECT COALESCE(MAX(Sequence), 0) + 1 FROM Collection "
                              "WHERE LibraryTypeId = 1 AND IsLibrary = 0").fetchone()[0]
            con.execute("INSERT INTO Collection (Id, ExternalId, Name, Emoji, Sequence, "
                        "LibraryTypeId, IsLibrary, IsFolder, ParentFolderId) "
                        "VALUES (?, NULL, ?, NULL, ?, 1, 0, 0, NULL)", (coll_id, name, seq))
        if songs:
            cols = list(songs[0])
            con.executemany(
                f"INSERT INTO Song ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [tuple(s[c] for c in cols) for s in songs])
        con.executemany("INSERT INTO SongCollectionMembership (Id, SongId, CollectionId, "
                        "Sequence) VALUES (?, ?, ?, ?)", members)
    in_coll = con.execute("SELECT COUNT(*) FROM SongCollectionMembership "
                          "WHERE CollectionId = ?", (coll_id,)).fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM Song").fetchone()[0]
    check = con.execute("PRAGMA integrity_check").fetchone()[0]
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    con.close()
    print(f"written    : collection '{name}' holds {in_coll} tracks; library has {total} songs")
    print(f"integrity  : {check}; foreign-key violations: {len(fk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
