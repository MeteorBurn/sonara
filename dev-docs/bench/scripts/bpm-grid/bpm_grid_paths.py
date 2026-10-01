"""Where the bpm-grid harness reads and writes.

The harness source lives here (tracked, `dev-docs/bench/scripts/bpm-grid/`).
Everything it reads or produces lives in the dataset directory, never next to
the scripts:

    <repo>/dev-docs/bench/datasets/bpm-grid/     (override: SONARA_BPM_GRID_DATA)
        data/            Sonara, Beat This! and MIK data, comparison table
        playlists/       playlist copies + SSD copy maps
        mik-backups/     MIKStore.db backups taken by mik_add_playlist.py (on demand)
        paths.json       machine-specific tool and drive locations (see below)

paths.json keys (all optional; %VAR% is expanded):
    ssd_root, ffmpeg, mik_db, beat_this_python, sonara_python, sonara_wheel
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[3]
DATASET = Path(
    os.environ.get("SONARA_BPM_GRID_DATA") or REPO / "dev-docs" / "bench" / "datasets" / "bpm-grid"
)

DATA = DATASET / "data"
PLAYLISTS = DATASET / "playlists"
MIK_BACKUPS = DATASET / "mik-backups"

PLAYLIST_BROKEN = PLAYLISTS / "djts-playlist-broken.txt"
PLAYLIST_STRAIGHT = PLAYLISTS / "djts-playlist-straight.txt"

_CONFIG_FILE = DATASET / "paths.json"
_CONFIG = json.loads(_CONFIG_FILE.read_text(encoding="utf-8")) if _CONFIG_FILE.is_file() else {}


def machine(key: str, default: str | None = None) -> Path | None:
    """A machine-specific location from paths.json, or `default`."""
    value = _CONFIG.get(key, default)
    return Path(os.path.expandvars(value)) if value else None


SSD_ROOT = machine("ssd_root")
FFMPEG = machine("ffmpeg", "ffmpeg")
MIK_DB = machine("mik_db", r"%LOCALAPPDATA%\Mixed In Key\Mixed In Key\11.0\MIKStore.db")
SONARA_WHEEL = machine("sonara_wheel")
