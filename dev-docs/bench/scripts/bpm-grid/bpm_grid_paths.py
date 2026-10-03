"""Where the bpm-grid harness reads and writes.

The harness source lives here (tracked, `dev-docs/bench/scripts/bpm-grid/`).
Everything it reads or produces lives in the dataset directory, never next to
the scripts:

    <repo>/dev-docs/bench/datasets/bpm-grid-2026-10/     (override: SONARA_BPM_GRID_DATA)
        data/            comparison table and one folder per program: 1_sonara,
                         2_mixed_in_key, 3_rekordbox, 4_beat_this; per-track JSON
                         in <folder>/json_<set>/
        sources/         the two bench playlists (line number = idx)
        mik-backups/     MIKStore.db backups taken by mik_add_playlist.py (on demand)
        README.md        dataset description; its "## Machine paths" section holds
                         the machine-specific tool and drive locations (see below)

Machine paths: the first ```json block under "## Machine paths" in the dataset
README. Keys (all optional; %VAR% is expanded): ssd_root, ffmpeg, mik_db,
beat_this_python, sonara_python, sonara_wheel, rekordbox_xml.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[3]
DATASET = Path(
    os.environ.get("SONARA_BPM_GRID_DATA") or REPO / "dev-docs" / "bench" / "datasets" / "bpm-grid-2026-10"
)

DATA = DATASET / "data"
PLAYLISTS = DATASET / "sources"
MIK_BACKUPS = DATASET / "mik-backups"

# One folder per program; per-track JSON in json_<set>/ (set = broken | straight).
SONARA_DIR = DATA / "1_sonara"
MIK_DIR = DATA / "2_mixed_in_key"
REKORDBOX_DIR = DATA / "3_rekordbox"
BEAT_THIS_DIR = DATA / "4_beat_this"

PLAYLIST_BROKEN = PLAYLISTS / "djts-playlist-broken.txt"
PLAYLIST_STRAIGHT = PLAYLISTS / "djts-playlist-straight.txt"

_CONFIG_FILE = DATASET / "README.md"
_CONFIG_BLOCK = re.compile(r"^## Machine paths\b.*?^```json[ \t]*\r?\n(.*?)^```", re.S | re.M)


def _machine_config() -> dict:
    """The JSON block of the dataset README's "## Machine paths" section ({} if absent)."""
    if not _CONFIG_FILE.is_file():
        return {}
    m = _CONFIG_BLOCK.search(_CONFIG_FILE.read_text(encoding="utf-8"))
    return json.loads(m.group(1)) if m else {}


_CONFIG = _machine_config()


def machine(key: str, default: str | None = None) -> Path | None:
    """A machine-specific location from the README's machine paths, or `default`."""
    value = _CONFIG.get(key, default)
    return Path(os.path.expandvars(value)) if value else None


SSD_ROOT = machine("ssd_root")
FFMPEG = machine("ffmpeg", "ffmpeg")
MIK_DB = machine("mik_db", r"%LOCALAPPDATA%\Mixed In Key\Mixed In Key\11.0\MIKStore.db")
SONARA_WHEEL = machine("sonara_wheel")
