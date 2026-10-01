# bpm-grid — BPM, beat grid and downbeat evaluation harness

Measures Sonara's tempo, beats and downbeats against external references,
Beat This! (beats/downbeats), Mixed In Key (BPM) and Rekordbox (BPM, grid), on one test bench: two
1000-track sets, "broken" and "straight" rhythm.
Plan and findings: `dev-docs/plans/bpm-grid.md` (local).

Research tooling, not part of CI. The enforceable checks are
`sonara/tests/bpm_accuracy.rs` and `tests/fidelity_gates.json`.

## Paths

Nothing is written next to these scripts. `bpm_grid_paths.py` (and
`bpm_grid_paths.ps1` for the launchers) resolves everything:

| Name | Default |
| --- | --- |
| dataset | `<repo>/dev-docs/bench/datasets/bpm-grid-2026-10` (override `SONARA_BPM_GRID_DATA`) |
| `data/` | one folder per program (`1_sonara`, `2_mixed_in_key`, `3_rekordbox`, `4_beat_this`; per-track JSON in `json_<set>/`), comparison table |
| `sources/` | `djts-playlist-{broken,straight}.txt`: the bench definition, line number = `idx` |
| `README.md` | dataset description; its `## Machine paths` JSON block is machine-specific: `ssd_root`, `ffmpeg`, `mik_db`, `beat_this_python`, `sonara_python`, `sonara_wheel`, `rekordbox_xml` |

Scripts run from any working directory.

## Pipeline

| Step | Script | Python |
| --- | --- | --- |
| add playlist to MIK library | `mik_add_playlist.py` (backs up `MIKStore.db` first), `mik_drag_source.ps1` | any 3.11+ |
| export MIK analysis | `mik_export.py --playlist … --out …` → `data/2_mixed_in_key/json_<set>/` | any 3.11+ |
| Beat This! beats/downbeats | `beat_this_bpm.py` (broken), `run_beatthis_straight.ps1` → `data/4_beat_this/json_<set>/` | `beat_this_python` |
| Sonara extraction | `extract_reference.py --name <set>_<revision>` (broken), `run_sonara_straight.ps1 -Name …` → `data/1_sonara/run-<name>.sqlite` + `json_<name>/` | `sonara_python` (wheel = `sonara_wheel`) |
| per-program tables | `build_tables.py [--only …]` (reads `json_<set>/`) → `data/<n>_<program>/<program>-<set>.{sqlite,xlsx}` | `beat_this_python` |
| compare both sets | `compare_all.py` (reads the four `<program>-<set>.sqlite`, Rekordbox as third validator) → `data/bpm_comparison_2000.{json,xlsx}` | `beat_this_python` |

The broken set is analysed from the `M:` paths of its playlist, the straight
set from the `S:` copies; the files are identical. Sonara and Beat This! runs
are resumable: replacing a track in a playlist slot needs only its old rows and
per-track JSON removed, then a rerun.

Retired 2026-10-01 (moved to `dev-docs/bin/bpm-grid-old-20261001/`): the numpy
what-if port `estimate/` with its 145-track downstream check and data, the
broken-only `compare_bpm.py` and the diagnostics that read its table, and the
synthetic-truth scripts.
