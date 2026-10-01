# bpm-grid — BPM, beat grid and downbeat evaluation harness

Measures Sonara's tempo, beats and downbeats against two external references,
Beat This! (beats/downbeats) and Mixed In Key (BPM), on two 1000-track sets
("broken" and "straight" rhythm), plus synthetic audio with exact truth.
Plan and findings: `dev-docs/plans/bpm-grid.md` (local).

Research tooling, not part of CI. The enforceable checks are
`sonara/tests/bpm_accuracy.rs` and `tests/fidelity_gates.json`.

## Paths

Nothing is written next to these scripts. `bpm_grid_paths.py` (and
`bpm_grid_paths.ps1` for the launchers) resolves everything:

| Name | Default |
| --- | --- |
| dataset | `<repo>/locals/datasets/bpm-grid` (override `SONARA_BPM_GRID_DATA`) |
| `data/` | Sonara SQLite/JSON, `beat_this*/`, `mik*/`, `synthetic/`, comparison tables |
| `playlists/` | `djts-playlist-{broken,straight}.txt`, SSD copy maps |
| `estimate/` | outputs of `estimate/` (`cache/`, `results/`, `synthetic/`, `downstream/`) |
| `paths.json` | machine-specific: `ssd_root`, `ffmpeg`, `mik_db`, `beat_this_python`, `sonara_python`, `sonara_wheel`, `regularity_eval` |

Scripts run from any working directory.

## Pipeline

| Step | Script | Python |
| --- | --- | --- |
| copy audio to SSD (optional) | `copy_to_ssd.ps1` | — |
| add playlist to MIK library | `mik_add_playlist.py` (backs up `MIKStore.db` first), `mik_drag_source.ps1` | any 3.11+ |
| export MIK analysis | `mik_export.py --playlist … --out …` | any 3.11+ |
| Beat This! beats/downbeats | `beat_this_bpm.py`, `run_beatthis_straight.ps1` | `beat_this_python` |
| Sonara extraction | `extract_reference.py`, `run_sonara_straight.ps1` | `sonara_python` (wheel = `sonara_wheel`) |
| synthetic truth | `synthetic_truth.py`, `synthetic_measure.py` | `sonara_python` |
| compare both sets | `compare_all.py` → `data/bpm_comparison_2000.{json,xlsx}` | `beat_this_python` |
| single-set / diagnostics | `compare_bpm.py`, `diagnose_sonara.py`, `recheck_real.py`, `verify_estimate.py`, `recompute_bt_bpm.py` | `beat_this_python` |

## estimate/ — offline what-if port

A numpy port of Sonara's tempo/beat/downbeat code (`est_common.py`), validated
against stored output (`validate_port.py`), used to estimate candidate fixes
before touching Rust. Never shipped. Run order: `read_tags.py`, `build_cache.py`,
`validate_port.py`, `synth_latency.py`, `fix1_bpm.py`, `fix3_downbeat.py`,
`fix4_tempo.py`, `synth_make.py`, `synth_run.py` (`sonara_python`),
`synth_eval.py`, `combined.py`, `robustness.py`, `sensitivity.py`,
`fix5_antiphase.py`, `report_tables.py`, `export_xlsx.py`.

`estimate/downstream/` re-derives downstream features (chords, regularity,
aggression, vocalness) for each fix variant on the 145 manually labelled
tracks; it reads the bundled models from this checkout.
