# PROJECT KNOWLEDGE BASE

**Generated:** 2026-09-23
**Commit:** 0c2a4b3
**Branch:** main

## OVERVIEW

Sonara is a Rust music-information-retrieval library (librosa-style DSP plus a
fused single-pass track analyzer) with PyO3/Maturin bindings (`import sonara`).
This repo is the MeteorBurn fork (0.3.7) of `kkollsga/sonara`: BPM fixes, opt-in
rhythm features, Symphonia pinned to `=0.6.1`, and a patched vendored Hound.

## STRUCTURE

```
sonara/                    # core crate; all analysis logic; PyO3-free by contract
  src/analyze.rs           # 6.2k lines: FEATURE_REGISTRY, modes, fused pass, augment
  src/core/                # numeric substrate: decode/resample, stft, fft, cqt, pitch
  tests/                   # only accuracy.rs + bpm_accuracy.rs; unit tests are inline
  examples/                # eval runners: accuracy_eval, rhythmic_regularity_eval, export_bpm_tsv
sonara-python/             # cdylib -> sonara._sonara; conversion and signatures only
python/sonara/             # Python facade, hand-written __init__.pyi, models/*.json
tests/                     # Python suite, reference_data, fixtures, fidelity_gates.json
scripts/                   # CI contract/release gates; run_python_tests.py is the entry point
workflow/skills/           # agent procedures (phased-plan, release, notify, ...)
docs/consumer-contract.md  # downstream API/storage contract - authoritative
dev-docs/bench/scripts/    # tracked research harnesses; rest of dev-docs/ is git-ignored
benchmarks/bpm/            # two tracked MIK-labeled BPM workbooks; dir otherwise ignored
vendor/hound-3.5.1/        # Hound + odd-length RIFF chunk padding fix via [patch.crates-io]
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add or rename an output feature | `sonara/src/analyze.rs` `FEATURE_REGISTRY` (:220) | then binding, `.pyi`, README, tests |
| Tempo / BPM | `sonara/src/beat.rs` `estimate_tempo` (:186) | parabolic ACF refinement, `bpm_min`/`bpm_max` |
| Rhythm timelines (0.3.7) | `onset.rs` (`onset_bands`), `rhythmic_regularity.rs` | opt-in, `FrameCurves` class |
| Decode / resample | `sonara/src/core/audio.rs` | Hound first for WAV, Symphonia fallback |
| Embedding / similarity | `sonara/src/similarity.rs` | 48-d, stored unweighted; profiles apply at query time |
| Aggression | `aggression.rs`, `aggression_dsp.rs` | cargo feature `aggression`; model files embedded from `src/` |
| Vocalness / genre models | `vocal_model.rs`, `genre.rs`, `python/sonara/models/` | JSON MLP; model id lands in provenance |
| Python signature or dict shape | `sonara-python/src/analyze.rs` + `python/sonara/__init__.pyi` | synced by hand |
| Stored-result rules | `docs/consumer-contract.md` | abstention, backfill, versioning |
| Accuracy gate routing | `tests/fidelity_gates.json`, `scripts/run_fidelity_gate.py` | most domains `blocked` |

## CODE MAP

rust-analyzer references (declaration excluded, tests included) at `0c2a4b3`.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|-----:|------|
| `analyze_signal` | fn | `sonara/src/analyze.rs:1227` | 137 | fused analysis of samples; file and batch paths end here |
| `AnalysisConfig` | struct | `analyze.rs:438` | 130 | mode, opt-in features, BPM range, models |
| `SIMILARITY_VERSION` | const `2` | `similarity.rs:74` | 48 | vector layout id; genre/vocal/aggression models must match |
| `AnalysisMode` | enum | `analyze.rs:90` | 46 | compact (default) / playlist / full |
| `TrackAnalysis` | struct | `analyze.rs:870` | 28 | result record; a dict in Python |
| `stft` | fn | `core/spectrum.rs:42` | 28 | widest DSP fan-out |
| `augment_analysis` | fn | `analyze.rs:3790` | 17 | merge missing features into a cached record |
| `resample` | fn | `core/audio.rs:662` | 16 | |
| `analyze_file` | fn | `analyze.rs:1164` | 14 | decode + `analyze_signal` |
| `load` | fn | `core/audio.rs:76` | 10 | decode entry |
| `ANALYSIS_SCHEMA_VERSION` | const `6` | `analyze.rs:745` | 8 | whole-record schema |
| `analyze_batch` | fn | `analyze.rs:1342` | 4 | rayon across files |

Import hubs (ast-grep): `types` (`Float = f32`) and `error` (`SonaraError`) are
imported by 26 modules each, including the binding; `core` by 8; `perceptual`
by 5 (`analyze`, `loudness_ext`, `mood`, `structure`, `aggression_dsp`).

## CONVENTIONS

- `Float = f32` throughout. Sample rate is always an explicit argument; no global SR.
- Opt-in families (`onset_bands`, `beatgrid`, `rhythmic_regularity`, ...) are
  never computed by a mode, only via `features=[...]`.
- Additive fields ship without an `ANALYSIS_SCHEMA_VERSION` bump; bump only when
  existing meaning or units change. Consumers key freshness per feature.
- Abstention is `None` with confidence kept: a stored `None` regularity score
  means "measured, undecidable", not "pending".
- Rust unit tests sit inline at the bottom of each module. Python suite files
  run as plain scripts (`python <file>`): pytest-style `def test_*` with no
  driver runs nothing and still exits 0.
- Versions move together in `sonara/Cargo.toml`, `sonara-python/Cargo.toml`,
  `pyproject.toml`, only via the release procedure.
- Wheel floor is abi3 `cp310`; repo tooling needs Python >= 3.11 (`tomllib`).

## ANTI-PATTERNS (THIS PROJECT)

- A feature that runs its own STFT: reuse the fused pass in `analyze_signal_inner`.
- PyO3/NumPy, or an ML runtime in default features, inside `sonara/`: contract
  boundary; downstream must be notified first.
- Reinterpreting a similarity/fingerprint layout without a version bump. Bumping
  `SIMILARITY_VERSION` also invalidates every genre/vocalness/aggression model.
- Running pytest directly: `scripts/run_python_tests.py` runs the contract
  checks CI gates on.
- Trusting green `cargo test -p sonara` for aggression code: the binding always
  enables `aggression`; run the feature build too.
- Static ACF score-ratio rules for tempo: they fix some rows and regress
  House/Techno controls.
- Committing audio, generated benchmark output, or sealed labels.
- Rewriting `BUILD-METADATA.md` for an unpublished build: it records the latest
  published fork release (`v0.3.7-meteorburn.1`) and its wheel hash.

## UNIQUE STYLES

- BPM/key/chord changes need before/after numbers on labeled data (octave-error
  rate, median BPM error) - see `CONTRIBUTING.md`.
- `tests/fidelity_gates.json` routes touched paths to accuracy domains; a
  `blocked` domain stops the change until evidence is supplied, by design.
- Text files are UTF-8 with em-dashes and arrows that some consoles print as
  `-` or `?`; check codepoints before "fixing" them.

## COMMANDS

```powershell
cargo test -p sonara                                  # unit + accuracy + bpm_accuracy
cargo test -p sonara --features aggression            # CI runs this on Linux only
cargo test -p sonara --test bpm_accuracy -- --ignored # documented known-failing cases
cargo check -p sonara-python
maturin develop --release -m sonara-python/Cargo.toml # in a venv; suites import the in-tree build
python scripts/run_python_tests.py                    # canonical suite; --list, --check-contract
python scripts/run_fidelity_gate.py --base origin/main --dry-run
cargo bench -p sonara --features bench-internals      # otherwise 5 of 10 benches skip silently
```

## NOTES

- Fork remotes: `origin` MeteorBurn/sonara, `upstream` kkollsga/sonara. Absorb
  upstream with a merge commit, never a fast-forward. CI `publish` is gated to
  `kkollsga/sonara`, so pushes here never reach PyPI.
- Commits, branches, pushes and PRs only on request; the branch/PR/push steps
  in `workflow/skills/` need explicit authorization in this fork.
- Fork BPM state: tempo-candidate selection fixed most half/double-tempo errors;
  `bpm_min`/`bpm_max` folds out-of-range values (typically 79-192); parabolic
  lag refinement cut 1-3 BPM drift. Still open: `LOW_inverse_ratio` near-misses;
  next idea is beat-grid/DP regularity scoring across top candidates. Benchmark
  on `benchmarks/bpm/*.xlsx` (target `bpm_mik`, audit `bpm_mik_raw`; never edit
  the workbooks) and confirm with the maintainer before changing tempo selection.
  Earlier campaign procedure: `git show 0c2a4b3:AGENTS.md`.
- `sonara-python/src/analyze.rs` shares only a name with `sonara/src/analyze.rs`.
- `.gitignore` ignores `AGENTS.md`; only this root file is tracked. Deeper
  per-directory notes may exist locally as git-ignored `*/AGENTS.md`.
