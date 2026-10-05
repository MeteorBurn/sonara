# MeteorBurn SONARA 0.3.7 release: precise BPM and beat grid

- Release date: 2026-10-05.
- GitHub tag: `v0.3.7-bpm-precision-20261005`. Fork tags name the version, the topic and the build date (`v<version>-<topic>-<yyyymmdd>`), so they never collide with upstream `v<version>` tags.
- Package version: `0.3.7` (`sonara/Cargo.toml`, `sonara-python/Cargo.toml` and `pyproject.toml` in lockstep; not bumped).
- Wheel source commit: `e7e666654b3c9ce3aa4e44bce2de61896afcb461`, exported with `git archive`; no source patches applied. The tagged commit adds only this file on top of it, and it is not a wheel input.
- Previous fork release: `v0.3.7-meteorburn.1` (`459bd3c`), the same package version with analysis schema 6.
- Upstream base: `15fb81d34e4e6ca856508e5520a84b1a4ab42b6e`, tag `v0.3.6`. `upstream/main` has not moved since, so nothing new was absorbed.
- Local build identity: `0.3.7-bpm-precision-20261005-e7e6666-20261005`.

## What this release changes

`ANALYSIS_SCHEMA_VERSION` moves from `6` to `7`. `bpm` and `bpm_raw` are the tempo of the tracked beats (beat-pair consensus on sub-frame beat positions), steady tracks report their exact integer tempo, and the tempo level is chosen among the autocorrelation peaks by how well each level's beats fit the onsets, without a fixed BPM window. Beats are tracked below 3.2 kHz and sit one frame earlier, `downbeats` follow the kick, and one-octave BPM ranges are accepted. Field meanings and units do not change, but stored schema-6 values differ, and `augment_analysis` rejects schema-6 records. Request, mood, augment and aggression changes ship without a schema bump, together with decoding and Python fixes. [CHANGELOG.md](CHANGELOG.md) has the details and the bench figures.

## Decoder overlay (unchanged from 0.3.6-meteorburn.1)

- Symphonia is pinned to exactly `0.6.1`, with `default-features = false` and features `all`, `opt-simd`.
- Hound is the first WAV decoder; Symphonia retries a WAV file that Hound rejects. External metadata probes are reduced when tags are not requested.
- Hound `3.5.1` is vendored at `vendor/hound-3.5.1` through `[patch.crates-io]`.
- `hound_patch_status`: `APPLIED` (vendored in-tree).
- Patch-kit status on the exported source: `PATCH_NOT_NEEDED`; the fix and its regression test are present.
- Unknown RIFF chunks consume their payload plus the pad byte of an odd length; `reads_data_after_odd_length_unknown_chunk` passes.

## Published artifacts

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| `sonara-0.3.7-cp310-abi3-win_amd64.whl` | 2,159,070 | `2b584a6775caf82454b8852f6fc7f2a4862346766e02e54b704bb337ff76239c` |
| `sonara-0.3.7-symphonia-0.6.1-hound-3.5.1-win_amd64.rlib` | 8,384,348 | `fc7dd464aa6c2aeadf03a3798a3abf40b49239cb30f7c984923b890071b75eb0` |

- The wheel supports Windows x64 and CPython 3.10+ (abi3). It was installed and exercised on Python 3.12 only.
- Toolchain: Rust/Cargo 1.98.1, Maturin 1.15.0, Python 3.12.11, target `x86_64-pc-windows-msvc`. Release profile: `opt-level = 3`, fat LTO, one codegen unit.
- The native module inside the wheel (5,230,592 bytes, SHA-256 `06a35d8867b9561a5329b6bc4d8a80186200d2954e6cb1202cfeaaf0e74166e2`) is byte-identical to the one that ran every Python check below.
- The `.rlib` was built with `cargo build --release --locked -p sonara --features aggression`, the Python binding's feature set. It is a Rust compiler artifact that carries LTO bitcode: it links only with rustc 1.98.1, the same dependency rlibs and fat LTO. A Cargo project should depend on the tagged source and mirror the vendored Hound patch in its own workspace root instead.
- Assets are published on GitHub only. The PyPI `sonara` package remains upstream's; this fork does not publish to PyPI or crates.io.

## Fresh verification on 2026-10-05

Every check ran on the exact exported source or on the exact wheel above.

| Check | Result |
| --- | --- |
| `cargo tree --locked -p sonara -i symphonia` | Exactly 0.6.1 |
| `cargo tree --locked -p sonara -i hound` | Vendored 3.5.1 |
| `cargo test --locked -p sonara` | 519 library + 33 accuracy + 6 BPM passed (the BPM suite includes 13 real-music tempo fixtures); 2 known upstream BPM tests ignored |
| `cargo test --locked -p sonara --features aggression --lib` | 536 passed |
| `cargo test --locked --manifest-path vendor/hound-3.5.1/Cargo.toml --lib` | 42 passed, including the odd-chunk regression |
| `python scripts/run_python_tests.py` with this wheel's native module | All six contract checks PASS; 16 of 16 suites passed, exit 0 |
| `python scripts/check_python_contract.py --check` | PASS |
| `python scripts/run_fidelity_gate.py --check-contract` | PASS |
| Wheel smoke in a fresh Python 3.12 venv | 16 of 16: version 0.3.7; the installed native module is the wheel's; schema 7, fingerprint version 1; a 31-feature library request on a real bench track returns 70 fields at 125.0 BPM, the validators' tempo; strided and float64 input; `os.PathLike`; `from sonara.feature import tempo`; FLAC tags; a damaged MP3 recovers; a missing file raises `OSError` |
| This wheel on the 13 real tracks behind the rhythm fixtures | 13 of 13 meet their expectations: exact integer tempos, fractional tempos and tempo levels |
| `.rlib` smoke: fat-LTO link of a small driver | The same real track at 125.00 BPM with the wheel's aggression score; a missing file is rejected |

The bench figures in the changelog (2,000 tracks with Mixed In Key and Rekordbox as validators, and 226 further tracks judged by Beat This!) were measured on the same library code by envelope replay, which is bit-identical to the analysis pipeline. This exact binary was checked on the 13 fixture tracks above.

## Hosted CI

The two causes that kept the fork's GitHub Actions on `main` red are fixed in this source: the fidelity-map test walks retired upstream files with `--full-history` (`31da9e9`), and the sdist ships `vendor/hound-3.5.1`, which the workspace `[patch.crates-io]` requires (`18f9d50`). CI runs when `main` moves to this release; its result is not recorded here. This release ships no sdist; build from a source checkout.

## Compatibility and downstream adoption

- Stored schema-6 results keep their meaning, but `bpm`, `bpm_raw`, `beats`, `downbeats` and the fields derived from them differ from a fresh run; re-analyse them. `augment_analysis` rejects a schema-6 record (`schema version mismatch`).
- `SIMILARITY_VERSION` stays `2` and `FINGERPRINT_VERSION` `1`. On tracks whose tempo level changed, `danceability`, `valence`, `mood_*`, `aggression_rhythm` and embedding components 35, 37, 42 and 47 move with the tempo.
- An explicit `features=[...]` request returns only the named groups plus the core signal scalars; name the spectral summaries (`mfcc`, `chroma`, `contrast`, `bandwidth`, `rolloff`, `flatness`) if you relied on them.
- `rhythmic_regularity` abstains on silence, ambient and drumless audio: the score, label and candidates are `None` while the confidence is reported. Key cache freshness on the presence of the confidence; see [docs/consumer-contract.md](docs/consumer-contract.md).
- In an `analyze_*` result, `onset_strength_bands` is a list of lists (JSON-safe); only the standalone `sonara.onset_strength_bands()` returns a `float32` numpy array.
- Publishing this package does not refresh any database or retrain any classifier.

Portable source build commands (run from the repository root):

```text
cargo test --locked -p sonara
cargo test --locked --manifest-path vendor/hound-3.5.1/Cargo.toml --lib
maturin build --release --locked --out dist
```

The vendored Hound patch is part of this source tree. Preserve it and its regression test when rebuilding.
