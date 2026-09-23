# MeteorBurn SONARA 0.3.7 release

- Release date: 2026-09-23.
- GitHub tag: `v0.3.7-meteorburn.1`.
- Package version: `0.3.7` (`sonara/Cargo.toml`, `sonara-python/Cargo.toml` and `pyproject.toml` in lockstep).
- Wheel source commit: `9327b16daa308f8e0c3088ed00d56ce1ba7fac3a`, exported with `git archive`; no source patches applied. The tagged commit adds only this file and an `AGENTS.md` note on top of it, and neither is a wheel input.
- Previous fork release: `v0.3.6-meteorburn.1` (`8fab461`), package 0.3.6.
- Upstream base: `15fb81d34e4e6ca856508e5520a84b1a4ab42b6e`, tag `v0.3.6`. `upstream/main` has not moved since, so nothing new was absorbed.
- Local build identity: `0.3.7-meteorburn.1-9327b16-20260923`.
- The fork is tagged `v<version>-meteorburn.<n>` so its tags never collide with upstream `v<version>` tags.

## What 0.3.7 adds

Two opt-in rhythm features, both additive: `onset_bands` (the multiband onset-strength timeline) and `rhythmic_regularity`. `ANALYSIS_SCHEMA_VERSION` stays `6`, no default mode computes either feature, and every 0.3.6 field keeps its meaning and range, so stored 0.3.6 results remain valid. [CHANGELOG.md](CHANGELOG.md) has the details and the validation figures.

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
| `sonara-0.3.7-cp310-abi3-win_amd64.whl` | 2,125,648 | `811627ba5d03fe87b79b68e2bfea241619ec3fe6269a4866ad7f2c0c7547a13c` |
| `sonara-0.3.7-symphonia-0.6.1-hound-3.5.1-win_amd64.rlib` | 8,137,956 | `5b00944908efc4bb80186efce4d7935152682aa709c00f0bf2ffc37af48bdc5f` |

- The wheel supports Windows x64 and CPython 3.10+ (abi3). It was installed and exercised on Python 3.12 only.
- Toolchain: Rust/Cargo 1.98.1, Maturin 1.15.0, Python 3.12.11, target `x86_64-pc-windows-msvc`. Release profile: `opt-level = 3`, fat LTO, one codegen unit.
- The native module inside the wheel (5,140,992 bytes, SHA-256 `b51edc32e4443c2a8f2d91d2d99c30a414b06a7764cd6476133065af8d035e8e`) is byte-identical to the one that ran every Python check below.
- The `.rlib` was built with `cargo build --release --locked -p sonara --features aggression`, the Python binding's feature set. It is a Rust compiler artifact that carries LTO bitcode: it links only with rustc 1.98.1, the same dependency rlibs and fat LTO. A Cargo project should depend on the tagged source and mirror the vendored Hound patch in its own workspace root instead.
- Assets are published on GitHub only. The PyPI `sonara` package remains upstream's; this fork does not publish to PyPI or crates.io.

## Fresh verification on 2026-09-23

Every check ran on the exact exported source or on the exact wheel above.

| Check | Result |
| --- | --- |
| `cargo tree --locked -p sonara -i symphonia` | Exactly 0.6.1 |
| `cargo tree --locked -p sonara -i hound` | Vendored 3.5.1 |
| `cargo test --locked -p sonara` | 505 library + 33 accuracy + 4 BPM passed; 2 known upstream BPM tests ignored |
| `cargo test --locked -p sonara --features aggression --lib` | 521 passed |
| `cargo test --locked --manifest-path vendor/hound-3.5.1/Cargo.toml --lib` | 42 passed, including the odd-chunk regression |
| `python scripts/run_python_tests.py` with this wheel's native module | All four contract checks PASS; 16 of 16 suites passed, exit 0 |
| `python scripts/run_fidelity_gate.py --check-contract` | PASS |
| Wheel smoke in a fresh Python 3.12 venv | Version 0.3.7; synthetic 124 BPM four-on-the-floor scores `regular` 0.775 (confidence 0.863); onset bands stay out of the result unless requested; WAV (Hound) and FLAC (Symphonia) decode; a missing file raises `OSError` |
| `.rlib` smoke: fat-LTO link of a small driver | FLAC analysis, both rhythm groups and the missing-file error pass |

The optional matplotlib display check was skipped because matplotlib is absent.

Real-music accuracy was not re-measured for this build. The `rhythmic_regularity` figures in the changelog (145 manually labelled tracks: ROC AUC 0.883, balanced accuracy 0.836 at the `0.5` boundary) come from its development measurement on the same code; the straight side still has no fresh held-out set.

## Hosted CI

The fork's GitHub Actions run on `main` is red for two reasons that do not involve the wheel; Rust tests and all four wheel builds pass there.

- `test_reviewed_bootstrap_transitions_match_exact_contents` looks for the recorded base of `scripts/mood_aggression_probe.py` with `git rev-list --all -- <path>`. That file exists only in upstream history, which the fork reaches through the 0.3.6 merge `8fab461`. The path is absent from both merge parents, so default history simplification never walks the upstream side. The test passes locally only because the `upstream/main` ref starts a walk inside upstream history; a CI clone of the fork has no such ref. With `--full-history`, `origin/main` alone reaches the revisions.
- The sdist omits `vendor/hound-3.5.1`, which the workspace `[patch.crates-io]` requires, so installing from the sdist fails in `cargo metadata`. This release ships no sdist; build from a source checkout.

## Compatibility and downstream adoption

- Existing callers and stored results are unaffected. Both features are enabled only through `features=[...]`.
- `rhythmic_regularity` abstains on silence, ambient and drumless audio: the score, label and candidates are `None` while the confidence is reported. Key cache freshness on the presence of the confidence; see [docs/consumer-contract.md](docs/consumer-contract.md).
- In an `analyze_*` result, `onset_strength_bands` is a list of lists (JSON-safe); only the standalone `sonara.onset_strength_bands()` returns a `float32` numpy array.
- Publishing this package does not refresh any database or retrain any classifier. `rhythmic_regularity` is not evidence for voice, live-instrument, valence or arousal classifiers without its own labelled, grouped evaluation.

Portable source build commands (run from the repository root):

```text
cargo test --locked -p sonara
cargo test --locked --manifest-path vendor/hound-3.5.1/Cargo.toml --lib
maturin build --release --locked --out dist
```

The vendored Hound patch is part of this source tree. Preserve it and its regression test when rebuilding.
