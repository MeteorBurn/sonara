# MeteorBurn SONARA 0.3.6 patched release

- Release date: 2026-09-16.
- GitHub tag: `v0.3.6-meteorburn.1`.
- Package version: `0.3.6` (the existing verified wheel is reused unchanged).
- Previous fork revision: `d0cf34ccfd27d566f5dac30798a246c9b353079a`, package 0.2.4.
- Upstream base: `15fb81d34e4e6ca856508e5520a84b1a4ab42b6e`, tag `v0.3.6`.
- Local build identity: `0.3.6-sym061-hound-riff-padding-15fb81d-20260915`.
- The merge preserves fork history, the tracked BPM workbooks and the TSV exporter.

## Decoder overlay

- Symphonia is pinned to exactly `0.6.1`, with `default-features = false` and features `all`, `opt-simd`.
- The verified local `sonara/src/core/audio.rs` port preserves Hound as the first WAV decoder and retries with Symphonia when Hound rejects a WAV file.
- External metadata probes are reduced when tags are not requested; the normal metadata path remains available when tags are requested.
- Hound `3.5.1` is vendored at `vendor/hound-3.5.1` through `[patch.crates-io]`.
- `hound_patch_status`: `APPLIED`.
- Patch-kit status before each compiling command: `PATCH_NOT_NEEDED`; the saved fix and regression test were already present.
- Unknown RIFF chunks consume their payload and the padding byte required for odd lengths. The `reads_data_after_odd_length_unknown_chunk` regression passes.
- BPM code is unchanged from upstream 0.3.6 and the verified local build. The optional `bpm_min`/`bpm_max` API is retained; no new BPM or beat-grid experiment is included.

## Published artifact

`sonara-0.3.6-cp310-abi3-win_amd64.whl` supports Windows x64 and Python 3.10+.

- Size: 2,154,140 bytes.
- SHA-256: `1e59d1b87a742563a5ff0222712af99ca94abe353760624ccbec46c53bbd7908`.
- Original build toolchain: Rust/Cargo 1.95.0, Maturin 1.14.1, Python 3.10.11, target `x86_64-pc-windows-msvc`.
- The 119 shipping source/dependency files checked for this release match the original build source after normalizing CRLF/LF. This includes Rust/Python runtime code, model data, manifests, lockfile and vendored Hound. Fork documentation, CI and the historical exporter are separate from those shipping inputs.
- The installed native module used for verification is byte-identical to the module inside this exact wheel.
- GitHub release assets identify this patched build separately from upstream. The PyPI package remains upstream; this fork does not publish to PyPI or crates.io.

## Fresh verification on 2026-09-16

| Check | Result |
| --- | --- |
| `cargo tree --locked --offline -p sonara -i symphonia` | Exactly 0.6.1 |
| `cargo tree --locked --offline -p sonara -i hound` | Vendored 3.5.1 |
| `cargo test --locked --offline -p sonara` | 467 library + 33 integration + 4 BPM passed; 2 known upstream BPM tests ignored |
| `cargo test --locked --offline --manifest-path vendor/hound-3.5.1/Cargo.toml --lib` | 42 passed, including odd-chunk regression |
| Exact wheel hash, native module identity, decoder/API smoke | Passed |
| All 14 standard Python API scripts against the installed wheel | Passed |
| Frozen similarity | 182 pairs and 28 neighbor orderings passed across two profiles |
| Comparison with the earlier August local wheel | All emitted fields equal on 12 real tracks |
| Python runtime/stub/ABI3 and release workflow contracts | Passed |

The optional matplotlib display checks were skipped because matplotlib is absent.

The upstream fidelity router remains enabled. A dry run over the complete fork update from 0.2.4 exits with status 2: BPM, beat-grid, fingerprint, genre, tonal, mood and cross-cutting domains lack mandatory labeled gates, and the broader vocalness corpus is local-only. The local checks above do not establish full-corpus accuracy or a green hosted CI run. These fidelity requirements were not disabled to publish this already-built artifact.

## Compatibility and downstream adoption

- Compared with 0.2.4, Python now requires 3.10+, input validation is stricter, MP3 recovery is improved, and a bundled vocalness model and optional aggression APIs are available.
- Upstream 0.3.6 adds per-feature augmentation and the `timbre` similarity profile. The 48-dimensional stored vector layout and default similarity metric remain unchanged from upstream 0.3.5.
- Analysis provenance/schema and model identities must be respected when comparing stored results from 0.2.4 with newer analyses; no database refresh is performed by publishing this package.
- This release does not establish improved accuracy for downstream voice, live-instrument, valence or arousal classifiers. Those decisions still require manual labels and grouped evaluation.

Portable source build commands (run from the repository root):

```text
cargo test --locked -p sonara
cargo test --locked --manifest-path vendor/hound-3.5.1/Cargo.toml --lib
maturin build --release --locked --interpreter python --out dist
```

The vendored Hound patch is part of this source tree. Preserve it and its regression test when rebuilding.
