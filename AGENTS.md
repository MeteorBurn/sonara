# Sonara Project Instructions

These instructions apply to work inside `E:\Projects\Sonara`.

## Current BPM Benchmarking Goal

- We are improving Sonara BPM detection for cases where Sonara reports roughly half of the target BPM shown by Mixed In Key.
- The benchmark source file is `benchmarks\bpm\label_x2\mik_bpm_and_sonara_bpm_x2.xlsx`.
- In that workbook:
  - `bpm_mik` and `bpm_mik_raw` are target BPM values.
  - `bpm_sonara` is the previously recorded Sonara BPM from the `E:\Projects\dj-track-similarity` project database.
  - `path` points to the audio file to re-analyze with Sonara.

## Workflow Rules

- Do not edit code, tests, docs, or benchmark files without explicit user confirmation in the current conversation.
- Before changing code, first benchmark and prove the behavior on real files from the workbook.
- Work in small steps and report the intended next action before doing it.
- Prefer read-only analysis unless the user explicitly asks for a file or code change.
- Do not delete or rewrite existing uncommitted work unless explicitly asked.
- Do not create one persistent artifact per 100-file batch. Consolidate benchmark results into one working file.

## BPM Benchmark Procedure

- Start with batches of 100 files from `benchmarks\bpm\label_x2\mik_bpm_and_sonara_bpm_x2.xlsx`.
- For each batch:
  - Run Sonara with the baseline/current logic without the new x2 optimization.
  - Compare output against `bpm_mik_raw` and `bpm_mik`.
  - Run the same files with the proposed x2 optimization.
  - Compare again against the same targets.
- If x2 optimization brings results close to target, continue with the next batch of 100.
- If remaining errors are around 2-3 BPM after x2 correction, treat that as a potentially separate optimization problem.
- During iterative work, keep at most one consolidated working report, preferably TSV or CSV.
- Produce a user-facing `.xlsx` report only at the end, unless the user asks for an intermediate workbook.

## Data Handling

- Treat audio paths and workbook rows as real user data.
- Do not commit large generated benchmark outputs or copied datasets unless the user explicitly asks.
- Keep label-x2 benchmark scripts and outputs under `benchmarks\bpm\label_x2\` unless the user requests another location.
- Remove temporary batch artifacts after their data is merged into the consolidated working report.
- For `.xlsx` analysis, prefer bundled Codex runtime tools or local read-only parsers. Do not create a project `.venv` unless the user approves it.

## Verification

- Use the cheapest verification that proves the current step.
- For Rust logic changes, use focused `cargo test -p sonara ...` first.
- For Python bindings changes, use `cargo check -p sonara-python` and only build/install the Python package when needed.
- Report exactly which commands were run and whether they passed.
