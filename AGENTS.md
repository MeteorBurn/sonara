# Sonara Project Instructions

These instructions apply to work inside `E:\Projects\Sonara`.

## Current BPM Benchmarking Goal

- We are improving Sonara BPM detection for cases where Sonara reports roughly half of the target BPM shown by Mixed In Key.
- The benchmark source file is `benchmarks\bpm\label_x2\mik_bpm_and_sonara_bpm_x2.xlsx`.
- In that workbook:
  - `bpm_mik` and `bpm_mik_raw` are target BPM values.
  - `bpm_sonara` is the previously recorded Sonara BPM from the `E:\Projects\dj-track-similarity` project database.
  - `path` points to the audio file to re-analyze with Sonara.
- Current fork changes compared with original `v0.1.7`:
  - tempo candidate selection in the beat tracker reduces the half-BPM/x2 mismatch substantially;
  - optional project BPM range (`bpm_min`, `bpm_max`) doubles or halves values outside the range.
- Current fork package version is `0.1.8`.
- On the first 1000 labeled rows, current optimized logic produced 998 successful analyses, 2 decode errors, 1 remaining x2-like result without BPM range, and 0 x2-like results after applying the 79-192 BPM range.

## Workflow Rules

- Do not edit code, tests, docs, or benchmark files without explicit user confirmation in the current conversation.
- Before changing code, first benchmark and prove the behavior on real files from the workbook.
- Work in small steps and report the intended next action before doing it.
- Prefer read-only analysis unless the user explicitly asks for a file or code change.
- Do not delete or rewrite existing uncommitted work unless explicitly asked.
- Do not create one persistent artifact per 100-file batch. Consolidate benchmark results into one working file.

## BPM Benchmark Procedure

- Future benchmark runs for this label should use the current optimized code only unless the user explicitly asks to compare against original `v0.1.7` again.
- Continue with batches of 200 files from `benchmarks\bpm\label_x2\mik_bpm_and_sonara_bpm_x2.xlsx` now that release-mode analysis has proven fast enough.
- For each batch:
  - Run Sonara with the current optimized logic.
  - Compare output against `bpm_mik_raw` and `bpm_mik`.
  - Also evaluate the deterministic BPM range post-process for the project range under discussion, currently 79-192 BPM.
- If the current x2 behavior remains clean, continue with the next batch.
- If remaining errors are around 2-3 BPM after x2 correction, treat that as a potentially separate optimization problem.
- During iterative work, keep at most one consolidated working report, preferably TSV or CSV.
- Produce a user-facing `.xlsx` report only at the end, unless the user asks for an intermediate workbook.
- Do not expand or modify the source workbook. Add analysis-only columns to the working/final report instead.
- For BPM values in reports:
  - `bpm_mik` is the UI-formatted target and should display with two decimals.
  - `bpm_mik_raw` is the raw audit value from the MIK/database source.
  - Keep analyzed Sonara values as both raw and UI forms, e.g. `bpm_sonara_run_raw` and `bpm_sonara_run`.
  - If carrying original database Sonara values into a report, keep both raw and UI forms, e.g. `bpm_sonara_db_raw` and `bpm_sonara_db`.
  - Primary error metrics should compare the UI-formatted Sonara run value against `bpm_mik`; raw metrics may be retained for audit.

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
