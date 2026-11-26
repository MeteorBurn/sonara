# Sonara project instructions

Sonara is the MeteorBurn fork of `kkollsga/sonara`, a Rust MIR library with a fused track analyzer and PyO3/Maturin bindings (`import sonara`). Apply the standing rules below to every task. Elsewhere, a rule is required when its stated condition holds. Defaults describe the usual starting point.

## Standing rules

- Keep project instructions in this root file. If a root `CLAUDE.md` is used, its only content is `@AGENTS.md`. Never add nested instruction files or let a workflow skill resync the two files.
- Before editing, inspect `git status --short` and the relevant diff. Preserve unrelated work. If UTF-8 text looks corrupted in the console, check its codepoints before editing it.
- Create branches, commit, push or open PRs only on explicit request, including when following a workflow skill. Never force-push `origin`, push to `upstream`, or publish this fork to crates.io or PyPI. When syncing upstream, use a merge commit and disable fast-forward.
- Never commit audio, generated benchmark output or sealed labels. At delivery, check for unexpected tracked changes and report what ran, what was skipped and why. Report passed mandatory checks in one line; name a check only when it failed or was skipped.
- Never fit a descriptor's method, constants or code paths to the user's tracks or to a verification dataset to make a check pass. Parameter values must come from the method's literature or an established implementation; data may confirm a choice, never produce it. For every parameter, report its source, the data that picked it and the data that only confirmed it. When a result falls short, report the shortfall; never tune it away.

## graphify

- For a codebase question, when the local graph and CLI exist, run `& .\.tools\graphify\Scripts\graphify.exe query "<question>"` from the repo root in PowerShell before the first source search or read. Include this rule in any delegated code exploration. Here `graphify` means `.tools/graphify/Scripts/graphify.exe` (`graphifyy==0.9.120`). Never run the skill's interpreter detection or installation steps, or install Graphify globally.
- Follow graph results or earlier reads to source lines. Use `path` for relationships, `explain` for symbols and `section` for indexed Markdown headings. Search unindexed files directly. For broad navigation, use `graphify-out/wiki/index.md` if present. Read `GRAPH_REPORT.md` only for broad architecture work or when focused queries are insufficient.
- Skip graph navigation if the tools are absent, the task concerns incorrect or stale graph output, or the user opts out. Dirty graph files alone do not qualify. After code changes, run `graphify update .` when available. A tracked `graph.json` is rewritten in full on rebuild; preserve the rebuild and its hooks.
- On an explicit `/graphify` request, consult the installed [Graphify skill](.claude/skills/graphify/SKILL.md) or the local [Graphify guide](dev-docs/agent-guides/graphify.md) first.

## Project documentation

- For setup, read the relevant [README](README.md) section. For numeric behavior or reference data, read [CONTRIBUTING.md](CONTRIBUTING.md). For stored fields, schema or similarity layout, read the [consumer contract](docs/consumer-contract.md). Local plans live under `dev-docs/plans/`. If required evidence or guidance is unavailable, identify the gap and leave the dependent conclusion unverified.

## Research

- When the user asks for research, or a task needs it (literature, algorithm definitions, reference implementations, datasets), use the Firecrawl research tools: `firecrawl_research_search_papers`, `firecrawl_research_read_paper`, `firecrawl_research_related_papers` and `firecrawl_research_inspect_paper` for papers, and `firecrawl_agent` for questions that span several sites. Read a known URL with `firecrawl_scrape`. For an ordinary lookup, use ordinary search. Include this rule in any delegated research.
- Record research per descriptor in the local research base `dev-docs/research/` (git-ignored): one folder per descriptor, holding the reports as delivered, an index of sources that says how far each was read, and the saved source texts. Later research on the same descriptor goes into that folder's existing files, never into a parallel set. Dataset and tool findings go to `dev-docs/research/datasets-and-tools.md`. Read the base before searching the web again.

## Engineering contracts

| When working on | Required behavior |
| --- | --- |
| DSP in `sonara/` | Use `Float = f32` and pass sample rate explicitly. When a new feature needs a spectrum, reuse the fused pass in `analyze_signal_inner`. Keep PyO3/NumPy out of the core crate and ML runtimes out of its default features. Any proposed boundary change requires prior downstream notification. |
| Feature selection | Opt-in families such as `onset_bands`, `beatgrid` and `rhythmic_regularity` are computed by `full` mode or through `features=[...]`. `compact` and `playlist` must never enable them. |
| Stored results | Bump `ANALYSIS_SCHEMA_VERSION` only when an existing field changes meaning or units. Additive fields use per-feature freshness. On abstention, keep confidence and store `None`; a stored `None` regularity score means the measurement was undecidable. |
| Similarity or fingerprints | Bump the corresponding layout version when reinterpreting stored values. A `SIMILARITY_VERSION` bump invalidates the genre, vocalness and aggression models. |
| Packaging or release | Change versions only during release, together in `sonara/Cargo.toml`, `sonara-python/Cargo.toml` and `pyproject.toml`. Wheels target abi3 `cp310`; repo tooling needs Python ≥ 3.11. The fork pins Symphonia to `=0.6.1` and uses a patched vendored Hound. Update `BUILD-METADATA.md` only for a published fork release. |
| Adding files | Follow upstream layout for code, bindings and tests. Keep the library description in `README.md`, user documentation in `docs/` and `docs/ru/`, and shipped changes in `CHANGELOG.md`. Put fork development materials in `dev-docs/`, including plans and benchmark harnesses with their datasets. Never introduce `locals/` or a root `benchmarks/`. |

## Choosing checks

**Default:** Start with the smallest relevant check. Widen coverage for shared APIs, binding changes, packaging or release risk, or an observed failure.

| Condition | Check |
| --- | --- |
| Delivery of a change under `sonara/`, `sonara-python/` or `python/` | Run `.\.venv\Scripts\python.exe scripts/run_gate.py` from the repo root. It runs the core tests in every build configuration, rebuilds the bindings, runs the Python suites and the fidelity gates, and prints one verdict. |
| Rust core | Place unit tests at the bottom of their module. While iterating, run the targeted `cargo test -p sonara --lib <filter>`. |
| Python API or bindings | For contract tests on Windows, build the bindings and run `.\.venv\Scripts\python.exe scripts/run_python_tests.py` from the repo root using the project environment. Suite files run as plain scripts: a bare pytest-style function without a driver silently runs zero tests. |
| BPM, key or chord behavior | Require before/after numbers on labeled data. A `blocked` domain in `tests/fidelity_gates.json` stops the change until evidence is supplied. For tempo, avoid static ACF score-ratio rules that regress House/Techno controls. |
| Analysis timing | Benchmark the Rust runner on real files. |
| Independent feature checks on the maintainer's Windows machine | Use the shared `audio-python` from `C:\Utils\tools\audio` for independent comparisons with librosa and pyloudnorm. Both Essentia executables, `ffmpeg` and `ffprobe` are on PATH. For native dynamics, run `audio-python dev-docs/bench/scripts/native-dynamics/verify.py ebu` from the repo root with bindings built. Use `itu` for BS.2217 or `tracks` to regenerate the 60-track report. |
