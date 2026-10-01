# Project instructions

This is the single instruction entry point for Sonara, the MeteorBurn fork of
`kkollsga/sonara`. Codex reads this file directly; Claude Code loads it through
`CLAUDE.md`, which only imports it (`@AGENTS.md`). Do not add nested `AGENTS.md` or
`CLAUDE.md` files: neither agent loads them reliably. Layer-specific instructions live in
`dev-docs/agent-guides/`; read them only when the matching task needs them. Keep this file
current by replacing obsolete guidance rather than layering rules.

## OVERVIEW

Sonara is a Rust music-information-retrieval library (librosa-style DSP plus a fused
single-pass track analyzer) with PyO3/Maturin bindings (`import sonara`). The fork (0.3.7)
adds BPM fixes, opt-in rhythm features (`onset_bands`, `rhythmic_regularity`), Symphonia
pinned to `=0.6.1` and a patched vendored Hound.

## TASK-BASED READING

Apply the rules below to every task. Before the matching work, read the relevant guide(s).
Combine routes when a task crosses layers; do not read all guides or the whole README at
session start. `dev-docs/` is git-ignored, so the guides and plans exist only in the
maintainer's checkout. If a required guide is unavailable, report it and keep the dependent
work unverified rather than inventing its instructions.

| Task | Read before acting |
| --- | --- |
| Explore or change repository code; locate unfamiliar code | [Graphify](dev-docs/agent-guides/graphify.md), [Architecture](dev-docs/agent-guides/architecture.md) |
| Core crate `sonara/`: features, fused pass, augment, cargo features, embedded models, `sonara/src/core/` DSP | [Rust core](dev-docs/agent-guides/rust-core.md) |
| Binding `sonara-python/`, facade `python/sonara/`, `.pyi` stub, signatures, result dict shape | [Python API](dev-docs/agent-guides/python-api.md) |
| Select checks; change `tests/`, fixtures, runners or `scripts/` gates | [Verification](dev-docs/agent-guides/verification.md) |
| Detection or numeric behavior; `tests/fidelity_gates.json`; reference data | [Accuracy](dev-docs/agent-guides/accuracy.md), [CONTRIBUTING.md](CONTRIBUTING.md) |
| Stored-result meaning, schema versions, similarity layout | [docs/consumer-contract.md](docs/consumer-contract.md) |
| Build a wheel or library; cut a fork release; Symphonia or Hound setup | [Build and release](dev-docs/agent-guides/build-release.md) |
| Commit, push or PR (on request); upstream sync; downstream adoption; builds, data refreshes, benchmarks or classifier training | [Git and upstream](dev-docs/agent-guides/git-upstream.md) |
| Research harnesses under `dev-docs/bench/` | [Research harnesses](dev-docs/agent-guides/research-harnesses.md) |
| Agent instructions, guides, skills or hooks; workflow skills in `workflow/`; Spec Kit; `dev-docs/` layout | [Agent layer](dev-docs/agent-guides/agent-layer.md) |
| Inspect or query SQLite data | [SQLite](dev-docs/agent-guides/sqlite-toolkit.md) |
| Open threads and plans | [dev-docs/todos.md](dev-docs/todos.md) |
| Need product/setup orientation | Relevant sections of [README.md](README.md) |

## EXECUTION AND SCOPE

- Inspect `git status --short` and the relevant diff before editing; preserve unrelated
  work.
- Commits, branches, pushes and PRs happen only on explicit request. Never force-push
  `origin`. `upstream` (`kkollsga/sonara`) has push disabled; absorb it with a merge commit,
  never a fast-forward. CI `publish` is gated to `kkollsga/sonara`, so pushes here never
  reach PyPI; nothing is published to crates.io or PyPI from this fork.
- This file overrides `workflow/skills/`: their branch, PR and push steps need explicit
  authorization. The `CLAUDE.md`/`AGENTS.md` resync in `dev-docs-cleanup` does not apply:
  this file is the authority and `CLAUDE.md` only imports it.
- Where content belongs: `README.md` is the library description as upstream wrote it plus
  this fork's features, without plans. `docs/` is user documentation only (English and
  `docs/ru/`). `dev-docs/` holds development material: agent guides, plans, designs,
  `todos.md`. `CHANGELOG.md` records shipped work only.
- Text files are UTF-8 with em-dashes and arrows that some consoles print as `-` or `?`, and
  PowerShell may show Cyrillic as mojibake; check codepoints before "fixing" text.

## CONVENTIONS

- `Float = f32` throughout. Sample rate is always an explicit argument; no global SR.
- Opt-in families (`onset_bands`, `beatgrid`, `rhythmic_regularity`, ...) are never
  computed by a mode, only via `features=[...]`.
- Additive fields ship without an `ANALYSIS_SCHEMA_VERSION` bump; bump only when existing
  meaning or units change. Consumers key freshness per feature.
- Abstention is `None` with confidence kept: a stored `None` regularity score means
  "measured, undecidable", not "pending".
- Rust unit tests sit inline at the bottom of each module. Python suite files run as plain
  scripts (`python <file>`): pytest-style `def test_*` with no driver runs nothing and still
  exits 0.
- Versions move together in `sonara/Cargo.toml`, `sonara-python/Cargo.toml` and
  `pyproject.toml`, only via the release procedure.
- Wheel floor is abi3 `cp310`; repo tooling needs Python >= 3.11 (`tomllib`).

## ANTI-PATTERNS

- A feature that runs its own STFT: reuse the fused pass in `analyze_signal_inner`.
- PyO3/NumPy, or an ML runtime in default features, inside `sonara/`: contract boundary;
  downstream must be notified first.
- Reinterpreting a similarity/fingerprint layout without a version bump. Bumping
  `SIMILARITY_VERSION` also invalidates every genre/vocalness/aggression model.
- Running pytest directly: `scripts/run_python_tests.py` runs the contract checks CI gates
  on.
- Trusting green `cargo test -p sonara` for aggression code: the binding always enables
  `aggression`; run the feature build too.
- Static ACF score-ratio rules for tempo: they fix some rows and regress House/Techno
  controls.
- Committing audio, generated benchmark output, or sealed labels.
- Rewriting `BUILD-METADATA.md` for an unpublished build: it records the latest published
  fork release and its wheel hash.
- Timing analysis through Python: benchmark with the Rust runner on real files.

## VERIFICATION AND DELIVERY

- Start at the smallest relevant check and widen only for shared APIs, cross-language
  bindings, packaging, release risk or an observed failure. Canonical gates:
  `cargo test -p sonara`, `cargo test -p sonara --features aggression` for aggression code,
  and `python scripts/run_python_tests.py` with the bindings built.
- BPM, key and chord changes need before/after numbers on labeled data; a `blocked`
  fidelity domain stops a change until evidence is supplied.
- Report which checks ran, which did not and why; never present expected behavior as a
  result. Leave no unexpected tracked changes.
