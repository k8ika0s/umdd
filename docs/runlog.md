# Run Log
Entries are newest-first. Each entry captures what changed and why the choice was made.

## 2025-11-20
- Added minimal multi-head model (Transformer encoder + codepage/tag/boundary heads) with synthetic labeled dataset, training loop, inference helpers, and CLI commands (`umdd train multitask`, `umdd infer`); rationale: establish an end-to-end, testable inference path before real data arrives.
- Documented multi-head training/inference flow and why synthetic labels are acceptable for now; refreshed README quickstart to show the new commands and their purpose.
- Refined README with CI/Eval badges, artifact download instructions, and explicit container-first rationale; highlighted data-request expectations for partners. Rationale: make operations/onboarding self-serve before opening a PR and keep the “why” visible for newcomers.
- Added copybook-driven synthetic generator, stubs for field-typing/boundary heads, and a micro-benchmark; dockerized checks + tests passing.
- Expanded copybook synth with overpunch support, endianness toggles, and added tests; containerized lint/type/tests remain green.
- Added CLI subcommand for copybook-driven dataset generation (with BDW/endianness/overpunch options) and heuristic field tagging baseline; rationale: richer synthetic coverage and a starting point for boundary/field-type labels.
- Added scheduled eval/benchmark workflow to produce logs/artifacts from synthetic runs; rationale: trend tracking while awaiting real data.
- Added copybook parser + synthetic generator hooks to build richer synthetic datasets (packed/zoned/binary fields) while awaiting real data; rationale: mirror copybook-driven structures and increase coverage.
- Stubbed field-typing and boundary training modules to align interfaces for future heads; rationale: prep integration paths before implementations land.
- Added a micro-benchmark script for heuristic throughput on synthetic data and documented in README/eval docs; rationale: track perf drift as heuristics evolve.
- Switched Docker test base to python:3.10-slim with pip-installed torch CPU; rationale: avoid missing pytorch/pytorch tags and keep a reproducible CPU environment.
- Hardened heuristic decoder to skip unavailable codepages (e.g., cp1047) to keep tests portable across runtimes.
- Updated Ruff config to new lint.* layout and ignored Typer’s B008 pattern; rationale: remove noise and align with current Ruff expectations.
- Added type-safety tweaks across eval/report/summarize and ensured dockerized `make check` + tests pass.
- Searched for public EBCDIC/mainframe binary datasets (Hugging Face queries for EBCDIC/mainframe/cobol/VSAM) and found no suitable raw byte corpora; action: rely on synthetic plus user-provided real datasets under `data/real`.
- Added `data-request.txt` to guide internal teams on required real datasets (codepages, RDW/BDW, packed/zoned/binary fields, metadata expectations, and privacy constraints).
- Enabled auto-logging for eval runs (defaults to logs/eval.csv/jsonl when --output is used) and added a log viewer script; rationale: keep reports and trend logs in sync and readable without new deps.
- Added eval log summarizer CLI command and Rich-based viewer; rationale: quick aggregates and human-friendly display without pulling plotting libraries.
- Enhanced codepage training to mix real RDW datasets per codepage, allow multiple files, and top up with synthetic records; rationale: prepare for hybrid training while preserving reproducibility.
- Added logging options to eval CLI (CSV/JSONL) and summarizer utilities; rationale: enable trend tracking without new dependencies and keep baselines visible in CI/local workflows.
- Extended codepage training to accept real datasets per codepage while keeping synthetic defaults; rationale: prepare for hybrid training once real data arrives without changing callers.
- Added eval summarizer CLI and report helpers; rationale: quick aggregates over logs for visibility.
- Updated README/docs to emphasize container-first testing (PyTorch inside Docker) and document new flags; rationale: reduce local setup friction and keep rationale explicit.
- Added codepage training scaffold with synthetic dataset, tiny embedding classifier, and CLI entry; rationale: validate training plumbing and checkpoint format before real data arrives.
- Created eval logging helpers (CSV/JSONL) to track heuristic baselines over time; rationale: simple trend tracking without dashboards.
- Added CI workflow (GitHub Actions) and refined Makefile to use fmtcheck for non-destructive checks; rationale: consistent automation across contributors and environments.
- Added training/eval documentation to explain usage and design choices; rationale: keep onboarding clear and decisions discoverable.
- Added Makefile with repeatable install/lint/type/test targets and Docker test harness; rationale: enforce consistency locally and in CI, and provide a cleanroom check path.
- Created test container (`docker/Dockerfile.test`) based on PyTorch CPU image; rationale: pinned environment to validate dependencies and PyTorch installs without polluting host machines.
- Added heuristic evaluation harness and CLI entry for baseline metrics; rationale: measure current behavior on synthetic data and provide a stable interface for future model evals.
- Introduced training skeleton for the codepage head; rationale: fix the API and config surface early so downstream scripts can build against it even before implementation.
- Added synthetic dataset generator and Typer subcommand to produce RDW-prefixed mixed-mode records for fixtures; rationale: need predictable, reproducible bytes to drive tests and early decoders before real mainframe data is available.
- Introduced heuristic decoder path in the CLI to guess codepages and provide a preview; rationale: keeps the tool usable for inspection and establishes baselines until ML heads ship.
- Added initial pytest coverage for the generator and heuristic decoder; rationale: lock in behaviors (RDW lengths, packed-decimal sign nibble, codepage preference) to prevent regressions while iterating.
- Expanded README to document rationale and usage of the Python/PyTorch stack and the new CLI commands; rationale: maintain context on design choices for new contributors.

## 2025-11-19
- Bootstrapped Python/PyTorch project scaffold with Typer CLI stub, planning docs, and stack decisions; rationale: align on the ML-first workflow and set a minimal but functional entrypoint for future heads.
