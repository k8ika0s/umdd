
# Universal Mainframe Data Decoder (UMDD)
## Full Technical Architecture Document
[![CI](https://github.com/k8ika0s/umdd/actions/workflows/ci.yml/badge.svg)](https://github.com/k8ika0s/umdd/actions/workflows/ci.yml) [![Eval & Benchmark](https://github.com/k8ika0s/umdd/actions/workflows/eval.yml/badge.svg)](https://github.com/k8ika0s/umdd/actions/workflows/eval.yml)

## Executive Abstract
Modernization pipelines consistently fail at their weakest link: raw mainframe data does not self-describe. Encoding varies by codepage, structure is hidden behind legacy formats, and mixed-mode binary/text datasets make traditional ASCII↔EBCDIC conversion brittle, lossy, and error-prone.

UMDD (Universal Mainframe Data Decoder) introduces an AI-driven, byte-level intelligence engine capable of detecting encoding, structure, record boundaries, numeric fields, packed-decimal formats, copybook-aligned layouts, and semantic meaning—without human intervention.

Running on IBM Z, LinuxONE, or distributed GPU systems, UMDD becomes a drop-in solution for high-confidence data extraction into UTF-8, Arrow, Parquet, JSON, SQL, or analytical pipelines.

UMDD transforms raw mainframe data into fully structured, validated modern data at scale.

## System Overview
UMDD architecture consists of:

- Byte Embedding Layer
- Codepage & Encoding Detector
- Field Type Classifier
- Boundary & Structure Inference Engine
- Contextual EBCDIC→UTF-8 Translator
- Structured Output Generator (Arrow, Parquet, JSON, SQL)

### High-Level Pipeline
RAW DATA → Byte Embeddings → Multi-Head Analysis → Translation → Structured Output

## Core Architecture

### Byte Embedding Layer
- 256-token vocabulary (one per byte)
- Embedding dimensions: 32–128
- Optimized for s390x vector instructions
- Optional positional encoding for RDW, BDW, copybook alignment

### Multi-Head Model Components

#### Codepage Detection Head
Predicts EBCDIC codepage (CP037, CP1047, CP500, etc.) with probability scoring.

#### Field-Type Classifier
Tags bytes as:
- TEXT_EBCDIC
- NUMERIC_PACKED (COMP-3)
- NUMERIC_BINARY (COMP/BIN)
- ZONED_DECIMAL
- CONTROL_CODE
- HEADER_FIELD
- STRUCTURAL_METADATA

Prevents corruption during conversion.

#### Numeric Structure Detection
Detects:
- packed decimal (COMP-3)
- binary int (COMP, COMP-5)
- zoned decimal (with signed overpunch)
- IBM date formats (CYYMMDD, YYDDD)

#### Record Boundary Detection
Uses:
- attention patterns
- sequential alignment
- heuristic + learned signals
- RL-enhanced synthetic training

Outputs byte ranges marking field boundaries.

### Structure Inference Engine
Identifies:
- field offsets and lengths
- repeating record groups
- nested structures (CICS, MQ, SMF, VSAM)
- PIC X / PIC 9 inference without copybooks

### Contextual Translation Head
Sequence-to-sequence EBCDIC→UTF-8 with:
- contextual disambiguation
- vocabulary priors
- anomaly correction
- robust handling of mixed-mode datasets

### Structured Output Generator
Can emit:
- JSON
- CSV
- SQL rowsets
- Apache Arrow tables
- Parquet files
- Iceberg records

## Training Pipeline

### Phase 1 — Byte Modeling
- masked byte prediction
- autoencoder reconstruction
- next-byte prediction

### Phase 2 — Mixed-Mode Labeling
- supervised tagging for text vs binary vs packed
- boundary detection
- segmentation of synthetic COBOL datasets

### Phase 3 — Translation Training
- parallel EBCDIC/UTF-8 corpora
- sequence alignment loss
- contextual reconstruction

### Phase 4 — Structure Inference
- synthetic copybook generator
- VSAM/DB2/SMF real-world samples
- multi-loss training for segmentation + classification

### Loss Functions
- Cross-entropy (codepage)
- Token CE (field-tagging)
- Boundary regression
- Seq2Seq + CTC (translation)
- Multi-label field boundary loss

## Runtime & Platform Architecture

### IBM Z / LinuxONE Runtime
Optimized for:
- Telum AI accelerator
- vector extensions
- USS deployment
- z/OSMF integration

### Cloud / Distributed Runtime
- Kubernetes
- Red Hat OpenShift Z
- GPU clusters
- watsonx.data integration

### Performance Targets
- Batch decode: 500–1200 MB/s
- Streaming latency: 2–5 ms
- Telum inference: <1 µs per 4KB block

### Memory Layout
- Arrow buffers for intermediate state
- zero-copy transforms
- vector-friendly memory alignment

## IBM Ecosystem Integration

### z/OS Integration
Supports:
- VSAM RLS
- SMF 30/70/110 flows
- DFSORT/ICETOOL pre/post-processing
- z/OS Connect EE pipelines

### CICS & MQ
Decodes:
- MQMD
- MQRFH2
- COMMAREA structures

### DB2
Handles:
- DSNTIAUL unload formats
- mixed-mode binary structures
- numeric compression

### watsonx.data
UMDD → Arrow/Parquet → watsonx.data ingest → analytics/AI

## Roadmap (MVP → GA)

### MVP
- Byte embeddings
- Codepage classifier
- Field classifier
- Basic translator
- JSON output

### Beta
- Packed decimal decoding
- Boundary detection
- Arrow output
- VSAM support

### GA
- Full structure inference
- Parquet output
- Telum acceleration
- watsonx.data connectors

## Risks & Mitigations
- Ambiguous codepages → confidence scoring
- Data corruption → anomaly head
- Complex COBOL → synthetic training
- Mixed-mode unforeseen formats → domain adapters

## Conclusion
UMDD transforms the modernization ecosystem by automating the hardest part: decoding legacy, unstructured, encoded mainframe data. It brings AI-driven understanding to byte streams, unlocking hybrid cloud, analytics, and AI workflows across IBM Z and LinuxONE.

## Why this stack (Python/PyTorch, Typer, PyArrow, orjson)
- Python + PyTorch: fastest iteration for byte-level modeling, rich ecosystem for token/sequence work, and easy CUDA opt-in when available; keeps training and early runtime in one language while we prototype.
- Typer + Rich: ergonomic CLI with typed params and readable console output; lowers friction for data scientists and platform teams.
- PyArrow: zero-copy buffers and a straight path to Arrow/Parquet outputs and watsonx.data ingest; matches the runtime target formats.
- orjson: fast, deterministic JSON for metadata and fixtures; useful in tests and quick previews.
- Make + Docker test harness: repeatable local targets and a clean, pinned CPU image (PyTorch base) ensure we can run lint/type/tests identically across contributors and CI without touching host environments.

## Operational status & automation (what runs, why it exists)
- CI (`ci.yml`, badge above) executes `make check` (format + lint + type + tests) on every push/PR for Python 3.10/3.11. Rationale: keep contributors honest and avoid local drift—PyTorch and Rich/Typer ergonomics stay reproducible across machines.
- Eval & Benchmark (`eval.yml`, badge above) runs daily at 08:00 UTC and on-demand. It: (1) generates synthetic fixtures, (2) runs the heuristic decoder with auto-logging to CSV/JSONL, and (3) captures a micro-benchmark of throughput. Rationale: we lack real data today, so we need visibility into heuristic drift and perf regressions on a fixed synthetic corpus.
- Artifacts: from any Eval run, download `eval-logs` in GitHub Actions → Eval & Benchmark → latest run; contains `logs/eval.csv`, `logs/eval.jsonl`, and `logs/benchmark.json`. These mirror local logs to keep trend tracking centralized.
- Container-first checks (`make docker-test`) mirror CI in a pinned PyTorch CPU image. Rationale: no host pollution and consistent dependency resolution (PyTorch wheels can be finicky); this is the recommended path even on laptops.
- Change log: `docs/runlog.md` records every meaningful change with the “why” so newcomers can catch up quickly and auditors can trace decisions.

## Current CLI capabilities (early scaffolding)
- `umdd decode <input> [-o output.json]`: reads raw bytes, runs a heuristic codepage preview (CP037/273/500/1047) to keep the tool useful before ML heads land, and emits structured metadata.
- `umdd dataset synthetic <output.bin> --metadata meta.json`: emits RDW-prefixed records that mix EBCDIC text, packed-decimal amounts, and big-endian integers; gives us reproducible fixtures for tests and decoder smoke checks.
- `umdd dataset copybook <copybook.cpy> <output.bin> [--bdw --binary-endian little --no-overpunch]`: generates synthetic data from a copybook with options for BDW blocks, COMP endianness, and zoned overpunch; rationale: mirror copybook-driven structures while we await real datasets.
- `umdd eval heuristic [--input file]`: runs the heuristic decoder over a supplied or synthetic dataset and summarizes codepage guesses/printability; rationale: establishes a measurable baseline for future ML heads. With `--output`, auto-logs to `logs/eval.csv` and `logs/eval.jsonl` by default (disable via `--no-auto-log`); override paths with `--log-csv/--log-jsonl`. Summaries: `umdd eval summarize logs/eval.csv`.
- `umdd train codepage --output-dir artifacts/codepage`: trains a tiny embedding + mean-pool classifier on synthetic data to fix the interface and checkpoint format early; rationale: validates data loaders and training plumbing before real datasets arrive. Accepts real RDW datasets per codepage via `--dataset CP037=path1.bin,path2.bin` and can top up with synthetic via `--synthetic-extra-per-codepage`.
- `umdd train multitask --output-dir artifacts/multihead`: trains a small shared encoder + heads for codepage, token tags, and field-boundary cues on synthetic RDW data; rationale: exercise the end-to-end training/inference path before real data lands.
- `umdd infer --checkpoint artifacts/multihead/multihead.pt --max-records 1 <input>`: runs the multi-head model on raw RDW data and emits predicted codepage + tagged spans + boundary positions; rationale: prove a working, testable inference path while we iterate on data/model fidelity.
- Notebook playground: `notebooks/umdd_playground.ipynb` with step-by-step synthetic generation, heuristic preview, tiny multi-head training, and inference. Rationale: give notebook-oriented users a single, heavily commented place to experiment without the CLI.
- Quick demos: sample copybooks live in `data/copybooks/sample.cpy`, `data/copybooks/smf_sample.cpy`, and `data/copybooks/mq_sample.cpy`; generate fixtures via `umdd dataset copybook data/copybooks/sample.cpy out.bin --metadata out.json`.

## Dev Setup (Python/PyTorch)
- Container-first recommended: use `make docker-test` to run formatting/lint/type/tests in a clean PyTorch CPU image pinned in `docker/Dockerfile.test`. This avoids local PyTorch installs.
- If you prefer local: Python 3.10+ with a virtual environment (`python -m venv .venv && source .venv/bin/activate`), then `pip install hatch` and `pip install -e .[dev]`. If you have GPU/CUDA, install the matching PyTorch build per https://pytorch.org/.
- Run the CLI: `umdd decode path/to/input.bin -o output.json` or generate fixtures with `umdd dataset synthetic data/synth.bin --metadata data/synth.json`.

## Quickstart (copy-paste)
1) Container checks: `make docker-test` (builds python:3.10-slim + torch CPU, runs format/lint/type/tests).
2) Generate synthetic RDW fixtures: `umdd dataset synthetic data/synth.bin --metadata data/synth.json`.
3) Copybook-driven fixture: `umdd dataset copybook data/copybooks/sample.cpy data/sample.bin --metadata data/sample.json` (flags: `--bdw`, `--binary-endian little`, `--no-overpunch`).
4) Heuristic decode + log: `umdd eval heuristic --input data/sample.bin --output logs/eval_latest.json` (auto-logs CSV/JSONL unless `--no-auto-log`).
5) Train multi-head model (codepage + tags + boundaries): `umdd train multitask --output-dir artifacts/multihead --epochs 1`.
6) Run inference on synthetic bytes: `umdd infer --checkpoint artifacts/multihead/multihead.pt --max-records 1 data/sample.bin`.
7) Train codepage head on synthetic: `umdd train codepage --output-dir artifacts/codepage`.
8) Benchmark: `python scripts/benchmark.py` to track heuristic throughput over time.

## Testing
- Run unit tests with `pytest`. Early coverage locks in RDW handling, packed-decimal encoding, and the heuristic codepage preview so we can refactor safely as ML components arrive.
- Use `make check` for the full local suite (format + lint + type + test). Make ensures repeatable commands and reduces drift across contributors.
- Dockerized test harness: `docker/Dockerfile.test` pins a clean PyTorch CPU environment so we can validate dependencies and tests without touching host machines. Build/run via `make docker-test`.
- CI: GitHub Actions workflow (`.github/workflows/ci.yml`) runs `make check` on Python 3.10/3.11 for consistent automation.
- Micro-benchmark: `python scripts/benchmark.py` to gauge heuristic throughput on synthetic data; useful to track changes as heuristics evolve.

## Evaluation & Logging
- Heuristic eval harness (`umdd eval heuristic`) outputs summary stats; log them over time with `--log-csv/--log-jsonl` (auto-enabled when `--output` is used) or lower-level helpers (`summary_to_row`, `append_csv`, `append_jsonl`) for trend tracking until ML heads land.
- Summaries of logs: `umdd eval summarize logs/eval.csv` computes aggregates without extra dependencies.
- Log viewer: `python scripts/view_eval_logs.py logs/eval.csv` prints aggregates and codepage counts via Rich; rationale: quick human-readable view without adding plotting deps.
- Rationale: CI-friendly, text-based logs keep baselines visible without bespoke dashboards. Visuals can be added later if needed.

## Real datasets
- Public search (Hugging Face queries for EBCDIC/mainframe/cobol/VSAM) did not surface raw EBCDIC/VSAM binaries; we are intentionally leaning on synthetic data until org-provided datasets arrive.
- When you have real RDW-prefixed binaries, drop them under `data/real/<CODEPAGE>/` and train with `--dataset CP037=data/real/CP037/sample.bin` (repeat per codepage). Rationale: keep real data separated, portable, and easily referenced from training CLI flags.
- Data ask for partners: see `data-request.txt` for what “good” looks like (codepages, RDW/BDW hints, packed/zoned/binary coverage, and masking expectations) so upstream teams can supply useful and compliant samples.
- Sample copybooks for fixtures: `data/copybooks/sample.cpy`, `data/copybooks/smf_sample.cpy`, `data/copybooks/mq_sample.cpy`. Generate data with `umdd dataset copybook <cpy> out.bin --metadata out.json` to mirror realistic layouts while we wait for real bytes.

## Change & Run Log
- Ongoing summary of changes and rationale: `docs/runlog.md`.
