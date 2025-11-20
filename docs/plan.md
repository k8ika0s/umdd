# UMDD Build Plan

## Goals
- Deliver an AI-driven decoder that converts raw mainframe byte streams into structured UTF-8/Arrow/Parquet/JSON safely and repeatably.
- Target first: high-confidence codepage detection, field typing, and basic translation with JSON/CSV output; iterate toward structure inference and Parquet.

## Workstreams
- **Data & Evaluation**: Collect/curate representative binaries; build synthetic mainframe dataset generator (copybook variants, mixed text/binary, packed decimals); define golden test vectors and fuzzing harness.
- **Modeling**: Byte embedding stack; heads for codepage detection, field typing, boundary detection, translation; multi-loss training scripts; eval dashboards.
- **Inference Runtime**: Stream-safe pipeline that reads blocks, runs heads, and emits typed spans; contextual EBCDIC→UTF-8 converter; JSON/Arrow writer; hooks for Parquet later; CPU-first, with GPU path.
- **Packaging & Interfaces**: CLI + library API; batch mode and streaming mode; connectors for files, MQ, VSAM/SMF dumps.
- **Performance & Safety**: Telemetry, anomaly/confidence scoring, guardrails for corruption detection; benchmarks for throughput/latency on Z/LinuxONE and x86/GPU.

## Milestones (first passes)
- **Foundation (this week)**: Finalize repo layout; **stack locked: Python 3.10+ with PyTorch for training and runtime**; add lint/format/test harness; dataset generator and golden fixtures; stub CLI for decode + stats.
- **MVP (weeks 2–4)**: Implement byte embeddings + codepage and field-type heads; basic translation head; JSON/CSV emitter; block/record boundary heuristics; regression tests on golden vectors.
- **Beta (weeks 4–8)**: Packed decimal and binary numeric decoding; Arrow output; VSAM/SMF adapters; confidence scoring + anomaly reporting; streaming performance tuning.
- **GA (post-beta)**: Full structure inference, Parquet/Iceberg outputs, Telum acceleration path, watsonx.data connector, expanded copybook inference.

## Next Actions to start
- Stack locked: Python + PyTorch end-to-end; Typer CLI; PyArrow/Parquet bindings; keep Rust/Go runtime option open later for perf.
- Stand up dataset generator + fixture catalog with expected decodes.
- Scaffold training scripts for codepage + field-type heads; wire basic CLI for file ingest → JSON decode using heuristics before ML is ready.
