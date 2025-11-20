# Stack & Tooling Decisions

## Language & Runtime
- Python 3.10+ end-to-end (training + initial runtime); revisit Rust/Go bindings later for hot paths.
- Target Linux (x86_64, s390x) and GPU/CUDA where available.

## ML & Modeling
- PyTorch 2.x for embeddings + multi-head model (codepage detection, field typing, boundaries, translation).
- Hugging Face tooling optional for experiment tracking/checkpointing.
- PyTorch Lightning or plain Trainer scripts; keep dependencies light at start.

## I/O & Data Formats
- PyArrow for Arrow/Parquet buffers and zero-copy intermediates.
- Typer CLI + Rich console for UX; orjson for fast JSON.
- Pandas only for inspection/tests, not hot path.

## Dev Tooling
- Packaging via `pyproject.toml` + `hatchling`.
- Ruff for lint/format; MyPy for types (strict on core).
- Pytest for unit/regression; Hypothesis/fuzz for byte-level cases.
- Pre-commit optional once hooks are stable.

## Environments
- GPU: CUDA builds of PyTorch where available; CPU fallback default.
- CI: matrix for Python 3.10/3.11, CPU; gated GPU jobs if infra allows.
