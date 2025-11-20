# Codepage Head Training (Synthetic Baseline)

## Purpose
- Provide a runnable training path today, even without real datasets.
- Fix the public API (`CodepageTrainingConfig`, `train_codepage_model`) so scripts/CI can integrate early.
- Use synthetic RDW records to validate the plumbing and produce a checkpoint format that we can later swap with real data.

## What it does now
- Builds a synthetic dataset per codepage (CP037/273/500/1047) using the same generator as fixtures; can ingest real RDW datasets per codepage and optionally top up with synthetic records for smoothing.
- Trains a tiny embedding + mean-pool classifier on CPU by default.
- Emits a checkpoint (`codepage_head.pt`) and metrics JSON (avg loss/accuracy, batches, epochs) to `--output-dir`.

## How to run
- CLI: `umdd train codepage --output-dir artifacts/codepage --epochs 2 --device cpu`
- Real data option: pass `--dataset CODEPAGE=path/to/file.bin` (repeatable) to mix real RDW-prefixed datasets with synthetic for that codepage.
- Programmatic:
  ```python
  from pathlib import Path
  from umdd.training.codepage import CodepageTrainingConfig, train_codepage_model

  cfg = CodepageTrainingConfig(output_dir=Path("artifacts/codepage"))
  metrics = train_codepage_model(cfg)
  print(metrics)
  ```

## Why synthetic first
- Blocks removed: we can validate loaders, model plumbing, and checkpoint I/O before securing real data.
- Reproducibility: synthetic generator is deterministic via seed, giving stable tests and CI runs.
- Swap-ready: when real datasets are ready, we can wire a new Dataset without changing callers.

## Real data intake (when available)
- Place RDW-prefixed binaries in `data/real/<CODEPAGE>/...` and pass to the CLI with `--dataset CP037=data/real/CP037/sample.bin`.
- Public searches (Hugging Face for EBCDIC/mainframe/cobol/VSAM) did not surface raw EBCDIC/VSAM binaries; expect to use org-provided datasets for true training/eval.
