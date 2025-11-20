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

# Multi-head Training (Codepage + Tags + Boundaries)

## Purpose
- Exercise a full training + inference path before real datasets arrive.
- Produce a single checkpoint with shared encoder + heads for codepage classification, token tagging (RDW/text/packed/binary), and field-boundary cues on synthetic RDW data.
- Keep defaults small enough for CPU so contributors can iterate without GPUs.

## What it does now
- Builds labeled synthetic samples from the RDW generator (known layout: RDW, text, packed decimal, binary ints) and pads to a fixed max sequence length.
- Uses a lightweight Transformer encoder (sinusoidal positional encoding) with three heads:
  - Codepage classifier (pooled)
  - Token tagger (PAD/RDW/TEXT/PACKED/BINARY)
  - Boundary detector (start-of-field flags)
- Emits a checkpoint (`multihead.pt`) and metrics JSON (`multihead_metrics.json`) into `--output-dir`.

## How to run
- CLI: `umdd train multitask --output-dir artifacts/multihead --epochs 1 --device cpu`
- Programmatic:
  ```python
  from pathlib import Path
  from umdd.training.multitask import MultiTaskConfig, train_multitask

  cfg = MultiTaskConfig(output_dir=Path("artifacts/multihead"), epochs=1, samples_per_codepage=64)
  metrics = train_multitask(cfg)
  print(metrics)
  ```

## Inference
- CLI: `umdd infer --checkpoint artifacts/multihead/multihead.pt --max-records 1 data/sample.bin`
- Programmatic:
  ```python
  from umdd.inference import infer_bytes
  results = infer_bytes(open("data/sample.bin", "rb").read(), Path("artifacts/multihead/multihead.pt"))
  print(results[0].codepage, results[0].tag_spans)
  ```

## Why synthetic first
- We can validate model shapes, padding/masking, and checkpoint compatibility with the CLI before plugging in real corpora.
- Synthetic labels come “for free” from the known RDW layout, giving us quick regression coverage.

## Next steps when real data arrives
- Swap the synthetic dataset with real RDW/BDW loaders and copybook-derived labels where available.
- Fine-tune or retrain with real codepage distributions and richer field structures; update metrics to include real-data validation splits.
