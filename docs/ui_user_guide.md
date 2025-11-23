# UMDD UI User Guide

This guide walks through the Streamlit UI for validation, inference, and quick demo training. The UI is intended for small samples and demos; use the CLI/notebook for full-scale runs.

## Launch
- From repo root: `streamlit run ui/app.py`
- Requirements: dev deps installed (`pip install -e '.[dev]'` or use the test container).

## Tabs
### Validate
- Upload a manifest (json/yaml). Fields should include path, codepage, BDW flag, copybook (optional), and checks.
- UI displays validation results (existence, hash, RDW/BDW parsing, printable ratios, copybook coverage).
- If the dataset exists, a heuristic eval summary is shown.
- Recent manifest paths are listed for convenience.
- For full validation, prefer `umdd manifest validate manifest.json`.

### Infer
- Upload a small RDW/BDW sample (UI warns if >5MB). Choose checkpoint, max records, and whether to include confidences.
- UI shows:
  - Codepage distribution bar chart.
  - Average codepage confidence.
  - Span viewer: pick a record, see spans with avg confidence, tag counts, boundary positions.
  - Performance summary (time, throughput).
  - Download JSON/JSONL/Arrow outputs.
- For larger files, use CLI: `umdd infer --format jsonl --output logs/infer.jsonl --checkpoint artifacts/multihead/multihead.pt data.bin`.

### Train
- Demo-only: runs a tiny synthetic multi-head training job (CPU-friendly defaults).
- Shows metrics, checkpoint path, and recent training runs.
- For real training (with manifests), use CLI: `umdd train multitask --real CODEPAGE=PATH[:copybook[:bdw]] --output-dir artifacts/multihead_real`.

## Guardrails & Tips
- UI is for small samples; prefer CLI/Docker for production-scale data.
- Keep manifests updated and validated before training.
- Use the notebook (`notebooks/umdd_playground.ipynb`) for step-by-step exploration and export options.
