"""Decode a dataset described by a manifest into Arrow/JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

from umdd.inference import infer_bytes, results_to_arrow, results_to_jsonl
from umdd.manifest import load_manifest, validate_manifest


def decode_manifest(manifest_path: Path, checkpoint: Path, output_dir: Path) -> dict:
    mf = load_manifest(manifest_path)
    validation = validate_manifest(mf)
    if validation.get("warnings"):
        raise RuntimeError(f"Manifest validation warnings: {validation['warnings']}")

    data = mf.path.read_bytes()
    results = infer_bytes(data, checkpoint=checkpoint, max_records=None, include_confidence=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{mf.name}_infer.jsonl"
    arrow_path = output_dir / f"{mf.name}_infer.arrow"

    results_to_jsonl(results, jsonl_path)
    results_to_arrow(results, arrow_path)

    return {
        "manifest": mf.name,
        "records": len(results),
        "jsonl": str(jsonl_path),
        "arrow": str(arrow_path),
        "validation": validation,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode a dataset described by a manifest.")
    parser.add_argument("manifest", type=Path, help="Path to manifest (json/yaml).")
    parser.add_argument("checkpoint", type=Path, help="Multi-head checkpoint.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/decoded"), help="Where to write outputs."
    )
    args = parser.parse_args()

    summary = decode_manifest(args.manifest, args.checkpoint, args.output_dir)
    print(json.dumps(summary, indent=2))
