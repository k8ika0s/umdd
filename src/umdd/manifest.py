from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from umdd.copybook.parser import parse_copybook
from umdd.data.loader import iter_bdw_records, iter_records_with_rdw, labels_from_copybook

PII_PATTERNS = {
    "email": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn_like": re.compile(rb"\b\d{3}-\d{2}-\d{4}\b"),
}


@dataclass
class Manifest:
    name: str
    codepage: str
    path: Path
    bdw: bool = False
    copybook: Path | None = None
    recfm: str | None = None
    lrecl: int | None = None
    hash: str | None = None
    notes: str | None = None
    checks: dict[str, Any] | None = None

    @staticmethod
    def from_mapping(payload: dict[str, Any]) -> Manifest:
        return Manifest(
            name=str(payload["name"]),
            codepage=str(payload["codepage"]),
            path=Path(payload["path"]),
            bdw=bool(payload.get("bdw", False)),
            copybook=Path(payload["copybook"]) if payload.get("copybook") else None,
            recfm=payload.get("recfm"),
            lrecl=payload.get("lrecl"),
            hash=payload.get("hash"),
            notes=payload.get("notes"),
            checks=payload.get("checks"),
        )


def _hash_file(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def _pii_counts(data: bytes) -> dict[str, int]:
    return {name: len(pattern.findall(data)) for name, pattern in PII_PATTERNS.items()}


def validate_manifest(manifest: Manifest) -> dict[str, Any]:
    path = manifest.path
    result: dict[str, Any] = {
        "name": manifest.name,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "codepage": manifest.codepage,
        "bdw": manifest.bdw,
        "recfm": manifest.recfm,
        "lrecl": manifest.lrecl,
        "copybook": str(manifest.copybook) if manifest.copybook else None,
        "hash_expected": manifest.hash,
        "hash_actual": None,
        "hash_match": None,
        "records": 0,
        "printable_ratio": 0.0,
        "pii_counts": {},
        "copybook_coverage": None,
        "warnings": [],
    }
    if not path.exists():
        result["warnings"].append("file_missing")
        return result

    if manifest.hash:
        algo, _hex = (
            manifest.hash.split(":", 1) if ":" in manifest.hash else ("sha256", manifest.hash)
        )
        actual = _hash_file(path, algo=algo)
        result["hash_actual"] = actual
        result["hash_match"] = actual == manifest.hash
        if not result["hash_match"]:
            result["warnings"].append("hash_mismatch")

    data = path.read_bytes()
    iterator = iter_bdw_records(data) if manifest.bdw else iter_records_with_rdw(data)
    records = list(iterator)
    result["records"] = len(records)
    if manifest.checks and manifest.checks.get("max_records"):
        max_rec = int(manifest.checks["max_records"])
        records = records[:max_rec]
        result["records_capped"] = max_rec

    total_body = b"".join(body for _length, body in records)
    printable = sum(1 for b in total_body if 32 <= b < 127)
    result["printable_ratio"] = round(printable / max(len(total_body), 1), 4)
    if manifest.checks and manifest.checks.get("min_printable_ratio") is not None:
        min_ratio = float(manifest.checks["min_printable_ratio"])
        if result["printable_ratio"] < min_ratio:
            result["warnings"].append("low_printability")

    if manifest.checks and manifest.checks.get("pii_scan"):
        result["pii_counts"] = _pii_counts(total_body)

    if manifest.copybook:
        fields = parse_copybook(manifest.copybook.read_text())
        coverages = []
        for _length, body in records[:5]:
            labels = labels_from_copybook(
                fields,
                body,
                max_len=256,
                tag_to_id={"PAD": 0, "RDW": 1, "TEXT": 2, "PACKED": 3, "BINARY": 4},
                boundary_pad=-100,
            )
            coverages.append(sum(1 for t in labels.tag_ids if t != 0) / len(labels.tag_ids))
        if coverages:
            result["copybook_coverage"] = round(sum(coverages) / len(coverages), 4)
            if result["copybook_coverage"] < 0.2:
                result["warnings"].append("copybook_low_coverage")
    return result


def load_manifest(path: Path) -> Manifest:
    if path.suffix.lower() in {".yml", ".yaml"}:
        payload = yaml.safe_load(path.read_text())
    else:
        payload = json.loads(path.read_text())
    return Manifest.from_mapping(payload)


def sample_manifest() -> dict[str, Any]:
    return {
        "name": "sample_cp037",
        "codepage": "cp037",
        "path": "data/real/CP037/sample.bin",
        "bdw": False,
        "copybook": "data/copybooks/sample.cpy",
        "recfm": "VB",
        "lrecl": None,
        "hash": "sha256:<hex>",
        "notes": "edit with real details",
        "checks": {"max_records": 20000, "min_printable_ratio": 0.2, "pii_scan": True},
    }


def render_validation_html(result: dict[str, Any], output: Path) -> None:
    """Render a simple HTML report for validation results."""
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings = result.get("warnings", [])
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in result.items()
        if k not in {"warnings", "pii_counts", "copybook_coverage"}
    )
    pii = result.get("pii_counts") or {}
    pii_rows = "".join(f"<li>{k}: {v}</li>" for k, v in pii.items())
    html = f"""<!DOCTYPE html>
<html><head><title>UMDD Manifest Validation</title></head>
<body>
<h1>UMDD Manifest Validation Report</h1>
<p><strong>Name:</strong> {result.get("name")}</p>
<p><strong>Warnings:</strong> {", ".join(warnings) if warnings else "None"}</p>
<table border="1" cellpadding="4" cellspacing="0">
{rows}
</table>
<h3>PII Scan</h3>
<ul>{pii_rows}</ul>
</body></html>
"""
    output.write_text(html)
