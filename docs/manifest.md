# Dataset Manifest Specification (draft)

Purpose: a small, human-editable manifest to describe real mainframe datasets so we can validate them (lengths, RDW/BDW, codepage, copybook alignment) before training/inference and keep a record of provenance/hashes.

Format: JSON or YAML. Example:

```json
{
  "name": "sample_cp037",
  "codepage": "cp037",
  "path": "data/real/CP037/sample.bin",
  "bdw": false,
  "copybook": "data/copybooks/sample.cpy",
  "recfm": "VB",
  "lrecl": null,
  "hash": "sha256:deadbeef...",
  "notes": "synthetic placeholder",
  "checks": {
    "max_records": 20000,
    "min_printable_ratio": 0.2,
    "pii_scan": true
  }
}
```

Fields:
- `name`: identifier for the dataset (required).
- `codepage`: expected codepage (required).
- `path`: path to the binary file (required).
- `bdw`: whether the file uses BDW wrapping (default false).
- `copybook`: optional copybook to derive tag/boundary labels.
- `recfm`: optional RECFM hint (F/FB/V/VB).
- `lrecl`: optional LRECL for fixed-length files.
- `hash`: optional SHA256 or similar (`sha256:<hex>`).
- `notes`: freeform.
- `checks` (optional):
  - `max_records`: stop after N records during validation.
  - `min_printable_ratio`: flag if printable ratio drops below this.
  - `pii_scan`: run simple PII pattern checks (email/SSN-ish).

Validator goals:
- Ensure file exists, hash matches (if provided).
- Confirm RDW/BDW structure parses sanely; flag truncated/oversized records.
- Estimate printable ratios; warn on very low printability.
- If copybook provided, attempt alignment and report coverage.
- Optionally run light PII-ish regex checks and report counts (no blocking).

CLI (planned):
- `umdd manifest validate manifest.json` -> JSON summary + exit status.
- `umdd manifest sample` -> emit a template manifest for editing.
