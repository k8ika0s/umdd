"""Synthetic mainframe-like dataset generator.

The generator builds mixed-mode records with:
- EBCDIC text (default CP037) for names/regions
- packed decimal (COMP-3 style) monetary fields
- big-endian binary integers for dates/IDs
- RDW (record descriptor word) prefix to mimic variable-length records

This is used for fixtures, fuzzing, and regression tests before real data arrives.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_CODEPAGE = "cp037"


@dataclass
class RecordSpec:
    customer_id: int
    region: str
    amount_cents: int
    start_date: date


def to_packed_decimal(value: int, digits: int) -> bytes:
    """Encode an integer into packed decimal (COMP-3 style).

    digits controls how many numeric digits (no decimal point) are encoded
    before the sign nibble. The sign nibble uses C (positive) or D (negative).
    """
    sign_nibble = "C" if value >= 0 else "D"
    digit_str = str(abs(value)).rjust(digits, "0")
    nibbles = list(digit_str) + [sign_nibble]
    if len(nibbles) % 2:
        nibbles.insert(0, "0")
    packed = bytearray()
    for hi, lo in zip(nibbles[0::2], nibbles[1::2], strict=False):
        packed.append((int(hi, 16) << 4) | int(lo, 16))
    return bytes(packed)


def _rdw_prefix(record_body: bytes) -> bytes:
    """RDW: 2-byte length (including RDW) + 2-byte reserved."""
    length = len(record_body) + 4
    return length.to_bytes(2, "big") + b"\x00\x00"


def _build_record(spec: RecordSpec, codepage: str = DEFAULT_CODEPAGE) -> tuple[bytes, dict]:
    """Assemble a single record body and metadata."""
    name_field = f"CUST{spec.customer_id:05d}"
    region_field = spec.region.ljust(4)[:4]
    text = f"{name_field}{region_field}"
    text_bytes = text.encode(codepage, errors="replace")

    amount_packed = to_packed_decimal(spec.amount_cents, digits=7)  # cents, 7 digits -> 4 bytes
    start_days = (spec.start_date - date(1970, 1, 1)).days
    start_bytes = start_days.to_bytes(4, "big", signed=True)

    record_body = text_bytes + amount_packed + start_bytes
    metadata = {
        "customer_id": spec.customer_id,
        "region": region_field.strip(),
        "amount_cents": spec.amount_cents,
        "start_date": spec.start_date.isoformat(),
        "codepage": codepage,
        "record_length": len(record_body),
    }
    return _rdw_prefix(record_body) + record_body, metadata


def generate_synthetic_dataset(
    count: int = 8, *, seed: int = 1234, codepage: str = DEFAULT_CODEPAGE
) -> tuple[bytes, list[dict]]:
    """Generate a dataset of RDW-prefixed records plus metadata."""
    rng = random.Random(seed)
    base_date = date(2023, 1, 1)
    regions: Sequence[str] = ("NYC", "LDN", "SFO", "TOR", "FRA", "TKY")
    records: list[bytes] = []
    metadata: list[dict] = []

    for i in range(count):
        spec = RecordSpec(
            customer_id=10_000 + i,
            region=rng.choice(regions),
            amount_cents=rng.randint(50_00, 250_00),
            start_date=base_date + timedelta(days=rng.randint(0, 365)),
        )
        record_bytes, record_meta = _build_record(spec, codepage=codepage)
        records.append(record_bytes)
        metadata.append(record_meta)

    return b"".join(records), metadata


def iter_records_with_rdw(
    dataset: bytes,
) -> Iterable[tuple[int, bytes]]:
    """Iterate over RDW-prefixed records, yielding (length, body)."""
    idx = 0
    total = len(dataset)
    while idx + 4 <= total:
        rdw = dataset[idx : idx + 4]
        length = int.from_bytes(rdw[:2], "big")
        if length < 4 or idx + length > total:
            break
        body = dataset[idx + 4 : idx + length]
        yield length, body
        idx += length
