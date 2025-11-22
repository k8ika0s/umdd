from pathlib import Path

from umdd.manifest import Manifest, validate_manifest


def test_manifest_validation_with_missing_file(tmp_path: Path) -> None:
    mf = Manifest(
        name="missing",
        codepage="cp037",
        path=tmp_path / "missing.bin",
    )
    result = validate_manifest(mf)
    assert result["warnings"] == ["file_missing"]


def test_manifest_validation_hash(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.bin"
    data_path.write_bytes(b"abcd")
    mf = Manifest(
        name="sample",
        codepage="cp037",
        path=data_path,
        hash="sha256:88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589",
    )
    result = validate_manifest(mf)
    assert result["hash_match"] is True
    assert result["records"] == 0  # not RDW, but we still parsed zero records
