from pathlib import Path

import pyarrow.ipc as pa_ipc

from umdd.inference import InferenceResult, results_to_arrow, results_to_jsonl


def test_results_to_jsonl_and_arrow(tmp_path: Path) -> None:
    results = [
        InferenceResult(
            record_index=0,
            byte_length=12,
            codepage="cp037",
            codepage_confidence=0.9,
            tag_spans=[{"start": 0, "end": 4, "tag": "RDW"}],
            boundary_positions=[0, 4],
            tag_confidences=[0.8, 0.7],
            boundary_confidences=[0.6, 0.4],
        )
    ]
    jsonl_path = tmp_path / "out.jsonl"
    arrow_path = tmp_path / "out.arrow"

    results_to_jsonl(results, jsonl_path)
    assert jsonl_path.exists()
    assert "cp037" in jsonl_path.read_text()

    results_to_arrow(results, arrow_path)
    assert arrow_path.exists()
    with pa_ipc.open_file(arrow_path) as reader:
        table = reader.read_all()
    assert table.num_rows == 1
