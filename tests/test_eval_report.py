from pathlib import Path

from umdd.eval.harness import EvalSummary, RecordEval
from umdd.eval.report import append_csv, append_jsonl, summary_to_row
from umdd.eval.summarize import summarize_log


def _sample_summary() -> EvalSummary:
    return EvalSummary(
        records=2,
        average_printable_ratio=0.95,
        codepage_counts={"cp037": 2},
        samples=[RecordEval(detected_codepage="cp037", printable_ratio=0.95, preview="HELLO")],
        notes="test summary",
    )


def test_summary_to_row_and_csv(tmp_path: Path):
    summary = _sample_summary()
    row = summary_to_row(summary, source="synthetic.bin", tag="test")
    out = tmp_path / "log.csv"
    append_csv(out, row)
    content = out.read_text()
    assert "synthetic.bin" in content
    assert "average_printable_ratio" in content


def test_summary_to_row_accepts_mapping():
    summary = {"records": 1, "average_printable_ratio": 0.5, "codepage_counts": {"cp037": 1}}
    row = summary_to_row(summary, source="foo")
    assert row["records"] == 1


def test_append_jsonl(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    payload = {"a": 1}
    append_jsonl(path, payload)
    assert path.exists()
    assert path.read_text().strip() == '{"a": 1}'


def test_append_jsonl_handles_dataclass(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    payload = {"summary": _sample_summary()}
    append_jsonl(path, payload)
    content = path.read_text().strip()
    assert "codepage_counts" in content


def test_summarize_log_over_csv(tmp_path: Path):
    summary = _sample_summary()
    row = summary_to_row(summary, source="synthetic.bin", tag="test")
    out = tmp_path / "log.csv"
    append_csv(out, row)
    agg = summarize_log(out)
    assert agg["entries"] == 1
    assert agg["records_total"] >= 2
