from umdd.eval.harness import evaluate_synthetic


def test_evaluate_synthetic_produces_summary():
    payload = evaluate_synthetic(count=2, seed=42)
    summary = payload["evaluation"]
    assert summary.records == 2
    assert summary.average_printable_ratio > 0
    assert payload["generator"]["seed"] == 42
