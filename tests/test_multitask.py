from pathlib import Path

from umdd.data.generator import generate_synthetic_dataset
from umdd.inference import infer_bytes
from umdd.training.multitask import MultiTaskConfig, train_multitask


def test_multitask_training_and_inference(tmp_path: Path) -> None:
    cfg = MultiTaskConfig(
        output_dir=tmp_path,
        samples_per_codepage=4,
        max_len=64,
        batch_size=2,
        epochs=1,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
    )
    metrics = train_multitask(cfg)
    ckpt = Path(metrics["checkpoint"])
    assert ckpt.exists()
    assert metrics["samples"] == len(cfg.codepages) * cfg.samples_per_codepage

    data, _ = generate_synthetic_dataset(count=2)
    results = infer_bytes(data, checkpoint=ckpt, max_records=1)
    assert results, "Inference should produce at least one record."
    first = results[0]
    assert first.tag_spans
    assert first.boundary_positions
