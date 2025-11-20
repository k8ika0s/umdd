import torch

from umdd.data.generator import generate_synthetic_dataset
from umdd.training.codepage import CodepageClassifier, CodepageDataset, CodepageTrainingConfig


def test_codepage_dataset_shapes_bytes_to_tensor():
    ds = CodepageDataset(codepages=["cp037", "cp273"], records_per_page=2, max_len=16)
    x, label = ds[0]
    assert x.shape == (16,)
    assert 0 <= label < 2
    assert int(x[0]) >= 0


def test_codepage_classifier_forward_smoke():
    model = CodepageClassifier(embed_dim=8, num_classes=2)
    batch = torch.zeros((4, 16), dtype=torch.long)
    logits = model(batch)
    assert logits.shape == (4, 2)


def test_training_config_defaults_are_reasonable(tmp_path):
    cfg = CodepageTrainingConfig(output_dir=tmp_path)
    assert cfg.batch_size == 64
    assert cfg.epochs >= 1
    assert len(cfg.codepages) >= 2


def test_codepage_dataset_accepts_external_paths(tmp_path):
    data, _ = generate_synthetic_dataset(count=1, codepage="cp273")
    p = tmp_path / "cp273.bin"
    p.write_bytes(data)
    ds = CodepageDataset(
        codepages=["cp037", "cp273"],
        records_per_page=1,
        max_len=16,
        dataset_paths={"cp273": [p]},
        synthetic_extra_per_codepage=1,
    )
    labels = {label for _, label in ds}
    assert labels == {0, 1}
