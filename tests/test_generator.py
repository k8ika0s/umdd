from umdd.data.generator import (
    generate_synthetic_dataset,
    iter_records_with_rdw,
    to_packed_decimal,
)


def test_to_packed_decimal_sets_positive_sign():
    packed = to_packed_decimal(12345, digits=5)
    # Low nibble of last byte should carry the sign (C for positive).
    assert (packed[-1] & 0x0F) == 0x0C


def test_generate_dataset_shapes_rdw_and_metadata():
    data, meta = generate_synthetic_dataset(count=3, seed=1)
    lengths = []
    for length, body in iter_records_with_rdw(data):
        lengths.append(length)
        assert len(body) == length - 4

    assert len(lengths) == 3
    assert len(meta) == 3
    assert meta[0]["start_date"].startswith("2023-")
