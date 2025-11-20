from umdd.data.generator import generate_synthetic_dataset, iter_records_with_rdw
from umdd.heuristic_tags import tag_fields


def test_tag_fields_marks_rdw_and_packed():
    data, _ = generate_synthetic_dataset(count=1, seed=1)
    length, body = next(iter_records_with_rdw(data))
    tags = tag_fields(body)
    assert any(span.tag == "PACKED_DECIMAL" for span in tags)


def test_tag_fields_emits_text_and_binary_spans():
    # Construct a record body with text then binary bytes.
    body = b"HELLO" + b"\x00\x01\x02"
    tags = tag_fields(body)
    assert any(span.tag == "TEXT" for span in tags)
    assert any(span.tag == "BINARY" for span in tags)
