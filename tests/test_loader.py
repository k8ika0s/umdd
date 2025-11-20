from umdd.copybook.parser import parse_copybook
from umdd.data.loader import iter_bdw_records, labels_from_copybook, rdw_record_with_header


def test_iter_bdw_records_parses_blocks():
    body1 = b"AAAA"
    body2 = b"BBBBB"
    rdw1 = len(body1) + 4
    rdw2 = len(body2) + 4
    rec1 = rdw1.to_bytes(2, "big") + b"\x00\x00" + body1
    rec2 = rdw2.to_bytes(2, "big") + b"\x00\x00" + body2
    block = rec1 + rec2
    bdw = len(block) + 4
    dataset = bdw.to_bytes(4, "big") + block
    records = list(iter_bdw_records(dataset))
    assert len(records) == 2
    assert records[0][1] == body1
    assert records[1][1] == body2


def test_copybook_labels_from_field_layout():
    copybook = """
    01 NAME PIC X(4).
    01 AMOUNT PIC 9(4) COMP-3.
    01 FLAG PIC X(2).
    """
    fields = parse_copybook(copybook)
    record_body = b"TEST" + b"\\x12\\x3C" + b"OK"
    labels = labels_from_copybook(
        fields=fields,
        body=record_body,
        max_len=32,
        tag_to_id={"PAD": 0, "RDW": 1, "TEXT": 2, "PACKED": 3, "BINARY": 4},
        boundary_pad=-100,
    )
    rdw_record = rdw_record_with_header(record_body)
    assert len(rdw_record) == len(record_body) + 4
    # RDW tagged positions
    assert labels.tag_ids[0] == 1
    # Text at offset 4
    assert labels.tag_ids[4] == 2
    # Packed at offset 8
    assert labels.tag_ids[8] == 3
