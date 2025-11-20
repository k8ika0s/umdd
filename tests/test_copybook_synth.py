from umdd.copybook.parser import Field
from umdd.copybook.synth import CopybookSynthConfig, extract_width, synthesize_from_copybook


def test_extract_width_parses_pic_lengths():
    assert extract_width("X(10)") == 10
    assert extract_width("9(5)") == 5
    assert extract_width("9(3)V99") >= 3


def test_synthesize_from_copybook_produces_rdw_records():
    fields = [
        Field(name="NAME", pic="X(8)", usage=None),
        Field(name="AMOUNT", pic="9(5)", usage="COMP-3"),
    ]
    data, meta = synthesize_from_copybook(fields, count=2, config=CopybookSynthConfig())
    # two RDW-prefixed records produced
    assert data[:2] != b"\x00\x00"  # length prefix present
    assert len(meta) == 2
