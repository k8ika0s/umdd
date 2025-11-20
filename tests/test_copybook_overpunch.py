from random import Random

from umdd.copybook.parser import Field
from umdd.copybook.synth import CopybookSynthConfig, _encode_numeric


def test_overpunch_positive_and_negative():
    cfg = CopybookSynthConfig(include_overpunch=True)
    field = Field(name="AMT", pic="9(4)", usage=None)
    rng = Random(1)

    pos = _encode_numeric(field, rng, cfg).decode("cp037")
    neg = _encode_numeric(field, rng, cfg).decode("cp037")

    assert pos[-1].isalpha() or pos[-1] in "{}"
    assert neg[-1].isalpha() or neg[-1] in "}"
