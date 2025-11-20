from umdd.decoder import heuristic_decode


def test_heuristic_prefers_cp037_for_cp037_data():
    text = "HELLO MAINFRAME"
    data = text.encode("cp037")
    result = heuristic_decode(data)
    assert result["detected_codepage"] == "cp037"
    assert result["printable_ratio"] > 0.9
    assert result["preview"].startswith("HELLO")
