from umdd.copybook.parser import parse_copybook


def test_parse_copybook_extracts_fields_and_occurs():
    text = """
     01  CUSTOMER-REC.
         05  NAME        PIC X(10).
         05  BALANCE     PIC 9(5)   USAGE COMP-3.
         05  BRANCH      PIC X(4)   OCCURS 2.
    """
    fields = parse_copybook(text)
    assert len(fields) == 3
    assert fields[0].name == "NAME"
    assert fields[1].usage == "COMP-3"
    assert fields[2].occurs == 2
