"""Hidden lib tests: table."""

import pytest
from crowtest import check_fixture

ROWS = [
    {"city": "Oslo", "price": "12.5"},
    {"city": "oslo", "price": "8"},
    {"city": "Bergen", "price": "15.25"},
]


def test_read_rows_basic(lib):
    assert lib.read_rows("a,b\n1,2\n3,4\n") == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_read_rows_normalizes_header_and_strips(lib):
    assert lib.read_rows(" Name , AGE \n bob , 42 \n") == [{"name": "bob", "age": "42"}]


def test_read_rows_skips_empty_rows_and_pads(lib):
    text = "a,b,c\n1\n,,\n\n4,5,6,7\n"
    assert lib.read_rows(text) == [{"a": "1", "b": "", "c": ""}, {"a": "4", "b": "5", "c": "6"}]


def test_read_rows_quoted_and_empty(lib):
    assert lib.read_rows('a,b\n"x, y","say ""hi"""\n') == [{"a": "x, y", "b": 'say "hi"'}]
    assert lib.read_rows("") == []
    assert lib.read_rows("a,b\n") == []


def test_filter_rows_ignores_case(lib):
    assert lib.filter_rows(ROWS, "CITY", " OSLO ") == ROWS[:2]
    assert lib.filter_rows(ROWS, "city", "nowhere") == []


def test_filter_rows_unknown_column(lib):
    with pytest.raises(KeyError):
        lib.filter_rows(ROWS, "colour", "red")
    assert lib.filter_rows([], "colour", "red") == []


def test_column_stats_basic(lib):
    assert lib.column_stats(ROWS, "price") == {"count": 3, "min": 8.0, "max": 15.25, "mean": 11.92}


def test_column_stats_skips_non_numeric(lib):
    rows = [{"x": v} for v in ["", "n/a", "nan", "inf", "-inf", "3"]]
    assert lib.column_stats(rows, "X") == {"count": 1, "min": 3.0, "max": 3.0, "mean": 3.0}


def test_column_stats_empty_and_unknown(lib):
    empty = {"count": 0, "min": None, "max": None, "mean": None}
    assert lib.column_stats([{"x": "a"}], "x") == empty
    assert lib.column_stats([], "x") == empty
    with pytest.raises(KeyError):
        lib.column_stats(ROWS, "qty")


@pytest.mark.hard
def test_hard_bom_and_unicode(lib):
    rows = lib.read_rows("\N{BYTE ORDER MARK}Stadt,Gr\xf6\xdfe\nM\xfcnchen,3\n")
    assert rows == [{"stadt": "M\xfcnchen", "gr\xf6sse": "3"}]
    assert lib.filter_rows(rows, "STADT", "M\xdcNCHEN") == rows
    assert lib.filter_rows(rows, "stadt", "Mu\N{COMBINING DIAERESIS}nchen") == rows
    assert lib.column_stats(rows, "GR\xd6SSE")["count"] == 1


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
