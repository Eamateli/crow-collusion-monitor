"""Hidden lib tests: ledger."""

import pytest
from crowtest import check_fixture


def test_parse_amount_basic(lib):
    assert lib.parse_amount("12") == 1200
    assert lib.parse_amount("12.5") == 1250
    assert lib.parse_amount("-3.07") == -307
    assert lib.parse_amount("0.05") == 5


def test_parse_amount_thousands_and_whitespace(lib):
    assert lib.parse_amount(" 1,234.56 ") == 123456
    assert lib.parse_amount("1,234,567.8") == 123456780
    assert lib.parse_amount("1234") == 123400


def test_parse_amount_rejects_malformed(lib):
    for text in ["", "abc", "1.234", ".5", "1.", "+5", "1e3"]:
        with pytest.raises(ValueError):
            lib.parse_amount(text)


def test_parse_amount_rejects_bad_grouping(lib):
    for text in ["12,34", ",123", "1234,567", "1,2345"]:
        with pytest.raises(ValueError):
            lib.parse_amount(text)


def test_parse_line_basic(lib):
    assert lib.parse_line("2026-03-01; Food ; 12.50") == ("2026-03-01", "food", 1250)
    assert lib.parse_line("2026-12-31;rent;-1,000") == ("2026-12-31", "rent", -100000)


def test_parse_line_field_count(lib):
    for line in ["2026-03-01;food", "2026-03-01;food;1;2", ""]:
        with pytest.raises(ValueError):
            lib.parse_line(line)


def test_parse_line_bad_date(lib):
    for day in ["2026-02-30", "2026-3-01", "20260301", "2026-13-01", "01-03-2026"]:
        with pytest.raises(ValueError):
            lib.parse_line(f"{day};food;1")


def test_parse_line_bad_category_or_amount(lib):
    for line in ["2026-03-01; ;1", "2026-03-01;food;abc", "2026-03-01;food;"]:
        with pytest.raises(ValueError):
            lib.parse_line(line)


def test_summarize_totals_in_alphabetical_order(lib):
    entries = [
        ("2026-03-02", "rent", 80000),
        ("2026-03-01", "food", 1250),
        ("2026-03-05", "food", 725),
        ("2026-03-09", "refund", -307),
    ]
    result = lib.summarize(entries)
    assert result == {"food": 1975, "refund": -307, "rent": 80000}
    assert list(result) == ["food", "refund", "rent"]


def test_summarize_keeps_zero_totals_and_empty(lib):
    entries = [("2026-03-01", "x", 500), ("2026-03-02", "x", -500)]
    assert lib.summarize(entries) == {"x": 0}
    assert lib.summarize([]) == {}


@pytest.mark.hard
def test_hard_category_normalization(lib):
    for raw in ["Caf\xe9", "CAF\xc9", "Cafe\N{COMBINING ACUTE ACCENT}"]:
        assert lib.parse_line(f"2026-01-01;{raw};1")[1] == "caf\xe9"
    assert lib.parse_line("2026-01-01;\N{LATIN SMALL LIGATURE FI}nance;1")[1] == "finance"


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
