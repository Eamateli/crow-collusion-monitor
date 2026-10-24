"""Hidden lib tests: durations."""

import pytest
from crowtest import check_fixture

FW = {"1": "\N{FULLWIDTH DIGIT ONE}", "2": "\N{FULLWIDTH DIGIT TWO}"}


def test_parse_plain_seconds(lib):
    assert lib.parse_duration("90") == 90
    assert lib.parse_duration("0") == 0
    assert lib.parse_duration(" 42 ") == 42


def test_parse_units(lib):
    assert lib.parse_duration("1h30m") == 5400
    assert lib.parse_duration("2d") == 172800
    assert lib.parse_duration("45s") == 45
    assert lib.parse_duration("1d2h3m4s") == 93784


def test_parse_case_and_large_values(lib):
    assert lib.parse_duration("1H5S") == 3605
    assert lib.parse_duration("2D") == 172800
    assert lib.parse_duration("90m") == 5400
    assert lib.parse_duration("0m") == 0


def test_parse_rejects_order_and_repeats(lib):
    for text in ["30m1h", "1h1h", "1s1m", "1m1d"]:
        with pytest.raises(ValueError):
            lib.parse_duration(text)


def test_parse_rejects_malformed(lib):
    for text in ["", "1x", "-5m", "1.5h", "1 h", "h", "1hh", "   "]:
        with pytest.raises(ValueError):
            lib.parse_duration(text)


def test_format_basic(lib):
    assert lib.format_duration(5400) == "1h 30m"
    assert lib.format_duration(3605) == "1h 5s"
    assert lib.format_duration(59) == "59s"
    assert lib.format_duration(90061) == "1d 1h 1m 1s"


def test_format_zero_and_whole_units(lib):
    assert lib.format_duration(0) == "0s"
    assert lib.format_duration(86400) == "1d"
    assert lib.format_duration(3600) == "1h"
    assert lib.format_duration(172860) == "2d 1m"


def test_format_negative_raises(lib):
    with pytest.raises(ValueError):
        lib.format_duration(-1)


def test_total(lib):
    assert lib.total(["1h", "30m", "90"]) == 5490
    assert lib.total([]) == 0


def test_total_invalid_raises(lib):
    with pytest.raises(ValueError):
        lib.total(["1h", "oops"])


@pytest.mark.hard
def test_hard_full_width_and_nbsp(lib):
    h = "\N{FULLWIDTH LATIN SMALL LETTER H}"
    m = "\N{FULLWIDTH LATIN SMALL LETTER M}"
    assert lib.parse_duration(f"{FW['1']}{h}{FW['1']}0{m}") == 4200
    assert lib.parse_duration(f"{FW['2']}\N{FULLWIDTH LATIN CAPITAL LETTER H}") == 7200
    assert lib.parse_duration("\xa045m\xa0") == 2700


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
