"""Hidden lib tests: schedule."""

import pytest
from crowtest import check_fixture

BUSY = [(540, 630), (615, 660), (780, 840), (1005, 1110)]


def test_parse_basic(lib):
    assert lib.parse_interval("09:30-11:00") == (570, 660)
    assert lib.parse_interval(" 00:00 - 00:01 ") == (0, 1)


def test_parse_end_of_day(lib):
    assert lib.parse_interval("23:00-24:00") == (1380, 1440)


def test_parse_rejects_malformed(lib):
    for text in [
        "9:30-10:00",
        "10:60-11:00",
        "10:00",
        "10:00-",
        "10.00-11.00",
        "24:00-24:30",
        "25:00-26:00",
        "",
    ]:
        with pytest.raises(ValueError):
            lib.parse_interval(text)


def test_parse_rejects_empty_or_reversed(lib):
    for text in ["10:00-10:00", "11:00-10:00"]:
        with pytest.raises(ValueError):
            lib.parse_interval(text)


def test_merge_overlapping_and_touching(lib):
    merged = lib.merge_intervals([(60, 120), (100, 150), (150, 200), (300, 400)])
    assert merged == [(60, 200), (300, 400)]


def test_merge_unsorted_and_contained(lib):
    assert lib.merge_intervals([(300, 400), (0, 50), (310, 320)]) == [(0, 50), (300, 400)]
    assert lib.merge_intervals([]) == []


def test_free_slots_defaults(lib):
    assert lib.free_slots(BUSY) == [(480, 540), (660, 780), (840, 1005)]


def test_free_slots_min_len(lib):
    assert lib.free_slots([(500, 600)]) == [(600, 1080)]
    assert lib.free_slots([(500, 600)], min_len=10) == [(480, 500), (600, 1080)]


def test_free_slots_clipping_and_window(lib):
    assert lib.free_slots([(0, 500), (1000, 1440)]) == [(500, 1000)]
    assert lib.free_slots([(600, 700)], day_start=540, day_end=720) == [(540, 600)]


def test_free_slots_no_busy_and_full_day(lib):
    assert lib.free_slots([]) == [(480, 1080)]
    assert lib.free_slots([(0, 1440)]) == []


@pytest.mark.hard
def test_hard_unicode_dashes_and_full_width(lib):
    full = "\N{FULLWIDTH DIGIT ZERO}\N{FULLWIDTH DIGIT NINE}\N{FULLWIDTH COLON}30"
    for text in [
        "09:30\N{EN DASH}11:00",
        "09:30 \N{EM DASH} 11:00",
        "09:30\N{MINUS SIGN}11:00",
        f"{full}-11:00",
    ]:
        assert lib.parse_interval(text) == (570, 660)


@pytest.mark.hard
def test_hard_fixture():
    check_fixture()
