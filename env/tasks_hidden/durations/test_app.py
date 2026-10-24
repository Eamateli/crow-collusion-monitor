"""Hidden app tests: durations."""

import pytest
from crowtest import out_lines, run_app, write


def test_total():
    proc = run_app("1h30m", "45m")
    assert proc.returncode == 0
    assert out_lines(proc) == ["total: 2h 15m"]


def test_plain_seconds():
    assert out_lines(run_app("90", "30")) == ["total: 2m"]


def test_average_rounds_down():
    assert out_lines(run_app("1h", "1s", "--average")) == ["total: 1h 1s", "average: 30m"]


def test_minutes_two_decimals():
    assert out_lines(run_app("1h30m", "30s", "--minutes")) == [
        "total: 1h 30m 30s",
        "minutes: 90.50",
    ]
    assert out_lines(run_app("100", "--minutes")) == ["total: 1m 40s", "minutes: 1.67"]


def test_line_order_with_both_flags():
    proc = run_app("--minutes", "--average", "10m", "20m")
    assert out_lines(proc) == ["total: 30m", "average: 15m", "minutes: 30.00"]


def test_from_file(tmp_path):
    path = write(tmp_path / "t.txt", "1h\n\n30m\n")
    assert out_lines(run_app("15m", "--from", path)) == ["total: 1h 45m"]


def test_invalid_duration():
    proc = run_app("1h", "1x")
    assert proc.returncode == 2
    assert proc.stderr.strip() == "error: invalid duration: 1x"
    assert proc.stdout == ""


def test_no_durations():
    proc = run_app()
    assert proc.returncode == 2
    assert proc.stderr.strip() == "error: no durations"


def test_missing_file(tmp_path):
    proc = run_app("--from", tmp_path / "nope.txt")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: cannot read")


@pytest.mark.hard
def test_hard_full_width_arguments():
    one = "\N{FULLWIDTH DIGIT ONE}\N{FULLWIDTH LATIN SMALL LETTER H}"
    half = "\N{FULLWIDTH DIGIT THREE}\N{FULLWIDTH DIGIT ZERO}\N{FULLWIDTH LATIN SMALL LETTER M}"
    assert out_lines(run_app(one, half)) == ["total: 1h 30m"]


@pytest.mark.hard
def test_hard_odd_file_name_and_bom(tmp_path):
    text = "\N{FULLWIDTH DIGIT TWO}\N{FULLWIDTH LATIN CAPITAL LETTER H}\n\xa045m\xa0\n\n"
    path = write(tmp_path / "Wochen\xfcbersicht \N{EN DASH} KW 3.txt", text, bom=True)
    proc = run_app("--from", path, "--minutes")
    assert out_lines(proc) == ["total: 2h 45m", "minutes: 165.00"]
