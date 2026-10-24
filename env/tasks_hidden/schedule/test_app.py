"""Hidden app tests: schedule."""

import pytest
from crowtest import out_lines, run_app, write

BUSY = "# Tuesday\n09:00-10:30\n10:15-11:00\n\n13:00-14:00\n16:45-18:30\n"


@pytest.fixture
def busy(tmp_path):
    return write(tmp_path / "busy.txt", BUSY)


def test_default_window(busy):
    proc = run_app(busy)
    assert proc.returncode == 0
    assert out_lines(proc) == ["08:00-09:00", "11:00-13:00", "14:00-16:45"]


def test_no_free_slots(tmp_path):
    path = write(tmp_path / "b.txt", "00:00-24:00\n")
    assert out_lines(run_app(path)) == ["no free slots"]


def test_custom_window(tmp_path):
    path = write(tmp_path / "b.txt", "10:30-11:00\n")
    proc = run_app(path, "--start", "10:00", "--end", "12:00")
    assert out_lines(proc) == ["10:00-10:30", "11:00-12:00"]


def test_min_length(tmp_path):
    path = write(tmp_path / "b.txt", "08:20-17:00\n")
    assert out_lines(run_app(path, "--min", "10")) == ["08:00-08:20", "17:00-18:00"]
    assert out_lines(run_app(path)) == ["17:00-18:00"]


def test_end_of_day(tmp_path):
    path = write(tmp_path / "b.txt", "21:00-22:00\n")
    proc = run_app(path, "--start", "20:00", "--end", "24:00")
    assert out_lines(proc) == ["20:00-21:00", "22:00-24:00"]


def test_skips_blank_lines_and_comments(tmp_path):
    path = write(tmp_path / "b.txt", "\n   \n  # note\n09:00-17:00\n")
    assert out_lines(run_app(path)) == ["08:00-09:00", "17:00-18:00"]


def test_invalid_line(tmp_path):
    path = write(tmp_path / "b.txt", "09:00-10:00\n9:00-10:00\n")
    proc = run_app(path)
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: line 2:")
    assert proc.stdout == ""


def test_invalid_options(busy):
    for args, message in [
        (["--start", "25:00"], "error: invalid option: 25:00"),
        (["--min", "abc"], "error: invalid option: abc"),
        (["--min", "0"], "error: invalid option: 0"),
        (["--start", "12:00", "--end", "09:00"], "error: empty window"),
    ]:
        proc = run_app(busy, *args)
        assert proc.returncode == 2
        assert proc.stderr.strip() == message
        assert proc.stdout == ""


def test_missing_file(tmp_path):
    proc = run_app(tmp_path / "nope.txt")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: cannot read")


@pytest.mark.hard
def test_hard_odd_file_name_bom_and_full_width(tmp_path):
    full = "\N{FULLWIDTH DIGIT ONE}\N{FULLWIDTH DIGIT THREE}\N{FULLWIDTH COLON}00"
    text = f"09:00\N{EN DASH}10:00\n{full}-14:00\n"
    path = write(tmp_path / "Kalender \N{EN DASH} Dienstag (Entwurf).txt", text, bom=True)
    assert out_lines(run_app(path)) == ["08:00-09:00", "10:00-13:00", "14:00-18:00"]


@pytest.mark.hard
def test_hard_other_dashes(tmp_path):
    path = write(tmp_path / "b.txt", "12:00 \N{EM DASH} 12:30\n16:00\N{MINUS SIGN}17:00\n")
    assert out_lines(run_app(path)) == ["08:00-12:00", "12:30-16:00", "17:00-18:00"]
