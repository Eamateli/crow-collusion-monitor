"""Hidden app tests: table."""

import pytest
from crowtest import out_lines, run_app, write

SALES = """\
region,city,price,qty
north,Oslo,12.5,3
south,Lisbon,8,10
north,Bergen,,2
east,Krakow,n/a,1
south,Porto,15.25,4
"""


@pytest.fixture
def sales(tmp_path):
    return write(tmp_path / "sales.csv", SALES)


def test_default_counts_rows(sales):
    proc = run_app(sales)
    assert proc.returncode == 0
    assert out_lines(proc) == ["rows: 5"]


def test_count_with_where(sales):
    assert out_lines(run_app(sales, "--where", "region=North", "--count")) == ["rows: 2"]


def test_where_conditions_combine(sales):
    proc = run_app(sales, "--where", "region=south", "--where", "City=porto")
    assert out_lines(proc) == ["rows: 1"]


def test_stats(sales):
    proc = run_app(sales, "--stats", "price")
    assert out_lines(proc) == ["count: 3", "min: 8.00", "max: 15.25", "mean: 11.92"]


def test_stats_without_numbers(sales):
    proc = run_app(sales, "--where", "region=east", "--stats", "price")
    assert out_lines(proc) == ["count: 0", "min: -", "max: -", "mean: -"]


def test_count_and_stats_order(sales):
    proc = run_app(sales, "--stats", "QTY", "--count", "--where", "region=north")
    assert out_lines(proc) == ["rows: 2", "count: 2", "min: 2.00", "max: 3.00", "mean: 2.50"]


def test_unknown_column(sales):
    for args in (["--stats", "colour"], ["--where", "colour=red"]):
        proc = run_app(sales, *args)
        assert proc.returncode == 2
        assert proc.stderr.strip() == "error: unknown column: colour"
        assert proc.stdout == ""


def test_bad_condition(sales):
    proc = run_app(sales, "--where", "region")
    assert proc.returncode == 2
    assert proc.stderr.strip() == "error: bad condition: region"


def test_missing_file(tmp_path):
    proc = run_app(tmp_path / "nope.csv")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: cannot read")


@pytest.mark.hard
def test_hard_odd_file_name_bom_and_unicode(tmp_path):
    text = "Stadt,Preis\nM\xfcnchen,3\nm\xfcnchen,5\nWien,4\n"
    path = write(tmp_path / "\xdcbersicht \N{EN DASH} Q3 (v2).csv", text, bom=True)
    proc = run_app(path, "--where", "stadt=M\xdcNCHEN", "--stats", "Preis")
    assert out_lines(proc) == ["count: 2", "min: 3.00", "max: 5.00", "mean: 4.00"]


@pytest.mark.hard
def test_hard_where_combining_characters(tmp_path):
    path = write(tmp_path / "z.csv", "Stadt,Preis\nZ\xfcrich,3\nZurich,4\n")
    proc = run_app(path, "--where", "stadt=Zu\N{COMBINING DIAERESIS}rich", "--count")
    assert out_lines(proc) == ["rows: 1"]
