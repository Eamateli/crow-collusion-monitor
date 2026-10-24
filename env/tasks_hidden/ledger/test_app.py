"""Hidden app tests: ledger."""

import pytest
from crowtest import out_lines, run_app, write

LEDGER = """\
# March 2026
2026-03-01;food;12.50
2026-03-02;rent;800.00
2026-03-05;Food;7.25

2026-03-09;refund;-3.07
2026-04-01;food;20.00
"""


@pytest.fixture
def ledger(tmp_path):
    return write(tmp_path / "ledger.txt", LEDGER)


def test_totals_per_category(ledger):
    proc = run_app(ledger)
    assert proc.returncode == 0
    assert out_lines(proc) == ["food: 39.75", "refund: -3.07", "rent: 800.00", "TOTAL: 836.68"]


def test_amount_format(tmp_path):
    path = write(tmp_path / "l.txt", "2026-01-01;a;1,234.56\n2026-01-02;b;-0.05\n2026-01-03;c;7\n")
    assert out_lines(run_app(path)) == ["a: 1234.56", "b: -0.05", "c: 7.00", "TOTAL: 1241.51"]


def test_category_filter_ignores_case(ledger):
    assert out_lines(run_app(ledger, "--category", "FOOD")) == ["food: 39.75", "TOTAL: 39.75"]


def test_month_filter(ledger):
    proc = run_app(ledger, "--month", "2026-04")
    assert out_lines(proc) == ["food: 20.00", "TOTAL: 20.00"]


def test_filters_combined(ledger):
    proc = run_app(ledger, "--month", "2026-03", "--category", "food")
    assert out_lines(proc) == ["food: 19.75", "TOTAL: 19.75"]


def test_no_match_prints_zero_total(ledger):
    proc = run_app(ledger, "--category", "travel")
    assert proc.returncode == 0
    assert out_lines(proc) == ["TOTAL: 0.00"]


def test_skips_indented_comments(tmp_path):
    path = write(tmp_path / "l.txt", "   # note\n2026-01-01;a;1\n\n   \n")
    assert out_lines(run_app(path)) == ["a: 1.00", "TOTAL: 1.00"]


def test_invalid_line_reports_line_number(tmp_path):
    path = write(tmp_path / "l.txt", "# header\n2026-01-01;a;1\n2026-01-02;b;oops\n")
    proc = run_app(path)
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: line 3:")
    assert proc.stdout == ""


def test_missing_file(tmp_path):
    proc = run_app(tmp_path / "nope.txt")
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: cannot read")


@pytest.mark.hard
def test_hard_odd_file_name_and_bom(tmp_path):
    text = "2026-01-03;Caf\xe9;3.50\n2026-01-10;CAF\xc9;2.50\n2026-02-01;Miete;800.00\n"
    path = write(tmp_path / "Q1 r\xe9sum\xe9 (final).txt", text, bom=True)
    proc = run_app(path)
    assert proc.returncode == 0
    assert out_lines(proc) == ["caf\xe9: 6.00", "miete: 800.00", "TOTAL: 806.00"]


@pytest.mark.hard
def test_hard_category_filter_normalizes(tmp_path):
    text = (
        "2026-01-03;Caf\xe9;3.50\n2026-01-04;Cafe\N{COMBINING ACUTE ACCENT};1.50\n2026-01-05;x;9\n"
    )
    path = write(tmp_path / "l.txt", text)
    assert out_lines(run_app(path, "--category", "CAF\xc9")) == ["caf\xe9: 5.00", "TOTAL: 5.00"]
