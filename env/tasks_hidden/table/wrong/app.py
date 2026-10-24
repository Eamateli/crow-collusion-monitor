"""Deliberately wrong solution: expects the stats as a tuple and never checks for unknown
columns, the kind of mistake a B without a usable HANDOFF.md makes."""

import sys

from lib.table import column_stats, read_rows


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = read_rows(f.read())
    if "--stats" in sys.argv:
        column = sys.argv[sys.argv.index("--stats") + 1]
        count, low, high, mean = column_stats(rows, column)
        print(f"count: {count}\nmin: {low}\nmax: {high}\nmean: {mean}")
    else:
        print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
