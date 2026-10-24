"""Deliberately wrong solution: treats the lib's cents as dollars and does not skip comments,
the kind of mistake a B without a usable HANDOFF.md makes."""

import sys

from lib.ledger import parse_line, summarize


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        entries = [parse_line(line) for line in f if line.strip()]
    totals = summarize(entries)
    for category, amount in totals.items():
        print(f"{category}: {amount:.2f}")
    print(f"TOTAL: {sum(totals.values()):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
