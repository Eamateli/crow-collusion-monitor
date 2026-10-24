"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import sys
import unicodedata

from lib.ledger import parse_line, summarize


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    parser.add_argument("--category")
    parser.add_argument("--month")
    args = parser.parse_args()
    try:
        with open(args.ledger, encoding="utf-8-sig") as f:
            lines = f.read().split("\n")
    except OSError:
        print(f"error: cannot read {args.ledger}", file=sys.stderr)
        return 2
    entries = []
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            entries.append(parse_line(line))
        except ValueError as e:
            print(f"error: line {n}: {e}", file=sys.stderr)
            return 2
    if args.category is not None:
        wanted = unicodedata.normalize("NFKC", args.category.strip()).casefold()
        entries = [e for e in entries if e[1] == wanted]
    if args.month is not None:
        entries = [e for e in entries if e[0][:7] == args.month]
    totals = summarize(entries)
    for category, cents in totals.items():
        print(f"{category}: {money(cents)}")
    print(f"TOTAL: {money(sum(totals.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
