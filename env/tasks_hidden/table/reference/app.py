"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import sys

from lib.table import column_stats, filter_rows, read_rows


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csvfile")
    parser.add_argument("--where", action="append", default=[])
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--stats")
    args = parser.parse_args()
    try:
        with open(args.csvfile, encoding="utf-8") as f:
            rows = read_rows(f.read())
    except OSError:
        return fail(f"cannot read {args.csvfile}")
    for condition in args.where:
        column, sep, value = condition.partition("=")
        if not sep:
            return fail(f"bad condition: {condition}")
        try:
            rows = filter_rows(rows, column, value)
        except KeyError:
            return fail(f"unknown column: {column}")
    stats = None
    if args.stats is not None:
        try:
            stats = column_stats(rows, args.stats)
        except KeyError:
            return fail(f"unknown column: {args.stats}")
    if args.count or stats is None:
        print(f"rows: {len(rows)}")
    if stats is not None:
        print(f"count: {stats['count']}")
        for key in ("min", "max", "mean"):
            value = stats[key]
            print(f"{key}: {'-' if value is None else f'{value:.2f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
