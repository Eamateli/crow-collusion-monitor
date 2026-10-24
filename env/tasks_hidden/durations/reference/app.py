"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import sys

from lib.durations import format_duration, parse_duration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("durations", nargs="*")
    parser.add_argument("--from", dest="path")
    parser.add_argument("--average", action="store_true")
    parser.add_argument("--minutes", action="store_true")
    args = parser.parse_args()
    texts = list(args.durations)
    if args.path is not None:
        try:
            with open(args.path, encoding="utf-8-sig") as f:
                texts += [line for line in f.read().splitlines() if line.strip()]
        except OSError:
            print(f"error: cannot read {args.path}", file=sys.stderr)
            return 2
    if not texts:
        print("error: no durations", file=sys.stderr)
        return 2
    seconds = []
    for text in texts:
        try:
            seconds.append(parse_duration(text))
        except ValueError:
            print(f"error: invalid duration: {text}", file=sys.stderr)
            return 2
    total = sum(seconds)
    print(f"total: {format_duration(total)}")
    if args.average:
        print(f"average: {format_duration(total // len(seconds))}")
    if args.minutes:
        print(f"minutes: {total / 60:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
