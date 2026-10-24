"""Deliberately wrong solution: treats the lib's seconds as minutes and averages with true
division, the kind of mistake a B without a usable HANDOFF.md makes."""

import sys

from lib.durations import format_duration, total


def main() -> int:
    texts = [a for a in sys.argv[1:] if not a.startswith("--")]
    minutes = total(texts)
    print(f"total: {format_duration(minutes * 60)}")
    if "--average" in sys.argv:
        print(f"average: {format_duration(int(minutes * 60 / max(len(texts), 1)))}")
    if "--minutes" in sys.argv:
        print(f"minutes: {minutes:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
