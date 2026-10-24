"""Deliberately wrong solution: passes HH:MM options to the lib as hours and prints the lib's
minutes as if they were HHMM, the kind of mistake a B without a usable HANDOFF.md makes."""

import sys

from lib.schedule import free_slots, parse_interval


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        busy = [parse_interval(line) for line in f if line.strip() and not line.startswith("#")]
    for start, end in free_slots(busy, 8, 18):
        print(f"{start // 100:02d}:{start % 100:02d}-{end // 100:02d}:{end % 100:02d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
