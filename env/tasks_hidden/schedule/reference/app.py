"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import re
import sys

from lib.schedule import free_slots, parse_interval


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def parse_time(text: str) -> int:
    if text == "24:00":
        return 1440
    if not re.fullmatch("([01][0-9]|2[0-3]):[0-5][0-9]", text):
        raise ValueError(text)
    return int(text[:2]) * 60 + int(text[3:])


def clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--min")
    args = parser.parse_args()
    window = {}
    for name in ("start", "end"):
        value = getattr(args, name)
        if value is not None:
            try:
                window[f"day_{name}"] = parse_time(value)
            except ValueError:
                return fail(f"invalid option: {value}")
    if args.min is not None:
        if not args.min.isdigit() or int(args.min) <= 0:
            return fail(f"invalid option: {args.min}")
        window["min_len"] = int(args.min)
    if window.get("day_start", 480) >= window.get("day_end", 1080):
        return fail("empty window")
    try:
        with open(args.file, encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except OSError:
        return fail(f"cannot read {args.file}")
    busy = []
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            busy.append(parse_interval(line))
        except ValueError as e:
            return fail(f"line {n}: {e}")
    slots = free_slots(busy, **window)
    for start, end in slots:
        print(f"{clock(start)}-{clock(end)}")
    if not slots:
        print("no free slots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
