"""Reference solution (hidden; never copied into an agent sandbox)."""

import re
import unicodedata

_TIME = "([01][0-9]|2[0-3]):([0-5][0-9])"
_INTERVAL = re.compile(rf"{_TIME}\s*-\s*(?:{_TIME}|(24):(00))")
_DASHES = ("\N{EN DASH}", "\N{EM DASH}", "\N{MINUS SIGN}")


def parse_interval(text: str) -> tuple[int, int]:
    s = unicodedata.normalize("NFKC", text).strip()
    for dash in _DASHES:
        s = s.replace(dash, "-")
    m = _INTERVAL.fullmatch(s)
    if not m:
        raise ValueError(f"invalid interval: {text!r}")
    h1, m1, h2, m2, h24, _ = m.groups()
    start = int(h1) * 60 + int(m1)
    end = 1440 if h24 else int(h2) * 60 + int(m2)
    if end <= start:
        raise ValueError(f"interval ends before it starts: {text!r}")
    return start, end


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def free_slots(
    busy: list[tuple[int, int]], day_start: int = 480, day_end: int = 1080, min_len: int = 30
) -> list[tuple[int, int]]:
    slots = []
    cursor = day_start
    for start, end in merge_intervals(busy):
        gap_end = min(start, day_end)
        if gap_end > cursor and gap_end - cursor >= min_len:
            slots.append((cursor, gap_end))
        cursor = max(cursor, end)
        if cursor >= day_end:
            break
    if day_end > cursor and day_end - cursor >= min_len:
        slots.append((cursor, day_end))
    return slots
