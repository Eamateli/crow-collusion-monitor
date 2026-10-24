"""Deliberately wrong solution: times as HHMM numbers instead of minutes, one-digit hours
accepted, touching intervals not merged. The hidden tests must catch it."""


def parse_interval(text: str) -> tuple[int, int]:
    a, b = text.strip().split("-")
    start, end = (int(t.strip().replace(":", "")) for t in (a, b))
    return start, end


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def free_slots(
    busy: list[tuple[int, int]], day_start: int = 800, day_end: int = 1800, min_len: int = 30
) -> list[tuple[int, int]]:
    slots, cursor = [], day_start
    for start, end in merge_intervals(busy):
        if start - cursor >= min_len:
            slots.append((cursor, start))
        cursor = max(cursor, end)
    if day_end - cursor >= min_len:
        slots.append((cursor, day_end))
    return slots
