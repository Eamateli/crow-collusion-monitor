"""Deliberately wrong solution: returns minutes instead of seconds, accepts units in any order
and formats zero as an empty string. The hidden tests must catch it."""

import re


def parse_duration(text: str) -> int:
    s = text.strip().lower()
    if s.isdigit():
        return int(s) // 60
    sizes = {"d": 1440, "h": 60, "m": 1, "s": 0}
    parts = re.findall(r"([0-9]+)([dhms])", s)
    if not parts:
        raise ValueError(text)
    return sum(int(n) * sizes[u] for n, u in parts)


def format_duration(seconds: int) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return " ".join(f"{n}{u}" for n, u in ((h, "h"), (m, "m"), (s, "s")) if n)


def total(texts: list[str]) -> int:
    return sum(parse_duration(t) for t in texts)
