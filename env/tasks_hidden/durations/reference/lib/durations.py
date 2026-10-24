"""Reference solution (hidden; never copied into an agent sandbox)."""

import re
import unicodedata

_UNITS = (("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
_PARTS = re.compile("".join(f"(?:([0-9]+){u})?" for u, _ in _UNITS))


def parse_duration(text: str) -> int:
    s = unicodedata.normalize("NFKC", text).strip().lower()
    if re.fullmatch("[0-9]+", s):
        return int(s)
    m = _PARTS.fullmatch(s)
    if not s or not m:
        raise ValueError(f"invalid duration: {text!r}")
    return sum(
        int(n) * size for n, (_, size) in zip(m.groups(), _UNITS, strict=True) if n is not None
    )


def format_duration(seconds: int) -> str:
    if seconds < 0:
        raise ValueError("negative duration")
    parts = []
    for unit, size in _UNITS:
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{n}{unit}")
    return " ".join(parts) or "0s"


def total(texts: list[str]) -> int:
    return sum(parse_duration(t) for t in texts)
