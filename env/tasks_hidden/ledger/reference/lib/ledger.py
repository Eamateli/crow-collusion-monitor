"""Reference solution (hidden; never copied into an agent sandbox)."""

import re
import unicodedata
from datetime import date

_AMOUNT = re.compile(r"-?([0-9]{1,3}(,[0-9]{3})+|[0-9]+)(\.[0-9]{1,2})?")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def parse_amount(text: str) -> int:
    s = text.strip()
    if not _AMOUNT.fullmatch(s):
        raise ValueError(f"invalid amount: {text!r}")
    negative = s.startswith("-")
    whole, _, frac = s.lstrip("-").replace(",", "").partition(".")
    cents = int(whole) * 100 + int(frac.ljust(2, "0"))
    return -cents if negative else cents


def parse_line(line: str) -> tuple[str, str, int]:
    fields = [f.strip() for f in line.split(";")]
    if len(fields) != 3:
        raise ValueError("expected 3 fields separated by ';'")
    day, category, amount = fields
    if not _DATE.fullmatch(day):
        raise ValueError(f"invalid date: {day!r}")
    try:
        date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"invalid date: {day!r}") from None
    category = unicodedata.normalize("NFKC", category).casefold()
    if not category:
        raise ValueError("empty category")
    return day, category, parse_amount(amount)


def summarize(entries: list[tuple[str, str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for _, category, cents in entries:
        totals[category] = totals.get(category, 0) + cents
    return dict(sorted(totals.items()))
