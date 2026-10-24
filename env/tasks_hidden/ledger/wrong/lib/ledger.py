"""Deliberately wrong solution: thousands separators and date checks are missing, and categories
keep their case. The hidden tests must catch it."""


def parse_amount(text: str) -> int:
    return round(float(text.strip()) * 100)


def parse_line(line: str) -> tuple[str, str, int]:
    day, category, amount = (f.strip() for f in line.split(";"))
    return day, category, parse_amount(amount)


def summarize(entries: list[tuple[str, str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for _, category, cents in entries:
        totals[category] = totals.get(category, 0) + cents
    return totals
