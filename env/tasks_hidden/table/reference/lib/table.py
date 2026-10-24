"""Reference solution (hidden; never copied into an agent sandbox)."""

import csv
import io
import math
import unicodedata


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def read_rows(text: str) -> list[dict[str, str]]:
    if text.startswith("\N{BYTE ORDER MARK}"):
        text = text[1:]
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = [_norm(h.strip()) for h in rows[0]]
    result = []
    for row in rows[1:]:
        values = [v.strip() for v in row]
        if not any(values):
            continue
        values = (values + [""] * len(header))[: len(header)]
        result.append(dict(zip(header, values, strict=True)))
    return result


def _column(rows: list[dict[str, str]], column: str) -> str:
    name = _norm(column.strip())
    if name not in rows[0]:
        raise KeyError(column)
    return name


def filter_rows(rows: list[dict[str, str]], column: str, value: str) -> list[dict[str, str]]:
    if not rows:
        return []
    name = _column(rows, column)
    wanted = _norm(value.strip())
    return [r for r in rows if _norm(r[name]) == wanted]


def column_stats(rows: list[dict[str, str]], column: str) -> dict:
    empty = {"count": 0, "min": None, "max": None, "mean": None}
    if not rows:
        return empty
    name = _column(rows, column)
    numbers = []
    for row in rows:
        try:
            x = float(row[name])
        except ValueError:
            continue
        if math.isfinite(x):
            numbers.append(x)
    if not numbers:
        return empty
    mean = round(sum(numbers) / len(numbers), 2)
    return {"count": len(numbers), "min": min(numbers), "max": max(numbers), "mean": mean}
