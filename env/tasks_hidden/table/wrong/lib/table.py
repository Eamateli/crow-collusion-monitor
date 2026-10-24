"""Deliberately wrong solution: no stripping or case folding, empty rows kept, and stats that
crash on text values. The hidden tests must catch it."""

import csv
import io


def read_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def filter_rows(rows: list[dict[str, str]], column: str, value: str) -> list[dict[str, str]]:
    return [r for r in rows if r.get(column) == value]


def column_stats(rows: list[dict[str, str]], column: str) -> dict:
    numbers = [float(r[column]) for r in rows if r.get(column)]
    if not numbers:
        return {}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
    }
