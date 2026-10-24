"""Deliberately wrong solution: keeps non-ASCII letters, cuts in the middle of words, numbers
duplicates from 1 and finds headings inside code fences. The hidden tests must catch it."""

import re


def slugify(title: str, max_len: int = 40) -> str:
    slug = re.sub(r"\W+", "-", title.lower()).strip("-")
    return slug[:max_len] or "untitled"


def unique_slug(slug: str, taken: set[str]) -> str:
    n = 1
    while f"{slug}-{n}" in taken:
        n += 1
    return slug if slug not in taken else f"{slug}-{n}"


def extract_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None
