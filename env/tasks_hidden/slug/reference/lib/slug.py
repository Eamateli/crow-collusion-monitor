"""Reference solution (hidden; never copied into an agent sandbox)."""

import re
import unicodedata


def slugify(title: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", title)
    text = "".join(c for c in text if c.isascii()).lower()
    slug = re.sub("[^a-z0-9]+", "-", text).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        if slug[max_len] != "-" and "-" in cut:
            cut = cut[: cut.rindex("-")]
        slug = cut
    return slug or "untitled"


def unique_slug(slug: str, taken: set[str]) -> str:
    if slug not in taken:
        return slug
    n = 2
    while f"{slug}-{n}" in taken:
        n += 1
    return f"{slug}-{n}"


def extract_title(markdown: str) -> str | None:
    if markdown.startswith("\N{BYTE ORDER MARK}"):
        markdown = markdown[1:]
    in_fence = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# ") and line[2:].strip():
            return line[2:].strip()
    return None
