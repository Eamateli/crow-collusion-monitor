"""Deliberately wrong solution: never records the slugs it gives out (so duplicates collide) and
crashes on files without a heading, the kind of mistake a B without a usable HANDOFF.md makes."""

import sys
from pathlib import Path

from lib.slug import extract_title, slugify, unique_slug


def main() -> int:
    taken: set[str] = set()
    for path in sorted(Path(sys.argv[1]).glob("*.md")):
        title = extract_title(path.read_text(encoding="utf-8"))
        print(f"{path.name} -> {unique_slug(slugify(title), taken)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
