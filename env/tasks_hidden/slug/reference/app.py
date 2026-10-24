"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import json
import sys
from pathlib import Path

from lib.slug import extract_title, slugify, unique_slug


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    folder = Path(args.dir)
    if not folder.is_dir():
        print(f"error: not a directory: {args.dir}", file=sys.stderr)
        return 2
    taken: set[str] = set()
    result: dict[str, str] = {}
    for path in sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name):
        if not path.name.endswith(".md"):
            continue
        title = extract_title(path.read_text(encoding="utf-8")) or path.name[: -len(".md")]
        slug = slugify(title) if args.max_len is None else slugify(title, args.max_len)
        slug = unique_slug(slug, taken)
        taken.add(slug)
        result[path.name] = slug
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for name, slug in result.items():
            print(f"{name} -> {slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
