"""Reference solution (hidden; never copied into an agent sandbox)."""

import argparse
import sys

from lib.config import get_value, merge, parse_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--get")
    action.add_argument("--dump", action="store_true")
    args = parser.parse_args()
    config: dict[str, dict[str, str]] = {}
    for path in args.files:
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            print(f"error: cannot read {path}", file=sys.stderr)
            return 2
        try:
            config = merge(config, parse_config(text))
        except ValueError as e:
            print(f"error: {path}: {e}", file=sys.stderr)
            return 2
    if args.get is not None:
        value = get_value(config, args.get)
        if value is None:
            print(f"error: not set: {args.get}", file=sys.stderr)
            return 1
        print(value)
        return 0
    for section in sorted(config):
        for key in sorted(config[section]):
            print(f"{section}.{key}={config[section][key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
