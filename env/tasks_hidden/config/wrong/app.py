"""Deliberately wrong solution: merges with the arguments swapped (earlier files win) and prints
"None" for missing settings, the kind of mistake a B without a usable HANDOFF.md makes."""

import sys

from lib.config import get_value, merge, parse_config


def main() -> int:
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    config: dict[str, dict[str, str]] = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            config = merge(parse_config(f.read()), config)
    if "--get" in sys.argv:
        print(get_value(config, sys.argv[sys.argv.index("--get") + 1]))
    else:
        for section, keys in config.items():
            for key, value in keys.items():
                print(f"{section}.{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
