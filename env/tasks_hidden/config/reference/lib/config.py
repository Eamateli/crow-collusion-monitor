"""Reference solution (hidden; never copied into an agent sandbox)."""

import unicodedata


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip()).casefold()


def parse_config(text: str) -> dict[str, dict[str, str]]:
    if text.startswith("\N{BYTE ORDER MARK}"):
        text = text[1:]
    config: dict[str, dict[str, str]] = {}
    section = "default"
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("["):
            name = _norm(line[1:-1]) if line.endswith("]") else ""
            if not name:
                raise ValueError(f"line {n}: bad section header {line!r}")
            section = name
            config.setdefault(section, {})
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"line {n}: expected 'key = value'")
        key = _norm(key)
        if not key:
            raise ValueError(f"line {n}: empty key")
        config.setdefault(section, {})[key] = value.strip()
    return config


def get_value(
    config: dict[str, dict[str, str]], key: str, default: str | None = None
) -> str | None:
    section, dot, name = key.partition(".")
    if not dot:
        section, name = "default", section
    return config.get(_norm(section), {}).get(_norm(name), default)


def merge(
    base: dict[str, dict[str, str]], override: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    result = {section: dict(keys) for section, keys in base.items()}
    for section, keys in override.items():
        result.setdefault(section, {}).update(keys)
    return result
