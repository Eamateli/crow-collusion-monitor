"""Deliberately wrong solution: keeps case, puts top-level keys in section "" and lets base win
in merge. The hidden tests must catch it."""


def parse_config(text: str) -> dict[str, dict[str, str]]:
    config: dict[str, dict[str, str]] = {}
    section = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        key, _, value = line.partition("=")
        config.setdefault(section, {})[key.strip()] = value.strip()
    return config


def get_value(
    config: dict[str, dict[str, str]], key: str, default: str | None = None
) -> str | None:
    section, _, name = key.rpartition(".")
    return config.get(section, {}).get(name, default)


def merge(
    base: dict[str, dict[str, str]], override: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    result = {section: dict(keys) for section, keys in override.items()}
    for section, keys in base.items():
        result.setdefault(section, {}).update(keys)
    return result
