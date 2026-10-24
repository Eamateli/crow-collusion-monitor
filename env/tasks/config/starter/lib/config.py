def parse_config(text: str) -> dict[str, dict[str, str]]:
    raise NotImplementedError


def get_value(
    config: dict[str, dict[str, str]], key: str, default: str | None = None
) -> str | None:
    raise NotImplementedError


def merge(
    base: dict[str, dict[str, str]], override: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    raise NotImplementedError
