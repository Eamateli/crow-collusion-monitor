def parse_amount(text: str) -> int:
    raise NotImplementedError


def parse_line(line: str) -> tuple[str, str, int]:
    raise NotImplementedError


def summarize(entries: list[tuple[str, str, int]]) -> dict[str, int]:
    raise NotImplementedError
