def read_rows(text: str) -> list[dict[str, str]]:
    raise NotImplementedError


def filter_rows(rows: list[dict[str, str]], column: str, value: str) -> list[dict[str, str]]:
    raise NotImplementedError


def column_stats(rows: list[dict[str, str]], column: str) -> dict:
    raise NotImplementedError
