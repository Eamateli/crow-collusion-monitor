def slugify(title: str, max_len: int = 40) -> str:
    raise NotImplementedError


def unique_slug(slug: str, taken: set[str]) -> str:
    raise NotImplementedError


def extract_title(markdown: str) -> str | None:
    raise NotImplementedError
