"""Task manifests: which workspace paths a task allows.

A manifest is a list of glob patterns over relative POSIX paths. Patterns match segment by
segment (`*` never crosses a `/`), so `tests/test_*.py` allows `tests/test_app.py` but not
`tests/sub/test_app.py`. Build artifacts (`__pycache__`, `.pytest_cache`) are kept apart: they
are not task content, and callers decide what to do with them.
"""

from fnmatch import fnmatchcase

ARTIFACT_DIRS = ("__pycache__", ".pytest_cache")


def matches(path: str, pattern: str) -> bool:
    parts, pat = path.strip("/").split("/"), pattern.strip("/").split("/")
    return len(parts) == len(pat) and all(
        fnmatchcase(p, q) for p, q in zip(parts, pat, strict=True)
    )


def allowed(path: str, manifest: list[str]) -> bool:
    """A file path the manifest allows."""
    return any(matches(path, pattern) for pattern in manifest)


def is_artifact(path: str) -> bool:
    return any(part in ARTIFACT_DIRS for part in path.strip("/").split("/"))
