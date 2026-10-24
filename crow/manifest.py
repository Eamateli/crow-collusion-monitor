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


def allowed_dir(path: str, manifest: list[str]) -> bool:
    """A directory the manifest needs: some pattern has a file below it."""
    parts = path.strip("/").split("/")
    for pattern in manifest:
        pat = pattern.strip("/").split("/")
        if len(pat) > len(parts) and all(
            fnmatchcase(p, q) for p, q in zip(parts, pat[: len(parts)], strict=True)
        ):
            return True
    return False


def fixed_paths(manifest: list[str]) -> set[str]:
    """Paths whose names the task fixes: exact entries, their parent directories, and the
    literal directories in front of glob patterns (`tests` in `tests/test_*.py`)."""
    fixed = set()
    for pattern in manifest:
        parts = pattern.strip("/").split("/")
        literal = []
        for part in parts:
            if any(ch in part for ch in "*?["):
                break
            literal.append(part)
        for i in range(1, len(literal) + 1):
            fixed.add("/".join(literal[:i]))
    return fixed


def is_artifact(path: str) -> bool:
    return any(part in ARTIFACT_DIRS for part in path.strip("/").split("/"))
