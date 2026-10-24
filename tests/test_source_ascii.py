"""Our own Python source is ASCII: invisible or unusual characters are written as escapes.

Task fixtures under env/tasks*/ are excluded (honest_hard legitimately contains unicode).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["crow", "env", "experiments", "analysis", "scripts", "tests"]


def test_python_source_is_ascii() -> None:
    offenders = []
    for d in DIRS:
        for path in (ROOT / d).rglob("*.py"):
            rel = path.relative_to(ROOT)
            if rel.parts[:2] in (("env", "tasks"), ("env", "tasks_hidden")):
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.isascii():
                    offenders.append(f"{rel}:{n}")
    assert not offenders, offenders
