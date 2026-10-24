"""All Python source in the repo is ASCII: invisible or unusual characters are written as
escapes (\\N{...} or \\x..), so they can be seen in review. This includes the task stubs, the
hidden tests and the reference solutions; task data files (.txt, .csv, .ini, .md) may hold
unicode on purpose.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["crow", "env", "experiments", "analysis", "scripts", "tests"]


def test_python_source_is_ascii() -> None:
    offenders = []
    for d in DIRS:
        for path in (ROOT / d).rglob("*.py"):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.isascii():
                    offenders.append(f"{path.relative_to(ROOT)}:{n}")
    assert not offenders, offenders
