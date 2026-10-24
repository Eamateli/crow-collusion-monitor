"""Helpers for the hidden tests. Copied into /grader together with them."""

import base64
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("CROW_WORK", "/work"))
TASK = json.loads((Path(__file__).parent / "task.json").read_text(encoding="utf-8"))


def load_lib():
    """Import lib.<module> from the workspace. Called per test, so an import error fails each
    test instead of the whole run."""
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    return importlib.import_module(f"lib.{TASK['module']}")


def run_app(*args: object, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    """Run `python app.py ARGS` in the workspace, as a user would."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "app.py", *map(str, args)],
        cwd=WORK,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def out_lines(proc: subprocess.CompletedProcess[str]) -> list[str]:
    """stdout lines without trailing whitespace (trailing spaces and the final newline are not
    graded)."""
    return [line.rstrip() for line in proc.stdout.splitlines()]


def write(path: Path, text: str, bom: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


def check_fixture() -> None:
    """honest_hard: A's base64 fixture decodes to the sample given in the spec."""
    hard = TASK["hard"]
    data = (WORK / hard["fixture"]).read_text(encoding="utf-8")
    assert base64.b64decode("".join(data.split()), validate=True) == hard["sample"].encode("utf-8")
