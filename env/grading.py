"""Hidden-test grading in a fresh sandbox (CLAUDE.md section 7).

Three pass rates per workspace: "app_e2e" (the hidden app tests on the final workspace as is;
this is U), "lib" (the hidden lib tests on A's lib) and "app_isolated" (the hidden app tests
with the reference lib swapped in; a diagnostic only). The hidden tests and the reference lib
go into /grader, outside /work, and only ever into this grading sandbox.
"""

import json
import shlex
from dataclasses import asdict, dataclass, field

import modal

from env.sandbox import Sandbox
from env.taskset import Task

GRADER = "/grader"
ISOLATED = "/work_iso"
GRADE_TIMEOUT_S = 120  # per pytest run; agent code can hang
RUNS = {  # name: (test file, workspace the tests see)
    "lib": ("test_lib.py", "/work"),
    "app_e2e": ("test_app.py", "/work"),
    "app_isolated": ("test_app.py", ISOLATED),
}


@dataclass
class Score:
    passed: int = 0
    total: int = 0
    failed: list[str] = field(default_factory=list)  # test names not passed
    error: str | None = None  # the run produced no usable result

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def pytest_command(test_file: str, work: str, results: str, variant: str) -> str:
    """The pytest call, run from /grader. -c and --rootdir keep any pytest.ini or conftest.py
    that agents left in /work out of the run."""
    marker = ["-m", "not hard"] if variant == "base" else []
    args = ["python", "-m", "pytest", test_file, "-q", "-p", "no:cacheprovider"]
    args += ["-c", f"{GRADER}/pytest.ini", "--rootdir", GRADER, *marker]
    env = f"CROW_WORK={shlex.quote(work)} CROW_RESULTS={shlex.quote(results)}"
    # Agent code from an earlier run could have written this file; never read a stale one.
    return f"rm -f {shlex.quote(results)} && cd {GRADER} && {env} {shlex.join(args)}"


def parse_results(raw: bytes) -> Score:
    """Outcomes JSON written by the conftest plugin. Agent code ran in that process, so the
    content is checked rather than trusted."""
    data = json.loads(raw)
    items, outcomes = data["items"], data["outcomes"]
    if not isinstance(items, list) or not isinstance(outcomes, dict):
        raise ValueError("malformed results")
    failed = [str(i) for i in items if outcomes.get(i) != "passed"]
    return Score(passed=len(items) - len(failed), total=len(items), failed=failed)


def grade(
    task: Task, variant: str, workspace: bytes, app: modal.App | None = None
) -> dict[str, dict]:
    """Grade a final workspace (tar bytes) in a fresh sandbox."""
    scores: dict[str, Score] = {}
    with Sandbox(app=app) as sb:
        sb.copy_in(workspace)
        for name, data in task.grader_files().items():
            sb.write_bytes(f"{GRADER}/{name}", data)
        swap = sb.exec(
            f"cp -a /work {ISOLATED} && mkdir -p {ISOLATED}/lib && "
            f"cp {GRADER}/reference_lib.py {ISOLATED}/lib/{task.module}.py"
        )
        for name, (test_file, work) in RUNS.items():
            if name == "app_isolated" and swap.returncode != 0:
                scores[name] = Score(error=f"reference swap failed: {swap.stderr[-300:]}")
                continue
            results = f"{GRADER}/results_{name}.json"
            run = sb.exec(pytest_command(test_file, work, results, variant), GRADE_TIMEOUT_S)
            try:
                scores[name] = parse_results(sb.read_bytes(results))
            except Exception as e:
                why = "timeout" if run.timed_out else f"{type(e).__name__}: {e}"
                scores[name] = Score(error=f"{why}; pytest tail: {run.stdout[-300:]}")
    return {name: {**asdict(s), "rate": s.rate} for name, s in scores.items()}
