"""Checks the tasks (Step 4): every reference solution passes its hidden tests, and the starter
alone and a deliberately wrong solution fail them.

Grades each task x variant x solution with env/grading.py in a fresh Modal sandbox, exactly as
an episode will. --local runs the same pytest commands on this machine instead (for fast
iteration; only ever on our own reference, starter and wrong files, never on agent output).

Run: python scripts/check_tasks.py [--local] [--task ID]
"""

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
from rich.console import Console
from rich.table import Table

from env.grading import RUNS, parse_results
from env.taskset import Task, load_task, task_ids

APP_NAME = "crow-dev"
SOLUTIONS = ("reference", "starter", "wrong")
console = Console()


def workspace_files(task: Task, variant: str, solution: str) -> dict[str, bytes]:
    files = task.starter_files(variant)
    if solution == "reference":
        files |= task.reference_files(variant)
    elif solution == "wrong":
        files |= task.wrong_files()
    return files


def grade_local(task: Task, variant: str, files: dict[str, bytes]) -> dict[str, dict]:
    """Same pytest calls as env/grading.py, on this machine, with temp dirs for /work etc."""
    scores = {}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        work, iso, grader = root / "work", root / "work_iso", root / "grader"
        for base, content in ((work, files), (grader, task.grader_files())):
            for rel, data in content.items():
                (base / rel).parent.mkdir(parents=True, exist_ok=True)
                (base / rel).write_bytes(data)
        iso_files = files | {f"lib/{task.module}.py": task.grader_files()["reference_lib.py"]}
        for rel, data in iso_files.items():
            (iso / rel).parent.mkdir(parents=True, exist_ok=True)
            (iso / rel).write_bytes(data)
        for name, (test_file, sandbox_work) in RUNS.items():
            results = grader / f"results_{name}.json"
            marker = ["-m", "not hard"] if variant == "base" else []
            cmd = [sys.executable, "-m", "pytest", test_file, "-q", "-p", "no:cacheprovider"]
            cmd += ["-c", str(grader / "pytest.ini"), "--rootdir", str(grader), *marker]
            target = iso if sandbox_work.endswith("_iso") else work
            env = {"CROW_WORK": str(target), "CROW_RESULTS": str(results), "PATH": "/usr/bin"}
            proc = subprocess.run(cmd, cwd=grader, env=env, capture_output=True, text=True)
            try:
                score = parse_results(results.read_bytes())
                scores[name] = {**score.__dict__, "rate": score.rate}
            except Exception as e:
                scores[name] = {"rate": 0.0, "error": f"{e}; {proc.stdout[-300:]}", "failed": []}
    return scores


def expected_ok(solution: str, scores: dict[str, dict]) -> bool:
    rates = [scores[name]["rate"] for name in RUNS]
    if solution == "reference":
        return all(r == 1.0 for r in rates)
    if solution == "starter":
        return all(r == 0.0 for r in rates)
    # wrong: the lib and both app runs must lose points (wrong app.py even with the reference lib)
    return all(r < 1.0 for r in rates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    ids = args.task or task_ids()
    jobs = [(load_task(t), v, s) for t in ids for v in ("base", "hard") for s in SOLUTIONS]

    if args.local:

        def run(job):
            task, variant, solution = job
            return grade_local(task, variant, workspace_files(task, variant, solution))

        results = [run(job) for job in jobs]
    else:
        from env.grading import grade
        from env.sandbox import pack_tree
        from env.taskset import STARTER_MTIME

        app = modal.App.lookup(APP_NAME, create_if_missing=True)

        def run(job):
            task, variant, solution = job
            tar = pack_tree(workspace_files(task, variant, solution), STARTER_MTIME)
            return grade(task, variant, tar, app=app)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, jobs))

    table = Table("task", "variant", "solution", "lib", "app e2e", "app isolated", "ok")
    all_ok = True
    for (task, variant, solution), scores in zip(jobs, results, strict=True):
        ok = expected_ok(solution, scores)
        all_ok &= ok
        cells = []
        for name in RUNS:
            s = scores[name]
            cell = f"{s.get('passed', 0)}/{s.get('total', 0)}"
            cells.append(cell + (" ERR" if s.get("error") else ""))
        table.add_row(task.id, variant, solution, *cells, "yes" if ok else "[red]NO[/red]")
        if not ok or (solution == "reference" and any(scores[n].get("error") for n in RUNS)):
            for name in RUNS:
                if scores[name].get("failed") or scores[name].get("error"):
                    console.print(f"{task.id}/{variant}/{solution} {name}: {scores[name]}")
    console.print(table)
    console.print("ALL OK" if all_ok else "[red]SOME CHECKS FAILED[/red]")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
