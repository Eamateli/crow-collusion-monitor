"""Checks the tasks (Step 4): every reference solution passes its hidden tests, and the starter
alone and a deliberately wrong solution fail them.

Grades each task x variant x solution with env/grading.py in a fresh Modal sandbox, exactly as
an episode will. --local runs the same pytest commands on this machine instead (for fast
iteration; only ever on our own reference, starter and wrong files, never on agent output).

Run: python scripts/check_tasks.py [--local] [--task ID]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import modal
from rich.console import Console
from rich.table import Table

from env.grading import RUNS, grade_local
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
        from crow.workspace import pack_tree
        from env.grading import grade
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
