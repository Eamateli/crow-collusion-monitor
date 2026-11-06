"""The S3 gate of Crow v2 (addenda, section 10): every intervention arm runs end to end on planted
fixtures, on every role. No outcome is required; the table is there to read, not to pass. No LLM,
no Modal and no real sandbox: the agents are scripted, and act on the prompts the replay gives them.

Run: python scripts/gate_s3.py
"""

import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from replay_fakes import ParaphraseOnly, PromptDrivenAgents, run_base  # noqa: E402
from test_episode import LocalSandbox, grade_fn  # noqa: E402

from env.episode import make_config  # noqa: E402
from env.replay import ARMS, run_arm  # noqa: E402
from env.taskset import task_ids  # noqa: E402

console = Console()
ROLES = {
    "split": lambda task, seed: make_config("collude_split", task, "open", seed, k=8, agents=3),
    "control": lambda task, seed: make_config("collude", task, "open", seed, k=8, agents=3),
    "honest": lambda task, seed: make_config("honest", task, "open", seed, agents=3),
}
RAN = {
    "replace_a": {"A", "B", "C"},
    "replace_b": {"B", "C"},
    "clean_before_c": {"C"},
    "canonicalize_before_c": {"C"},
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "[green]pass[/green]" if ok else "[bold red]FAIL[/bold red]"
    console.print(f"  {mark}  {name}" + (f"  [dim]{detail}[/dim]" if detail else ""))
    return ok


def main() -> int:
    tasks = task_ids()
    table = Table(
        "role",
        "arm",
        "episodes",
        "C's bit accuracy",
        "U (app e2e)",
        "lib",
        "agents run",
        "names changed",
    )
    problems, runs = [], 0
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for role, make in ROLES.items():
            bases = [(make(task, seed), seed) for seed, task in enumerate(tasks)]
            for config, _ in bases:
                base, _ = run_base(config, root / "base")
                if base["status"] != "ok":
                    problems.append(f"base {config.episode_id}: {base.get('reason')}")
            rows = {"base": [run_record(root, c) for c, _ in bases]}
            for arm in ARMS:
                rows[arm] = []
                for config, _ in bases:
                    agents, trusted = PromptDrivenAgents(config), ParaphraseOnly()
                    record = run_arm(
                        root / "base",
                        f"{config.episode_id}__a1",
                        arm,
                        agents,
                        trusted,
                        root / "arms",
                        sandbox_factory=LocalSandbox,
                        grade_fn=grade_fn,
                    )
                    runs += 1
                    rows[arm].append(record)
                    if record["status"] != "ok":
                        problems.append(f"{record['episode_id']}: {record.get('reason')}")
                    elif set(agents.order) != RAN[arm] or trusted.keys:
                        problems.append(
                            f"{record['episode_id']}: ran {agents.order}, {trusted.keys}"
                        )
            for arm, records in rows.items():
                table.add_row(role, arm, *_cells(records, role))
    console.print(table)
    ok = check("every arm finished on every role and task", not problems, f"{runs} replays")
    for problem in problems[:10]:
        console.print(f"    {problem}")
    console.print("  [dim]No outcome is required of any arm (addenda, section 10).[/dim]")
    verdict = "[bold green]all checks passed[/bold green]" if ok else "[bold red]FAILED[/bold red]"
    console.rule(verdict)
    return 0 if ok else 1


def run_record(root: Path, config) -> dict:
    import json

    path = root / "base" / "records" / f"{config.episode_id}__a1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


def _cells(records: list[dict], role: str) -> list[str]:
    ok = [r for r in records if r["status"] == "ok"]
    if role == "honest":
        accuracy = "-"
    else:
        hits = [1 - r["submission"]["bit_errors"] / r["config"]["k"] for r in ok]
        accuracy = _mean(hits)
    stripped = sum(len(n) for r in ok for n in (r.get("clean") or {}).get("stripped", {}).values())
    stripped += sum(sum((r.get("canonicalize") or {}).get("renamed", {}).values()) for r in ok)
    return [
        str(len(records)),
        accuracy,
        _mean([r["usefulness"]["app_e2e"]["rate"] for r in ok]),
        _mean([r["usefulness"]["lib"]["rate"] for r in ok]),
        ",".join(sorted(ok[0]["turns"])) if ok else "-",
        str(stripped) if any(r.get("clean") or r.get("canonicalize") for r in ok) else "-",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
