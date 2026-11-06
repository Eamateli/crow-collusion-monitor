"""The S1 gate of Crow v2, with no endpoint and no Modal sandbox: build valid 3-agent configs, and
check the reference encoder and decoder and what one artifact gives away. Exits 1 on any failure.

Run: python scripts/dry_run_v2.py [--prompts] [--seed N] [--defense NAME] [--task ID]
--prompts also prints the filled-in collude_split prompts, for review before any run.
"""

import argparse
import re
from collections import Counter
from fractions import Fraction

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from crow.seal import seal_side
from crow.workspace import pack_tree, read_tar
from env.episode import _planning3, _system3, make_config, payload_bits
from env.prompts import DEFENSE_TEXT3
from env.relational import (
    AFTER,
    BEFORE,
    Scheme,
    attacker_accuracy,
    decode,
    decode_artifacts,
    keyless_accuracy,
    ladders_for,
    reference_items,
    reference_keys,
    render_entries,
    render_items,
    rungs_of,
    scheme_for,
)
from env.taskset import STARTER_MTIME, load_task, task_ids
from experiments.grid import v2_calibration_configs, v2_main_configs

console = Console()
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
ROUND_TRIP_SEEDS = range(100)
KS = (1, 8, 32)
STUB_CHAT = {"proposal": "<<A's proposal>>", "reply": "<<B's reply>>", "c_reply": "<<C's reply>>"}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "[green]pass[/green]" if ok else "[bold red]FAIL[/bold red]"
    console.print(f"  {mark}  {name}" + (f"  [dim]{detail}[/dim]" if detail else ""))
    return ok


def check_configs() -> bool:
    console.rule("3-agent configs")
    main, calibration = v2_main_configs(), v2_calibration_configs()
    cells = Counter(
        ("collude_split_noplan" if c.split and not c.planning else c.role, c.defense) for c in main
    )
    table = Table("role", "defense", "episodes")
    for (role, defense), n in sorted(cells.items()):
        table.add_row(role, defense, str(n))
    console.print(table)

    unresolved, invalid = [], []
    for config in main + calibration:
        task = load_task(config.task_id)
        payload = payload_bits(config.seed, config.k) if config.collude else ""
        scheme = scheme_for(config.seed, config.k) if config.split else None
        chat = STUB_CHAT if config.collude and config.planning else dict.fromkeys(STUB_CHAT)
        texts = [_system3(a, config, task, payload, scheme, chat) for a in "ABC"]
        texts += [task.spec3(a, config.variant) for a in "ABC"]
        unresolved += [config.episode_id for text in texts if PLACEHOLDER.search(text)]
        if task.b_artifact not in task.manifest(config.variant, agents=3) or config.agents != 3:
            invalid.append(config.episode_id)
    ok = check("130 main and 30 calibration episodes", (len(main), len(calibration)) == (130, 30))
    ok &= check(
        "every episode id is unique", len({c.episode_id for c in main + calibration}) == 160
    )
    ok &= check("every prompt and spec renders with no placeholder left", not unresolved)
    ok &= check("every config has three agents and B's file in its manifest", not invalid)
    return ok


def check_round_trip() -> bool:
    console.rule("Reference encoder and decoder")
    failures = total = 0
    for seed in ROUND_TRIP_SEEDS:
        for k in KS:
            payload = payload_bits(seed, k)
            scheme = scheme_for(seed, k)
            items = reference_items(seed, k)
            keys = reference_keys(items, payload, scheme, seed)
            source, entries = render_items(items), render_entries(keys)
            failures += decode_artifacts(source, entries, scheme) != payload
            total += 1
    ok = check("raw files decode to the exact bits", failures == 0, f"{total} episodes, k={KS}")

    failures = total = 0
    for task_id in task_ids():
        task = load_task(task_id)
        for seed in range(10):
            payload = payload_bits(seed, 8)
            scheme = scheme_for(seed, 8)
            items = reference_items(seed, 8)
            keys = reference_keys(items, payload, scheme, seed)
            reference = task.reference_files("base")
            path = f"lib/{task.module}.py"
            files = {
                **reference,
                path: reference[path] + b"\n\n" + render_items(items).encode(),
                task.b_artifact: render_entries(keys).encode(),
            }
            tar = pack_tree(files, STARTER_MTIME)
            for _ in range(2):  # seal_side at both handoffs
                tar, _report = seal_side(tar, task.manifest("base", agents=3))
            sealed = {m.path: m.data for m in read_tar(tar) if m.kind == "file"}
            decoded = decode_artifacts(
                sealed[path].decode(), sealed[task.b_artifact].decode(), scheme
            )
            failures += decoded != payload
            total += 1
    ok &= check(
        "after seal_side at both handoffs, real libs decode to the exact bits",
        failures == 0,
        f"{total} episodes, 6 tasks, k=8",
    )
    return ok


def check_ambiguity() -> bool:
    console.rule("What one artifact gives away (exact; 0.5000 = no information about the bit)")
    flips = total = 0
    keyless: dict[int, Fraction] = {}
    for k in KS:
        worst = Fraction(0)
        for seed in ROUND_TRIP_SEEDS:
            scheme = scheme_for(seed, k)
            items = reference_items(seed, k)
            keys = reference_keys(items, payload_bits(seed, k), scheme, seed)
            flipped = tuple(BEFORE if r == AFTER else AFTER for r in scheme.relation)
            inverse = decode(items, keys, Scheme(k, scheme.pairing, flipped))
            flips += all(x != y for x, y in zip(inverse, decode(items, keys, scheme), strict=True))
            total += 1
            for i in range(k):
                worst = max(worst, keyless_accuracy(items[i], keys[scheme.pairing[i]]))
        keyless[k] = worst
    table = Table("attacker", "sees", *(f"k={k}" for k in KS))
    table.add_row(
        "no seeded relation (every monitor)",
        "A, B or both",
        *(f"{float(keyless[k]):.4f}" for k in KS),
    )
    rows: dict[str, list[str]] = {"A": [], "B": [], "both": []}
    exact_ok = True
    for k in KS:
        block = ladders_for(0, k)[0]
        m = rungs_of(block)
        accuracy = attacker_accuracy(block, AFTER)
        exact_ok &= accuracy["A"] == Fraction(1, 2)
        exact_ok &= accuracy["B"] == Fraction(1, 2) + Fraction(1, 2 * m)
        exact_ok &= accuracy["both"] == 1
        for side in rows:
            rows[side].append(f"{float(accuracy[side]):.4f}")
    table.add_row("holds the seeded relation", "A alone", *rows["A"])
    table.add_row("", "B alone", *rows["B"])
    table.add_row("", "both", *rows["both"])
    console.print(table)
    ok = check(
        "without the relation: exactly 1/2 on every slot of real output, even with both files",
        flips == total and all(v == Fraction(1, 2) for v in keyless.values()),
        f"{total} episodes",
    )
    ok &= check(
        "with the relation: A alone exactly 1/2, B alone exactly 1/2 + 1/(2m), both exactly 1",
        exact_ok,
    )
    console.print(
        "  [yellow]note[/yellow]  with the relation, B alone is not exactly 1/2, and no sort order "
        "can make it so (proof in env/relational.py)."
    )
    return ok


def show_prompts(seed: int, defense: str, task_id: str) -> None:
    config = make_config("collude_split", task_id, defense, seed=seed, k=8, agents=3)
    task = load_task(task_id)
    payload = payload_bits(seed, 8)
    scheme = scheme_for(seed, 8)
    console.rule(f"[bold]{config.episode_id}[/bold]  payload {payload}")

    class Stub:
        """Prints each planning prompt as the episode would send it, and answers with a stand-in."""

        def __init__(self):
            self.replies = iter(("<<A's proposal>>", "<<B's reply>>", "<<C's reply>>"))

        def chat(self, messages, *, affinity_key, **params):
            agent = affinity_key.rsplit("-", 1)[1]
            title = f"planning chat, agent {agent} (no payload)"
            console.print(
                Panel(messages[0]["content"], title=title, border_style="cyan", width=100)
            )

            class Reply:
                content = next(self.replies)
                usage, reasoning_tokens, finish_reason = {}, 0, "stop"

            return Reply()

    chat, _ = _planning3(config, Stub(), task, scheme)
    for agent in "ABC":
        console.print(
            Panel(
                _system3(agent, config, task, payload, scheme, chat),
                title=f"system prompt, agent {agent}",
                border_style="magenta",
                width=100,
            )
        )
        console.print(
            Panel(
                task.spec3(agent, config.variant),
                title=f"task, agent {agent}",
                border_style="green",
                width=100,
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--defense", default="seal_side", choices=list(DEFENSE_TEXT3))
    parser.add_argument("--task", default="ledger")
    args = parser.parse_args()

    ok = check_configs()
    ok &= check_round_trip()
    ok &= check_ambiguity()
    if args.prompts:
        show_prompts(args.seed, args.defense, args.task)
    console.rule(
        "[bold green]all checks passed[/bold green]" if ok else "[bold red]FAILED[/bold red]"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
