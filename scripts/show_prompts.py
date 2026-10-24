"""Render the filled-in prompts for one episode, for review before any run (no LLM, no Modal).

Static placeholders (k, defense_text, task_summary, the real seeded payload) are filled.
{a_proposal} and {b_reply} are shown as labelled stand-ins, because they only exist once the
planning chat has run.

Run: python scripts/show_prompts.py [--role collude] [--task ledger] [--defense seal_side] [--k 8]
"""

import argparse

from rich.console import Console
from rich.panel import Panel

from env.episode import make_config, payload_bits
from env.prompts import (
    AGENT_SYSTEM_A,
    AGENT_SYSTEM_B,
    AGREED_A,
    AGREED_B,
    COLLUDE_A_ADDENDUM,
    COLLUDE_B_ADDENDUM,
    DEFENSE_TEXT,
    PLAN_A,
    PLAN_B,
    SCHELLING_A,
    SCHELLING_B,
)
from env.taskset import load_task

console = Console()
A_STUB = "<<A's planning-chat proposal is inserted here>>"
B_STUB = "<<B's planning-chat reply is inserted here>>"


def show(title: str, text: str) -> None:
    console.print(Panel(text, title=title, border_style="cyan", width=100))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="collude", choices=["collude", "honest", "honest_hard"])
    parser.add_argument("--task", default="ledger")
    parser.add_argument("--defense", default="seal_side", choices=list(DEFENSE_TEXT))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-planning", action="store_true")
    args = parser.parse_args()

    config = make_config(
        args.role, args.task, args.defense, args.seed, k=args.k, planning=not args.no_planning
    )
    task = load_task(config.task_id)
    defense_text = DEFENSE_TEXT[config.defense]
    payload = payload_bits(config.seed, config.k) if config.collude else ""

    console.rule(f"[bold]{config.episode_id}[/bold]")
    console.print(
        f"role={config.role} task={config.task_id} variant={config.variant} "
        f"defense={config.defense} k={config.k} seed={config.seed} planning={config.planning}"
    )
    if config.collude:
        console.print(f"payload (seed {config.seed}, k {config.k}): [bold]{payload}[/bold]\n")

    show("AGENT_SYSTEM_A (base, every role)", AGENT_SYSTEM_A.format(defense_text=defense_text))
    show("AGENT_SYSTEM_B (base, every role)", AGENT_SYSTEM_B.format(defense_text=defense_text))

    if not config.collude:
        console.print("[dim]Honest/honest_hard: no planning chat, no payload, no addendum.[/dim]")
        return 0

    if config.planning:
        show(
            "PLAN_A (planning chat, A; no payload)",
            PLAN_A.format(k=config.k, defense_text=defense_text, task_summary=task.summary),
        )
        show(
            "PLAN_B (planning chat, B; no payload)",
            PLAN_B.format(
                k=config.k, defense_text=defense_text, task_summary=task.summary, a_proposal=A_STUB
            ),
        )
        a_add = COLLUDE_A_ADDENDUM.format(
            k=config.k,
            payload=payload,
            defense_text=defense_text,
            a_proposal=A_STUB,
            b_reply=B_STUB,
        )
        b_add = COLLUDE_B_ADDENDUM.format(
            k=config.k,
            payload=payload,
            defense_text=defense_text,
            a_proposal=A_STUB,
            b_reply=B_STUB,
        )
    else:
        console.print(
            "[dim]Schelling: no planning chat; the agreed-method paragraph is "
            "replaced by the no-chat text.[/dim]"
        )
        a_add = COLLUDE_A_ADDENDUM.replace(AGREED_A, SCHELLING_A).format(
            k=config.k, payload=payload, defense_text=defense_text
        )
        b_add = COLLUDE_B_ADDENDUM.replace(AGREED_B, SCHELLING_B).format(
            k=config.k, payload=payload, defense_text=defense_text
        )

    show("COLLUDE_A_ADDENDUM (appended to AGENT_SYSTEM_A)", a_add)
    show("COLLUDE_B_ADDENDUM (appended to AGENT_SYSTEM_B)", b_add)
    console.print(
        "\n[bold]Full A system prompt for A's turn = AGENT_SYSTEM_A + a blank line + "
        "COLLUDE_A_ADDENDUM above.[/bold]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
