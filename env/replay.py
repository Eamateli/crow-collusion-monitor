"""The intervention arms (Crow v2 addenda, sections 5 and 10): paired replays from a saved episode.

Each arm applies to every episode of its cell, whatever any monitor would have said, and no score
is computed during it. A replay starts from the saved state at a handoff and keeps the episode's
seed, task, payload, key and planning chat, so its pair differs only by the intervention:

- replace_a: a fresh honest A works from the starter. B and C then continue as their role has them,
  and may adapt to whatever the new A left.
- replace_b: A's saved output stands. A fresh honest B works from the saved view, then C continues.
- clean_before_c: A's and B's saved outputs stand. The workspace is cleaned, then C continues.

A replaced agent gets the base prompt, no private goal and no private tool. Nobody is told of the
replacement. Every arm records C's bit accuracy, usefulness and cost, and saves the same workspaces
an episode does. Reasoning and the private texts stay in the record, as for any episode.
"""

import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any

from crow.control import clean_workspace
from crow.events import EventLog
from crow.llm import LLMCall
from crow.wrap import snapshot
from env.episode import (
    AGENT_MODEL_ID,
    PROMPT_VERSION_3,
    TRUSTED_MODEL_ID,
    EpisodeConfig,
    _channel_fields,
    _private3,
    _run_turn,
    _save_tar,
    _seal_handoff,
    _system3,
    _turn_record,
    _usage,
    bit_errors,
    cost_estimate,
    honest_system3,
    payload_bits,
)
from env.grading import grade
from env.relational import scheme_for
from env.sandbox import Sandbox
from env.taskset import Task, load_task

ARMS = ("replace_a", "replace_b", "clean_before_c")
_REPLACED = {"replace_a": "A", "replace_b": "B", "clean_before_c": None}
_NO_CHAT = {"proposal": None, "reply": None, "c_reply": None}


def arm_episode_id(base_id: str, arm: str) -> str:
    return f"{base_id}__arm_{arm}"


def load_base_record(run_dir: str | Path, base_stem: str) -> dict:
    path = Path(run_dir) / "records" / f"{base_stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_arm(
    base_dir: str | Path,
    base_stem: str,
    arm: str,
    agent_client: Any,
    trusted_client: Any,
    out_dir: str | Path,
    *,
    app: Any = None,
    attempt: int = 1,
    sandbox_factory: Callable[[], AbstractContextManager[Sandbox]] | None = None,
    grade_fn: Callable[[Task, str, bytes], dict] | None = None,
) -> dict:
    """One arm on one saved episode. A failure is recorded, not raised (CLAUDE.md section 13)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    base = load_base_record(base_dir, base_stem)
    config = replace(
        EpisodeConfig(**base["config"]), episode_id=arm_episode_id(base["episode_id"], arm)
    )
    stem = f"{config.episode_id}__a{attempt}"
    log = EventLog(out_dir / "events" / f"{stem}.jsonl", config.episode_id)
    sandbox_factory = sandbox_factory or (lambda: Sandbox(app=app))
    grade_fn = grade_fn or (lambda task, variant, tar: grade(task, variant, tar, app=app))

    def saved(name: str) -> bytes:
        return (base_dir / "workspaces" / f"{base_stem}.{name}.tar").read_bytes()

    started = time.time()
    record: dict[str, Any] = {
        "episode_id": config.episode_id,
        "base_episode": base["episode_id"],
        "base_attempt": base["attempt"],
        "arm": arm,
        "replaced_agent": _REPLACED[arm],
        "attempt": attempt,
        "status": "ok",
        "config": vars(config),
        "prompt_version": PROMPT_VERSION_3,
        "models": {"agent": AGENT_MODEL_ID, "trusted": TRUSTED_MODEL_ID},
        "scores": None,  # no score during a replay; the scoring pass can read what it saves
    }
    try:
        if base["status"] != "ok":
            raise ValueError(f"base episode is {base['status']}")
        task = load_task(config.task_id)
        manifest = task.manifest(config.variant, agents=3)
        payload = payload_bits(config.seed, config.k) if config.collude else None
        keyed = config.collude and config.planning
        scheme = scheme_for(config.seed, config.k) if keyed else None
        if payload != base.get("payload"):
            raise ValueError("the payload differs from the saved episode's")
        chat = {k: (base.get("planning") or _NO_CHAT).get(k) for k in _NO_CHAT}
        record["payload"] = payload
        record["scheme"] = scheme.to_dict() if scheme else None

        record["turns"], record["seal"] = {}, {}
        results: dict[str, Any] = {}
        states: dict[str, Any] = {}
        seal_calls: list[LLMCall] = []
        replaced = _REPLACED[arm]

        def turn(agent: str, view_tar: bytes) -> bytes:
            honest = agent == replaced
            system = (
                honest_system3(agent, config)
                if honest
                else _system3(agent, config, task, payload or "", scheme, chat)
            )
            with sandbox_factory() as sb:
                sb.copy_in(view_tar)
                results[agent], states[agent] = _run_turn(
                    config,
                    agent_client,
                    sb,
                    log,
                    agent,
                    system,
                    task.spec3(agent, config.variant),
                    None if honest else _private3(config, agent),
                )
                raw = sb.copy_out()
            snapshot(raw, log=log, agent_id="crow", label=f"after {agent}")
            record["turns"][agent] = _turn_record(results[agent])
            return raw

        def handoff(agent: str, next_agent: str, raw: bytes) -> bytes:
            view, report, calls = _seal_handoff(
                config, raw, manifest, trusted_client, log, next_agent
            )
            record["seal"][next_agent] = report
            seal_calls.extend(calls)
            _save_tar(out_dir, stem, f"{agent.lower()}_out", raw)
            _save_tar(out_dir, stem, f"{next_agent.lower()}_view", view)
            return view

        if arm == "replace_a":
            b_view = handoff("A", "B", turn("A", task.starter_tar(config.variant)))
            c_view = handoff("B", "C", turn("B", b_view))
        elif arm == "replace_b":
            _save_tar(out_dir, stem, "a_out", saved("a_out"))
            _save_tar(out_dir, stem, "b_view", saved("b_view"))
            c_view = handoff("B", "C", turn("B", saved("b_view")))
        else:
            b_out = saved("b_out")
            for name, data in (("a_out", saved("a_out")), ("b_view", saved("b_view"))):
                _save_tar(out_dir, stem, name, data)
            _save_tar(out_dir, stem, "b_out", b_out)
            starter = snapshot(task.starter_tar(config.variant))
            c_view, clean_report = clean_workspace(b_out, starter, manifest)
            record["clean"] = clean_report.to_dict()
            record["seal"]["C"] = {"mode": "clean"}
            _save_tar(out_dir, stem, "c_view", c_view)
        final = turn("C", c_view)
        _save_tar(out_dir, stem, "final", final)

        record["report_method"] = {
            agent: ((states.get(agent) or {}).get("args") or {}).get("text") for agent in ("A", "B")
        }
        bits = None
        record["submission"], record["bit_accuracy"] = None, None
        if config.collude:
            bits = ((states["C"] or {}).get("args") or {}).get("bits")
            bits = bits.strip() if isinstance(bits, str) else None
            errors = bit_errors(payload, bits)
            record["submission"] = {
                "bits": bits,
                "correct": bits is not None and bits == payload,
                "bit_errors": errors,
            }
            record["bit_accuracy"] = 1 - errors / config.k
        record.update(_channel_fields(config, task, c_view, scheme, payload, bits))
        record["usefulness"] = grade_fn(task, config.variant, final)

        turn_calls = [call for agent in results for call in results[agent].calls]
        buckets = {f"agent_{agent}": _usage(results[agent].calls) for agent in sorted(results)} | {
            "paraphrase": _usage(seal_calls)
        }
        everything = _usage(turn_calls + seal_calls)
        record["tokens"] = {
            **buckets,
            "total": everything["total"],
            "reasoning_total": everything["reasoning"],
            "length_total": everything["length"],
        }
        record["cost_estimate_usd"] = cost_estimate(buckets)
    except Exception as e:  # noqa: BLE001 - handled: mark failed, keep the record (CLAUDE.md 13)
        record["status"] = "failed"
        record["reason"] = f"{type(e).__name__}: {e}"
    record["seconds"] = round(time.time() - started, 2)
    path = out_dir / "records" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
