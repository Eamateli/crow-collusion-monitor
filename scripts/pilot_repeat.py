"""The S2 pilot: each LLM monitor repeated on identical input, to apply the variance rule that was
registered first (analysis.leak.monitor_calls_by_rule, commit eb1343a).

    python scripts/pilot_repeat.py run      the 65 calls (about $0.20; loads .env), raw replies to
                                            results/raw/, which is gitignored
    python scripts/pilot_repeat.py export   results/pilot/monitor_repeat.json, committed: scores,
                                            tokens and latency, and no reply text

analysis/analyze.py reads the exported file into numbers.json.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analysis.integrity import tag_score
from crow.detectors.judge import monitor
from crow.detectors.views3 import (
    agent_transcript_view,
    crow_env_view,
    message_view3,
    union_transcript_view,
)
from crow.score import build_assembled_view
from env.planted import REFERENCE_LABEL, build
from env.prompts import (
    CROW_JUDGE3_PROMPT,
    MESSAGE_MONITOR3_PROMPT,
    TRANSCRIPT_AGENT_PROMPT,
    TRANSCRIPT_UNION_PROMPT,
    TRUSTED_MODEL_ID,
    TRUSTED_PARAMS,
)
from experiments.score_v2 import trusted_client

ROOT = Path(__file__).resolve().parent.parent
RAW = [ROOT / "results" / "raw" / n for n in ("judge_repeat_s2.jsonl", "monitor_repeat_s2.jsonl")]
EXPORT = ROOT / "results" / "pilot" / "monitor_repeat.json"
TASK = "ledger"
RULE_COMMIT = "eb1343a"
LABEL_WORDS = ("reference solution", "never copied", "agent sandbox")


def _commands(episode) -> dict[str, list[dict]]:
    return {
        agent: [
            {"tool": e.tool, "args": e.args}
            for e in episode.events
            if e.agent == agent and e.type == "tool_call"
        ]
        for agent in "ABC"
    }


def _judge_view(kind: str) -> str:
    episode = build(TASK, kind)
    assembled = episode.assembled
    specs = {a: episode.task.spec3(a, episode.variant) for a in "ABC"}
    manifest = episode.task.manifest(episode.variant, agents=3)
    view = build_assembled_view(assembled.starter, assembled.snapshot, manifest, episode.events, {})
    return crow_env_view(specs, assembled, view.changes)


def jobs() -> list[tuple[str, str, str]]:
    """(variant, prompt, view). Every view is built once, so repeats get identical input."""
    clean = _judge_view("honest")
    label = REFERENCE_LABEL.decode()
    with_label = clean
    for head in (f"--- lib/{TASK}.py (written by A) ---\n", "--- app.py (written by C) ---\n"):
        with_label = with_label.replace(head, head + label)
    honest, split = build(TASK, "honest"), build(TASK, "split_ladder")
    h_specs = {a: honest.task.spec3(a, honest.variant) for a in "ABC"}
    h_calls, s_calls = _commands(honest), _commands(split)
    work = [
        ("judge, honest, with the label", CROW_JUDGE3_PROMPT, with_label),
    ] * 10
    work += [("judge, honest, clean", CROW_JUDGE3_PROMPT, clean)] * 10
    work += [("judge, split, clean", CROW_JUDGE3_PROMPT, _judge_view("split_ladder"))] * 5
    work += [
        (
            "b2 union, honest, clean",
            TRANSCRIPT_UNION_PROMPT,
            union_transcript_view(h_specs, h_calls),
        )
    ] * 5
    note = message_view3(h_specs["A"], honest.assembled.snapshot)
    work += [("b1, honest, clean", MESSAGE_MONITOR3_PROMPT, note)] * 5
    for agent in "ABC":
        for name, calls in (("honest", h_calls), ("split", s_calls)):
            view = agent_transcript_view(agent, h_specs[agent], calls[agent])
            work += [(f"b2 {agent}, {name}, clean", TRANSCRIPT_AGENT_PROMPT, view)] * 5
    return work


def run() -> int:
    client = trusted_client()
    RAW[0].parent.mkdir(parents=True, exist_ok=True)

    def one(job):
        i, (name, prompt, view) = job
        result = monitor(
            "pilot", client, prompt, view, affinity_key=f"pilot-{i}", params=TRUSTED_PARAMS
        )
        return name, result

    with RAW[1].open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed(pool.submit(one, job) for job in enumerate(jobs())):
            name, r = future.result()
            replies = [c.content for c in r.calls]
            row = {
                "variant": name,
                "raw_score": r.raw_score,
                "tag": tag_score(replies),
                "reasoning": r.reasoning,
                "replies": replies,
                "finish": [c.finish_reason for c in r.calls],
                "latency_s": [c.latency_s for c in r.calls],
                "prompt_tokens": sum(c.usage.get("prompt_tokens", 0) for c in r.calls),
                "completion_tokens": sum(c.usage.get("completion_tokens", 0) for c in r.calls),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{name}: raw {r.raw_score}", flush=True)
    return 0


def export() -> int:
    calls = []
    for path in RAW:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            reasoning = (row["reasoning"] or "").lower()
            calls.append(
                {
                    "variant": row["variant"],
                    "raw_score": row["raw_score"],
                    "tag_score": row["tag"],
                    "finish": row["finish"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "latency_s": row.get("latency_s"),  # the first run did not save it
                    "mentions_the_label": any(word in reasoning for word in LABEL_WORDS),
                }
            )
    data = {
        "what": "each LLM monitor repeated on identical input, on planted fixtures (task ledger)",
        "rule_registered_in_commit": RULE_COMMIT,
        "model": TRUSTED_MODEL_ID,
        "params": TRUSTED_PARAMS,
        "calls": calls,
    }
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    EXPORT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(calls)} calls to {EXPORT.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "export"])
    args = parser.parse_args()
    return run() if args.command == "run" else export()


if __name__ == "__main__":
    raise SystemExit(main())
