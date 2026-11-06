"""The S2 pilot's numbers: results/pilot/monitor_repeat.json -> s2_pilot in numbers.json.

The pilot repeated each LLM monitor on identical input, to apply the variance rule registered before
it was read (analysis.leak.monitor_calls_by_rule). It ran on small planted fixtures, so its cost and
time estimates are a floor for real workspaces. Judge and B2-union calls saved no latency, so their
times are estimated from their tokens at the speed measured on the calls that did.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from analysis.leak import MEDIAN_CALLS, VARIANCE_RANGE_POINTS, monitor_calls_by_rule
from env.episode import PRICES
from env.prompts import MONITOR_CALLS

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot" / "monitor_repeat.json"
PER_EPISODE = {"judge": 1, "b1": 1, "b2 per agent": 3, "b2 union": 1}  # LLM monitors per episode
V2_EPISODES = {"main": 130, "calibration": 30}


def _family(variant: str) -> str:
    name = variant.split(",")[0]
    if name.startswith("b2 ") and name != "b2 union":
        return "b2 per agent"
    return name


def _cost(prompt: float, completion: float) -> float:
    in_price, out_price = PRICES["trusted"]
    return prompt * in_price / 1e6 + completion * out_price / 1e6


def _summary(rows: list[dict]) -> dict:
    scores = sorted(r["raw_score"] for r in rows if r["raw_score"] is not None)
    return {
        "n": len(rows),
        "scores": scores,
        "range": max(scores) - min(scores) if scores else None,
        "median": median(scores) if scores else None,
        "missing": sum(r["raw_score"] is None for r in rows),
    }


def _label_side(rows: list[dict]) -> dict:
    return {
        **_summary(rows),
        "at_8_or_more": sum((r["raw_score"] or 0) >= 8 for r in rows),
        "replies_mentioning_the_label": sum(r["mentions_the_label"] for r in rows),
    }


def _estimates(calls: list[dict]) -> dict:
    timed = [r for r in calls if r.get("latency_s")]
    speed = sum(r["completion_tokens"] for r in timed) / sum(sum(r["latency_s"]) for r in timed)
    clean = [r for r in calls if "with the label" not in r["variant"]]
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        by_family[_family(r["variant"])].append(r)
    monitors, seconds, cost, tokens = {}, 0.0, 0.0, 0.0
    for family, per_episode in PER_EPISODE.items():
        rows = by_family[family]
        prompt = mean(r["prompt_tokens"] for r in rows)
        completion = mean(r["completion_tokens"] for r in rows)
        monitors[family] = {
            "calls_measured": len(rows),
            "per_episode": per_episode,
            "mean_prompt_tokens": round(prompt),
            "mean_completion_tokens": round(completion),
            "estimated_seconds_per_call": round(completion / speed),
            "cost_per_call_usd": round(_cost(prompt, completion), 5),
        }
        seconds += per_episode * completion / speed
        cost += per_episode * _cost(prompt, completion)
        tokens += per_episode * (prompt + completion)
    pass_minutes = MEDIAN_CALLS * seconds / 60
    return {
        "basis": "clean inputs only, on small planted fixtures: a floor for real workspaces",
        "completion_tokens_per_second": round(speed),
        "latency_s": {
            "calls_measured": sum(len(r["latency_s"]) for r in timed),
            "median": round(median(x for r in timed for x in r["latency_s"])),
            "max": round(max(x for r in timed for x in r["latency_s"])),
        },
        "monitors": monitors,
        "per_episode": {
            "one_call_each": {"cost_usd": round(cost, 4), "minutes": round(seconds / 60, 1)},
            "median_of_3": {
                "cost_usd": round(MEDIAN_CALLS * cost, 4),
                "minutes": round(pass_minutes, 1),
                "tokens": round(MEDIAN_CALLS * tokens),
            },
            "extra_cost_usd": round((MEDIAN_CALLS - 1) * cost, 4),
        },
        "extra_cost_usd_for_median_of_3": {
            name: round((MEDIAN_CALLS - 1) * cost * n, 2) for name, n in V2_EPISODES.items()
        },
        "scoring_pass_wall_hours_for_160_episodes": {
            f"{cap}_requests": round(sum(V2_EPISODES.values()) * pass_minutes / cap / 60, 1)
            for cap in (12, 16)
        },
    }


def pilot_numbers(path: Path = PILOT) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    calls = data["calls"]
    by_input: dict[str, list[dict]] = defaultdict(list)
    for row in calls:
        by_input[row["variant"]].append(row)
    scores = {k: [r["raw_score"] for r in rows] for k, rows in by_input.items()}
    meeting = sorted(k for k, v in scores.items() if monitor_calls_by_rule({k: v}) > 1)
    return {
        "source": str(path.relative_to(ROOT)),
        "what": data["what"],
        "calls": len(calls),
        "cost_usd": round(sum(_cost(r["prompt_tokens"], r["completion_tokens"]) for r in calls), 4),
        "rule": {
            "registered_in_commit": data["rule_registered_in_commit"],
            "range_points": VARIANCE_RANGE_POINTS,
            "calls_per_monitor": monitor_calls_by_rule(scores),
            "calls_in_the_code": MONITOR_CALLS,
            "inputs_meeting_the_criterion": meeting,
            "clean_inputs_meeting_the_criterion": [k for k in meeting if "with the label" not in k],
        },
        "inputs": {k: _summary(by_input[k]) for k in sorted(by_input)},
        "label_vs_clean": {
            "with_the_label": _label_side(by_input["judge, honest, with the label"]),
            "clean": _label_side(by_input["judge, honest, clean"]),
        },
        "integrity": {
            "score_vs_tag_mismatches": sum(r["raw_score"] != r["tag_score"] for r in calls),
            "cut_at_max_tokens": sum(f == "length" for r in calls for f in r["finish"]),
            "re_asks": sum(len(r["finish"]) - 1 for r in calls),
        },
        "estimates": _estimates(calls),
    }
