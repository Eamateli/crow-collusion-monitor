"""Does every recorded monitor score equal the <score> tag in its saved reply? (no I/O, no LLM)

Written apart from crow.detectors.judge.monitor, with its own extraction, so that this is a check on
the parser and on the wiring of the record rather than a repeat of them.
"""

import re
from statistics import median

_TAG = re.compile(r"(?is)<score>(.*?)</score>")


def tag_score(replies: list[str | None]) -> float | None:
    """The score a run of replies holds: the last numeric tag of the first reply that has one, on
    the 0 to 10 scale, clamped. None if no reply has one."""
    for reply in replies:
        numbers = []
        for text in _TAG.findall(reply or ""):
            try:
                numbers.append(float(text.strip()))
            except ValueError:
                continue
        if numbers:
            return min(10.0, max(0.0, numbers[-1]))
    return None


def _expected(responses: list[dict]) -> float | None:
    """The score a monitor's saved replies hold: the median over its samples that gave one."""
    samples: dict[int, list[str | None]] = {}
    for reply in responses:
        samples.setdefault(reply["sample"], []).append(reply["content"])
    scored = [s for s in (tag_score(replies) for replies in samples.values()) if s is not None]
    return median(scored) if scored else None


def score_mismatches(scores: dict) -> list[dict]:
    """Every monitor in a 3-agent record's scores whose raw_score differs from its replies' tags."""
    per_agent = scores["b2_per_agent"]
    records = {
        "judge": scores["crow_env"]["judge"],
        "judge_files": scores["judge_files"],
        "b1": scores["b1"],
        "b2_union": scores["b2_union"],
        **{f"b2_{agent}": per_agent[agent] for agent in ("A", "B", "C")},
    }
    return [
        {"monitor": name, "recorded": r["raw_score"], "tag": _expected(r["responses"])}
        for name, r in records.items()
        if r["raw_score"] != _expected(r["responses"])
    ]
