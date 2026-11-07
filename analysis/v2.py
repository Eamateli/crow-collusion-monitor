"""Crow v2 (S7): raw 3-agent records and scores -> the v2 section of analysis/numbers.json, the v2
tables, Figure 1 and the findings list. Every number in the v2 paper comes from here.

It runs as registered in crow/frozen_v2.json ("registered") and in the addenda (sections 1, 2, 9
to 13), in one pass, and makes no model call. Anything beyond the registration is marked
exploratory in numbers.json and in the tables. Agent work is read as data only: records, scores,
event logs, and workspace tars parsed with ast and json; nothing an agent wrote is run.

    python analysis/analyze.py --v2 results/raw/v2

results/raw/v2 holds v2_main/{records,scores,events,workspaces,scoring_summaries},
v2_main_arms/records and v2_calibration/scores, copied from the crow-results Volume.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from analysis.integrity import score_mismatches
from analysis.leak import (
    best_bit_accuracy,
    command_leak_share,
    cost_of_hiding,
    helpers_called,
    kinds_where_built,
    payload_in_commands,
    pooled_leak_test,
)
from analysis.metrics import (
    auroc,
    bit_error_rate,
    bootstrap_ci,
    capacity,
    paired_comparison,
    tpr_at_fpr,
)

ROOT = Path(__file__).resolve().parent.parent
FROZEN_V2 = ROOT / "crow" / "frozen_v2.json"
PREEMPTIONS = ROOT / "results" / "v2_main" / "preemptions.json"
BILLING = ROOT / "results" / "v2_main" / "billing.json"
DEFENSES = ("open", "seal_side", "seal_side_para")
RESAMPLES, SEED = 2000, 0  # CLAUDE.md section 7, equation 5
RESCORE_FIX_COMMIT = "b011a91"  # the logged fix after which 52 lost results were scored again
MIN_NEGATIVES_FOR_1PCT = 100

# Each monitor: its threshold in frozen_v2.json, how to read its score, and its name in the paper.
MONITORS: dict[str, tuple[str, Callable[[dict], float | None], str]] = {
    "per_agent": ("per_agent_max", lambda s: s["b2_per_agent"]["max"], "per-agent B2 (max)"),
    "pooled": ("b2_union", lambda s: s["b2_union"]["score"], "pooled B2"),
    "crow": ("judge", lambda s: s["crow_env"]["judge"]["score"], "Crow judge"),
    "postcondition": ("postcondition", lambda s: s["postcondition"]["score"], "postcondition"),
    "composite": ("composite", lambda s: s["crow_env"]["composite"], "Crow composite"),
    "b1": ("b1", lambda s: s["b1"]["score"], "B1 message"),
    "judge_files": ("judge_files", lambda s: s["judge_files"]["score"], "files-only judge"),
}
# The four ways of watching in the primary comparison (addenda, section 11); the rest are secondary.
PRIMARY = ("per_agent", "pooled", "crow", "postcondition")
SECONDARY = ("composite", "b1", "judge_files")
REPLY_GROUPS: dict[str, Callable[[dict], dict]] = {
    "judge": lambda s: s["crow_env"]["judge"],
    "judge_files": lambda s: s["judge_files"],
    "b1": lambda s: s["b1"],
    "b2_union": lambda s: s["b2_union"],
    "b2_A": lambda s: s["b2_per_agent"]["A"],
    "b2_B": lambda s: s["b2_per_agent"]["B"],
    "b2_C": lambda s: s["b2_per_agent"]["C"],
}
COLLUDE_CELLS = (
    *[("split", d) for d in DEFENSES],
    *[("control", d) for d in DEFENSES],
    ("noplan", "open"),
    ("careless_split", "open"),
    ("careless_control", "open"),
)
HONEST_CELLS = (*[("honest", d) for d in DEFENSES], *[("honest_hard", d) for d in DEFENSES])
ROLE_OF_FAMILY = {"split": "collude_split", "control": "collude", "honest": "honest"}
PAYLOAD_HOLDER_ARM = {"split": "replace_b", "control": "replace_a"}  # addenda, section 10
ARMS = ("replace_a", "replace_b", "clean_before_c", "canonicalize_before_c")
EXPLORATORY = {
    "auroc_ci": "stratified bootstrap CI of AUROC (2000 resamples, seed 0): not registered",
    "cost_of_hiding_ci": "bootstrap CI over seed pairs (2000 resamples, seed 0): not registered",
    "flagged_agent_ties": (
        "the flagged agent is the unique argmax of the three per-agent scores; ties and a flagged "
        "C have no replay and are counted, not imputed"
    ),
    "per_agent_mean_auroc": "AUROC of the mean of the three per-agent scores, which has no "
    "frozen threshold",
    "false_alarms_by_defense": "each monitor's false alarms on the main honest episodes, by role "
    "and defense; calibration ran under open only (added after S7, at the user's request)",
    "flagged_agents": "which agent the per-agent monitor flags (the unique highest of the three "
    "per-agent scores) in the careful split and control cells, ties and C counted (added after "
    "S7, at the user's request)",
}


# --- loading ---


@dataclass
class V2Data:
    run_dir: Path
    records: list[dict]  # ok main episodes, the latest attempt of each
    failed: list[dict]
    scores: dict[str, dict]  # episode_id -> its ok score file
    failed_scores: list[dict]
    arms: list[dict]  # ok replays
    failed_arms: list[dict]
    calibration_scores: list[dict]
    summaries: list[dict]  # the scoring passes over the main run, in order


def _latest(paths: Iterable[Path], key: str = "episode_id") -> list[dict]:
    latest: dict[str, dict] = {}
    for path in sorted(paths):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item[key] not in latest or item.get("attempt", 1) >= latest[item[key]].get("attempt", 1):
            latest[item[key]] = item
    return list(latest.values())


def load(raw: Path) -> V2Data:
    main = raw / "v2_main"
    records = _latest((main / "records").glob("*.json"))
    scores = _latest((main / "scores").glob("*.json"))
    arms = _latest((raw / "v2_main_arms" / "records").glob("*.json"))
    calibration = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((raw / "v2_calibration" / "scores").glob("*.json"))
    ]
    summaries = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((main / "scoring_summaries").glob("*.json"))
    ]
    return V2Data(
        run_dir=main,
        records=[r for r in records if r["status"] == "ok"],
        failed=[r for r in records if r["status"] != "ok"],
        scores={s["episode_id"]: s for s in scores if s["status"] == "ok"},
        failed_scores=[s for s in scores if s["status"] != "ok"],
        arms=[a for a in arms if a["status"] == "ok"],
        failed_arms=[a for a in arms if a["status"] != "ok"],
        calibration_scores=[s for s in calibration if s["status"] == "ok"],
        summaries=summaries,
    )


# --- small measures ---


def rate(flags: Iterable[bool]) -> dict[str, Any]:
    """A share with its 95% percentile bootstrap CI (CLAUDE.md section 7, equation 5)."""
    flags = [bool(f) for f in flags]
    if not flags:
        return {"hits": 0, "n": 0, "rate": None, "ci": None}
    values = [float(f) for f in flags]
    return {
        "hits": sum(flags),
        "n": len(flags),
        "rate": sum(values) / len(values),
        "ci": list(bootstrap_ci(values, resamples=RESAMPLES, seed=SEED)),
    }


def mean_ci(values: Iterable[float | None]) -> dict[str, Any]:
    given = [float(v) for v in values if v is not None]
    if not given:
        return {"n": 0, "mean": None, "ci": None}
    return {
        "n": len(given),
        "mean": sum(given) / len(given),
        "ci": list(bootstrap_ci(given, resamples=RESAMPLES, seed=SEED)),
    }


def auroc_ci(pos: list[float], neg: list[float]) -> list[float] | None:
    """Stratified bootstrap: positives and negatives are resampled apart (exploratory)."""
    if not pos or not neg:
        return None
    rng = np.random.default_rng(SEED)
    p, n = np.array(pos, dtype=float), np.array(neg, dtype=float)
    draws = []
    for _ in range(RESAMPLES):
        a = list(rng.choice(p, size=len(p), replace=True))
        b = list(rng.choice(n, size=len(n), replace=True))
        draws.append(auroc(a, b))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def bit_accuracy(record: dict) -> float | None:
    submission = record.get("submission")
    if not submission:
        return None
    return 1 - submission["bit_errors"] / record["config"]["k"]


def recovered(record: dict) -> bool:
    return bool((record.get("submission") or {}).get("correct"))


def usefulness(record: dict) -> float:
    return record["usefulness"]["app_e2e"]["rate"]


def capped(record: dict) -> bool:
    """A turn ended at the agent's max_tokens cap (addenda, section 13)."""
    return any(t["stop_reason"] == "length" for t in record["turns"].values())


def cell_of(config: dict) -> tuple[str, str]:
    role = config["role"]
    if role in ("honest", "honest_hard"):
        return role, config["defense"]
    family = "split" if role == "collude_split" else "control"
    if not config["planning"]:
        return "noplan", config["defense"]
    if config["careless"]:
        return f"careless_{family}", config["defense"]
    return family, config["defense"]


def is_primary(record: dict) -> bool:
    """The planned careful split cell under its three defenses, with the channel built (addenda,
    sections 11 and 13). No-plan and the careless contrast are outside it."""
    return cell_of(record["config"])[0] == "split" and record.get("channel_built") is True


# --- the monitors ---


class Monitors:
    """Scores and alarms at the frozen thresholds, for every scored episode."""

    def __init__(self, scores: dict[str, dict], thresholds: dict[str, float]):
        self.scores, self.thresholds = scores, thresholds

    def score(self, episode_id: str, monitor: str) -> float | None:
        scored = self.scores.get(episode_id)
        return None if scored is None else MONITORS[monitor][1](scored["scores"])

    def threshold(self, monitor: str) -> float:
        return self.thresholds[MONITORS[monitor][0]]

    def alarm(self, episode_id: str, monitor: str) -> bool | None:
        s = self.score(episode_id, monitor)
        return None if s is None else s >= self.threshold(monitor)

    def per_agent(self, episode_id: str) -> dict[str, float | None]:
        per = self.scores[episode_id]["scores"]["b2_per_agent"]
        return {agent: per[agent]["score"] for agent in "ABC"}


def detection(ids_pos: list[str], ids_neg: list[str], monitors: Monitors) -> dict[str, Any]:
    """Each monitor on one population against the main honest episodes."""
    out: dict[str, Any] = {}
    for m in MONITORS:
        pos = [s for e in ids_pos if (s := monitors.score(e, m)) is not None]
        neg = [s for e in ids_neg if (s := monitors.score(e, m)) is not None]
        tau = monitors.threshold(m)
        out[m] = {
            "name": MONITORS[m][2],
            "family": "primary" if m in PRIMARY else "secondary",
            "threshold": tau,
            "positives": len(ids_pos),
            "positives_missing": len(ids_pos) - len(pos),
            "negatives": len(ids_neg),
            "negatives_missing": len(ids_neg) - len(neg),
            "detection": rate(s >= tau for s in pos),
            "false_alarm": rate(s >= tau for s in neg),
            "auroc": auroc(pos, neg),
            "auroc_ci": auroc_ci(pos, neg),
            "tpr_at_5pct_fpr_on_main_negatives": tpr_at_fpr(pos, neg, 0.05),
        }
    means_pos = [monitors.scores[e]["scores"]["b2_per_agent"]["mean"] for e in ids_pos]
    means_neg = [monitors.scores[e]["scores"]["b2_per_agent"]["mean"] for e in ids_neg]
    out["per_agent_mean_auroc"] = auroc(
        [s for s in means_pos if s is not None], [s for s in means_neg if s is not None]
    )
    return out


def false_alarms(data: V2Data, monitors: Monitors, frozen: dict) -> dict[str, Any]:
    """Every monitor's false-alarm rate on the main honest episodes, by role and together (addenda,
    section 2), and in calibration (addenda, section 12)."""
    by_role = defaultdict(list)
    for r in data.records:
        if r["config"]["role"] in ("honest", "honest_hard"):
            by_role[r["config"]["role"]].append(r["episode_id"])
    calibration = frozen["provenance"]["calibration_alarms"]
    out = {}
    for m in MONITORS:
        row = {}
        for role in ("honest", "honest_hard"):
            row[role] = rate(a for e in by_role[role] if (a := monitors.alarm(e, m)) is not None)
        both = by_role["honest"] + by_role["honest_hard"]
        row["all"] = rate(a for e in both if (a := monitors.alarm(e, m)) is not None)
        cal = calibration[MONITORS[m][0]]
        row["calibration"] = rate(
            [True] * cal["at_threshold"] + [False] * (cal["scored"] - cal["at_threshold"])
        )
        out[m] = row
    return out


def cut_shares(score_files: Iterable[dict]) -> dict[str, Any]:
    """Each monitor's share of model calls cut at max_tokens (addenda, section 12)."""
    calls, cut = Counter(), Counter()
    for scored in score_files:
        for name, read in REPLY_GROUPS.items():
            for reply in read(scored["scores"])["responses"]:
                calls[name] += 1
                cut[name] += reply["finish_reason"] == "length"
    out = {
        name: {"cut": cut[name], "calls": calls[name], "share": cut[name] / calls[name]}
        for name in REPLY_GROUPS
        if calls[name]
    }
    per_agent = [n for n in ("b2_A", "b2_B", "b2_C") if n in out]
    if per_agent:
        c, n = sum(cut[x] for x in per_agent), sum(calls[x] for x in per_agent)
        out["b2_per_agent"] = {"cut": c, "calls": n, "share": c / n}
    total_cut, total_calls = sum(cut.values()), sum(calls.values())
    out["all"] = {
        "cut": total_cut,
        "calls": total_calls,
        "share": total_cut / total_calls if total_calls else None,
    }
    return out


# --- the manipulation check, capacity and success ---


def by_cell(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        cells[cell_of(r["config"])].append(r)
    return cells


def manipulation(records: list[dict]) -> dict[str, Any]:
    """One cell's manipulation check (addenda, sections 1, 9, 12 and 13)."""
    checks = [r["channel_check"] for r in records if r.get("channel_check")]
    pooled = pooled_leak_test(checks)
    label = {True: "leaky", False: "relational", None: "not tested"}[pooled["flagged"]]
    k = records[0]["config"]["k"] if records else 0
    errors = [r["submission"]["bit_errors"] for r in records if r.get("submission")]
    epsilon = bit_error_rate(errors, [k] * len(errors)) if errors else None
    return {
        "n": len(records),
        "built": rate(r.get("channel_built") is True for r in records),
        "bit_accuracy": mean_ci(bit_accuracy(r) for r in records),
        "recovered": rate(recovered(r) for r in records),
        "usefulness": mean_ci(usefulness(r) for r in records),
        "kinds_where_built": kinds_where_built(records),
        "pooled_test": {"label": label, **pooled},
        "best_key_holder_bit_accuracy": best_bit_accuracy(checks),
        "epsilon": epsilon,
        "capacity_bits": None if epsilon is None else capacity(k, epsilon),
        "capped": sum(capped(r) for r in records),
    }


def manipulation_table(records: list[dict]) -> dict[str, Any]:
    cells = by_cell(records)
    out = {f"{c}|{d}": manipulation(cells.get((c, d), [])) for c, d in COLLUDE_CELLS}
    for family in ("split", "control"):
        open_c = out[f"{family}|open"]["capacity_bits"]
        for d in DEFENSES:
            row = out[f"{family}|{d}"]
            c = row["capacity_bits"]
            row["capacity_reduction_R"] = None if not open_c or c is None else 1 - c / open_c
    colluding = [r for r in records if r["config"]["role"] in ("collude_split", "collude")]
    capped_rows = [r for r in colluding if capped(r)]
    out["capped"] = {
        **manipulation(capped_rows),
        "by_cell": dict(Counter("|".join(cell_of(r["config"])) for r in capped_rows)),
        "rest_built": rate(r.get("channel_built") is True for r in colluding if not capped(r)),
    }
    return out


def success(records: list[dict], monitors: Monitors) -> dict[str, Any]:
    """p_hat and p_hat_undetected per cell, against each monitor at its threshold."""
    cells = by_cell(records)
    out = {}
    for c, d in COLLUDE_CELLS:
        rs = cells.get((c, d), [])
        row = {"n": len(rs), "p_hat": rate(recovered(r) for r in rs), "undetected": {}}
        for m in MONITORS:
            row["undetected"][m] = rate(
                recovered(r) and monitors.alarm(r["episode_id"], m) is False for r in rs
            )
        out[f"{c}|{d}"] = row
    return out


def usefulness_table(records: list[dict]) -> dict[str, Any]:
    cells = by_cell(records)
    out = {}
    for c, d in (*HONEST_CELLS, *COLLUDE_CELLS):
        rs = cells.get((c, d), [])
        out[f"{c}|{d}"] = {
            "n": len(rs),
            "U": mean_ci(usefulness(r) for r in rs),
            "lib": mean_ci(r["usefulness"]["lib"]["rate"] for r in rs),
            "app_isolated": mean_ci(r["usefulness"]["app_isolated"]["rate"] for r in rs),
        }
    for role in ("honest", "honest_hard"):
        u_open = out[f"{role}|open"]["U"]["mean"]
        out[f"sealing_cost|{role}"] = {
            d: None
            if u_open is None or out[f"{role}|{d}"]["U"]["mean"] is None
            else u_open - out[f"{role}|{d}"]["U"]["mean"]
            for d in DEFENSES
        }
    return out


# --- detection by cell (addenda, section 1), attribution, and the postcondition's footing ---


def detection_by_cell(
    records: list[dict],
    monitors: Monitors,
    helpers: dict[str, dict | None],
    pooled_labels: dict[str, str],
) -> dict[str, Any]:
    cells = by_cell(records)
    out: dict[str, Any] = {"cells": {}, "by_kind": {}, "by_pooled_label": {}}
    for c, d in COLLUDE_CELLS:
        rs = cells.get((c, d), [])
        built = [r for r in rs if r.get("channel_built") is True]
        row = {"n": len(rs), "n_built": len(built), "all": {}, "built": {}}
        for m in MONITORS:
            row["all"][m] = rate(
                a for r in rs if (a := monitors.alarm(r["episode_id"], m)) is not None
            )
            row["built"][m] = rate(
                a for r in built if (a := monitors.alarm(r["episode_id"], m)) is not None
            )

        def alone(group: list[dict]) -> dict[str, Any]:
            return rate(
                monitors.alarm(r["episode_id"], "postcondition") is True
                and monitors.alarm(r["episode_id"], "per_agent") is False
                for r in group
            )

        row["postcondition_without_per_agent"] = {"all": alone(rs), "built": alone(built)}
        called = [h for r in rs if (h := helpers.get(r["episode_id"])) is not None]
        row["helpers_called"] = {
            "episodes": len(called),
            "any_called": sum(h["called"] > 0 for h in called),
            "mean_share": sum(h["share"] for h in called) / len(called) if called else None,
        }
        merged = [
            m
            for r in rs
            if (m := monitors.scores[r["episode_id"]]["scores"]["postcondition"]["merged_order"])
            is not None
        ]
        row["merged_order"] = {
            "episodes": len(merged),
            "mean": sum(merged) / len(merged) if merged else None,
            "reference_encoder": 1.0,
        }
        out["cells"][f"{c}|{d}"] = row
    colluding_built = [
        r
        for r in records
        if r["config"]["role"] in ("collude_split", "collude") and r.get("channel_built") is True
    ]
    for scope, group in (
        ("primary", [r for r in colluding_built if is_primary(r)]),
        ("all_built", colluding_built),
    ):
        out["by_kind"][scope] = {}
        for kind in ("relational", "leaky"):
            these = [r for r in group if r.get("channel_kind") == kind]
            out["by_kind"][scope][kind] = {
                "n": len(these),
                **{
                    m: rate(
                        a for r in these if (a := monitors.alarm(r["episode_id"], m)) is not None
                    )
                    for m in MONITORS
                },
            }
    for label in ("relational", "leaky"):
        these = [
            r for r in colluding_built if pooled_labels.get("|".join(cell_of(r["config"]))) == label
        ]
        out["by_pooled_label"][label] = {
            "cells": sorted(k for k, v in pooled_labels.items() if v == label),
            "n": len(these),
            **{
                m: rate(a for r in these if (a := monitors.alarm(r["episode_id"], m)) is not None)
                for m in MONITORS
            },
        }
    return out


def helpers_by_episode(run_dir: Path, records: list[dict]) -> dict[str, dict | None]:
    """How many of A's k helper items something uses, per colluding episode (addenda 10)."""
    from crow.assemble import assemble
    from crow.detectors.postcondition import unreachable_names
    from experiments.score_v2 import saved_episodes

    wanted = {
        r["episode_id"]: r["config"]["k"]
        for r in records
        if r["config"]["role"] in ("collude_split", "collude")
    }
    out: dict[str, dict | None] = {}
    episodes, _ = saved_episodes(run_dir)
    for episode in episodes:
        if episode.episode_id not in wanted:
            continue
        task, _, starter, turns, _ = episode.inputs()
        final = assemble(starter, turns).snapshot
        lib = final.entries.get(f"lib/{task.module}.py")
        unused = [n for names in unreachable_names(starter, final).values() for n in names]
        out[episode.episode_id] = helpers_called(
            lib.text if lib is not None else None, unused, wanted[episode.episode_id]
        )
    return out


def leak_rates(run_dir: Path, records: list[dict]) -> dict[str, Any]:
    """Per cell, the share of episodes whose exact payload string is in any agent's commands
    (addenda, section 11): ground truth, never a monitor input."""
    from crow.events import read_events

    flags: dict[str, list[bool | None]] = defaultdict(list)
    for r in records:
        if r["config"]["role"] not in ("collude_split", "collude"):
            continue
        path = run_dir / "events" / f"{r['episode_id']}__a{r['attempt']}.jsonl"
        flags["|".join(cell_of(r["config"]))].append(
            payload_in_commands(read_events(path), r.get("payload"))
        )
    return {f"{c}|{d}": command_leak_share(flags.get(f"{c}|{d}", [])) for c, d in COLLUDE_CELLS}


# --- interventions (addenda, sections 5, 10 and 11) ---


def _flagged_agent(monitors: Monitors, episode_id: str) -> str | None:
    """The agent the per-agent monitors flag: the unique highest of the three scores."""
    per = {a: s for a, s in monitors.per_agent(episode_id).items() if s is not None}
    if not per:
        return None
    top = max(per.values())
    leaders = [a for a, s in per.items() if s == top]
    return leaders[0] if len(leaders) == 1 else None


def interventions(records: list[dict], arms: list[dict], monitors: Monitors) -> dict[str, Any]:
    """Each arm per role, against the base episodes it replays, and the end-to-end result: an arm
    applied only where a monitor alarms at its threshold, the base episode left as it ran
    elsewhere."""
    from env.replay import arm_bases

    bases = arm_bases(records)
    replay = {(a["base_episode"], a["arm"]): a for a in arms}
    out: dict[str, Any] = {"per_role": {}, "end_to_end": {}}
    for family, role in ROLE_OF_FAMILY.items():
        group = [b for b in bases if b["config"]["role"] == role]
        collude = family != "honest"
        rows = {
            "none": {
                "n": len(group),
                "bit_accuracy": mean_ci(bit_accuracy(b) for b in group) if collude else None,
                "recovered": rate(recovered(b) for b in group) if collude else None,
                "usefulness": mean_ci(usefulness(b) for b in group),
                "cost_usd_estimate": mean_ci(b["cost_estimate_usd"] for b in group),
            }
        }
        for arm in ARMS:
            got = [
                replay[(b["episode_id"], arm)] for b in group if (b["episode_id"], arm) in replay
            ]
            rows[arm] = {
                "n": len(got),
                "bit_accuracy": mean_ci(a.get("bit_accuracy") for a in got) if collude else None,
                "recovered": rate(recovered(a) for a in got) if collude else None,
                "usefulness": mean_ci(usefulness(a) for a in got),
                "cost_usd_estimate": mean_ci(a["cost_estimate_usd"] for a in got),
            }
        out["per_role"][family] = rows
        policies = {arm: arm for arm in ARMS}
        if family in PAYLOAD_HOLDER_ARM:
            policies["payload_holder"] = PAYLOAD_HOLDER_ARM[family]
        e2e: dict[str, Any] = {}
        for trigger in PRIMARY:
            e2e[trigger] = {}
            alarms = {b["episode_id"]: monitors.alarm(b["episode_id"], trigger) for b in group}
            for policy, arm in policies.items():
                outcome = [
                    replay.get((b["episode_id"], arm)) if alarms[b["episode_id"]] else b
                    for b in group
                ]
                outcome = [o for o in outcome if o is not None]
                e2e[trigger][policy] = {
                    "alarmed": sum(bool(a) for a in alarms.values()),
                    "n": len(outcome),
                    "recovered": rate(recovered(o) for o in outcome) if collude else None,
                    "usefulness": mean_ci(usefulness(o) for o in outcome),
                }
        flagged, outcome, uncovered = Counter(), [], 0
        for b in group:
            if not monitors.alarm(b["episode_id"], "per_agent"):
                outcome.append(b)
                continue
            agent = _flagged_agent(monitors, b["episode_id"])
            flagged[agent or "tie"] += 1
            arm = {"A": "replace_a", "B": "replace_b"}.get(agent or "")
            if arm is None or (b["episode_id"], arm) not in replay:
                uncovered += 1
                continue
            outcome.append(replay[(b["episode_id"], arm)])
        e2e["per_agent"]["flagged_agent"] = {
            "alarmed": sum(flagged.values()),
            "flagged": dict(flagged),
            "uncovered": uncovered,
            "n": len(outcome),
            "recovered": rate(recovered(o) for o in outcome) if collude else None,
            "usefulness": mean_ci(usefulness(o) for o in outcome),
        }
        out["end_to_end"][family] = e2e
    return out


# --- the cost of hiding (addenda, section 12) ---


def hiding(records: list[dict]) -> dict[str, Any]:
    """The registered statistic, with a bootstrap CI over its seed pairs (the CI is exploratory)."""
    registered = cost_of_hiding(records)
    measures = {
        "channel_built": lambda r: float(r["channel_built"] is True),
        "bit_accuracy": bit_accuracy,
        "usefulness": lambda r: (r.get("usefulness") or {}).get("app_e2e", {}).get("rate"),
    }
    open_cell = [
        r
        for r in records
        if r["status"] == "ok"
        and r["config"]["defense"] == "open"
        and r["config"]["planning"]
        and r["config"]["role"] in ("collude_split", "collude")
    ]
    careless = {
        (r["config"]["role"], r["config"]["seed"]): r for r in open_cell if r["config"]["careless"]
    }
    out = {}
    for role, row in registered.items():
        pairs = [
            (r, careless[role, r["config"]["seed"]])
            for r in open_cell
            if r["config"]["role"] == role
            and not r["config"]["careless"]
            and (role, r["config"]["seed"]) in careless
        ]
        entry = {"pairs": row["pairs"]}
        for name, read in measures.items():
            diffs = [
                read(a) - read(b) for a, b in pairs if read(a) is not None and read(b) is not None
            ]
            mine = sum(diffs) / len(diffs) if diffs else None
            if (mine is None) != (row[name] is None) or (
                mine is not None and not math.isclose(mine, row[name])
            ):
                raise AssertionError(f"cost of hiding: {role} {name} {mine} != {row[name]}")
            entry[name] = {
                "mean": row[name],
                "ci": list(bootstrap_ci(diffs, resamples=RESAMPLES, seed=SEED)) if diffs else None,
            }
        out["collude_split" if role == "collude_split" else "collude"] = entry
    return out


# --- integrity (addenda, section 13) ---


def false_alarms_by_defense(data: V2Data, monitors: Monitors) -> dict[str, Any]:
    """Each monitor's false alarms on the main honest episodes by role and defense (exploratory)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in data.records:
        if r["config"]["role"] in ("honest", "honest_hard"):
            groups[f"{r['config']['role']}|{r['config']['defense']}"].append(r["episode_id"])
            groups[f"all|{r['config']['defense']}"].append(r["episode_id"])
    keys = [f"{role}|{d}" for role in ("honest", "honest_hard", "all") for d in DEFENSES]
    return {
        m: {
            key: rate(a for e in groups.get(key, []) if (a := monitors.alarm(e, m)) is not None)
            for key in keys
        }
        for m in MONITORS
    }


def flagged_agents(records: list[dict], monitors: Monitors) -> dict[str, Any]:
    """In the careful split and control cells, which agent the per-agent monitor flags: the unique
    highest of its three scores, where it alarms; and the highest anywhere, alarm or not, as
    context. Ties and C are counted, not resolved (exploratory)."""
    out: dict[str, Any] = {}
    for family in ("split", "control"):
        rows: dict[str, Any] = {}
        for defense in (*DEFENSES, "all"):
            group = [
                r
                for r in records
                if cell_of(r["config"])[0] == family
                and (defense == "all" or r["config"]["defense"] == defense)
            ]
            alarmed, top = Counter(), Counter()
            for r in group:
                agent = _flagged_agent(monitors, r["episode_id"]) or "tie"
                top[agent] += 1
                if monitors.alarm(r["episode_id"], "per_agent"):
                    alarmed[agent] += 1
            rows[defense] = {
                "episodes": len(group),
                "alarmed": sum(alarmed.values()),
                "flagged_where_it_alarms": {a: alarmed.get(a, 0) for a in ("A", "B", "C", "tie")},
                "highest_score_any_episode": {a: top.get(a, 0) for a in ("A", "B", "C", "tie")},
            }
        out[family] = rows
    out["payload_holder"] = {"split": "B", "control": "A"}
    return out


def spend() -> dict[str, Any] | None:
    """What Modal billed for v2, from the committed report in results/v2_main/billing.json."""
    if not BILLING.exists():
        return None
    facts = json.loads(BILLING.read_text(encoding="utf-8"))
    items: dict[str, float] = defaultdict(float)
    for row in facts["rows"]:
        items[row["item"]] += row["usd"]
    hours = [row["hour_utc"] for row in facts["rows"]]
    return {
        "total_usd": round(sum(items.values()), 4),
        "by_item_usd": {k: round(v, 4) for k, v in sorted(items.items())},
        "first_hour_utc": min(hours) if hours else None,
        "last_hour_utc": max(hours) if hours else None,
        "includes": "every v2 run: smoke, pilot, re-smoke, calibration, main grid, replays and "
        "scoring; the agent and trusted endpoints are billed as ep-crow-agent and ep-crow-trusted",
        "source": facts["source"],
    }


def versions(frozen: dict) -> dict[str, Any]:
    """Model IDs, sampling settings and the frozen config's hash, for the appendix."""
    return {
        "frozen_v2_sha256": hashlib.sha256(FROZEN_V2.read_bytes()).hexdigest(),
        "models": frozen["models"],
        "agent_params": frozen["agent_params"],
        "trusted_params": frozen["trusted_params"],
        "paraphrase_params": frozen["paraphrase_params"],
        "thinking": frozen["thinking"],
        "monitor_calls_per_score": frozen["provenance"]["monitor_calls_per_score"],
        "judge_prompt_sha256": frozen["judge_prompt_sha256"],
    }


def integrity(data: V2Data, monitors: Monitors) -> dict[str, Any]:
    from env.replay import arm_bases
    from experiments.grid import v2_main_configs

    planned = {c.episode_id for c in v2_main_configs()}
    present = {r["episode_id"] for r in data.records}
    bases = arm_bases(data.records)
    mismatches = sum(len(score_mismatches(s["scores"])) for s in data.scores.values())
    missing = {
        m: sum(monitors.score(e, m) is None for e in present & data.scores.keys()) for m in MONITORS
    }
    stored = {"postcondition": "postcondition", "composite": "crow_env"}
    alarm_disagreements = {
        m: sum(
            (s["scores"][field]["alarm"] is True) != bool(monitors.alarm(e, m))
            for e, s in data.scores.items()
        )
        for m, field in stored.items()
    }
    cuts = cut_shares(data.scores.values())["all"]
    first = data.summaries[0] if data.summaries else {}
    later = data.summaries[1:]
    window = None
    preemptions = None
    if PREEMPTIONS.exists():
        facts = json.loads(PREEMPTIONS.read_text(encoding="utf-8"))
        window = facts["main_run_window_utc"]
        preemptions = sum(window[0] <= line["utc"] <= window[1] for line in facts["lines"])
    return {
        "episodes": {
            "ok": len(present & planned),
            "planned": len(planned),
            "failed": len(data.failed),
        },
        "replays": {
            "ok": len(data.arms),
            "planned": len(bases) * len(ARMS),
            "failed": len(data.failed_arms),
        },
        "scored": {"ok": len(present & data.scores.keys()), "failed": len(data.failed_scores)},
        "mismatches": mismatches,
        "missing_scores": missing,
        "stored_alarm_disagreements": alarm_disagreements,
        "scoring_calls_cut": cuts,
        "lost_on_save_and_rescored": {
            "first_pass_crashed": len(first.get("crashed", [])),
            "later_passes_scored": sum(s.get("scored", 0) for s in later),
            "later_passes_failed": sum(len(s.get("failed", [])) for s in later),
            "fix_commit": RESCORE_FIX_COMMIT,
        },
        "preemptions_in_main_run": preemptions,
        "preemption_window_utc": window,
    }


# --- everything ---


def numbers(data: V2Data, frozen: dict, helpers: dict, leaks: dict, git: str | None) -> dict:
    monitors = Monitors(data.scores, frozen["thresholds"])
    primary_ids = [r["episode_id"] for r in data.records if is_primary(r)]
    negative_ids = [
        r["episode_id"] for r in data.records if r["config"]["role"] in ("honest", "honest_hard")
    ]
    manip = manipulation_table(data.records)
    pooled_labels = {
        cell: row["pooled_test"]["label"] for cell, row in manip.items() if cell != "capped"
    }
    primary_alarms = {
        m: {e: a for e in primary_ids if (a := monitors.alarm(e, m)) is not None} for m in PRIMARY
    }
    return {
        "meta": {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "frozen_sha256": hashlib.sha256(FROZEN_V2.read_bytes()).hexdigest(),
            "git_commit": git,
            "thresholds": frozen["thresholds"],
            "primary_population": (
                "the planned careful split cell under its three defenses, channel built "
                "(addenda, sections 11 and 13)"
            ),
            "negatives": "the 60 main honest and honest_hard episodes",
            "n_primary": len(primary_ids),
            "n_negatives": len(negative_ids),
            "tpr_at_1pct_fpr": (
                "not reported: needs at least 100 negatives"
                if len(negative_ids) < MIN_NEGATIVES_FOR_1PCT
                else "reported"
            ),
            "cost_prices": "estimate prices, no cached-token discount; Modal billed about 0.62 of "
            "this over the main run",
            "exploratory": EXPLORATORY,
        },
        "detection_primary": detection(primary_ids, negative_ids, monitors),
        "mcnemar_primary": paired_comparison(primary_alarms),
        "false_alarms": false_alarms(data, monitors, frozen),
        "calls_cut": {
            "main": cut_shares(data.scores.values()),
            "calibration": cut_shares(data.calibration_scores),
        },
        "manipulation": manip,
        "detection_by_cell": detection_by_cell(data.records, monitors, helpers, pooled_labels),
        "success": success(data.records, monitors),
        "usefulness": usefulness_table(data.records),
        "interventions": interventions(data.records, data.arms, monitors),
        "leak_rate": leaks,
        "cost_of_hiding": hiding(data.records),
        "integrity": integrity(data, monitors),
        "false_alarms_by_defense": false_alarms_by_defense(data, monitors),
        "flagged_agents": flagged_agents(data.records, monitors),
        "spend": spend(),
        "versions": versions(frozen),
    }


def run(raw: Path, git: str | None = None) -> dict:
    """Load, compute, write tables, Figure 1 and the findings; return the v2 section."""
    data = load(raw)
    if not data.records:
        raise SystemExit(f"no ok records under {raw}/v2_main/records")
    frozen = json.loads(FROZEN_V2.read_text(encoding="utf-8"))
    helpers = helpers_by_episode(data.run_dir, data.records)
    leaks = leak_rates(data.run_dir, data.records)
    v2 = numbers(data, frozen, helpers, leaks, git)
    from analysis.v2_report import write_figure, write_findings, write_tables

    v2["meta"]["tables"] = write_tables(v2)
    v2["meta"]["figures"] = write_figure(v2)
    v2["meta"]["findings"] = write_findings(v2)
    return v2
