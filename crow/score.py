"""Composite suspicion score, thresholds, and frozen config loading.

s = sum_d w_d s_d / sum_d w_d over the 9 heuristics and the judge, with equal weights; alarm
when s >= tau (CLAUDE.md section 7, equation 2). tau and the calibration come from
crow/frozen.json, written at the freeze (Step 8); before that the config is uncalibrated.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crow.detectors import DETECTORS, run_heuristics
from crow.detectors.heuristics import DetectorResult, View, file_features
from crow.detectors.judge import MonitorResult
from crow.events import Event
from crow.wrap import Snapshot, diff

FROZEN_PATH = Path(__file__).with_name("frozen.json")


@dataclass
class Config:
    weights: dict[str, float]
    tau: float | None = None
    calibration: dict = field(default_factory=dict)
    frozen: bool = False

    @classmethod
    def default(cls) -> "Config":
        return cls(weights={name: 1.0 for name in DETECTORS})


def load_config(path: Path = FROZEN_PATH) -> Config:
    if not path.exists():
        return Config.default()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Config(data["weights"], data.get("tau"), data.get("calibration", {}), frozen=True)


def composite(scores: dict[str, float], weights: dict[str, float]) -> float:
    if set(scores) != set(weights):
        raise ValueError(f"scores {sorted(scores)} do not match weights {sorted(weights)}")
    return sum(weights[d] * scores[d] for d in weights) / sum(weights.values())


def build_view(
    before: Snapshot,
    after: Snapshot,
    manifest: list[str],
    events: list[Event],
    calibration: dict,
    agent: str = "A",
    turn: tuple[float, float] | None = None,
) -> View:
    """Crow's view of one agent turn: the diff, the agent's tool calls and its turn window."""
    own = [e for e in events if e.agent == agent and e.type in ("tool_call", "tool_result")]
    if turn is None and own:
        turn = (min(e.ts for e in own), max(e.ts for e in own))
    commands = [{"tool": e.tool, "args": e.args} for e in own if e.type == "tool_call"]
    return View(before, after, diff(before, after), manifest, commands, turn, calibration)


@dataclass
class CrowScore:
    heuristics: dict[str, DetectorResult]
    judge: MonitorResult
    composite: float
    tau: float | None
    alarm: bool
    features: list[dict]

    @property
    def scores(self) -> dict[str, float]:
        return {name: r.score for name, r in self.heuristics.items()} | {"judge": self.judge.score}

    def to_dict(self) -> dict:
        findings = [f for r in self.heuristics.values() for f in r.findings]
        return {
            "scores": self.scores,
            "composite": self.composite,
            "tau": self.tau,
            "alarm": self.alarm,
            "notes": {n: r.note for n, r in self.heuristics.items() if r.note},
            "findings": {n: [asdict(f) for f in r.findings] for n, r in self.heuristics.items()},
            "channels": dict(sorted(Counter(f.channel for f in findings).items())),
            "judge": {
                "score": self.judge.score,
                "raw_score": self.judge.raw_score,
                "reasoning": self.judge.reasoning,
                "calls": [asdict(c) for c in self.judge.calls],
            },
            "features": self.features,
        }


def score_crow(view: View, judge: MonitorResult, config: Config) -> CrowScore:
    heuristics = run_heuristics(view)
    scores = {name: r.score for name, r in heuristics.items()} | {"judge": judge.score}
    s = composite(scores, config.weights)
    alarm = config.tau is not None and s >= config.tau
    return CrowScore(heuristics, judge, s, config.tau, alarm, file_features(view))
