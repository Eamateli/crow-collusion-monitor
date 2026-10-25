"""Calibration math (no Modal, no endpoint): fitting mu/sigma and the judge repeatability stats."""

import pytest

from crow.llm import LLMCall
from experiments.calibrate import fit_calibration, judge_repeatability


class FakeEpisode:
    def __init__(self, eid, features=()):
        self.id = eid
        self.record = {"scores": {"crow": {"features": list(features)}}}

    def monitor_views(self):
        return {"judge": f"view of {self.id}"}


def feature(type_, entropy=None, density=None):
    return {"type": type_, "entropy": entropy, "comment_density": density}


def test_fit_calibration_per_type_sample_std_and_drops_thin_types():
    eps = [
        FakeEpisode("a", [feature("py", 4.0, 0.2), feature("md", 5.0), feature("data", 6.0)]),
        FakeEpisode("b", [feature("py", 5.0, 0.4), feature("md", 5.0)]),
    ]
    cal = fit_calibration(eps)
    assert cal["entropy"]["py"] == pytest.approx([4.5, 0.7071067811865476])
    assert cal["entropy"]["md"] == [5.0, 0.0]  # sigma 0: the detector then skips this type
    assert "data" not in cal["entropy"]  # one sample only
    assert cal["comments"] == pytest.approx([0.3, 0.1414213562373095])


class ScriptedClient:
    def __init__(self, scores):
        self.scores = scores

    def chat(self, messages, *, affinity_key, **params):
        eid = affinity_key.removeprefix("repeat-judge-")
        content = f"<reasoning>r</reasoning><score>{self.scores[eid]}</score>"
        return LLMCall(
            "t",
            "t",
            params,
            {"completion_tokens": 5},
            0,
            0,
            0.0,
            1,
            content,
            None,
            None,
            [],
            "stop",
        )


def test_judge_repeatability_stats():
    eps = [FakeEpisode(f"e{i}") for i in range(4)]
    first = {"e0": 0.0, "e1": 0.2, "e2": 0.5, "e3": None}  # e3 missing in the first scoring
    client = ScriptedClient({"e0": 0, "e1": 2, "e2": 3, "e3": 1})
    rep = judge_repeatability(eps, first, 4, client)
    assert rep["n"] == 3 and rep["missing"] == 1
    assert rep["mean_abs_diff"] == pytest.approx((0 + 0 + 0.2) / 3)
    assert rep["exact_agreement"] == pytest.approx(2 / 3)
    assert rep["pearson_r"] is not None and rep["pearson_r"] > 0.9


def test_judge_repeatability_without_variance_has_no_pearson():
    eps = [FakeEpisode(f"e{i}") for i in range(3)]
    first = dict.fromkeys(("e0", "e1", "e2"), 0.0)
    rep = judge_repeatability(eps, first, 3, ScriptedClient(dict.fromkeys(first, 0)))
    assert (
        rep["pearson_r"] is None and rep["mean_abs_diff"] == 0.0 and rep["exact_agreement"] == 1.0
    )
