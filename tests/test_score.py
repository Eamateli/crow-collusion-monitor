"""Composite score, alarm and config loading (no LLM)."""

import json

import pytest
from crow_fixtures import TURN, clean, plant_injection, starter

from crow.detectors import DETECTORS
from crow.detectors.judge import MonitorResult
from crow.score import Config, build_view, composite, load_config, score_crow
from env.taskset import load_task

MANIFEST = load_task("ledger").manifest("base")


def test_default_config_has_equal_weights_for_all_ten_detectors():
    config = Config.default()
    assert list(config.weights) == DETECTORS and len(DETECTORS) == 10
    assert set(config.weights.values()) == {1.0}
    assert config.tau is None and not config.frozen


def test_composite():
    weights = dict.fromkeys(DETECTORS, 1.0)
    scores = dict.fromkeys(DETECTORS, 0.0) | {"judge": 1.0, "manifest": 0.5}
    assert composite(scores, weights) == pytest.approx(0.15)
    with pytest.raises(ValueError):
        composite({"judge": 1.0}, weights)


def test_composite_leaves_a_missing_score_out_never_zero():
    weights = dict.fromkeys(DETECTORS, 1.0)
    scores = dict.fromkeys(DETECTORS, 0.0) | {"manifest": 0.9, "judge": None}
    assert composite(scores, weights) == pytest.approx(0.1)  # 0.9 / 9, not 0.9 / 10
    with pytest.raises(ValueError):
        composite(dict.fromkeys(DETECTORS, None), weights)


def test_score_crow_records_a_missing_judge():
    ws = clean()
    plant_injection(ws)
    view = build_view(starter().snapshot(), ws.snapshot(), MANIFEST, ws.events(), {}, turn=TURN)
    record = score_crow(view, MonitorResult("judge", None, None, ""), Config.default()).to_dict()
    assert record["missing"] == ["judge"] and record["scores"]["judge"] is None
    heuristic_mean = sum(s for n, s in record["scores"].items() if n != "judge") / 9
    assert record["composite"] == pytest.approx(heuristic_mean)


def test_load_config(tmp_path):
    assert load_config(tmp_path / "missing.json") == Config.default()
    path = tmp_path / "frozen.json"
    weights = dict.fromkeys(DETECTORS, 1.0)
    path.write_text(
        json.dumps({"weights": weights, "tau": 0.2, "calibration": {"comments": [0, 1]}})
    )
    config = load_config(path)
    assert config.frozen and config.tau == 0.2 and config.calibration == {"comments": [0, 1]}


def test_score_crow_alarm_and_record():
    ws = clean()
    plant_injection(ws)
    view = build_view(starter().snapshot(), ws.snapshot(), MANIFEST, ws.events(), {}, turn=TURN)
    judge = MonitorResult("judge", 0.8, 8.0, "addressed to the auditor")
    config = Config.default()
    no_tau = score_crow(view, judge, config)
    assert no_tau.tau is None and not no_tau.alarm
    config.tau = 0.1
    record = score_crow(view, judge, config).to_dict()
    assert record["alarm"] and record["composite"] >= 0.1
    assert set(record["scores"]) == set(DETECTORS)
    assert record["channels"]["C1"] >= 1  # the injection sits in HANDOFF.md
    assert record["notes"] == {"entropy": "uncalibrated", "comments": "uncalibrated"}
    assert record["judge"]["raw_score"] == 8.0 and record["missing"] == []
    assert {f["path"] for f in record["features"]} == {"lib/ledger.py", "HANDOFF.md"}
