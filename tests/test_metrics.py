"""Pure analysis metrics (no I/O)."""

import math

import numpy as np
import pytest

from analysis.metrics import (
    auroc,
    bit_error_rate,
    bootstrap_ci,
    capacity,
    cohen_kappa,
    h2,
    roc_curve,
    tpr_at_fpr,
)


def test_auroc_perfect_and_reversed_and_chance():
    assert auroc([1.0, 0.9], [0.1, 0.2]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 1.0]) == 0.0
    assert auroc([1.0, 0.0], [1.0, 0.0]) == 0.5  # all tied across groups
    assert auroc([], [0.1]) is None


def test_auroc_matches_brute_force_with_ties():
    pos, neg = [0.5, 0.5, 0.9], [0.1, 0.5, 0.7]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    assert auroc(pos, neg) == pytest.approx(wins / (len(pos) * len(neg)))


def test_tpr_at_fpr():
    pos, neg = [0.9, 0.8, 0.6], [0.5, 0.4, 0.3, 0.2]  # neg all below pos
    assert tpr_at_fpr(pos, neg, 0.0) == 1.0
    # one negative at 0.65 forces a higher threshold to keep FPR at 0
    assert tpr_at_fpr([0.9, 0.8, 0.6], [0.65, 0.4, 0.3, 0.2], 0.0) == pytest.approx(2 / 3)
    assert tpr_at_fpr([0.9, 0.8, 0.6], [0.65, 0.4, 0.3, 0.2], 0.25) == 1.0


def test_roc_curve_monotone_endpoints():
    fpr, tpr = roc_curve([0.9, 0.6], [0.5, 0.2])
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert fpr == sorted(fpr) and tpr == sorted(tpr)


def test_h2_and_capacity():
    assert h2(0.5) == pytest.approx(1.0)
    assert h2(0.0) == 0.0 and h2(1.0) == 0.0
    assert capacity(8, 0.0) == pytest.approx(8.0)  # noiseless: full k bits
    assert capacity(8, 0.5) == pytest.approx(0.0)
    assert capacity(32, 0.75) == 0.0  # epsilon > 0.5
    assert capacity(8, 0.11) == pytest.approx(8 * (1 - h2(0.11)))


def test_bit_error_rate_pools_over_the_cell():
    # a missing submission counts k*0.5 errors; here two 8-bit episodes, one perfect, one missing
    assert bit_error_rate([0.0, 4.0], [8, 8]) == pytest.approx(4.0 / 16)
    assert bit_error_rate([], []) == 0.0


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    values = [1.0] * 7 + [0.0] * 3  # p_hat = 0.7
    lo, hi = bootstrap_ci(values)
    assert lo <= 0.7 <= hi and 0.0 <= lo < hi <= 1.0
    assert bootstrap_ci(values) == bootstrap_ci(values)  # deterministic (seed 0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_cohen_kappa():
    assert cohen_kappa(["C1", "C2", "C3"], ["C1", "C2", "C3"]) == 1.0
    assert cohen_kappa(["C1"] * 4, ["C1"] * 4) == 1.0  # constant and identical
    half = cohen_kappa(["C1", "C1", "C2", "C2"], ["C1", "C2", "C1", "C2"])
    assert half == pytest.approx(0.0, abs=1e-9)  # observed == expected
    assert cohen_kappa(["C1"], ["C1"]) is None
    with pytest.raises(ValueError):
        cohen_kappa(["C1"], ["C1", "C2"])


def test_bootstrap_matches_numpy_percentile_shape():
    rng = np.random.default_rng(0)
    values = list(rng.random(20))
    lo, hi = bootstrap_ci(values)
    assert lo < np.mean(values) < hi


def test_mcnemar_exact():
    from analysis.metrics import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0  # no discordant pairs: no evidence either way
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(6, 0) == mcnemar_exact(0, 6)  # two-sided, so symmetric
    assert mcnemar_exact(3, 3) == 1.0  # capped at 1
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)


def test_holm_steps_down_and_stays_monotone():
    from analysis.metrics import holm

    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)  # 3 x 0.01
    assert adjusted["c"] == pytest.approx(0.06)  # 2 x 0.03
    assert adjusted["b"] == pytest.approx(0.06)  # 1 x 0.04, raised to stay monotone
    assert holm({"a": 0.9, "b": 0.8}) == {"b": 1.0, "a": 1.0}  # capped at 1
    assert holm({}) == {}


def test_paired_comparison_runs_all_six_pairs_among_four_monitors():
    from analysis.metrics import mcnemar_exact, paired_comparison

    episodes = [f"e{i}" for i in range(8)]
    alarms = {
        "crow": {e: True for e in episodes},
        "per_agent": {e: False for e in episodes},
        "pooled": {e: True for e in episodes},
        "postcondition": {e: i < 4 for i, e in enumerate(episodes)},
    }
    pairs = paired_comparison(alarms)
    assert len(pairs) == 6
    row = pairs["crow vs per_agent"]
    assert (row["episodes"], row["only_first"], row["only_second"]) == (8, 8, 0)
    assert row["p"] == pytest.approx(mcnemar_exact(8, 0))
    assert pairs["crow vs pooled"]["p"] == 1.0
    assert all(r["p_holm"] >= r["p"] for r in pairs.values())


def test_a_missing_score_leaves_an_episode_out_of_that_monitors_pairs_only():
    from analysis.metrics import paired_comparison

    alarms = {
        "crow": {"e1": True, "e2": True},
        "pooled": {"e1": False},  # no score on e2
        "postcondition": {"e1": True, "e2": False},
    }
    pairs = paired_comparison(alarms)
    assert pairs["crow vs pooled"]["episodes"] == 1
    assert pairs["crow vs postcondition"]["episodes"] == 2
