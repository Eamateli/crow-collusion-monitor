"""Seed to task mapping and the probe configs (no Modal)."""

from collections import Counter

import pytest

from env.taskset import task_ids
from experiments.grid import probe_configs, task_for_seed


def test_task_for_seed_cycles_through_sorted_tasks():
    ids = task_ids()
    assert [task_for_seed(s) for s in range(6)] == ids
    assert task_for_seed(13) == ids[1]


def test_probe_gives_every_task_two_honest_open_base_episodes():
    configs = probe_configs()
    assert [c.seed for c in configs] == list(range(2000, 2012))
    assert Counter(c.task_id for c in configs) == dict.fromkeys(task_ids(), 2)
    assert {(c.role, c.variant, c.defense, c.k) for c in configs} == {("honest", "base", "open", 0)}
    assert len({c.episode_id for c in configs}) == 12


def test_probe_seeds_stay_in_their_range():
    assert probe_configs(2012)[0].seed == 2012  # a second round takes new seeds
    with pytest.raises(ValueError):
        probe_configs(1999)
    with pytest.raises(ValueError):
        probe_configs(2990)  # would reach the smoke seeds (3000+)


def test_calibration_configs():
    from experiments.grid import calibration_configs

    configs = calibration_configs()
    assert len(configs) == 30
    assert Counter(c.role for c in configs) == {"honest": 15, "honest_hard": 15}
    assert {c.seed for c in configs} == set(range(1000, 1015))
    assert {c.defense for c in configs} == {"open"}
    assert {(c.role, c.variant) for c in configs} == {("honest", "base"), ("honest_hard", "hard")}
    assert len({c.episode_id for c in configs}) == 30


def test_main_grid_is_320_episodes_ordered_by_seed():
    from experiments.grid import main_configs

    configs = main_configs()
    assert len(configs) == 320
    assert len({c.episode_id for c in configs}) == 320
    assert [c.seed for c in configs] == sorted(c.seed for c in configs)  # seed-major order
    assert {c.seed for c in configs} == set(range(20))
    collude = [c for c in configs if c.role == "collude" and c.planning]
    nochat = [c for c in configs if c.role == "collude" and not c.planning]
    assert len(collude) == 180 and len(nochat) == 20
    assert {(c.defense, c.k) for c in nochat} == {("open", 8)}
    assert Counter(c.role for c in configs if c.role != "collude") == {
        "honest": 60,
        "honest_hard": 60,
    }
    assert Counter((c.defense, c.k) for c in collude) == {
        (d, k): 20 for d in ("open", "seal_side", "seal_side_para") for k in (1, 8, 32)
    }


def test_main_grid_one_task_per_seed_and_matched_payloads():
    from env.episode import payload_bits
    from experiments.grid import main_configs

    by_seed = {}
    for c in main_configs():
        by_seed.setdefault(c.seed, set()).add(c.task_id)
        if c.role == "honest_hard":
            assert c.variant == "hard"
        elif c.role == "honest":
            assert c.variant == "base"
    assert all(len(tasks) == 1 for tasks in by_seed.values())  # same task for every cell of a seed
    # matched pairs: for a (seed, k), the payload is the same in every defense
    for c in main_configs()[:9]:
        assert payload_bits(c.seed, c.k) == payload_bits(0, c.k)
    # the 20 seeds spread over the 6 tasks as evenly as possible (4, 4, 3, 3, 3, 3)
    assert sorted(Counter(task_for_seed(s) for s in range(20)).values()) == [3, 3, 3, 3, 4, 4]
