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
