"""Config builder: the seed to task mapping, and the configs of each run.

The seed fixes the task and the payload, so the same seed gives the same task and payload in
every defense (CLAUDE.md section 5, matched pairs). Probe configs are Step 6b; calibration
(Step 8) and the main grid (Step 9) are added here in those steps.
"""

from env.episode import EpisodeConfig, make_config
from env.taskset import task_ids

PROBE_FIRST_SEED, PROBE_EPISODES = 2000, 12  # 2 per task; a new round starts after the last
PROBE_LIMIT = 3000  # smoke seeds start at 3000; probe seeds stay below


def task_for_seed(seed: int) -> str:
    """The (seed mod 6)-th task id in sorted order."""
    ids = task_ids()
    return ids[seed % len(ids)]


def probe_configs(first_seed: int = PROBE_FIRST_SEED) -> list[EpisodeConfig]:
    """Difficulty probe (Step 6b): honest, base version, defense open, 12 consecutive seeds."""
    seeds = range(first_seed, first_seed + PROBE_EPISODES)
    if first_seed < PROBE_FIRST_SEED or seeds[-1] >= PROBE_LIMIT:
        raise ValueError(f"probe seeds must lie in [{PROBE_FIRST_SEED}, {PROBE_LIMIT})")
    return [make_config("honest", task_for_seed(s), "open", seed=s) for s in seeds]
