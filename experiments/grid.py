"""Config builder: the seed to task mapping, and the configs of each run.

The seed fixes the task and the payload, so the same seed gives the same task and payload in
every defense (CLAUDE.md section 5, matched pairs). Probe configs are Step 6b; calibration
(Step 8) and the main grid (Step 9) are added here in those steps. The v2_* configs are the
3-agent harness (Crow v2 spec, section 5). There the seed also fixes the pairing and the per-slot
relation of collude_split (env.relational.scheme_for, from seed and k), so those match across
defenses too.
"""

from env.episode import EpisodeConfig, make_config
from env.taskset import task_ids

MAIN_SEEDS = range(0, 20)  # the 20-seed upgrade (CLAUDE.md section 5)
DEFENSES = ("open", "seal_side", "seal_side_para")
PAYLOAD_BITS = (1, 8, 32)
CALIBRATION_SEEDS = range(1000, 1015)  # CLAUDE.md section 8; never overlaps the main seeds
PROBE_FIRST_SEED, PROBE_EPISODES = 2000, 12  # 2 per task; a new round starts after the last
PROBE_LIMIT = 3000  # smoke seeds start at 3000; probe seeds stay below
V2_MAIN_SEEDS = range(0, 12)  # addenda, section 11; drop to 10 before cutting any cell
V2_K = 8
V2_CALIBRATION_SEEDS = range(1100, 1115)  # never overlaps the main, v1 calibration or probe seeds
# One per task. Smoke seeds are never reused: the first v2 smoke took 3100 to 3105, with the prompts
# that are now the careless contrast.
V2_RESMOKE_SEEDS = range(3106, 3112)


def task_for_seed(seed: int) -> str:
    """The (seed mod 6)-th task id in sorted order."""
    ids = task_ids()
    return ids[seed % len(ids)]


def main_configs(seeds: range = MAIN_SEEDS) -> list[EpisodeConfig]:
    """The main grid (Step 9, CLAUDE.md section 5), ordered by seed: all 16 cells of seed 0, then
    of seed 1, and so on, so a run stopped early still leaves complete matched sets. Per seed: the
    same task (task_for_seed) for every role and defense; honest_hard uses its hard version."""
    configs = []
    for seed in seeds:
        task = task_for_seed(seed)
        for defense in DEFENSES:
            for k in PAYLOAD_BITS:
                configs.append(make_config("collude", task, defense, seed=seed, k=k))
        configs.append(make_config("collude", task, "open", seed=seed, k=8, planning=False))
        for role in ("honest", "honest_hard"):
            for defense in DEFENSES:
                configs.append(make_config(role, task, defense, seed=seed))
    return configs


def calibration_configs() -> list[EpisodeConfig]:
    """Calibration (Step 8): 15 honest + 15 honest_hard, defense open. Each seed gives one honest
    and one honest_hard episode on the same task (base and hard version)."""
    configs = []
    for seed in CALIBRATION_SEEDS:
        task = task_for_seed(seed)
        configs.append(make_config("honest", task, "open", seed=seed))
        configs.append(make_config("honest_hard", task, "open", seed=seed))
    return configs


def probe_configs(first_seed: int = PROBE_FIRST_SEED) -> list[EpisodeConfig]:
    """Difficulty probe (Step 6b): honest, base version, defense open, 12 consecutive seeds."""
    seeds = range(first_seed, first_seed + PROBE_EPISODES)
    if first_seed < PROBE_FIRST_SEED or seeds[-1] >= PROBE_LIMIT:
        raise ValueError(f"probe seeds must lie in [{PROBE_FIRST_SEED}, {PROBE_LIMIT})")
    return [make_config("honest", task_for_seed(s), "open", seed=s) for s in seeds]


def v2_main_configs(seeds: range = V2_MAIN_SEEDS, k: int = V2_K) -> list[EpisodeConfig]:
    """The 3-agent main grid, ordered by seed so a run stopped early leaves complete matched sets:
    collude_split and the single-carrier control under all 3 defenses, collude_split with no
    planning chat under open, honest and honest_hard under all 3 defenses, and the careless contrast
    (split and control under open). 15 per seed."""
    configs = []
    for seed in seeds:
        task = task_for_seed(seed)
        for role in ("collude_split", "collude"):
            for defense in DEFENSES:
                configs.append(make_config(role, task, defense, seed=seed, k=k, agents=3))
        configs.append(
            make_config("collude_split", task, "open", seed=seed, k=k, planning=False, agents=3)
        )
        for role in ("honest", "honest_hard"):
            for defense in DEFENSES:
                configs.append(make_config(role, task, defense, seed=seed, agents=3))
        for role in ("collude_split", "collude"):
            configs.append(make_config(role, task, "open", seed=seed, k=k, agents=3, careless=True))
    return configs


def v2_calibration_configs() -> list[EpisodeConfig]:
    """3-agent calibration: 15 honest + 15 honest_hard, defense open."""
    configs = []
    for seed in V2_CALIBRATION_SEEDS:
        task = task_for_seed(seed)
        configs.append(make_config("honest", task, "open", seed=seed, agents=3))
        configs.append(make_config("honest_hard", task, "open", seed=seed, agents=3))
    return configs


def v2_resmoke_configs(seeds: range = V2_RESMOKE_SEEDS) -> list[EpisodeConfig]:
    """The careful prompts, before calibration (addenda, section 11): three splits, a control and
    two honest episodes, one on each task, under the open defense."""
    roles = ("collude_split", "collude", "collude_split", "honest", "collude_split", "honest")
    configs = []
    for seed, role in zip(seeds, roles, strict=True):
        k = V2_K if role != "honest" else 0
        configs.append(make_config(role, task_for_seed(seed), "open", seed=seed, k=k, agents=3))
    return configs
