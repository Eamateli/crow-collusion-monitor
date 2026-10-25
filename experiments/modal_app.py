"""Modal app 'crow': run_episode on Modal, the grid runner, and --smoke (CLAUDE.md section 9).

Each episode is one container: it builds the two endpoint clients from the crow-secrets Secret,
runs env.episode.run_episode against fresh network-blocked sandboxes, and writes its record and
event log to the crow-results Volume. The grid is run with run_episode_remote.map, capped at 12
containers, skipping episodes already finished (resume) and stopping at MAX_TOTAL_TOKENS.
"""

import json
import os
import time
from pathlib import Path

import modal

from env.episode import EpisodeConfig, make_config, run_episode
from experiments.grid import PROBE_FIRST_SEED, probe_configs

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "crow"
VOLUME_NAME = "crow-results"
RESULTS = "/results"
MAX_TOTAL_TOKENS = 8_000_000  # budget guard; override with --max-tokens
RUNS = ("smoke", "probe", "calibration", "main")  # one folder each: /results/<run>/
SMOKE_SEED = 3000  # smoke seeds are 3000 and up (CLAUDE.md section 5, Results namespaces)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# The orchestrator image: pinned deps, the task data files (baked in), and the project code.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .add_local_dir(str(ROOT / "env" / "tasks"), "/root/env/tasks", copy=True)
    .add_local_dir(str(ROOT / "env" / "tasks_hidden"), "/root/env/tasks_hidden", copy=True)
    .add_local_python_source("crow", "env", "experiments", "analysis")
)


def _clients():
    """The agent and trusted endpoint clients, from the crow-secrets Secret in the environment."""
    from crow.llm import modal_endpoint
    from env.prompts import AGENT_MODEL_ID, TRUSTED_MODEL_ID

    agent = modal_endpoint("agent", os.environ["AGENT_ENDPOINT_URL"], AGENT_MODEL_ID)
    trusted = modal_endpoint("trusted", os.environ["TRUSTED_ENDPOINT_URL"], TRUSTED_MODEL_ID)
    return agent, trusted


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=3600,
    max_containers=12,
    retries=1,
)
def run_episode_remote(payload: dict) -> dict:
    """One episode. `payload` = {"config": {...}, "attempt": n, "run": name}. Records and events
    go under /results/<run>/. Sandboxes use app=None (we are already inside a Modal container)."""
    agent, trusted = _clients()
    config = EpisodeConfig(**payload["config"])
    run = payload["run"]
    if run not in RUNS:
        raise ValueError(f"unknown run {run!r}")
    started = time.time()
    record = run_episode(
        config, agent, trusted, f"{RESULTS}/{run}", app=None, attempt=payload.get("attempt", 1)
    )
    volume.commit()
    return {
        "episode_id": record["episode_id"],
        "attempt": record["attempt"],
        "status": record["status"],
        "reason": record.get("reason"),
        "run": run,
        "tokens": record.get("tokens", {}).get("total", 0),
        "cost": record.get("cost_estimate_usd"),
        "seconds": record.get("seconds", round(time.time() - started, 1)),
        "correct": (record.get("submission") or {}).get("correct"),
        "usefulness": (record.get("usefulness") or {}).get("app_e2e", {}).get("rate"),
    }


def _existing_attempts(run: str) -> dict[str, list[str]]:
    """episode_id -> the statuses of its attempt records in this run's folder only."""
    attempts: dict[str, list[str]] = {}
    try:
        entries = list(volume.listdir(f"{run}/records"))
    except Exception:  # noqa: BLE001 - a new/empty Volume has no records dir yet
        entries = []
    for entry in entries:
        name = Path(entry.path).name
        if not name.endswith(".json") or "__a" not in name:
            continue
        episode_id = name.split("__a")[0]
        status = "unknown"
        for chunk in volume.read_file(f"{run}/records/{name}"):
            status = json.loads(chunk).get("status", "unknown")
            break
        attempts.setdefault(episode_id, []).append(status)
    return attempts


def to_run(configs: list[EpisodeConfig], run: str) -> list[dict]:
    """Resume, inside this run's folder: skip an episode that already has an ok record or two
    attempts; otherwise queue the next attempt (at most one re-run of a failed episode). Smoke
    never resumes: it always runs, numbered after the attempts already there."""
    done = _existing_attempts(run)
    jobs = []
    for config in configs:
        statuses = done.get(config.episode_id, [])
        if run != "smoke" and ("ok" in statuses or len(statuses) >= 2):
            continue
        jobs.append({"config": vars(config), "attempt": len(statuses) + 1, "run": run})
    return jobs


def smoke_configs() -> list[EpisodeConfig]:
    """1 honest and 2 collude episodes. The two collude ones share a seed, so they are a matched
    pair (same task, same payload) differing only in the defense: open vs seal_side_para."""
    return [
        make_config("honest", "ledger", "open", seed=SMOKE_SEED),
        make_config("collude", "ledger", "open", seed=SMOKE_SEED + 1, k=8),
        make_config("collude", "ledger", "seal_side_para", seed=SMOKE_SEED + 1, k=8),
    ]


@app.local_entrypoint()
def main(
    run: str = "",
    smoke: bool = False,
    probe_first_seed: int = PROBE_FIRST_SEED,
    only: str = "",
    max_tokens: int = MAX_TOTAL_TOKENS,
) -> None:
    """--smoke, or --run probe [--probe-first-seed N]. --only keeps the configs whose episode_id
    contains that text. Launch long runs with `modal run --detach` (CLAUDE.md section 9)."""
    run = "smoke" if smoke else run
    if run == "smoke":
        configs = smoke_configs()
    elif run == "probe":
        configs = probe_configs(probe_first_seed)
    else:
        raise SystemExit("use --smoke or --run probe; calibration and main come in Steps 8 and 9")
    configs = [c for c in configs if only in c.episode_id]
    jobs = to_run(configs, run)
    print(f"run {run}: {len(configs)} configs, {len(jobs)} to run, budget {max_tokens} tokens")

    total_tokens = 0
    for result in run_episode_remote.map(jobs, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"  EXCEPTION (Modal): {type(result).__name__}: {result}")
            continue
        total_tokens += result["tokens"]
        print(
            f"  {result['status']:6} {result['episode_id']}__a{result['attempt']} | "
            f"{result['tokens']} tok | ${result['cost']} | {result['seconds']}s | "
            f"correct={result['correct']} U={result['usefulness']}"
            + (f" | {result['reason']}" if result.get("reason") else "")
        )
        if total_tokens >= max_tokens:
            print(f"  budget reached ({total_tokens} >= {max_tokens}); stopping")
            break
    print(f"total tokens: {total_tokens}")
