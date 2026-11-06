"""Modal app 'crow': run_episode on Modal, the grid runner, and --smoke (CLAUDE.md section 9).

Each episode is one container: it builds the two endpoint clients from the crow-secrets Secret,
runs env.episode.run_episode against fresh network-blocked sandboxes, and writes its record and
event log to the crow-results Volume. The grid is run with run_episode_remote.map, capped at 12
containers, skipping episodes already finished (resume) and stopping at MAX_TOTAL_TOKENS.

A 3-agent run (v2_*) makes no monitor call in its episodes. One more container, score_run_remote,
follows the run and scores episodes as they finish, at 12 requests while they run and 16 after.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import modal

from env.episode import EpisodeConfig, make_config, run_episode
from env.replay import ARMS, arm_bases, arm_episode_id, run_arm
from experiments.grid import (
    PROBE_FIRST_SEED,
    calibration_configs,
    main_configs,
    probe_configs,
    v2_calibration_configs,
    v2_main_configs,
    v2_smoke_configs,
)
from experiments.score_v2 import DONE_MARKER, follow_run, run_finished, score_run
from experiments.score_v2 import MAX_TOTAL_TOKENS as MAX_SCORING_TOKENS

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "crow"
VOLUME_NAME = "crow-results"
RESULTS = "/results"
MAX_TOTAL_TOKENS = 8_000_000  # budget guard; override with --max-tokens
V2_GRIDS = {
    "v2_smoke": v2_smoke_configs,
    "v2_calibration": v2_calibration_configs,
    "v2_main": v2_main_configs,
}
V2_ARMS_RUN = "v2_intervention"
RUNS = ("smoke", "probe", "calibration", "main", *V2_GRIDS, V2_ARMS_RUN)  # /results/<run>/ each
# Modal allows up to 24 hours per attempt. A pass that dies resumes when launched again.
SCORING_TIMEOUT = 12 * 3600
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
# add_local_python_source ships only .py files, so the frozen config is added explicitly once it
# exists (after Step 8); without it the containers would score with the uncalibrated default.
FROZEN = ROOT / "crow" / "frozen.json"
if FROZEN.exists():
    image = image.add_local_file(str(FROZEN), "/root/crow/frozen.json")
FROZEN_V2 = ROOT / "crow" / "frozen_v2.json"
if FROZEN_V2.exists():
    image = image.add_local_file(str(FROZEN_V2), "/root/crow/frozen_v2.json")


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
    if run == "main":  # frozen-config guard, inside the container (CLAUDE.md section 8)
        from crow.score import load_config

        if not load_config().frozen:
            raise RuntimeError("main run refused: this container has no crow/frozen.json")
    if run == "v2_main":
        from crow.score import FROZEN_V2_PATH, load_config

        if not load_config(FROZEN_V2_PATH).frozen:
            raise RuntimeError("v2 main run refused: this container has no crow/frozen_v2.json")
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


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=SCORING_TIMEOUT,
    max_containers=1,
)
def score_run_remote(run: str, episode_ids: list[str] | None, max_tokens: int) -> dict:
    """The scoring pass in a container. With `episode_ids` it follows a grid as episodes finish;
    with None it scores what is on the Volume and stops. Reload and commit share a lock: a Volume's
    reload fails while a file is open, and follow_run tries a failed one again at the next poll."""
    _, trusted = _clients()
    run_dir = f"{RESULTS}/{run}"
    lock = threading.Lock()

    def refresh() -> None:
        with lock:
            volume.reload()

    def commit() -> None:
        with lock:
            volume.commit()

    refresh()
    if episode_ids is None:
        summary = score_run(run_dir, trusted, max_tokens=max_tokens)
    else:
        summary = follow_run(
            run_dir,
            trusted,
            lambda: run_finished(run_dir, episode_ids),
            max_tokens=max_tokens,
            refresh=refresh,
            commit=commit,
        )
    commit()
    results = summary.pop("results")
    summary["seconds_per_episode"] = sorted(r["seconds"] for r in results)
    summary["cost_per_episode"] = sorted(r.get("cost_estimate_usd", 0) for r in results)
    return summary


@app.function(image=image, volumes={RESULTS: volume}, timeout=120)
def set_episodes_done(run: str, done: bool) -> None:
    """Tell the scoring pass whether this run's episodes have all ended."""
    marker = Path(RESULTS) / run / DONE_MARKER
    if done:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    volume.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=3600,
    max_containers=12,
    retries=1,
)
def run_arm_remote(payload: dict) -> dict:
    """One intervention arm on one saved episode. Writes under /results/v2_intervention/."""
    agent, trusted = _clients()
    record = run_arm(
        f"{RESULTS}/{payload['base_run']}",
        payload["base_stem"],
        payload["arm"],
        agent,
        trusted,
        f"{RESULTS}/{V2_ARMS_RUN}",
        app=None,
        attempt=payload.get("attempt", 1),
    )
    volume.commit()
    return {
        "episode_id": record["episode_id"],
        "attempt": record["attempt"],
        "status": record["status"],
        "reason": record.get("reason"),
        "tokens": record.get("tokens", {}).get("total", 0),
        "cost": record.get("cost_estimate_usd"),
        "seconds": record.get("seconds"),
        "bit_accuracy": record.get("bit_accuracy"),
        "usefulness": (record.get("usefulness") or {}).get("app_e2e", {}).get("rate"),
    }


@app.function(image=image, timeout=120)
def frozen_info() -> dict:
    """Which crow config a container sees (checks that frozen.json reaches the grid)."""
    from crow.score import load_config

    config = load_config()
    return {"frozen": config.frozen, "tau": config.tau, "sha256": config.sha256}


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


def _records(run: str) -> list[dict]:
    """Every record of a run folder on the Volume."""
    records = []
    for entry in volume.listdir(f"{run}/records"):
        name = Path(entry.path).name
        if name.endswith(".json"):
            data = b"".join(volume.read_file(f"{run}/records/{name}"))
            records.append(json.loads(data))
    return records


def run_arms(base_run: str, only: str, arms: tuple[str, ...], max_tokens: int) -> None:
    """Every arm on every matching finished episode of `base_run` (addenda, section 10)."""
    bases = [r for r in arm_bases(_records(base_run)) if only in r["episode_id"]]
    done = _existing_attempts(V2_ARMS_RUN)
    jobs = []
    for base in bases:
        for arm in arms:
            statuses = done.get(arm_episode_id(base["episode_id"], arm), [])
            if "ok" in statuses or len(statuses) >= 2:
                continue
            jobs.append(
                {
                    "base_run": base_run,
                    "base_stem": f"{base['episode_id']}__a{base['attempt']}",
                    "arm": arm,
                    "attempt": len(statuses) + 1,
                }
            )
    print(f"arms on {base_run}: {len(bases)} episodes, {len(jobs)} replays to run")
    total_tokens = 0
    for result in run_arm_remote.map(jobs, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"  EXCEPTION (Modal): {type(result).__name__}: {result}")
            continue
        total_tokens += result["tokens"]
        print(
            f"  {result['status']:6} {result['episode_id']} | {result['tokens']} tok | "
            f"${result['cost']} | {result['seconds']}s | bits={result['bit_accuracy']} "
            f"U={result['usefulness']}" + (f" | {result['reason']}" if result.get("reason") else "")
        )
        if total_tokens >= max_tokens:
            print(f"  budget reached ({total_tokens} >= {max_tokens}); stopping")
            break
    print(f"total arm tokens: {total_tokens}")


@app.local_entrypoint()
def main(
    run: str = "",
    smoke: bool = False,
    probe_first_seed: int = PROBE_FIRST_SEED,
    only: str = "",
    max_tokens: int = MAX_TOTAL_TOKENS,
    scoring_tokens: int = MAX_SCORING_TOKENS,
    base_run: str = "",
    arms: str = ",".join(ARMS),
) -> None:
    """--smoke, or --run probe [--probe-first-seed N] | calibration | main | v2_smoke |
    v2_calibration | v2_main, or --run v2_intervention --base-run v2_smoke [--arms a,b]. --only
    keeps the configs whose episode_id contains that text. Launch long runs with
    `modal run --detach` (CLAUDE.md section 9)."""
    run = "smoke" if smoke else run
    if run == V2_ARMS_RUN:
        if not base_run:
            raise SystemExit("--run v2_intervention needs --base-run")
        run_arms(base_run, only, tuple(arms.split(",")), max_tokens)
        return
    if run == "smoke":
        configs = smoke_configs()
    elif run == "probe":
        configs = probe_configs(probe_first_seed)
    elif run == "calibration":
        configs = calibration_configs()
    elif run == "main":
        # Frozen-config guard, before launch: the local file must exist and a container must see
        # exactly it (add_local_python_source alone would not ship it; CLAUDE.md section 9).
        if not FROZEN.exists():
            raise SystemExit("main run refused: crow/frozen.json is missing (Step 8)")
        local_sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
        info = frozen_info.remote()
        if not info["frozen"] or info["sha256"] != local_sha:
            raise SystemExit(f"main run refused: container sees {info}, local sha {local_sha}")
        print(f"frozen config ok in the container: tau {info['tau']}, sha {local_sha[:12]}")
        configs = main_configs()
    elif run in V2_GRIDS:
        if run == "v2_main" and not FROZEN_V2.exists():
            raise SystemExit("v2 main run refused: crow/frozen_v2.json is missing (S5)")
        configs = V2_GRIDS[run]()
    else:
        raise SystemExit(f"use --smoke or --run with one of {RUNS}")
    configs = [c for c in configs if only in c.episode_id]
    jobs = to_run(configs, run)
    print(f"run {run}: {len(configs)} configs, {len(jobs)} to run, budget {max_tokens} tokens")

    # Scored as episodes finish, from the moment the first one does (addenda, section 10).
    scoring = None
    if run in V2_GRIDS:
        ids = [c.episode_id for c in configs]
        set_episodes_done.remote(run, False)
        scoring = score_run_remote.spawn(run, ids, scoring_tokens)
        print(f"scoring pass following {len(ids)} episodes: {scoring.object_id}")

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
    if scoring is not None:
        set_episodes_done.remote(run, True)
        print("episodes done; waiting for the scoring pass")
        print(f"scoring pass: {json.dumps(scoring.get())}")
