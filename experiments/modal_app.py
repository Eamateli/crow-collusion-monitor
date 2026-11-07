"""Modal app 'crow': run_episode on Modal, the grid runner, and --smoke (CLAUDE.md section 9).

Each episode is one container: it builds the two endpoint clients from the crow-secrets Secret,
runs env.episode.run_episode against fresh network-blocked sandboxes, and writes its record and
event log to the crow-results Volume. The grid is run with run_episode_remote.map, capped at 12
containers, skipping episodes already finished (resume) and stopping at MAX_TOTAL_TOKENS.

A 3-agent run (v2_*) makes no monitor call in its episodes. One more container, score_run_remote,
follows the run and scores episodes as they finish, at 12 requests while they run and 16 after.
These runs are driven from inside Modal, by launch_run_remote on the deployed app, so nothing local
can end them. Start one with scripts/launch_v2.py.
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from env.episode import EpisodeConfig, make_config, run_episode
from env.replay import ARMS, arm_bases, arm_episode_id, arms_run_for, run_arm
from experiments.grid import (
    ARMS_STAGE,
    BILLED_RATIO,
    PROBE_FIRST_SEED,
    SCORING_COST,
    arm_budget,
    calibration_configs,
    main_configs,
    probe_configs,
    token_budget,
    v2_calibration_configs,
    v2_main_configs,
    v2_main_stages,
    v2_resmoke_configs,
)
from experiments.score_v2 import DONE_MARKER, follow_run, run_finished, score_run

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "crow"
VOLUME_NAME = "crow-results"
RESULTS = "/results"
MAX_TOTAL_TOKENS = 8_000_000  # budget guard; override with --max-tokens
V2_GRIDS = {
    "v2_resmoke": v2_resmoke_configs,
    "v2_calibration": v2_calibration_configs,
    "v2_main": v2_main_configs,
}
# v2_intervention is the first v2 smoke's replays: kept, and never written again.
V2_ARM_RUNS = (arms_run_for("v2_resmoke"), arms_run_for("v2_main"))
# v2_smoke is the first v2 smoke's folder: it can be a base for arms, but is never run again.
RUNS = ("smoke", "probe", "calibration", "main", "v2_smoke", "v2_intervention", *V2_GRIDS)
# Modal allows up to 24 hours per attempt. A pass that dies resumes when launched again.
SCORING_TIMEOUT = 12 * 3600
LAUNCHER_TIMEOUT = 20 * 3600  # it waits for the pass, so it outlasts it
MAX_RUNNING = 12  # episodes or replays at a time (CLAUDE.md section 9)
NO_TOKEN_LIMIT = 10**12  # a dollar limit is in force instead
# Modal restarts a preempted function on the same input (modal.com/docs/guide/preemption). The two
# drivers can't be, at 3x the CPU and memory price of a nearly idle container: a restarted launcher
# would send episodes that are still running a second time. A preempted episode writes its record
# once, when it ends, so episodes stay preemptible.
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
    retries=1,
    nonpreemptible=True,
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
    # A later pass over the same run must not erase what this one failed (CLAUDE.md section 13).
    _write_json(run, f"scoring_summaries/{int(time.time())}.json", summary)
    _write_json(run, "scoring_summary.json", summary)
    return summary


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=3600,
    max_containers=12,
    retries=1,
)
def run_arm_remote(payload: dict) -> dict:
    """One intervention arm on one saved episode. Writes under /results/<base run>_arms/."""
    if payload["out_run"] not in V2_ARM_RUNS:
        raise ValueError(f"unknown arms folder {payload['out_run']!r}")
    agent, trusted = _clients()
    record = run_arm(
        f"{RESULTS}/{payload['base_run']}",
        payload["base_stem"],
        payload["arm"],
        agent,
        trusted,
        f"{RESULTS}/{payload['out_run']}",
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
def frozen_v2_info() -> dict:
    """Which 3-agent config a container sees (checks that frozen_v2.json reaches the grid)."""
    from crow.score import FROZEN_V2_PATH, load_config

    config = load_config(FROZEN_V2_PATH)
    return {"frozen": config.frozen, "tau": config.tau, "sha256": config.sha256}


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
        data = b"".join(volume.read_file(f"{run}/records/{name}"))  # a record can span chunks
        attempts.setdefault(episode_id, []).append(json.loads(data).get("status", "unknown"))
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


def _write_json(run: str, name: str, data: dict) -> None:
    """A status file in the run folder, committed so that anything can read it from outside."""
    path = Path(RESULTS) / run / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    volume.commit()


def _set_episodes_done(run: str, done: bool) -> None:
    """Tell the scoring pass whether this run's episodes have all ended."""
    marker = Path(RESULTS) / run / DONE_MARKER
    if done:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    volume.commit()


def _new_state() -> dict:
    return {
        "finished": 0,
        "failed": [],
        "exceptions": [],
        "tokens": 0,
        "cost": 0.0,  # finished episodes and replays, at estimate prices
        "stopped_by_budget": False,
        "not_sent": {},
    }


class DollarGuard:
    """One limit in real dollars over the episodes, the replays and the scoring pass (addenda,
    section 12). The pass runs hours behind the episodes, so a running total would see the money
    too late: an episode's scoring counts from the moment it is sent, at the estimate for its role
    until its real cost is known. An episode still running adds its own cost on top of the limit,
    which is why the limit leaves room for that."""

    def __init__(self, limit_usd: float, read_scored=None, refresh_s: float = 60.0, clock=None):
        self.limit_usd = limit_usd
        self.sent: dict[str, str] = {}  # episode_id -> role
        self.scored: dict[str, float] = {}  # episode_id -> what its scoring cost
        self._read_scored = read_scored
        self._refresh_s = refresh_s
        self._clock = clock or time.monotonic
        self._last_read: float | None = None

    def seed(self, records: list[dict], state: dict) -> None:
        """What a run on the Volume has already spent, so that a re-launch counts it."""
        for record in records:
            state["cost"] += record.get("cost_estimate_usd") or 0
            config = record.get("config") or {}
            if "role" in config and "__intervene_" not in record["episode_id"]:
                self.sent[record["episode_id"]] = config["role"]

    def note_sent(self, job: dict) -> None:
        config = job.get("config")  # a replay has no config: the pass never scores it
        if config:
            self.sent[config["episode_id"]] = config["role"]

    def _refresh(self) -> None:
        now = self._clock()
        if self._read_scored is None:
            return
        if self._last_read is not None and now - self._last_read < self._refresh_s:
            return
        self._last_read = now
        try:
            self._read_scored(self.scored)
        except Exception as error:  # noqa: BLE001 - the estimates stand until the next read
            print(f"  scoring costs not read: {type(error).__name__}: {error}")

    def projected_usd(self, state: dict) -> float:
        scoring = sum(self.scored.get(e, SCORING_COST[role]) for e, role in self.sent.items())
        return BILLED_RATIO * (state["cost"] + scoring)

    def over(self, state: dict) -> bool:
        self._refresh()
        return self.projected_usd(state) >= self.limit_usd


def _read_scored_costs(run: str, known: dict[str, float]) -> None:
    """Add what the pass has spent on each newly scored episode of this run."""
    for entry in volume.listdir(f"{run}/scores"):
        name = Path(entry.path).name
        episode_id = name.split("__a")[0]
        if name.endswith(".json") and episode_id not in known:
            data = json.loads(b"".join(volume.read_file(f"{run}/scores/{name}")))
            if data.get("status") == "ok":
                known[episode_id] = data["cost_estimate_usd"]


def _run_jobs(
    function,
    jobs: list[dict],
    max_tokens: int,
    on_progress,
    *,
    state: dict | None = None,
    workers: int = MAX_RUNNING,
    guard: DollarGuard | None = None,
) -> dict:
    """Run jobs at most `workers` at a time. The budget guard is checked before each job is sent,
    so a stop leaves nothing queued; jobs already running finish and are counted. `state` lets
    several calls draw on one budget, and a job a stop keeps back is counted under its stage."""
    lock = threading.Lock()
    state = _new_state() if state is None else state

    def report() -> None:
        try:
            on_progress(state)
        except Exception as error:  # noqa: BLE001 - a status file must not stop an episode
            print(f"  status not written: {type(error).__name__}: {error}")

    def work(job: dict) -> None:
        with lock:
            if state["tokens"] >= max_tokens or (guard is not None and guard.over(state)):
                state["stopped_by_budget"] = True
                stage = state.get("stage", "")
                state["not_sent"][stage] = state["not_sent"].get(stage, 0) + 1
                return
            if guard is not None:
                guard.note_sent(job)
        try:
            result = function.remote(job)
        except Exception as error:  # noqa: BLE001 - kept and reported, never dropped
            with lock:
                state["exceptions"].append(f"{type(error).__name__}: {error}")
                report()
            return
        with lock:
            state["finished"] += 1
            state["tokens"] += result["tokens"]
            state["cost"] += result.get("cost") or 0
            if guard is not None:
                state["projected_usd"] = round(guard.projected_usd(state), 2)
            if result["status"] != "ok":
                state["failed"].append(f"{result['episode_id']}: {result.get('reason')}")
            print(
                f"  {result['status']:6} {result['episode_id']} | {result['tokens']} tok | "
                f"${result['cost']} | {result['seconds']}s",
                flush=True,
            )
            report()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, jobs))
    if state["stopped_by_budget"]:
        print(f"  budget reached ({state['tokens']} >= {max_tokens}); stopped")
    return state


def _run_stages(stages: dict[str, list], state: dict, send_episodes, send_arms) -> dict:
    """Each stage runs to its end before the next starts, and all draw on one budget, so a budget
    stop only ever cuts the last stages (addenda, section 12)."""
    for stage, configs in stages.items():
        state["stage"] = stage
        if stage == ARMS_STAGE:
            send_arms(state)
        else:
            send_episodes(configs, state)
    return state


def _launch_arms(
    base_run: str,
    only: str,
    arms: tuple[str, ...],
    max_tokens: int | None,
    state: dict | None = None,
    guard: DollarGuard | None = None,
) -> dict:
    """Every arm on every matching finished episode of `base_run` (addenda, section 10)."""
    out_run = arms_run_for(base_run)
    bases = [r for r in arm_bases(_records(base_run)) if only in r["episode_id"]]
    done = _existing_attempts(out_run)
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
                    "out_run": out_run,
                }
            )
    budget = max_tokens or arm_budget(len(bases))
    print(f"arms on {base_run}: {len(bases)} episodes, {len(jobs)} replays, budget {budget} tokens")
    status = {"state": "running", "bases": len(bases), "to_run": len(jobs), "budget": budget}
    _write_json(out_run, "launcher.json", status)
    state = _run_jobs(
        run_arm_remote,
        jobs,
        budget,
        lambda s: _write_json(out_run, "launcher.json", {**status, **s}),
        state=state,
        guard=guard,
    )
    _write_json(out_run, "launcher.json", {**status, **state, "state": "done"})
    return state


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=LAUNCHER_TIMEOUT,
    max_containers=1,
    retries=1,
    nonpreemptible=True,
)
def launch_run_remote(
    run: str,
    seeds: int | None = None,
    only: str = "",
    max_tokens: int | None = None,
    scoring_tokens: int | None = None,
    max_usd: float | None = None,
) -> dict:
    """A whole 3-agent run, driven from inside Modal so that nothing local can end it: v1 lost 65
    episodes when its local driver stopped (addenda, section 12). It starts the scoring pass, runs
    the stages in order, says when the episodes are done, and waits for the pass. A second launch
    resumes from the Volume. `max_tokens` and `scoring_tokens` replace the estimate plus 10%. The
    arms are a stage of v2_main, and are left out when `only` picks some of its episodes. With
    `max_usd`, one limit in real dollars over everything replaces the episode token budget, and the
    pass keeps its token budget only as a backstop.
    """
    if run not in V2_GRIDS:
        raise ValueError(f"not a 3-agent run: {run!r}")
    if seeds and run != "v2_main":
        raise ValueError("only v2_main takes a number of seeds")
    if run == "v2_main":
        from crow.score import FROZEN_V2_PATH, load_config

        if not load_config(FROZEN_V2_PATH).frozen:
            raise RuntimeError("v2 main run refused: this container has no crow/frozen_v2.json")
        stages = v2_main_stages(range(seeds)) if seeds else v2_main_stages()
    else:
        stages = {run: V2_GRIDS[run]()}
    stages = {
        name: [c for c in configs if only in c.episode_id] for name, configs in stages.items()
    }
    if only:
        stages.pop(ARMS_STAGE, None)
    configs = [c for group in stages.values() for c in group]
    planned = [{"episode_id": c.episode_id, "status": "ok", "config": vars(c)} for c in configs]
    bases = len(arm_bases(planned)) if ARMS_STAGE in stages else 0
    budget = token_budget(configs, bases)
    episode_budget = max_tokens or (NO_TOKEN_LIMIT if max_usd else budget["episodes"])
    state = _new_state()
    guard = None
    if max_usd:
        guard = DollarGuard(max_usd, lambda known: _read_scored_costs(run, known))
        for folder in (run, arms_run_for(run)):
            try:
                guard.seed(_records(folder), state)
            except Exception as error:  # noqa: BLE001 - a new run has no records folder yet
                print(f"  nothing spent in {folder} yet: {type(error).__name__}")
    status = {
        "state": "running",
        "configs": len(configs),
        "stages": {name: len(group) for name, group in stages.items()},
        "arm_bases": bases,
        "seeds": seeds,
        "episode_budget": episode_budget,
        "scoring_budget": scoring_tokens or budget["scoring"],
        "max_usd": max_usd,
    }
    print(f"run {run}: {status}")
    _write_json(run, "launcher.json", status)
    _set_episodes_done(run, False)
    scoring = score_run_remote.spawn(
        run, [c.episode_id for c in configs], scoring_tokens or budget["scoring"]
    )
    status["scoring_call"] = scoring.object_id

    def progress(current: dict) -> None:
        _write_json(run, "launcher.json", {**status, **current})

    def send_arms(current: dict) -> None:
        progress(current)  # the replays report to their own folder: show the stage here too
        _launch_arms(run, "", ARMS, episode_budget, current, guard)
        progress(current)

    state = _run_stages(
        stages,
        state,
        lambda group, current: _run_jobs(
            run_episode_remote,
            to_run(group, run),
            episode_budget,
            progress,
            state=current,
            guard=guard,
        ),
        send_arms,
    )
    _set_episodes_done(run, True)
    _write_json(run, "launcher.json", {**status, **state, "state": "episodes done"})
    try:
        state["scoring"] = scoring.get()
    except Exception as error:  # noqa: BLE001 - reported in the status file, never dropped
        state["scoring_error"] = f"{type(error).__name__}: {error}"
    _write_json(run, "launcher.json", {**status, **state, "state": "done"})
    return state


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("crow-secrets")],
    volumes={RESULTS: volume},
    timeout=LAUNCHER_TIMEOUT,
    max_containers=1,
    retries=1,
    nonpreemptible=True,
)
def launch_arms_remote(base_run: str, only: str, arms: list[str]) -> dict:
    """Intervention arms alone, from inside Modal: for a check on a run that has already ended."""
    return _launch_arms(base_run, only, tuple(arms), None)


@app.local_entrypoint()
def main(
    run: str = "",
    smoke: bool = False,
    probe_first_seed: int = PROBE_FIRST_SEED,
    only: str = "",
    max_tokens: int = MAX_TOTAL_TOKENS,
) -> None:
    """The v1 runs: --smoke, or --run probe [--probe-first-seed N] | calibration | main. --only
    keeps the configs whose episode_id contains that text. Launch long runs with
    `modal run --detach` (CLAUDE.md section 9). 3-agent runs start with scripts/launch_v2.py."""
    run = "smoke" if smoke else run
    if run in V2_GRIDS or run in V2_ARM_RUNS:
        raise SystemExit("3-agent runs are driven server-side: use scripts/launch_v2.py")
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
    else:
        raise SystemExit(f"use --smoke or --run with one of {RUNS}")
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
