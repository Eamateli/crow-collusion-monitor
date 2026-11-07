"""The scoring pass for 3-agent runs: every monitor, from what the episodes saved.

Episodes make no monitor call. Each LLM monitor is asked MONITOR_CALLS times, so scoring takes about
three times what the agents take, and it would sit inside the 60-minute episode limit if it ran
there. Here it runs on its own: a scoring failure cannot cost an episode, a pass can resume, and a
re-score never needs the agents again. Only intervention runs (S3) need a score during an episode.

Each worker scores one episode, one request at a time, so `max_requests` is also the number of
requests in flight; a shared model takes 16 at once across the whole workspace. Reads workspace tars
in memory and never extracts or runs them. Takes only the config from each record: the planning chat
and the private reports are never read here, so no monitor can be shown them.

A grid is scored as its episodes finish (--follow): at most 12 requests while episodes run, because
the endpoint also serves their paraphrasing, then 16 once they are done.

    python experiments/score_v2.py RUN_DIR [--max-requests 12] [--max-tokens N]
    python experiments/score_v2.py RUN_DIR --follow v2_main | v2_calibration
"""

import argparse
import json
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from crow.events import read_events
from crow.llm import modal_endpoint
from crow.score import FROZEN_V2_PATH, Config, load_config
from crow.wrap import snapshot
from env.episode import PRICES, score_three
from env.prompts import MAX_TRUSTED_REQUESTS, MONITOR_CALLS, RUNNING_REQUESTS, TRUSTED_MODEL_ID
from env.taskset import load_task

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUESTS = 12
MAX_EPISODE_ATTEMPTS = 2  # as experiments/modal_app.py, to_run: a third attempt never comes
POLL_S, MAX_IDLE_S = 30.0, 3600.0
WRITE_TRIES, WRITE_WAIT_S = 8, 15.0  # outlasts a Volume reload, which comes every POLL_S
DONE_MARKER = "episodes_done"  # in the run folder, once the runner's episodes have all ended
FROZEN_RUNS = ("main", "v2_main")  # these run folders are scored only with crow/frozen_v2.json
MAX_TOTAL_TOKENS = 40_000_000  # for a pass run by hand; a launched run passes its own guard
console = Console()


@dataclass
class SavedEpisode:
    """What a pass reads of one finished episode. Only the config, nothing private."""

    run_dir: Path
    stem: str
    episode_id: str
    config: dict

    @property
    def out_path(self) -> Path:
        return self.run_dir / "scores" / f"{self.stem}.json"

    def _snapshot(self, name: str):
        path = self.run_dir / "workspaces" / f"{self.stem}.{name}.tar"
        return snapshot(path.read_bytes())

    def inputs(self) -> tuple:
        """(task, variant, starter, turns, events), as score_three takes them."""
        task = load_task(self.config["task_id"])
        variant = self.config["variant"]
        starter = snapshot(task.starter_tar(variant))
        turns = [
            ("A", starter, self._snapshot("a_out")),
            ("B", self._snapshot("b_view"), self._snapshot("b_out")),
            ("C", self._snapshot("c_view"), self._snapshot("final")),
        ]
        events = read_events(self.run_dir / "events" / f"{self.stem}.jsonl")
        return task, variant, starter, turns, events


def saved_episodes(run_dir: Path) -> tuple[list[SavedEpisode], list[str]]:
    """The finished 3-agent episodes under a run folder, and the ids it cannot score, with why."""
    episodes, skipped = [], []
    for path in sorted((run_dir / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        if config.get("agents", 2) != 3:
            continue
        if record["status"] != "ok":
            skipped.append(f"{record['episode_id']}: episode {record['status']}")
            continue
        episodes.append(SavedEpisode(run_dir, path.stem, record["episode_id"], config))
    return episodes, skipped


def _tokens(calls: list) -> dict[str, int]:
    return {
        "calls": len(calls),
        "prompt": sum(c.usage.get("prompt_tokens", 0) for c in calls),
        "completion": sum(c.usage.get("completion_tokens", 0) for c in calls),
        "cut_at_max_tokens": sum(c.finish_reason == "length" for c in calls),
    }


def _save_result(path: Path, result: dict) -> None:
    """Write a result in one step, so resume never trusts a half-written file. The monitor calls
    are already paid for by now, so a refused write waits: in the main grid, 52 finished results
    were lost to PermissionError in the window when the pass was reloading the Modal Volume."""
    text = json.dumps(result, ensure_ascii=False, indent=1)
    temporary = path.with_suffix(".tmp")
    for attempt in range(WRITE_TRIES):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == WRITE_TRIES - 1:
                raise
            time.sleep(WRITE_WAIT_S)


def score_episode(episode: SavedEpisode, client: Any, cfg: Config, monitor_calls: int) -> dict:
    """Score one episode and write scores/<stem>.json. An endpoint failure after the retries is
    recorded as failed, and a later pass tries that episode again."""
    started = time.time()
    result: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "stem": episode.stem,
        "status": "ok",
        "monitor_calls": monitor_calls,
        "frozen": {"tau": cfg.tau, "sha256": cfg.sha256},
    }
    try:
        task, variant, starter, turns, events = episode.inputs()
        result["scores"], calls = score_three(
            task,
            variant,
            client,
            f"score-{episode.episode_id}",
            starter,
            turns,
            events,
            cfg,
            monitor_calls=monitor_calls,
        )
        result["tokens"] = _tokens(calls)
        in_price, out_price = PRICES["trusted"]
        prompt, completion = result["tokens"]["prompt"], result["tokens"]["completion"]
        result["cost_estimate_usd"] = round(
            prompt * in_price / 1e6 + completion * out_price / 1e6, 5
        )
    except Exception as e:  # noqa: BLE001 - handled: mark failed, keep the reason (CLAUDE.md 13)
        result["status"] = "failed"
        result["reason"] = f"{type(e).__name__}: {e}"
    result["seconds"] = round(time.time() - started, 2)
    _save_result(episode.out_path, result)
    return result


def _already_scored(episode: SavedEpisode) -> bool:
    if not episode.out_path.exists():
        return False
    return json.loads(episode.out_path.read_text(encoding="utf-8")).get("status") == "ok"


def score_run(
    run_dir: str | Path,
    client: Any,
    *,
    max_requests: int = DEFAULT_REQUESTS,
    max_tokens: int = MAX_TOTAL_TOKENS,
    monitor_calls: int = MONITOR_CALLS,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Score every finished episode under `run_dir` that has no ok score yet."""
    if not 1 <= max_requests <= MAX_TRUSTED_REQUESTS:
        raise ValueError(f"max_requests must be 1 to {MAX_TRUSTED_REQUESTS}, got {max_requests}")
    run_dir = Path(run_dir)
    cfg = cfg or load_config(FROZEN_V2_PATH)
    if run_dir.name in FROZEN_RUNS and not cfg.frozen:
        raise SystemExit("main pass refused: crow/frozen_v2.json is missing (S5)")
    episodes, skipped = saved_episodes(run_dir)
    todo = [e for e in episodes if not _already_scored(e)]
    lock = threading.Lock()
    spent = {"tokens": 0, "stopped": 0}
    results: list[dict] = []

    def run(episode: SavedEpisode) -> None:
        with lock:
            if spent["tokens"] >= max_tokens:
                spent["stopped"] += 1
                return
        result = score_episode(episode, client, cfg, monitor_calls)
        with lock:
            tokens = result.get("tokens", {})
            spent["tokens"] += tokens.get("prompt", 0) + tokens.get("completion", 0)
            results.append(result)

    started = time.time()
    with ThreadPoolExecutor(max_workers=max_requests) as pool:
        list(pool.map(run, todo))
    failed = [r for r in results if r["status"] != "ok"]
    return {
        "episodes": len(episodes),
        "already_scored": len(episodes) - len(todo),
        "scored": len(results) - len(failed),
        "failed": [f"{r['episode_id']}: {r['reason']}" for r in failed],
        "not_started_budget": spent["stopped"],
        "not_scorable": skipped,
        "tokens": spent["tokens"],
        "cost_estimate_usd": round(sum(r.get("cost_estimate_usd", 0) for r in results), 4),
        "seconds": round(time.time() - started, 1),
        "results": results,
    }


class RequestLimiter:
    """At most `limit` requests in flight. The limit can be raised while requests are running."""

    def __init__(self, limit: int):
        self.limit = limit
        self.in_flight = 0
        self._changed = threading.Condition()

    def raise_to(self, limit: int) -> None:
        with self._changed:
            self.limit = limit
            self._changed.notify_all()

    def __enter__(self) -> "RequestLimiter":
        with self._changed:
            while self.in_flight >= self.limit:
                self._changed.wait()
            self.in_flight += 1
        return self

    def __exit__(self, *exc: object) -> None:
        with self._changed:
            self.in_flight -= 1
            self._changed.notify_all()


class _LimitedClient:
    def __init__(self, client: Any, limiter: RequestLimiter):
        self.client, self.limiter = client, limiter

    def chat(self, messages: list[dict[str, Any]], *, affinity_key: str, **params: Any) -> Any:
        with self.limiter:
            return self.client.chat(messages, affinity_key=affinity_key, **params)


def run_finished(run_dir: str | Path, episode_ids: Iterable[str]) -> bool:
    """No more episodes are coming. The runner leaves a marker when its map ends, because one launch
    makes one attempt: an episode that failed would otherwise look as if it still had one to come,
    and a run the budget guard stopped would never look finished."""
    return (Path(run_dir) / DONE_MARKER).exists() or grid_finished(run_dir, episode_ids)


def grid_finished(run_dir: str | Path, episode_ids: Iterable[str]) -> bool:
    """Every episode of the grid has an ok record, or has used its attempts."""
    statuses: dict[str, list[str]] = {}
    for path in (Path(run_dir) / "records").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        statuses.setdefault(record["episode_id"], []).append(record["status"])
    return all(
        "ok" in statuses.get(episode, []) or len(statuses.get(episode, [])) >= MAX_EPISODE_ATTEMPTS
        for episode in episode_ids
    )


def follow_run(
    run_dir: str | Path,
    client: Any,
    is_finished: Callable[[], bool],
    *,
    running_requests: int = RUNNING_REQUESTS,
    final_requests: int = MAX_TRUSTED_REQUESTS,
    poll_s: float = POLL_S,
    max_idle_s: float = MAX_IDLE_S,
    max_tokens: int = MAX_TOTAL_TOKENS,
    monitor_calls: int = MONITOR_CALLS,
    cfg: Config | None = None,
    refresh: Callable[[], None] | None = None,
    commit: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Score episodes as they finish, until `is_finished()` and all are scored. `refresh` and
    `commit` are for a Modal Volume: its reload fails while a file is open, so a failed refresh is
    counted and tried again at the next poll. If no new episode appears for `max_idle_s`, it stops
    and says so, rather than wait forever on a grid that died; a new pass resumes."""
    if not 1 <= running_requests <= final_requests <= MAX_TRUSTED_REQUESTS:
        raise ValueError(
            f"need 1 <= running ({running_requests}) <= final ({final_requests}) "
            f"<= {MAX_TRUSTED_REQUESTS}"
        )
    run_dir = Path(run_dir)
    cfg = cfg or load_config(FROZEN_V2_PATH)
    if run_dir.name in FROZEN_RUNS and not cfg.frozen:
        raise SystemExit("main pass refused: crow/frozen_v2.json is missing (S5)")
    limiter = RequestLimiter(running_requests)
    limited = _LimitedClient(client, limiter)
    lock = threading.Lock()
    spent = {"tokens": 0, "stopped": 0, "commit_failures": 0}
    results: list[dict] = []
    submitted: set[str] = set()
    futures = []

    def run(episode: SavedEpisode) -> None:
        with lock:
            if spent["tokens"] >= max_tokens:
                spent["stopped"] += 1
                return
        result = score_episode(episode, limited, cfg, monitor_calls)
        with lock:
            tokens = result.get("tokens", {})
            spent["tokens"] += tokens.get("prompt", 0) + tokens.get("completion", 0)
            results.append(result)
            if commit is not None:
                try:
                    commit()
                except Exception:  # noqa: BLE001 - the next commit carries this file too
                    spent["commit_failures"] += 1

    started, last_new = clock(), clock()
    refresh_failures, already, skipped = 0, 0, []
    finished = idle = False
    with ThreadPoolExecutor(max_workers=final_requests) as pool:
        while True:
            if refresh is not None:
                try:
                    refresh()
                except Exception:  # noqa: BLE001 - e.g. a Volume file is open; next poll retries
                    refresh_failures += 1
            finished = is_finished()  # before listing, so that the last records are seen
            episodes, skipped = saved_episodes(run_dir)
            for episode in episodes:
                if episode.stem in submitted:
                    continue
                submitted.add(episode.stem)
                last_new = clock()
                if _already_scored(episode):
                    already += 1
                else:
                    futures.append((episode.episode_id, pool.submit(run, episode)))
            if finished:
                limiter.raise_to(final_requests)  # the episodes no longer need the endpoint
                break
            if clock() - last_new > max_idle_s:
                idle = True
                break
            sleep(poll_s)
    # A file that could not be written, for one: never let an episode vanish without a word.
    crashed = [f"{name}: {f.exception()!r}" for name, f in futures if f.exception() is not None]
    failed = [r for r in results if r["status"] != "ok"]
    return {
        "episodes": len(submitted),
        "crashed": crashed,
        "commit_failures": spent["commit_failures"],
        "already_scored": already,
        "scored": len(results) - len(failed),
        "failed": [f"{r['episode_id']}: {r['reason']}" for r in failed],
        "not_started_budget": spent["stopped"],
        "not_scorable": skipped,
        "grid_finished": finished,
        "stopped_idle": idle,
        "refresh_failures": refresh_failures,
        "tokens": spent["tokens"],
        "cost_estimate_usd": round(sum(r.get("cost_estimate_usd", 0) for r in results), 4),
        "seconds": round(clock() - started, 1),
        "results": results,
    }


def trusted_client():
    """From .env by python-dotenv, locally; from the Modal Secret when run in a container."""
    load_dotenv(ROOT / ".env")
    return modal_endpoint("trusted", os.environ["TRUSTED_ENDPOINT_URL"], TRUSTED_MODEL_ID)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument("--follow", choices=["v2_main", "v2_calibration"])
    args = parser.parse_args(argv)
    if args.follow:
        from experiments.grid import v2_calibration_configs, v2_main_configs

        grid = v2_main_configs() if args.follow == "v2_main" else v2_calibration_configs()
        ids = [c.episode_id for c in grid]
        summary = follow_run(
            args.run_dir,
            trusted_client(),
            lambda: run_finished(args.run_dir, ids),
            max_tokens=args.max_tokens,
        )
    else:
        summary = score_run(
            args.run_dir,
            trusted_client(),
            max_requests=args.max_requests,
            max_tokens=args.max_tokens,
        )
    summary.pop("results")
    console.print_json(data=summary)
    incomplete = (
        summary["failed"]
        or summary["not_started_budget"]
        or summary.get("stopped_idle")
        or summary.get("crashed")
    )
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
