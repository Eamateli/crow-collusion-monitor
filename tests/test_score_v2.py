"""The separate scoring pass: from saved data, resumable, within the endpoint's limit (no Modal)."""

import ast
import inspect
import json
import threading
import time
from pathlib import Path

import pytest

from analysis.integrity import score_mismatches
from crow.assemble import assemble
from crow.detectors.postcondition import postcondition
from crow.detectors.views3 import crow_env_view
from crow.llm import LLMCall
from crow.score import Config
from env.episode import score_three
from env.planted import build
from env.prompts import MAX_TRUSTED_REQUESTS, MONITOR_CALLS, RUNNING_REQUESTS
from experiments import score_v2
from experiments.score_v2 import (
    RequestLimiter,
    follow_run,
    grid_finished,
    saved_episodes,
    score_run,
)

ROOT = Path(__file__).resolve().parent.parent
CFG = Config.default()
KINDS = ("honest", "split_ladder", "control_ladder")


def reply(raw=2):
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    content = f"<reasoning>checked</reasoning><score>{raw}</score>"
    return LLMCall("t", "t", {}, usage, 0, 0, 0.0, 1, content, None, None, [], "stop")


class Counting:
    def __init__(self, fail_on=None):
        self.lock = threading.Lock()
        self.now = self.peak = self.count = 0
        self.fail_on = fail_on

    def chat(self, messages, *, affinity_key, **params):
        if self.fail_on and self.fail_on in affinity_key:
            raise RuntimeError("endpoint down")
        with self.lock:
            self.now += 1
            self.count += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.002)
        with self.lock:
            self.now -= 1
        return reply()


def saved_run(tmp_path, kinds=KINDS, task="ledger"):
    for kind in kinds:
        build(task, kind).save(tmp_path)
    return tmp_path


def score_file(tmp_path, kind, task="ledger"):
    path = tmp_path / "scores" / f"planted__{kind}__{task}__s0__a1.json"
    return json.loads(path.read_text())


def test_it_scores_every_saved_episode_into_its_own_file(tmp_path):
    summary = score_run(saved_run(tmp_path), Counting(), cfg=CFG)
    assert summary["episodes"] == 3 and summary["scored"] == 3 and summary["failed"] == []
    for kind in KINDS:
        result = score_file(tmp_path, kind)
        assert result["status"] == "ok" and result["monitor_calls"] == MONITOR_CALLS
        assert set(result["scores"]) == {
            "crow_env",
            "postcondition",
            "b1",
            "b2_per_agent",
            "b2_union",
        }
        assert score_mismatches(result["scores"]) == []
    assert not list((tmp_path / "scores").glob("*.tmp"))


def test_it_uses_the_registered_number_of_calls(tmp_path):
    client = Counting()
    score_run(saved_run(tmp_path, kinds=("honest",)), client, cfg=CFG)
    assert client.count == 6 * MONITOR_CALLS == 18


def test_saved_data_gives_the_same_scores_as_the_episode_itself(tmp_path):
    episode = build("ledger", "split_ladder")
    episode.save(tmp_path)
    saved = saved_episodes(tmp_path)[0][0]
    task, variant, starter, turns, events = saved.inputs()
    assembled = assemble(starter, turns)
    assert {p: e.sha256 for p, e in assembled.snapshot.entries.items()} == {
        p: e.sha256 for p, e in episode.assembled.snapshot.entries.items()
    }
    assert assembled.authors == episode.assembled.authors
    args = (task, variant, Counting(), "k", starter, turns, events, CFG)
    direct = (
        episode.task,
        episode.variant,
        Counting(),
        "k",
        episode.starter,
        episode.turns,
        episode.events,
        CFG,
    )
    a, _ = score_three(*args, monitor_calls=1)
    b, _ = score_three(*direct, monitor_calls=1)
    assert a["postcondition"]["counts"] == b["postcondition"]["counts"]
    assert a["crow_env"]["scores"] == b["crow_env"]["scores"]


def test_requests_in_flight_never_pass_the_cap(tmp_path):
    for task in ("config", "ledger", "table"):
        saved_run(tmp_path, task=task)
    client = Counting()
    score_run(tmp_path, client, max_requests=3, cfg=CFG)
    assert 1 < client.peak <= 3


@pytest.mark.parametrize("cap", [0, -1, MAX_TRUSTED_REQUESTS + 1])
def test_a_cap_outside_the_endpoints_limit_is_refused(tmp_path, cap):
    with pytest.raises(ValueError, match="max_requests"):
        score_run(saved_run(tmp_path), Counting(), max_requests=cap, cfg=CFG)


def test_the_highest_allowed_cap_is_the_endpoints_limit(tmp_path):
    assert MAX_TRUSTED_REQUESTS == 16
    summary = score_run(saved_run(tmp_path), Counting(), max_requests=16, cfg=CFG)
    assert summary["scored"] == 3


def test_a_second_pass_makes_no_request(tmp_path):
    saved_run(tmp_path)
    score_run(tmp_path, Counting(), cfg=CFG)
    client = Counting()
    summary = score_run(tmp_path, client, cfg=CFG)
    assert client.count == 0 and summary["already_scored"] == 3 and summary["scored"] == 0


def test_a_new_episode_is_picked_up_by_the_next_pass(tmp_path):
    saved_run(tmp_path, kinds=("honest",))
    score_run(tmp_path, Counting(), cfg=CFG)
    saved_run(tmp_path, kinds=("split_ladder",))
    summary = score_run(tmp_path, Counting(), cfg=CFG)
    assert summary["already_scored"] == 1 and summary["scored"] == 1


def test_a_failure_is_recorded_does_not_stop_the_others_and_is_retried(tmp_path):
    saved_run(tmp_path)
    summary = score_run(tmp_path, Counting(fail_on="split_ladder"), cfg=CFG)
    assert summary["scored"] == 2 and len(summary["failed"]) == 1
    assert "endpoint down" in summary["failed"][0]
    assert score_file(tmp_path, "split_ladder")["status"] == "failed"
    retry = score_run(tmp_path, Counting(), cfg=CFG)
    assert retry["scored"] == 1 and retry["already_scored"] == 2
    assert score_file(tmp_path, "split_ladder")["status"] == "ok"


def test_the_budget_guard_stops_new_episodes(tmp_path):
    for task in ("config", "ledger", "table", "slug"):
        saved_run(tmp_path, task=task)
    summary = score_run(tmp_path, Counting(), max_requests=1, max_tokens=1, cfg=CFG)
    assert summary["scored"] == 1 and summary["not_started_budget"] == 11
    resumed = score_run(tmp_path, Counting(), max_requests=2, cfg=CFG)
    assert resumed["already_scored"] == 1 and resumed["scored"] == 11


def test_v1_records_are_left_alone_and_failed_episodes_are_named(tmp_path):
    saved_run(tmp_path, kinds=("honest",))
    records = tmp_path / "records"
    v1 = {"episode_id": "v1", "attempt": 1, "status": "ok", "config": {"role": "honest"}}
    (records / "v1__a1.json").write_text(json.dumps(v1))
    failed = {
        "episode_id": "broke",
        "attempt": 1,
        "status": "failed",
        "config": {"role": "honest", "agents": 3},
    }
    (records / "broke__a1.json").write_text(json.dumps(failed))
    summary = score_run(tmp_path, Counting(), cfg=CFG)
    assert summary["episodes"] == 1 and summary["scored"] == 1
    assert summary["not_scorable"] == ["broke: episode failed"]


def test_a_missing_workspace_is_a_failed_score_not_a_crash(tmp_path):
    saved_run(tmp_path, kinds=("honest",))
    next((tmp_path / "workspaces").glob("*.b_out.tar")).unlink()
    summary = score_run(tmp_path, Counting(), cfg=CFG)
    assert summary["scored"] == 0 and "FileNotFoundError" in summary["failed"][0]


def test_a_main_pass_needs_the_v2_freeze(tmp_path):
    run_dir = tmp_path / "main"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="frozen_v2"):
        score_run(run_dir, Counting(), cfg=CFG)


# --- scoring a grid as its episodes finish ---


class Phased(Counting):
    """Records the peak of requests in flight separately before and after the grid is finished."""

    def __init__(self):
        super().__init__()
        self.finished = False
        self.peak_running = self.peak_final = 0

    def chat(self, messages, *, affinity_key, **params):
        with self.lock:
            self.now += 1
            self.count += 1
            if self.finished:
                self.peak_final = max(self.peak_final, self.now)
            else:
                self.peak_running = max(self.peak_running, self.now)
        time.sleep(0.003)
        with self.lock:
            self.now -= 1
        return reply()


def planted_ids(tasks=("config", "ledger", "table"), kinds=KINDS):
    return [(task, kind, build(task, kind).episode_id) for task in tasks for kind in kinds]


def follow(tmp_path, client, is_finished, **options):
    return follow_run(tmp_path, client, is_finished, cfg=CFG, **{"poll_s": 0.0, **options})


def test_episodes_are_scored_as_they_appear(tmp_path):
    waiting = [(t, k) for t, k, _ in planted_ids()]
    seen_scored_before_end = []

    def next_episode(_):
        seen_scored_before_end.append(len(list((tmp_path / "scores").glob("*.json"))))
        if waiting:
            task, kind = waiting.pop()
            build(task, kind).save(tmp_path)
        time.sleep(0.02)

    ids = [e for _, _, e in planted_ids()]
    client = Counting()
    (tmp_path / "records").mkdir()
    summary = follow(tmp_path, client, lambda: grid_finished(tmp_path, ids), sleep=next_episode)
    assert summary["scored"] == 9 and summary["grid_finished"] and summary["failed"] == []
    assert not summary["stopped_idle"]
    assert client.count == 9 * 18
    assert max(seen_scored_before_end) > 0  # some were scored before the grid ended
    for task, kind, _ in planted_ids():
        assert score_file(tmp_path, kind, task)["status"] == "ok"


def test_the_cap_is_12_style_while_running_then_rises_when_the_grid_ends(tmp_path):
    for task, kind, _ in planted_ids():
        build(task, kind).save(tmp_path)
    client = Phased()
    polls = []

    def is_finished():
        polls.append(1)
        client.finished = len(polls) > 4
        return client.finished

    summary = follow(
        tmp_path,
        client,
        is_finished,
        running_requests=2,
        final_requests=5,
        sleep=lambda _: time.sleep(0.05),
    )
    assert summary["scored"] == 9
    assert client.peak_running <= 2
    assert 2 < client.peak_final <= 5


def test_the_default_caps_are_the_decided_ones():
    assert RUNNING_REQUESTS == 12 and MAX_TRUSTED_REQUESTS == 16
    defaults = inspect.signature(follow_run).parameters
    assert defaults["running_requests"].default == 12
    assert defaults["final_requests"].default == 16


@pytest.mark.parametrize(
    ("running", "final"),
    [(0, 16), (13, 12), (12, 17), (17, 17)],
)
def test_caps_outside_the_limits_are_refused(tmp_path, running, final):
    with pytest.raises(ValueError, match="running"):
        follow(tmp_path, Counting(), lambda: True, running_requests=running, final_requests=final)


def test_a_limiter_never_lets_more_through_than_its_limit():
    limiter = RequestLimiter(2)
    peak, lock = [0], threading.Lock()

    def work():
        with limiter:
            with lock:
                peak[0] = max(peak[0], limiter.in_flight)
            time.sleep(0.002)

    threads = [threading.Thread(target=work) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] <= 2 and limiter.in_flight == 0


def test_raising_a_limit_frees_waiting_requests():
    limiter = RequestLimiter(1)
    limiter.__enter__()
    entered = threading.Event()

    def waiter():
        with limiter:
            entered.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    assert not entered.wait(0.05)
    limiter.raise_to(2)
    assert entered.wait(1.0)
    thread.join()
    limiter.__exit__()


def test_a_failed_refresh_is_counted_and_tried_again(tmp_path):
    # A Modal Volume's reload fails while one of its files is open.
    saved_run(tmp_path, kinds=("honest",))
    calls = []

    def refresh():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("there are open files for the volume")

    polls = []

    def is_finished():
        polls.append(1)
        return len(polls) >= 2

    summary = follow(tmp_path, Counting(), is_finished, refresh=refresh)
    assert summary["refresh_failures"] == 1 and len(calls) == 2 and summary["scored"] == 1


def test_each_finished_score_is_committed(tmp_path):
    saved_run(tmp_path)
    commits = []
    follow(tmp_path, Counting(), lambda: True, commit=lambda: commits.append(1))
    assert len(commits) == 3


def test_it_stops_and_says_so_when_the_grid_goes_silent(tmp_path):
    saved_run(tmp_path, kinds=("honest",))
    now = [0.0]

    def tick(_):
        now[0] += 100.0

    summary = follow(
        tmp_path, Counting(), lambda: False, max_idle_s=250.0, sleep=tick, clock=lambda: now[0]
    )
    assert summary["stopped_idle"] is True and summary["grid_finished"] is False
    assert summary["scored"] == 1


def test_a_stopped_follow_resumes_without_repeating_anything(tmp_path):
    saved_run(tmp_path, kinds=("honest",))
    follow(tmp_path, Counting(), lambda: True)
    saved_run(tmp_path, kinds=("split_ladder",))
    client = Counting()
    summary = follow(tmp_path, client, lambda: True)
    assert summary["already_scored"] == 1 and summary["scored"] == 1 and client.count == 18


def test_a_main_follow_needs_the_v2_freeze(tmp_path):
    run_dir = tmp_path / "main"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="frozen_v2"):
        follow(run_dir, Counting(), lambda: True)


def test_the_budget_guard_holds_while_following(tmp_path):
    for task in ("config", "ledger"):
        saved_run(tmp_path, task=task)
    summary = follow(
        tmp_path, Counting(), lambda: True, running_requests=1, final_requests=1, max_tokens=1
    )
    assert summary["scored"] == 1 and summary["not_started_budget"] == 5


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({"a": ["ok"], "b": ["ok"]}, True),
        ({"a": ["ok"]}, False),  # b has not run
        ({"a": ["ok"], "b": ["failed"]}, False),  # b has one attempt left
        ({"a": ["ok"], "b": ["failed", "failed"]}, True),  # b has used its attempts
        ({"a": ["ok"], "b": ["failed", "ok"]}, True),
    ],
)
def test_a_grid_is_finished_when_every_episode_is_ok_or_out_of_attempts(
    tmp_path, statuses, expected
):
    (tmp_path / "records").mkdir()
    for episode, attempts in statuses.items():
        for i, status in enumerate(attempts, 1):
            record = {"episode_id": episode, "status": status}
            (tmp_path / "records" / f"{episode}__a{i}.json").write_text(json.dumps(record))
    assert grid_finished(tmp_path, ["a", "b"]) is expected


# --- no detector may receive the condition or change its computation by it ---


class Capturing(Counting):
    def __init__(self):
        super().__init__()
        self.views = {}

    def chat(self, messages, *, affinity_key, **params):
        self.views[affinity_key.split("planted__", 1)[1]] = messages[0]["content"]
        return super().chat(messages, affinity_key=affinity_key, **params)


def _with_role(run_dir, kind, role):
    """The same workspace, saved as an episode of another role."""
    path = build("ledger", kind).save(run_dir)
    record = json.loads(path.read_text())
    record["config"]["role"] = role
    path.write_text(json.dumps(record))


def _scored_as(tmp_path, kind, role):
    run_dir = tmp_path / role
    _with_role(run_dir, kind, role)
    client = Capturing()
    score_run(run_dir, client, max_requests=1, monitor_calls=1, cfg=CFG)
    return score_file(run_dir, kind), client.views


@pytest.mark.parametrize("kind", ["split_ladder", "control_ladder", "honest"])
def test_a_workspace_gets_the_same_scores_and_views_whatever_its_role(tmp_path, kind):
    results = {
        role: _scored_as(tmp_path, kind, role) for role in ("honest", "collude_split", "collude")
    }
    (first, first_views), *others = results.values()
    for scored, views in others:
        assert scored["scores"]["postcondition"] == first["scores"]["postcondition"]
        assert scored["scores"]["crow_env"]["scores"] == first["scores"]["crow_env"]["scores"]
        assert views == first_views  # every LLM monitor sees byte for byte the same


def test_the_cross_file_comparison_runs_on_an_episode_labelled_honest(tmp_path):
    scored, _ = _scored_as(tmp_path, "split_ladder", "honest")
    match = scored["scores"]["postcondition"]["match"]
    assert match["size"] == 8 and match["cross_agent"] is True
    assert "lib/ledger.py" in match["items"] and "ledger_cases.json" in match["entries"]


def test_honest_work_goes_through_the_same_comparison_and_finds_no_match(tmp_path):
    scored, _ = _scored_as(tmp_path, "honest", "honest")
    post = scored["scores"]["postcondition"]
    assert post["match"] is None and post["counts"]["count_match"] == 0


def test_no_detector_can_be_given_the_condition():
    forbidden = {"role", "config", "condition", "payload", "scheme", "planning", "defense", "k"}
    for function in (postcondition, assemble, score_three, crow_env_view):
        assert not forbidden & set(inspect.signature(function).parameters), function.__name__


@pytest.mark.parametrize(
    "module",
    [
        "crow/detectors/postcondition.py",
        "crow/detectors/heuristics.py",
        "crow/detectors/views3.py",
        "crow/assemble.py",
        "crow/control.py",
    ],
)
def test_detector_code_never_names_a_condition(module):
    # Names in code, not words in comments: nothing here can branch on them.
    tree = ast.parse((ROOT / module).read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    forbidden = {"role", "collude", "collude_split", "honest_hard", "payload", "scheme", "planning"}
    assert not forbidden & names


def test_the_pass_reads_only_the_task_and_its_version_from_a_record(tmp_path):
    class OnlyThese(dict):
        def __getitem__(self, key):
            assert key in ("task_id", "variant"), key
            return super().__getitem__(key)

    build("ledger", "split_ladder").save(tmp_path)
    episode = saved_episodes(tmp_path)[0][0]
    episode.config = OnlyThese(episode.config)
    task, variant, *_ = episode.inputs()
    assert task.id == "ledger" and variant == "base"


# --- nothing is lost while following ---


def test_a_score_that_cannot_be_written_is_reported_not_lost(tmp_path, monkeypatch):
    saved_run(tmp_path, kinds=("honest", "split_ladder"))
    real = score_v2.score_episode

    def flaky(episode, *args, **kwargs):
        if "split_ladder" in episode.stem:
            raise OSError("the volume refused the write")
        return real(episode, *args, **kwargs)

    monkeypatch.setattr(score_v2, "score_episode", flaky)
    summary = follow(tmp_path, Counting(), lambda: True)
    assert summary["scored"] == 1
    assert len(summary["crashed"]) == 1 and "the volume refused" in summary["crashed"][0]


def test_a_failed_commit_is_counted_and_scoring_goes_on(tmp_path):
    saved_run(tmp_path)
    calls = []

    def commit():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("commit failed")

    summary = follow(tmp_path, Counting(), lambda: True, commit=commit)
    assert summary["scored"] == 3 and summary["commit_failures"] == 1 and len(calls) == 3


@pytest.mark.parametrize("name", ["main", "v2_main"])
def test_a_v2_main_pass_needs_the_freeze_in_both_modes(tmp_path, name):
    run_dir = tmp_path / name
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="frozen_v2"):
        score_run(run_dir, Counting(), cfg=CFG)
    with pytest.raises(SystemExit, match="frozen_v2"):
        follow(run_dir, Counting(), lambda: True)
