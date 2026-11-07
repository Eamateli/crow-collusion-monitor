"""The Modal wiring of Crow v2, checked without Modal: no container is started here."""

import ast
import inspect

import pytest

from experiments import modal_app


def test_every_v2_run_has_its_own_folder():
    for run in ("v2_smoke", "v2_resmoke", "v2_calibration", "v2_main", "v2_intervention"):
        assert run in modal_app.RUNS
    assert modal_app.V2_ARM_RUNS == ("v2_resmoke_arms", "v2_main_arms")
    assert set(modal_app.V2_GRIDS) == {"v2_resmoke", "v2_calibration", "v2_main"}


def test_the_scoring_timeout_is_within_modals_limit_and_covers_the_estimate():
    # Modal allows 1 second to 24 hours per attempt (modal.com/docs/guide/timeouts).
    assert 4 * 3600 <= modal_app.SCORING_TIMEOUT <= 24 * 3600


def test_the_grids_are_the_v2_ones():
    assert len(modal_app.V2_GRIDS["v2_main"]()) == 150
    assert len(modal_app.V2_GRIDS["v2_calibration"]()) == 30
    assert len(modal_app.V2_GRIDS["v2_resmoke"]()) == 6


def _function(name: str) -> str:
    # Modal wraps its functions, so read them from the module's source.
    tree = ast.parse(inspect.getsource(modal_app))
    return ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == name))


def test_the_launcher_tells_the_pass_when_its_episodes_have_ended():
    source = _function("launch_run_remote")
    assert source.index("_set_episodes_done(run, False)") < source.index("score_run_remote.spawn")
    assert source.index("_run_stages(") < source.index("_set_episodes_done(run, True)")
    assert source.index("_set_episodes_done(run, True)") < source.index("scoring.get()")


def test_3_agent_runs_cannot_start_from_a_local_driver():
    # v1 lost 65 episodes when its local driver stopped (addenda, section 12).
    source = _function("main")
    assert "V2_GRIDS" in source and "launch_v2.py" in source
    assert "score_run_remote" not in source


def test_the_launcher_stays_within_modals_limit_and_outlasts_the_pass():
    assert modal_app.SCORING_TIMEOUT <= modal_app.LAUNCHER_TIMEOUT <= 24 * 3600


def _result(job, status="ok", tokens=100):
    return {
        "status": status,
        "episode_id": job,
        "reason": "x" if status != "ok" else None,
        "tokens": tokens,
        "cost": 0,
        "seconds": 1,
    }


class Remote:
    def __init__(self, fail=()):
        self.sent, self.fail = [], fail

    def remote(self, job):
        self.sent.append(job)
        if job in self.fail:
            raise RuntimeError("container died")
        return _result(job)


def test_a_budget_stop_sends_nothing_more_and_is_reported():
    fake, progress = Remote(), []
    state = modal_app._run_jobs(fake, ["a", "b", "c", "d"], 200, progress.append, workers=1)
    assert fake.sent == ["a", "b"] and state["finished"] == 2
    assert state["stopped_by_budget"] is True and len(progress) == 2


def test_failures_and_exceptions_are_kept_not_dropped():
    class Failing(Remote):
        def remote(self, job):
            return _result(job, status="failed") if job == "a" else super().remote(job)

    state = modal_app._run_jobs(Failing(fail=("b",)), ["a", "b", "c"], 10**6, lambda s: None)
    assert state["failed"] == ["a: x"] and state["exceptions"] == ["RuntimeError: container died"]
    assert state["finished"] == 2 and state["stopped_by_budget"] is False


def test_a_status_file_that_cannot_be_written_does_not_stop_the_run():
    def broken(state):
        raise OSError("commit failed")

    state = modal_app._run_jobs(Remote(), ["a", "b"], 10**6, broken)
    assert state["finished"] == 2 and state["exceptions"] == []


def test_at_most_twelve_run_at_once():
    assert modal_app.MAX_RUNNING == 12
    assert "max_containers=12" in inspect.getsource(modal_app)


def test_the_drivers_cannot_be_preempted_and_episodes_can():
    for name in ("launch_run_remote", "launch_arms_remote", "score_run_remote"):
        assert "nonpreemptible=True" in _function(name), name
    assert "nonpreemptible" not in _function("run_episode_remote")


# --- a budget stop cuts only the last stages (addenda, section 12) ---


def _staged(budget, sent_tokens=100):
    stages = {"honest": ["h1", "h2"], "careful": ["c1", "c2"], "arms": [], "noplan": ["n1"]}
    fake = Remote()

    def episodes(jobs, state):
        modal_app._run_jobs(fake, jobs, budget, lambda s: None, state=state, workers=1)

    def arms(state):
        modal_app._run_jobs(fake, ["a1", "a2"], budget, lambda s: None, state=state, workers=1)

    state = modal_app._run_stages(stages, modal_app._new_state(), episodes, arms)
    return fake, state


def test_with_budget_left_every_stage_runs_in_order():
    fake, state = _staged(10**6)
    assert fake.sent == ["h1", "h2", "c1", "c2", "a1", "a2", "n1"]
    assert state["stopped_by_budget"] is False and state["not_sent"] == {}


def test_a_budget_stop_cuts_the_last_stages_and_counts_what_it_cut():
    fake, state = _staged(400)
    assert fake.sent == ["h1", "h2", "c1", "c2"]
    assert state["stopped_by_budget"] is True
    assert state["not_sent"] == {"arms": 2, "noplan": 1}


def test_a_stop_inside_a_stage_keeps_the_earlier_ones_whole():
    fake, state = _staged(300)
    assert fake.sent == ["h1", "h2", "c1"]
    assert state["not_sent"] == {"careful": 1, "arms": 2, "noplan": 1}


def test_stages_share_one_budget():
    _, state = _staged(10**6)
    assert state["tokens"] == 700 and state["finished"] == 7


def test_the_arms_draw_on_the_runs_budget_not_their_own():
    source = _function("launch_run_remote")
    assert "_launch_arms(run, '', ARMS, episode_budget, current, guard)" in source


def test_a_later_pass_keeps_what_an_earlier_one_failed():
    source = _function("score_run_remote")
    assert "scoring_summaries/" in source and "scoring_summary.json" in source


# --- the dollar limit over the whole run (addenda, section 12) ---


class Priced:
    """A remote function whose episodes each cost `cost` at estimate prices."""

    def __init__(self, cost=1.0):
        self.sent, self.cost = [], cost

    def remote(self, job):
        self.sent.append(job["config"]["episode_id"] if "config" in job else job["arm"])
        return {**_result("x"), "cost": self.cost}


def _episode(name, role="honest"):
    return {"config": {"episode_id": name, "role": role}}


def _guard(limit, **options):
    return modal_app.DollarGuard(limit, **options)


def test_the_guard_counts_an_episodes_scoring_from_the_moment_it_is_sent():
    guard, state = _guard(100.0), modal_app._new_state()
    assert guard.projected_usd(state) == 0
    guard.note_sent(_episode("a", "honest_hard"))  # still running: nothing has been billed for it
    expected = modal_app.BILLED_RATIO * modal_app.SCORING_COST["honest_hard"]
    assert guard.projected_usd(state) == pytest.approx(expected)


def test_a_real_scoring_cost_replaces_the_estimate():
    guard, state = _guard(100.0), modal_app._new_state()
    guard.note_sent(_episode("a"))
    guard.scored["a"] = 0.5
    assert guard.projected_usd(state) == pytest.approx(modal_app.BILLED_RATIO * 0.5)


def test_the_guard_applies_the_billed_ratio_to_episodes_too():
    guard, state = _guard(100.0), modal_app._new_state()
    state["cost"] = 10.0
    assert guard.projected_usd(state) == pytest.approx(10.0 * modal_app.BILLED_RATIO)


def test_replays_spend_money_but_commit_no_scoring():
    guard, state = _guard(100.0), modal_app._new_state()
    guard.note_sent({"arm": "replace_a", "base_stem": "x"})
    assert guard.sent == {} and guard.projected_usd(state) == 0


def test_a_dollar_stop_sends_nothing_more_and_counts_what_it_cut():
    # Each episode: 1.0 of its own plus its scoring, at the billed ratio.
    per_episode = modal_app.BILLED_RATIO * (1.0 + modal_app.SCORING_COST["honest"])
    guard, fake, state = _guard(per_episode * 2 - 0.01), Priced(), modal_app._new_state()
    state["stage"] = "careless"
    jobs = [_episode(f"e{i}") for i in range(5)]
    modal_app._run_jobs(fake, jobs, 10**12, lambda s: None, state=state, workers=1, guard=guard)
    assert fake.sent == ["e0", "e1"]
    assert state["stopped_by_budget"] is True and state["not_sent"] == {"careless": 3}
    assert state["projected_usd"] == pytest.approx(per_episode * 2, abs=0.01)


def test_with_the_real_grid_only_the_last_stage_is_cut_by_a_small_overrun():
    from experiments.grid import v2_main_stages

    def jobs(name):
        return [{"config": vars(c)} for c in v2_main_stages()[name]]

    cost = {"collude_split": 0.1249, "collude": 0.1468, "honest": 0.0677, "honest_hard": 0.0875}

    class ByRole:
        def __init__(self, overrun):
            self.overrun, self.sent = overrun, []

        def remote(self, job):
            self.sent.append(job)
            role = job["config"]["role"] if "config" in job else None
            amount = cost[role] if role else 0.1693 / 4
            return {**_result("x"), "cost": amount * self.overrun}

    def run(overrun, limit=27.73):
        guard, fake, state = _guard(limit), ByRole(overrun), modal_app._new_state()
        stages = {name: jobs(name) for name in ("honest", "careful", "noplan", "careless")}
        stages = {
            "honest": stages["honest"],
            "careful": stages["careful"],
            "arms": [],
            "noplan": stages["noplan"],
            "careless": stages["careless"],
        }
        arm_jobs = [{"arm": str(i)} for i in range(120)]

        def episodes(group, current):
            modal_app._run_jobs(
                fake, group, 10**12, lambda s: None, state=current, workers=1, guard=guard
            )

        def arms(current):
            modal_app._run_jobs(
                fake, arm_jobs, 10**12, lambda s: None, state=current, workers=1, guard=guard
            )

        return modal_app._run_stages(stages, state, episodes, arms)

    assert run(1.0)["not_sent"] == {}  # the estimate fits under the limit
    assert set(run(1.15)["not_sent"]) == {"careless"}
    assert set(run(1.40)["not_sent"]) >= {"noplan", "careless"}  # a limit, not a promise


def test_a_failed_read_of_the_scoring_costs_keeps_the_estimates():
    def broken(known):
        raise OSError("volume busy")

    guard, state = _guard(100.0, read_scored=broken), modal_app._new_state()
    guard.note_sent(_episode("a"))
    assert guard.over(state) is False
    assert "a" not in guard.scored


def test_the_scoring_costs_are_read_at_most_once_a_minute():
    reads, now = [], [0.0]
    guard = _guard(100.0, read_scored=lambda known: reads.append(1), clock=lambda: now[0])
    state = modal_app._new_state()
    guard.over(state), guard.over(state)
    assert len(reads) == 1
    now[0] = 61.0
    guard.over(state)
    assert len(reads) == 2


def test_a_relaunch_counts_what_the_run_already_spent():
    guard, state = _guard(100.0), modal_app._new_state()
    guard.seed(
        [
            {"episode_id": "a", "cost_estimate_usd": 2.0, "config": {"role": "honest"}},
            {
                "episode_id": "b__intervene_replace_a",
                "cost_estimate_usd": 1.0,
                "config": {"role": "honest"},
            },
            {"episode_id": "c", "cost_estimate_usd": None, "config": {"role": "collude"}},
        ],
        state,
    )
    assert state["cost"] == 3.0
    assert guard.sent == {"a": "honest", "c": "collude"}  # a replay is never scored
