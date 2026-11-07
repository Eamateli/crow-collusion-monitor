"""The Modal wiring of Crow v2, checked without Modal: no container is started here."""

import ast
import inspect

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
    assert len(modal_app.V2_GRIDS["v2_main"]()) == 180
    assert len(modal_app.V2_GRIDS["v2_calibration"]()) == 30
    assert len(modal_app.V2_GRIDS["v2_resmoke"]()) == 6


def _function(name: str) -> str:
    # Modal wraps its functions, so read them from the module's source.
    tree = ast.parse(inspect.getsource(modal_app))
    return ast.unparse(next(n for n in tree.body if getattr(n, "name", None) == name))


def test_the_launcher_tells_the_pass_when_its_episodes_have_ended():
    source = _function("launch_run_remote")
    assert source.index("_set_episodes_done(run, False)") < source.index("score_run_remote.spawn")
    assert source.index("_run_jobs(") < source.index("_set_episodes_done(run, True)")
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
