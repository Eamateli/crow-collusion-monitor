"""The Modal wiring of Crow v2, checked without Modal: no container is started here."""

import ast
import inspect

from experiments import modal_app


def test_every_v2_run_has_its_own_folder():
    for run in ("v2_smoke", "v2_calibration", "v2_main", "v2_intervention"):
        assert run in modal_app.RUNS
    assert set(modal_app.V2_GRIDS) == {"v2_smoke", "v2_calibration", "v2_main"}


def test_the_scoring_timeout_is_within_modals_limit_and_covers_the_estimate():
    # Modal allows 1 second to 24 hours per attempt (modal.com/docs/guide/timeouts).
    assert 4 * 3600 <= modal_app.SCORING_TIMEOUT <= 24 * 3600


def test_entrypoint_takes_the_new_options():
    # Modal wraps the entrypoint, so read its signature from the source.
    tree = ast.parse(inspect.getsource(modal_app))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert {"base_run", "arms", "scoring_tokens"} <= {a.arg for a in main.args.args}


def test_the_grids_are_the_v2_ones():
    assert len(modal_app.V2_GRIDS["v2_main"]()) == 130
    assert len(modal_app.V2_GRIDS["v2_calibration"]()) == 30
    assert len(modal_app.V2_GRIDS["v2_smoke"]()) == 12


def test_the_runner_tells_the_pass_when_its_episodes_have_ended():
    source = inspect.getsource(modal_app)
    start = source.index("scoring = score_run_remote.spawn")
    end = source.index("scoring.get()")
    assert source.rindex("set_episodes_done.remote(run, False)", 0, start) < start
    assert source.index("set_episodes_done.remote(run, True)", start) < end
