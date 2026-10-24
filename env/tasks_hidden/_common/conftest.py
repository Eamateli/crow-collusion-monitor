"""Hidden-test plugin: the `lib` fixture, and each test's outcome written to $CROW_RESULTS."""

import json
import os

import pytest
from crowtest import load_lib

_outcomes: dict[str, str] = {}


@pytest.fixture
def lib():
    return load_lib()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.failed:
        _outcomes[report.nodeid] = "failed"  # setup, call or teardown
    elif report.skipped:
        _outcomes[report.nodeid] = "skipped"
    elif report.when == "call":
        _outcomes.setdefault(report.nodeid, "passed")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    path = os.environ.get("CROW_RESULTS")
    if path:
        items = [item.nodeid for item in session.items]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": items, "outcomes": _outcomes, "exitstatus": int(exitstatus)}, f)
