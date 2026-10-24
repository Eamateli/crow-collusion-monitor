"""Every scaffold module imports cleanly."""

import importlib

import pytest

MODULES = [
    "crow",
    "crow.events",
    "crow.llm",
    "crow.manifest",
    "crow.workspace",
    "crow.wrap",
    "crow.detectors",
    "crow.detectors.heuristics",
    "crow.detectors.judge",
    "crow.seal",
    "crow.score",
    "crow.cli",
    "env",
    "env.prompts",
    "env.agent",
    "env.sandbox",
    "env.episode",
    "env.taskset",
    "env.grading",
    "experiments",
    "experiments.modal_app",
    "experiments.grid",
    "experiments.calibrate",
    "analysis",
    "analysis.analyze",
    "analysis.human_check",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
