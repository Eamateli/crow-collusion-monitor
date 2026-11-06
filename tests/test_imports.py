"""Every scaffold module imports cleanly."""

import importlib

import pytest

MODULES = [
    "crow",
    "crow.events",
    "crow.llm",
    "crow.manifest",
    "crow.workspace",
    "env.prompts",
    "crow.wrap",
    "crow.detectors",
    "crow.detectors.heuristics",
    "crow.detectors.judge",
    "crow.detectors.postcondition",
    "crow.detectors.views3",
    "crow.assemble",
    "crow.seal",
    "crow.score",
    "crow.cli",
    "env",
    "env.prompts",
    "env.relational",
    "env.planted",
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
    "analysis.leak",
    "analysis.integrity",
    "analysis.analyze",
    "analysis.human_check",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
