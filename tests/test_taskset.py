"""The task files are consistent (no Modal, no LLM)."""

import importlib.util
import inspect
import io
import re
import sys
import tarfile
from pathlib import Path

import pytest

from crow.manifest import allowed, is_artifact, matches
from env.taskset import STARTER_MTIME, TASKS_DIR, VARIANTS, load_task, task_ids

TASKS = task_ids()
CASES = [(t, v) for t in TASKS for v in VARIANTS]


def load(path: Path):
    sys.dont_write_bytecode = True  # no __pycache__ inside env/tasks
    spec = importlib.util.spec_from_file_location(f"m_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_six_tasks() -> None:
    assert TASKS == ["config", "durations", "ledger", "schedule", "slug", "table"]


@pytest.mark.parametrize(("task_id", "variant"), CASES)
def test_specs_render(task_id: str, variant: str) -> None:
    task = load_task(task_id)
    for agent in "AB":
        text = task.spec(agent, variant)
        assert "{{" not in text
        assert f"lib/{task.module}.py" in text
        assert "HANDOFF.md" in text
    if variant == "hard":
        spec_a = task.spec("A", variant)
        assert task.meta["hard"]["fixture"] in spec_a
        assert task.meta["hard"]["sample"] in spec_a


@pytest.mark.parametrize(("task_id", "variant"), CASES)
def test_manifest_covers_starter_and_reference(task_id: str, variant: str) -> None:
    task = load_task(task_id)
    manifest = task.manifest(variant)
    files = task.starter_files(variant) | task.reference_files(variant)
    files |= {"HANDOFF.md": b"", "tests/test_app.py": b"", "tests/__init__.py": b""}
    assert [p for p in files if not allowed(p, manifest)] == []


@pytest.mark.parametrize("task_id", TASKS)
def test_base_manifest_excludes_hard_files(task_id: str) -> None:
    task = load_task(task_id)
    hard_only = set(task.starter_files("hard")) - set(task.starter_files("base"))
    hard_only.add(task.meta["hard"]["fixture"])
    assert hard_only
    assert not [p for p in hard_only if allowed(p, task.manifest("base"))]


@pytest.mark.parametrize("task_id", TASKS)
def test_reference_keeps_stub_signatures(task_id: str) -> None:
    task = load_task(task_id)
    stub = load(task.dir / "starter" / "lib" / f"{task.module}.py")
    reference = load(task.hidden_dir / "reference" / "lib" / f"{task.module}.py")
    names = [n for n, f in inspect.getmembers(stub, inspect.isfunction)]
    assert len(names) == 3
    for name in names:
        assert inspect.signature(getattr(reference, name)) == inspect.signature(getattr(stub, name))
        with pytest.raises(NotImplementedError):
            getattr(stub, name)(*[None] * len(inspect.signature(getattr(stub, name)).parameters))


def test_agent_visible_dirs_hold_no_hidden_material() -> None:
    for path in TASKS_DIR.rglob("*"):
        assert not path.name.startswith("test_"), path
        assert "reference" not in path.parts, path
        if path.is_file():
            assert b"tasks_hidden" not in path.read_bytes(), path


def test_starter_tar_is_deterministic() -> None:
    task = load_task("ledger")
    tar = task.starter_tar("hard")
    assert tar == task.starter_tar("hard")
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for member in t.getmembers():
            assert member.mode == 0o644
            assert member.mtime == STARTER_MTIME


def test_manifest_matching() -> None:
    assert matches("tests/test_app.py", "tests/test_*.py")
    assert not matches("tests/sub/test_app.py", "tests/test_*.py")
    assert not matches("lib/ledger.py.bak", "lib/ledger.py")
    assert is_artifact("lib/__pycache__/ledger.cpython-312.pyc")
    assert is_artifact(".pytest_cache/v/cache/nodeids")
    assert not is_artifact("lib/ledger.py")


# --- Crow v2 (three agents) ---


@pytest.mark.parametrize(("task_id", "variant"), CASES)
def test_three_agent_specs_render(task_id: str, variant: str) -> None:
    task = load_task(task_id)
    for agent in "ABC":
        text = task.spec3(agent, variant)
        assert f"(you are agent {agent})" in text
        assert "{" + "{" not in text and "{module}" not in text and "{artifact}" not in text
        assert f"lib/{task.module}.py" in text and "HANDOFF.md" in text
        assert task.b_artifact in text


@pytest.mark.parametrize(("task_id", "variant"), CASES)
def test_three_agent_rename_only_touches_the_agent(task_id: str, variant: str) -> None:
    task = load_task(task_id)
    spec_a, spec_c = task.spec3("A", variant), task.spec3("C", variant)
    # In A's spec the app is C's now, and B appears only in the note about the cases file.
    body = spec_a.rsplit("\n\n", 1)[0]
    assert not re.search(r"\bB\b", body)
    assert "Another agent, C, will write a command-line app" in body
    assert "(you are agent B)" not in spec_c
    if variant == "hard":
        assert task.meta["hard"]["sample"] in spec_a  # the sample data is left alone
        assert task.meta["hard"]["fixture"] in spec_a


@pytest.mark.parametrize(("task_id", "variant"), CASES)
def test_two_agent_specs_and_manifests_are_unchanged(task_id: str, variant: str) -> None:
    task = load_task(task_id)
    assert "you are agent A" in task.spec("A", variant)
    assert "you are agent B" in task.spec("B", variant)
    assert task.b_artifact not in task.spec("A", variant) + task.spec("B", variant)
    assert not allowed(task.b_artifact, task.manifest(variant))
    assert allowed(task.b_artifact, task.manifest(variant, agents=3))
    assert task.manifest(variant, agents=3) == [*task.manifest(variant), task.b_artifact]
