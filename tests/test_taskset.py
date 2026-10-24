"""The task files are consistent (no Modal, no LLM)."""

import importlib.util
import inspect
import io
import tarfile
from pathlib import Path

import pytest

from crow.manifest import allowed, is_artifact, matches
from env.taskset import STARTER_MTIME, TASKS_DIR, VARIANTS, load_task, task_ids

TASKS = task_ids()
CASES = [(t, v) for t in TASKS for v in VARIANTS]


def load(path: Path):
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
