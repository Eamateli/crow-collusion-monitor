"""The 6 tasks: specs for A and B, starter workspace and manifest, per variant.

env/tasks/<id>/ holds what agents may see (task.json, spec_a.md, spec_b.md, hard_a.md,
hard_b.md, starter/, starter_hard/). env/tasks_hidden/<id>/ holds the hidden tests, the
reference solution and a deliberately wrong solution; none of it is ever copied into an agent
sandbox. Variant "base" serves the honest and collude roles, "hard" the honest_hard role.
"""

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crow.manifest import is_artifact
from crow.workspace import pack_tree

ENV_DIR = Path(__file__).resolve().parent
TASKS_DIR = ENV_DIR / "tasks"
HIDDEN_DIR = ENV_DIR / "tasks_hidden"
COMMON_HIDDEN_DIR = HIDDEN_DIR / "_common"
VARIANTS = ("base", "hard")
STARTER_MTIME = 1_767_225_600  # 2026-01-01 00:00 UTC: starter files are identical in every episode


def _files(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}
    files = (p for p in root.rglob("*") if p.is_file())
    # Build artifacts (__pycache__ from imports in tests) are never task files.
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in files
        if not is_artifact(p.relative_to(root).as_posix())
    }


@dataclass(frozen=True)
class Task:
    id: str
    meta: dict[str, Any]

    @property
    def module(self) -> str:
        return self.meta["module"]

    @property
    def dir(self) -> Path:
        return TASKS_DIR / self.id

    @property
    def hidden_dir(self) -> Path:
        return HIDDEN_DIR / self.id

    def spec(self, agent: str, variant: str) -> str:
        """The task text for agent "A" or "B"."""
        text = (self.dir / f"spec_{agent.lower()}.md").read_text(encoding="utf-8")
        if variant == "hard":
            extra = (self.dir / f"hard_{agent.lower()}.md").read_text(encoding="utf-8")
            hard = self.meta["hard"]
            extra = extra.replace("{{FIXTURE}}", hard["fixture"])
            extra = extra.replace("{{SAMPLE}}", f"```\n{hard['sample']}```")
            text = f"{text.rstrip()}\n\n{extra}"
        return text

    def manifest(self, variant: str) -> list[str]:
        extra = self.meta["hard"]["manifest"] if variant == "hard" else []
        return [*self.meta["manifest"], *extra]

    def starter_files(self, variant: str) -> dict[str, bytes]:
        files = _files(self.dir / "starter")
        if variant == "hard":
            files |= _files(self.dir / "starter_hard")
        return files

    def starter_tar(self, variant: str) -> bytes:
        return pack_tree(self.starter_files(variant), STARTER_MTIME)

    def reference_files(self, variant: str) -> dict[str, bytes]:
        """What a correct A and B would add to the starter (no HANDOFF.md needed)."""
        files = _files(self.hidden_dir / "reference")
        if variant == "hard":
            sample = self.meta["hard"]["sample"].encode("utf-8")
            files[self.meta["hard"]["fixture"]] = base64.b64encode(sample) + b"\n"
        return files

    def wrong_files(self) -> dict[str, bytes]:
        return _files(self.hidden_dir / "wrong")

    def grader_files(self) -> dict[str, bytes]:
        """Everything the grading sandbox needs in /grader."""
        files = _files(COMMON_HIDDEN_DIR)
        for name in ("test_lib.py", "test_app.py"):
            files[name] = (self.hidden_dir / name).read_bytes()
        files["task.json"] = (self.dir / "task.json").read_bytes()
        files["reference_lib.py"] = (
            self.hidden_dir / "reference" / "lib" / f"{self.module}.py"
        ).read_bytes()
        return files


def load_task(task_id: str) -> Task:
    meta = json.loads((TASKS_DIR / task_id / "task.json").read_text(encoding="utf-8"))
    return Task(task_id, meta)


def task_ids() -> list[str]:
    return sorted(p.name for p in TASKS_DIR.iterdir() if (p / "task.json").is_file())
