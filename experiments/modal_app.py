"""Modal app 'crow': run_episode(config) -> dict, grid runner, and --smoke."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# Same pinned versions as the local venv: both install from pyproject.toml.
image = modal.Image.debian_slim(python_version="3.12").pip_install_from_pyproject(
    str(ROOT / "pyproject.toml")
)
