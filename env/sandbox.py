"""Modal Sandbox wrapper.

Creates a network-blocked sandbox, runs commands with a timeout, copies /work in and out,
and terminates the sandbox.

Workspaces move as POSIX tar bytes (modes, sub-second mtimes and odd names survive), so
agent-written files are never extracted outside a sandbox.
"""

import io
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import modal
import modal.exception

WORKDIR = "/work"
SANDBOX_TIMEOUT_S = 2400  # long reasoning must not kill a turn (Step 6)
EXEC_TIMEOUT_S = 30

# Agent sandboxes: no project code, no secrets. curl only so the network block can be tested.
AGENT_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("pytest==9.1.1")  # same pin as the dev extra in pyproject.toml
    .run_commands(f"mkdir -p {WORKDIR}")
)

# NotFoundError is what calls on a terminated sandbox raise (seen in scripts/check_sandbox.py).
# Missing files raise SandboxFilesystemNotFoundError, which is not a subclass, so stay non-fatal.
_FATAL = (
    modal.exception.SandboxTerminatedError,
    modal.exception.SandboxTimeoutError,
    modal.exception.NotFoundError,
)


class SandboxGone(Exception):
    """The sandbox was terminated or hit its lifetime limit; the turn cannot continue."""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool


def resolve(path: str) -> str:
    """Absolute sandbox path; relative paths are resolved against /work."""
    return posixpath.normpath(posixpath.join(WORKDIR, path))


def pack_dir(path: Path) -> bytes:
    """Local directory -> tar bytes for copy_in. Members are relative to the directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for child in sorted(path.iterdir()):
            tar.add(child, arcname=child.name)
    return buf.getvalue()


class Sandbox:
    """One network-blocked sandbox. Use as a context manager so it is always terminated."""

    def __init__(self, app: modal.App | None = None, image: modal.Image = AGENT_IMAGE) -> None:
        # app may be None only inside a Modal container (Modal docs, Sandboxes guide).
        self._sb = modal.Sandbox.create(
            app=app,
            image=image,
            block_network=True,
            timeout=SANDBOX_TIMEOUT_S,
            workdir=WORKDIR,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.terminate()

    def exec(self, cmd: str, timeout: int = EXEC_TIMEOUT_S) -> ExecResult:
        try:
            proc = self._sb.exec("bash", "-c", cmd, timeout=timeout, workdir=WORKDIR, text=False)
            stdout, stderr = proc.stdout.read(), proc.stderr.read()
            returncode = proc.wait()
        except _FATAL as e:
            raise SandboxGone(str(e)) from e
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            returncode=returncode,
            # On timeout Modal raises nothing: output so far comes back and wait() returns -1
            # (seen in scripts/check_sandbox.py). Exit codes from bash are 0 to 255.
            timed_out=returncode == -1,
        )

    def read_bytes(self, path: str) -> bytes:
        try:
            return self._sb.filesystem.read_bytes(resolve(path))
        except _FATAL as e:
            raise SandboxGone(str(e)) from e

    def write_bytes(self, path: str, data: bytes) -> None:
        try:
            self._sb.filesystem.write_bytes(data, resolve(path))
        except _FATAL as e:
            raise SandboxGone(str(e)) from e

    def copy_in(self, tar: bytes) -> None:
        """Extract tar bytes into /work, keeping modes and mtimes."""
        tmp = self._run("mktemp").stdout.strip()
        self.write_bytes(tmp, tar)
        self._run(f"tar -x -p --no-same-owner -C {WORKDIR} -f {tmp} && rm -f {tmp}")

    def copy_out(self) -> bytes:
        """/work as POSIX tar bytes."""
        tmp = self._run("mktemp").stdout.strip()
        self._run(f"tar --format=posix -C {WORKDIR} -cf {tmp} .")
        data = self.read_bytes(tmp)
        self._run(f"rm -f {tmp}")
        return data

    def terminate(self) -> None:
        self._sb.terminate()

    def _run(self, cmd: str) -> ExecResult:
        """Internal command that must succeed."""
        result = self.exec(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"sandbox command failed ({result.returncode}): {result.stderr}")
        return result
