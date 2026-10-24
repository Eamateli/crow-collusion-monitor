"""Checks the agent sandbox on Modal (Step 3).

(a) an agent writes a file and reads it back, (b) the network is blocked, (c) the sandbox
environment holds no token or secret. Also shows how exec timeouts surface, checks that
copy_in/copy_out keep bytes, modes and mtimes, and prints token use per model call.

Loads .env with python-dotenv and never prints its values. Run: python scripts/check_sandbox.py
"""

import io
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from crow.llm import modal_endpoint
from env.agent import run_agent, sandbox_tools
from env.prompts import AGENT_MODEL_ID, AGENT_PARAMS
from env.sandbox import Sandbox, SandboxGone, pack_dir

ROOT = Path(__file__).resolve().parent.parent
APP_NAME = "crow-dev"
MTIME = 1_700_000_000.25
ENV_KEYS = [
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
    "AGENT_ENDPOINT_URL",
    "AGENT_MODEL",
    "TRUSTED_ENDPOINT_URL",
    "TRUSTED_MODEL",
]
# ENV GPG_KEY in github.com/docker-library/python 3.12/slim-bookworm/Dockerfile (public).
PYTHON_RELEASE_FINGERPRINT = "7169605F62C751356D054A26A821E680E5FA6305"

console = Console()
results: dict[str, bool] = {}


def record(name: str, ok: bool, detail: object = "") -> None:
    results[name] = ok
    tag = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"{tag} {name} {detail}")


def check_network(sb: Sandbox) -> None:
    console.rule("(b) network")
    which = sb.exec("command -v curl")
    record("curl is installed", which.returncode == 0, which.stdout.strip())
    probes = {
        "curl": sb.exec("curl -sS -m 10 https://example.com"),
        "python urllib": sb.exec(
            "python -c \"import urllib.request; urllib.request.urlopen('https://example.com', "
            'timeout=10)"'
        ),
        "dns (getent)": sb.exec("getent hosts example.com"),
    }
    for name, r in probes.items():
        console.print(f"  {name}: exit {r.returncode}, stderr tail {r.stderr.strip()[-160:]!r}")
    ok = all(r.returncode not in (0, 127) for r in probes.values())
    record("(b) curl, urllib and DNS all fail", ok)


def check_env(sb: Sandbox) -> None:
    console.rule("(c) environment")
    out = sb.exec("env -0").stdout
    sandbox_env = dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
    names = sorted(sandbox_env)
    console.print(f"  variable names: {names}")  # names only, never values
    flagged = [n for n in names if any(w in n.upper() for w in ("TOKEN", "SECRET", "KEY"))]
    # GPG_KEY comes from the official python base image (FROM python:3.12-slim-...): the
    # public fingerprint of the CPython release-signing key, used to verify the source tarball.
    if "GPG_KEY" in flagged:
        r = sb.exec(f'[ "$GPG_KEY" = "{PYTHON_RELEASE_FINGERPRINT}" ]')
        console.print(
            f"  GPG_KEY equals the python image's public release-key fingerprint: "
            f"{r.returncode == 0}"
        )
        if r.returncode == 0:
            flagged.remove("GPG_KEY")
    record("(c) no TOKEN, SECRET or KEY variables (GPG_KEY checked)", not flagged, flagged)
    # None of our six .env values may appear anywhere in the sandbox environment.
    ours = [v for k in ENV_KEYS if (v := os.environ.get(k))]
    leaked = [k for k, v in sandbox_env.items() if any(o in v for o in ours)]
    record(
        "(c) no .env value appears in the sandbox environment", bool(ours) and not leaked, leaked
    )


def check_timeout(sb: Sandbox) -> None:
    console.rule("exec timeout")
    # Raw Modal behaviour (private handle on purpose: this is what Sandbox.exec must handle).
    t0 = time.perf_counter()
    proc = sb._sb.exec("bash", "-c", "echo start; sleep 15; echo end", timeout=5, text=False)
    raw: dict[str, object] = {}
    for step in ("stdout", "wait"):
        try:
            raw[step] = proc.stdout.read() if step == "stdout" else proc.wait()
        except Exception as e:
            raw[step] = f"raised {type(e).__module__}.{type(e).__name__}"
    console.print(f"  raw modal: {raw}, elapsed {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    r = sb.exec("echo start; sleep 15; echo end", timeout=5)
    elapsed = time.perf_counter() - t0
    console.print(f"  Sandbox.exec: {r}, elapsed {elapsed:.1f}s")
    record("timeout stops the command and is reported", elapsed < 12 and r.timed_out)


def check_files(sb: Sandbox) -> None:
    console.rule("copy_in / copy_out / write_file")
    data = "echo hi\u200b\u00e9\n".encode()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "lib").mkdir()
        script = root / "lib" / "run.sh"
        script.write_bytes(data)
        script.chmod(0o755)
        os.utime(script, (MTIME, MTIME))
        (root / "odd name \u2713.txt").write_text("x")
        sb.copy_in(pack_dir(root))

    with tarfile.open(fileobj=io.BytesIO(sb.copy_out())) as tar:
        members = {m.name.removeprefix("./"): m for m in tar.getmembers()}
        m = members.get("lib/run.sh")
        got = tar.extractfile(m).read() if m else None
    console.print(f"  members: {sorted(members)}")
    if m:
        console.print(f"  lib/run.sh mode {oct(m.mode & 0o777)} mtime {m.mtime}")
    ok = (
        m is not None
        and got == data
        and m.mode & 0o777 == 0o755
        and abs(m.mtime - MTIME) < 1e-6
        and "odd name \u2713.txt" in members
    )
    record("round trip keeps bytes, mode, sub-second mtime, odd names", ok)

    tools = sandbox_tools(sb)
    text = "a\u200bb\ufeffc\u00a0d\n"
    console.print(f"  write_file: {tools['write_file'].fn('notes/zw.txt', text)}")
    same = sb.read_bytes("notes/zw.txt") == text.encode()
    mode = sb.exec("stat -c %a /work/notes/zw.txt").stdout.strip()
    record("write_file keeps exact UTF-8 bytes", same, f"(new file mode {mode})")


def check_agent(app: modal.App) -> None:
    console.rule("(a) agent writes and reads back a file")
    client = modal_endpoint("agent", os.environ["AGENT_ENDPOINT_URL"], AGENT_MODEL_ID)
    system = "You are a coding agent. You work in /work through the tools you are given."
    user = (
        "Create the file notes/hello.txt containing exactly: hello from crow\n"
        "Then read it back with read_file to confirm, and call done()."
    )
    t0 = time.perf_counter()
    with Sandbox(app=app) as sb:
        result = run_agent(
            client,
            system,
            user,
            sandbox_tools(sb),
            affinity_key="check-sandbox",
            params=AGENT_PARAMS,
        )
        with tarfile.open(fileobj=io.BytesIO(sb.copy_out())) as tar:
            members = {m.name.removeprefix("./"): m for m in tar.getmembers()}
            m = members.get("notes/hello.txt")
            content = tar.extractfile(m).read().decode() if m and m.isfile() else None
    wall = time.perf_counter() - t0

    table = Table("#", "finish", "tools", "prompt", "completion", "reasoning", "latency s")
    for i, c in enumerate(result.calls):
        names = ",".join(tc["function"]["name"] for tc in c.tool_calls) or "-"
        table.add_row(
            str(i),
            str(c.finish_reason),
            names,
            str(c.usage.get("prompt_tokens")),
            str(c.usage.get("completion_tokens")),
            str(c.reasoning_tokens),
            f"{c.latency_s:.1f}",
        )
    console.print(table)
    prompt = sum(c.usage.get("prompt_tokens", 0) for c in result.calls)
    completion = sum(c.usage.get("completion_tokens", 0) for c in result.calls)
    lengths = sum(c.finish_reason == "length" for c in result.calls)
    cost = (prompt * 0.30 + completion * 1.20) / 1e6
    console.print(
        f"  stop={result.stop_reason} error={result.error} nudges={result.nudges} "
        f"work_tool_calls={result.work_tool_calls} wall={wall:.1f}s"
    )
    console.print(
        f"  tokens: prompt {prompt}, completion {completion}; finish_reason=length "
        f"{lengths}/{len(result.calls)}; cost estimate without cache discount ${cost:.4f}"
    )
    console.print(f"  notes/hello.txt in copy_out: {content!r}")

    used = [tc["function"]["name"] for c in result.calls for tc in c.tool_calls]
    ok = (
        content is not None
        and content.strip() == "hello from crow"
        and "write_file" in used
        and "read_file" in used[used.index("write_file") :]
        and result.stop_reason == "done"
    )
    record("(a) agent wrote notes/hello.txt, read it back, called done()", ok)


def main() -> int:
    load_dotenv(ROOT / ".env")
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    with Sandbox(app=app) as sb:
        check_network(sb)
        check_env(sb)
        check_timeout(sb)
        check_files(sb)
    try:
        sb.exec("true")
        record("calls after terminate() raise SandboxGone", False, "no exception")
    except SandboxGone as e:
        record("calls after terminate() raise SandboxGone", True, f"({str(e)[:80]})")
    except Exception as e:
        record("calls after terminate() raise SandboxGone", False, f"{type(e).__name__}: {e}")
    check_agent(app)

    console.rule("summary")
    for name, ok in results.items():
        console.print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
