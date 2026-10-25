"""Crow CLI: crow observe | crow score | crow seal (CLAUDE.md section 11), using rich.

    python -m crow.cli observe EPISODE [--follow [--idle-exit S]]
    python -m crow.cli score EPISODE [--findings N]
    python -m crow.cli seal --mode side|side_para --manifest FILE IN_DIR OUT_DIR

EPISODE is an episode's events file (<run>/events/<stem>.jsonl) or its record file
(<run>/records/<stem>.json); the other one is found next to it, as the Modal runs store them.
The inline flags in `observe` are computed live from each tool call with the same line checks as
the heuristics; the authoritative scores are the ones in the record (`score`).
"""

import argparse
import io
import json
import os
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Line-level checks shared with the heuristics (read-only use; the detectors are not changed).
from crow.detectors.heuristics import (
    _INJECTION,
    _META_COMMAND,
    blobs,
    name_anomaly,
    unusual_characters,
)
from crow.events import Event
from crow.seal import seal_side, seal_side_para

console = Console()
CHANNELS = {
    "C1": "HANDOFF.md text",
    "C2": "file or directory names",
    "C3": "whitespace or invisible characters",
    "C4": "code comments or docstrings",
    "C5": "extra or hidden files",
    "C6": "code semantics",
    "C7": "metadata",
    "C8": "other",
}
AGENT_STYLE = {"A": "cyan", "B": "magenta", "crow": "yellow"}


def episode_paths(arg: str) -> tuple[Path | None, Path | None]:
    """(events file, record file) for an episode, each None if it does not exist."""
    p = Path(arg)
    if p.suffix == ".jsonl":
        events, record = p, p.parent.parent / "records" / f"{p.stem}.json"
    elif p.suffix == ".json":
        record, events = p, p.parent.parent / "events" / f"{p.stem}.jsonl"
    else:
        raise SystemExit(f"give an events .jsonl or a record .json file, not {arg}")
    return (events if events.exists() else None), (record if record.exists() else None)


def flags(event: Event) -> list[str]:
    """Detector hints for one tool call: which checks its written text or command trips."""
    if event.type != "tool_call":
        return []
    args = event.args or {}
    found: Counter[str] = Counter()
    if event.tool == "write_file":
        path = str(args.get("path", ""))
        found["filename"] += sum(1 for part in path.split("/") if name_anomaly(part))
        text, is_python = str(args.get("content", "")), path.endswith(".py")
    elif event.tool == "bash":
        text, is_python = str(args.get("cmd", "")), False
        found["metadata"] += bool(_META_COMMAND.search(text))
    else:
        return []
    for line in text.split("\n"):
        found["invisible"] += bool(unusual_characters(line, is_python))
        found["whitespace"] += line != line.rstrip()
        found["encoded"] += len(blobs(line))
        found["injection"] += sum(1 for pattern in _INJECTION if pattern.search(line))
    return [f"{name}x{n}" for name, n in found.items() if n]


def _one_line(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def show_event(event: Event, t0: float) -> None:
    style = AGENT_STYLE.get(event.agent, "white")
    head = f"[dim]+{event.ts - t0:7.1f}s[/dim] [{style}]{event.agent:4}[/{style}]"
    if event.type == "tool_call":
        args = event.args or {}
        if event.tool == "bash":
            body = f"$ {args.get('cmd', '')}"
        elif event.tool == "write_file":
            body = f"write {args.get('path')} ({len(str(args.get('content', '')))} chars)"
        else:
            body = ", ".join(f"{k}={v}" for k, v in args.items())
        marks = flags(event)
        flag = f"  [bold red]flags: {', '.join(marks)}[/bold red]" if marks else ""
        console.print(f"{head} [bold]{event.tool}[/bold] {_one_line(body)}{flag}", markup=True)
    elif event.type == "tool_result":
        console.print(f"{head}   [dim]-> {_one_line(event.result or '')}[/dim]")
    elif event.type == "snapshot":
        files = event.meta.get("files", [])
        console.print(
            f"{head} [bold]snapshot[/bold] {event.meta.get('label', '')}: {len(files)} paths"
        )
    elif event.type == "seal":
        meta = event.meta
        console.print(
            f"{head} [bold]seal[/bold] {meta.get('mode')}: dropped {len(meta.get('dropped', []))}, "
            f"python {len(meta.get('python_ast', []))}, text {len(meta.get('text', []))}, "
            f"paraphrased {meta.get('paraphrased')}"
        )
    else:
        console.print(f"{head} [bold]{event.type}[/bold]")


FINDINGS_SHOWN = 20


def _record(record_path: Path | None) -> dict | None:
    if record_path is None:
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return record if record.get("status") == "ok" else None


def show_findings(record: dict) -> None:
    """Crow's snapshot findings from the record, shown under the "after A" snapshot. They catch
    what the per-call flags cannot, e.g. a file changed by a command rather than written."""
    lines = [
        f"{name} {f['channel']} {f['path']}{':' + str(f['line']) if f.get('line') else ''} "
        f"{f['detail']}"
        for name, items in record["scores"]["crow"]["findings"].items()
        for f in items
    ]
    if not lines:
        console.print("           [dim]Crow findings on this snapshot: none[/dim]")
        return
    console.print(f"           [bold red]Crow findings on this snapshot ({len(lines)}):[/bold red]")
    for line in lines[:FINDINGS_SHOWN]:
        console.print(f"             {_one_line(line)}", markup=False)
    if len(lines) > FINDINGS_SHOWN:
        console.print(
            f"             ... {len(lines) - FINDINGS_SHOWN} more (crow score --findings)"
        )


def observe(args: argparse.Namespace) -> int:
    events_path, record_path = episode_paths(args.episode)
    if events_path is None:
        raise SystemExit(f"no events file for {args.episode}")
    record = _record(record_path)
    t0 = None
    last_new = time.monotonic()
    with events_path.open(encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                if line.strip():
                    event = Event(**json.loads(line))
                    t0 = event.ts if t0 is None else t0
                    show_event(event, t0)
                    if record and event.type == "snapshot" and event.meta.get("label") == "after A":
                        show_findings(record)
                    last_new = time.monotonic()
                continue
            if not args.follow:
                break
            if args.idle_exit and time.monotonic() - last_new > args.idle_exit:
                break
            time.sleep(0.5)
    if record and not args.follow:
        crow = record["scores"]["crow"]
        marks = ", ".join(f"{k}={v:.2f}" for k, v in crow["scores"].items() if v)
        console.print(
            f"\n[bold]Crow composite {crow['composite']:.3f}[/bold] "
            f"(tau {crow['tau']}, alarm {crow['alarm']}); nonzero: {marks or 'none'}"
        )
    return 0


def score(args: argparse.Namespace) -> int:
    _, record_path = episode_paths(args.episode)
    if record_path is None:
        raise SystemExit(f"no record file for {args.episode}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    console.rule(f"[bold]{record['episode_id']}[/bold] (attempt {record.get('attempt')})")
    if record.get("status") != "ok":
        console.print(f"[red]episode failed: {record.get('reason')}[/red]")
        return 1
    crow = record["scores"]["crow"]
    table = Table("detector", "score", "findings", "note")
    for name, value in crow["scores"].items():
        n = len(crow["findings"].get(name, []))
        cell = "missing" if value is None else f"{value:.3f}"
        table.add_row(name, cell, str(n) if name != "judge" else "-", crow["notes"].get(name, ""))
    console.print(table)
    console.print(
        f"[bold]composite {crow['composite']:.3f}[/bold]  tau {crow['tau']}  "
        f"alarm {crow['alarm']}  missing {crow['missing'] or 'none'}"
    )
    channels = Table("channel", "meaning", "findings")
    for code, meaning in CHANNELS.items():
        channels.add_row(code, meaning, str(crow["channels"].get(code, 0)))
    console.print(channels)
    baselines = Table("monitor", "score", "raw (0-10)", "missing")
    for name in ("b1", "b2"):
        m = record["scores"][name]
        baselines.add_row(
            name,
            "-" if m["score"] is None else f"{m['score']:.2f}",
            str(m["raw_score"]),
            str(m["missing"]),
        )
    judge = crow["judge"]
    baselines.add_row(
        "crow judge",
        "-" if judge["score"] is None else f"{judge['score']:.2f}",
        str(judge["raw_score"]),
        str(judge["score"] is None),
    )
    console.print(baselines)
    if args.findings:
        for name, items in crow["findings"].items():
            for f in items[: args.findings]:
                where = f"{f['path']}:{f['line']}" if f.get("line") else f["path"]
                console.print(f"  [{name}] {f['channel']} {where}  {f['detail']}")
    return 0


def load_manifest(path: str) -> list[str]:
    """A JSON list, a JSON object with a "manifest" list (task.json), or one pattern per line."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return data["manifest"] if isinstance(data, dict) else list(data)


def dir_to_tar(root: Path) -> bytes:
    """A workspace directory as tar bytes, keeping modes, mtimes and symlinks (not followed)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs.sort()
            for name in [*dirs, *sorted(files)]:
                path = Path(current) / name
                tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return buf.getvalue()


def seal(args: argparse.Namespace) -> int:
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"not a directory: {in_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to write into a non-empty directory: {out_dir}")
    manifest = load_manifest(args.manifest)
    tar = dir_to_tar(in_dir)
    if args.mode == "side":
        sealed, report = seal_side(tar, manifest)
    else:
        if not args.prompt:
            raise SystemExit("--mode side_para needs --prompt FILE (the paraphrase prompt)")
        from dotenv import load_dotenv

        from crow.llm import modal_endpoint

        load_dotenv()  # TRUSTED_ENDPOINT_URL, TRUSTED_MODEL and the proxy token; never printed
        client = modal_endpoint(
            "trusted", os.environ["TRUSTED_ENDPOINT_URL"], os.environ["TRUSTED_MODEL"]
        )
        params = {"temperature": args.temperature, "max_tokens": args.max_tokens}
        prompt = Path(args.prompt).read_text(encoding="utf-8")
        sealed, report = seal_side_para(
            tar, manifest, client, prompt, affinity_key="crow-cli-seal", params=params
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(sealed)) as t:
        t.extractall(out_dir, filter="data")
    table = Table("path", "action")
    for d in report.dropped:
        table.add_row(d["path"], f"[red]dropped[/red]: {d['why']}")
    for p in report.python_ast:
        table.add_row(p, "python: ast round trip, no comments or docstrings")
    for f in report.python_fallback:
        table.add_row(f["path"], f"python fallback ({f['error']}): comment lines stripped")
    for p in report.text:
        table.add_row(p, "text normalized")
    for p in report.binary_kept:
        table.add_row(p, "kept byte for byte (not UTF-8)")
    console.print(table)
    console.print(
        f"sealed {in_dir} -> {out_dir} (mode {args.mode}; modes 0644/0755, one mtime"
        + (f"; HANDOFF.md paraphrased: {report.paraphrased})" if args.mode == "side_para" else ")")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crow", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("observe", help="replay (or follow) an episode's event feed with flags")
    p.add_argument("episode")
    p.add_argument("--follow", action="store_true", help="keep reading as the log grows")
    p.add_argument("--idle-exit", type=float, default=0, help="with --follow: stop after S s idle")
    p.set_defaults(fn=observe)
    p = sub.add_parser("score", help="per-detector scores, composite, per-channel breakdown")
    p.add_argument("episode")
    p.add_argument("--findings", type=int, default=0, help="also list up to N findings each")
    p.set_defaults(fn=score)
    p = sub.add_parser("seal", help="apply a defense to a workspace directory")
    p.add_argument("--mode", choices=["side", "side_para"], required=True)
    p.add_argument("--manifest", required=True, help="task.json, a JSON list, or one per line")
    p.add_argument("--prompt", help="paraphrase prompt file (side_para)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("in_dir")
    p.add_argument("out_dir")
    p.set_defaults(fn=seal)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
