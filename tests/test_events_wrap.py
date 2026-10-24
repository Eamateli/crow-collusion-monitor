"""Event log, tool wrapper, snapshot and diff (no Modal, no LLM)."""

import hashlib
import json

import pytest

from crow.events import RESULT_LIMIT, EventLog, read_events
from crow.workspace import Member, write_tar
from crow.wrap import TEXT_LIMIT, diff, snapshot, wrap


def test_event_log_appends_json_lines(tmp_path):
    log = EventLog(tmp_path / "e" / "events.jsonl", "ep1")
    log.emit("A", "tool_call", tool="bash", args={"cmd": "ls \N{ZERO WIDTH SPACE}"})
    log.emit("A", "tool_result", tool="bash", result="x" * (RESULT_LIMIT + 10))
    lines = (tmp_path / "e" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(line) for line in lines)
    assert lines[0].isascii()  # escapes keep one event per line
    events = read_events(tmp_path / "e" / "events.jsonl")
    assert [e.type for e in events] == ["tool_call", "tool_result"]
    assert events[0].args == {"cmd": "ls \N{ZERO WIDTH SPACE}"}
    assert events[1].result.endswith("[truncated: 10 more characters]")
    with pytest.raises(ValueError):
        log.emit("A", "nope")


def test_wrap_logs_call_before_running_and_result_after(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, "ep1")
    seen_when_running = []

    def bash(cmd: str) -> str:
        seen_when_running.append([e.type for e in read_events(path)])
        return f"ran {cmd}"

    tools = wrap({"bash": bash}, log, "A")
    assert tools["bash"](cmd="ls") == "ran ls"
    assert seen_when_running == [["tool_call"]]
    events = read_events(path)
    assert [(e.type, e.tool, e.agent) for e in events] == [
        ("tool_call", "bash", "A"),
        ("tool_result", "bash", "A"),
    ]
    assert events[0].args == {"cmd": "ls"} and events[1].result == "ran ls"


def test_wrap_logs_errors_and_reraises(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", "ep1")

    def broken(path: str) -> str:
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError):
        wrap({"read_file": broken}, log, "A")["read_file"](path="nope")
    result = read_events(tmp_path / "events.jsonl")[1]
    assert result.result == "error: FileNotFoundError: nope" and result.meta == {"error": True}


def test_snapshot_fields(tmp_path):
    members = [
        Member("lib", "dir", 0o755, 10.0),
        Member("lib/a.py", "file", 0o644, 12.5, b"x = 1\n"),
        Member("blob.bin", "file", 0o600, 13.0, b"\xff\xfe\x00"),
        Member("big.txt", "file", 0o644, 14.0, b"a" * (TEXT_LIMIT + 1)),
    ]
    log = EventLog(tmp_path / "events.jsonl", "ep1")
    snap = snapshot(write_tar(members), log=log, label="end of A")
    a = snap.entries["lib/a.py"]
    assert (a.kind, a.size, a.mode, a.mtime, a.text) == ("file", 6, 0o644, 12.5, "x = 1\n")
    assert a.sha256 == hashlib.sha256(b"x = 1\n").hexdigest()
    assert snap.entries["blob.bin"].text is None and snap.entries["blob.bin"].mode == 0o600
    assert snap.entries["big.txt"].text is None and snap.entries["big.txt"].size == TEXT_LIMIT + 1
    assert snap.entries["lib"].kind == "dir"
    event = read_events(tmp_path / "events.jsonl")[0]
    assert event.type == "snapshot" and event.meta["label"] == "end of A"
    assert all("text" not in f for f in event.meta["files"])
    assert snapshot(write_tar(members)).to_dict() == type(snap).from_dict(snap.to_dict()).to_dict()


def _snap(*members: Member):
    return snapshot(write_tar(members))


def test_diff_content_metadata_and_renames():
    before = _snap(
        Member("data", "dir", 0o755, 1.0),
        Member("data/a.txt", "file", 0o644, 1.0, b"a"),
        Member("data/b.txt", "file", 0o644, 1.0, b"b"),
        Member("data/c.txt", "file", 0o644, 1.0, b"c"),
        Member("data/d.txt", "file", 0o644, 1.0, b"d"),
        Member("data/e.txt", "file", 0o644, 1.0, b"e"),
        Member("data/f.txt", "file", 0o644, 1.0, b"f"),
    )
    after = _snap(
        Member("data", "dir", 0o755, 99.0),  # dir mtime moves when files change: ignored
        Member("data/a.txt", "file", 0o644, 1.0, b"a"),  # unchanged
        Member("data/b.txt", "file", 0o644, 5.0, b"B"),  # content (and mtime)
        Member("data/c.txt", "file", 0o644, 7.0, b"c"),  # mtime only
        Member("data/d.txt", "file", 0o755, 1.0, b"d"),  # mode only
        Member("data/e-renamed.txt", "file", 0o644, 1.0, b"e"),  # rename
        Member("data/new.txt", "file", 0o644, 9.0, b"n"),  # created; f.txt deleted
    )
    changes = {c.path: c for c in diff(before, after)}
    assert set(changes) == {
        "data/b.txt",
        "data/c.txt",
        "data/d.txt",
        "data/e-renamed.txt",
        "data/new.txt",
        "data/f.txt",
    }
    assert changes["data/b.txt"].reasons == ["content", "mtime"]
    assert changes["data/c.txt"].reasons == ["mtime"] and not changes["data/c.txt"].content_changed
    assert changes["data/d.txt"].reasons == ["mode"]
    assert changes["data/e-renamed.txt"].status == "renamed"
    assert changes["data/e-renamed.txt"].old_path == "data/e.txt"
    assert changes["data/new.txt"].status == "created" and changes["data/new.txt"].content_changed
    assert changes["data/f.txt"].status == "deleted"


def test_diff_directory_mode_counts():
    before = _snap(Member("d", "dir", 0o755, 1.0))
    after = _snap(Member("d", "dir", 0o700, 1.0))
    assert [(c.path, c.reasons) for c in diff(before, after)] == [("d", ["mode"])]
