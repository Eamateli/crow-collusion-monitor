"""crow.wrap(tools, log, agent_id) -> monitored tools; snapshot(); diff().

Each wrapped call logs the event before executing, executes, then logs the result.
snapshot() records paths, sizes, modes, mtimes, sha256 and text contents up to 200 KB.
diff() compares two snapshots on content and metadata (CLAUDE.md section 6, Scope).
"""

import functools
import hashlib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from crow.events import EventLog
from crow.workspace import byte_entropy, read_tar

TEXT_LIMIT = 200_000  # bytes; larger or non-UTF-8 files keep metadata but no text


def wrap(tools: dict[str, Callable[..., Any]], log: EventLog, agent_id: str) -> dict:
    """Monitored versions of `tools`. Only pass tools that monitors may see."""

    def monitored(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            logged = {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}
            log.emit(agent_id, "tool_call", tool=name, args=logged)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                text = f"error: {type(e).__name__}: {e}"
                log.emit(agent_id, "tool_result", tool=name, result=text, meta={"error": True})
                raise
            log.emit(agent_id, "tool_result", tool=name, result=str(result))
            return result

        return call

    return {name: monitored(name, fn) for name, fn in tools.items()}


@dataclass
class Entry:
    path: str
    kind: str  # file | dir | symlink | other
    size: int
    mode: int
    mtime: float
    sha256: str
    entropy: float  # bits per byte of the content
    link: str | None = None
    text: str | None = None


@dataclass
class Snapshot:
    entries: dict[str, Entry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"entries": [asdict(e) for e in self.entries.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        return cls({e["path"]: Entry(**e) for e in data["entries"]})


def snapshot(
    tar: bytes, log: EventLog | None = None, agent_id: str = "crow", label: str = ""
) -> Snapshot:
    entries = {}
    for m in read_tar(tar):
        text = None
        if m.kind == "file" and len(m.data) <= TEXT_LIMIT:
            try:
                text = m.data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        entries[m.path] = Entry(
            path=m.path,
            kind=m.kind,
            size=len(m.data),
            mode=m.mode,
            mtime=m.mtime,
            sha256=hashlib.sha256(m.data).hexdigest(),
            entropy=byte_entropy(m.data),
            link=m.link,
            text=text,
        )
    snap = Snapshot(entries)
    if log is not None:
        files = [
            {k: v for k, v in asdict(e).items() if k not in ("text", "entropy")}
            for e in entries.values()
        ]
        log.emit(agent_id, "snapshot", meta={"label": label, "files": files})
    return snap


@dataclass
class Change:
    path: str
    status: str  # created | deleted | modified | renamed
    reasons: list[str] = field(default_factory=list)  # content, mode, mtime
    old_path: str | None = None  # for renamed

    @property
    def content_changed(self) -> bool:
        return self.status == "created" or (self.status == "modified" and "content" in self.reasons)


def _reasons(a: Entry, b: Entry) -> list[str]:
    reasons = []
    if a.kind != b.kind or a.sha256 != b.sha256 or a.link != b.link:
        reasons.append("content")
    if a.mode != b.mode:
        reasons.append("mode")
    if a.mtime != b.mtime and b.kind != "dir":  # creating a file inside a dir updates its mtime
        reasons.append("mtime")
    return reasons


def diff(before: Snapshot, after: Snapshot) -> list[Change]:
    """Changes from `before` to `after`, on content and metadata. A deleted and a created file
    with identical content are reported once, as a rename."""
    old, new = before.entries, after.entries
    changes = []
    for path in sorted(old.keys() & new.keys()):
        reasons = _reasons(old[path], new[path])
        if reasons:
            changes.append(Change(path, "modified", reasons))
    deleted = sorted(old.keys() - new.keys())
    created = sorted(new.keys() - old.keys())
    unpaired = [p for p in deleted if old[p].kind == "file"]
    for path in created:
        e = new[path]
        match = next(
            (
                p
                for p in unpaired
                if e.kind == "file" and (old[p].sha256, old[p].size) == (e.sha256, e.size)
            ),
            None,
        )
        if match is None:
            changes.append(Change(path, "created"))
        else:
            unpaired.remove(match)
            reasons = [r for r in _reasons(old[match], e) if r != "content"]
            changes.append(Change(path, "renamed", reasons, old_path=match))
    renamed_from = {c.old_path for c in changes if c.status == "renamed"}
    changes += [Change(p, "deleted") for p in deleted if p not in renamed_from]
    return changes
