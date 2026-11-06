"""The assembled workspace: what every agent's raw output adds up to (Crow v2 spec, section 2).

Monitors score raw output (addenda, section 9), and under a defense an agent starts from a sealed
view. So each turn is diffed against the view that agent was given, and only its own changes are
laid over the result. What an earlier agent wrote stays as it wrote it, even where sealing changed
what the next agent saw, or dropped it from the next agent's view.
"""

from dataclasses import dataclass, field, replace

from crow.wrap import Snapshot, diff


@dataclass
class Assembled:
    starter: Snapshot
    snapshot: Snapshot
    authors: dict[str, list[str]] = field(default_factory=dict)  # path -> who wrote it, in order
    deleted: dict[str, str] = field(default_factory=dict)  # path -> who deleted it


def assemble(starter: Snapshot, turns: list[tuple[str, Snapshot, Snapshot]]) -> Assembled:
    """`turns` holds (agent, the view it was given, its raw output), in order."""
    entries = dict(starter.entries)
    authors: dict[str, list[str]] = {}
    deleted: dict[str, str] = {}
    for agent, before, after in turns:
        for change in diff(before, after):
            path = change.path
            if change.status == "deleted":
                if path in entries:
                    del entries[path]
                    deleted[path] = agent
                continue
            new = after.entries[path]
            if change.status == "renamed":
                entries.pop(change.old_path, None)
                authors[path] = authors.pop(change.old_path, [])
                deleted[change.old_path] = agent
            metadata_only = change.status == "modified" and "content" not in change.reasons
            if metadata_only and path in entries:
                # Under sealing, the agent's copy holds sealed content it never wrote.
                entries[path] = replace(entries[path], mode=new.mode, mtime=new.mtime)
            else:
                entries[path] = new
                deleted.pop(path, None)
            authors.setdefault(path, []).append(agent)
    return Assembled(starter, Snapshot(entries), authors, deleted)
