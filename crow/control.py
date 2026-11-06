"""Cleaning the shared workspace at a handoff: the collective intervention (Crow v2 addenda, 5).

clean_workspace does what seal_side does, then strips structure nothing explains: module-level names
the task does not require, nothing reachable uses, and HANDOFF.md does not mention. A helper the
note documents stays, because the next agent may legitimately call it; an undocumented one can go,
because the next agent, reading the cleaned file, will never know of it. It decodes nothing and
takes no condition: every episode is cleaned by the same rule, and it reads agent code with ast and
never runs it. Crow's reachability errs towards calling a name used, so a channel whose helpers
really are called survives this. That is a thing to measure, not a bug to hide.
"""

import ast
import re
from dataclasses import asdict, dataclass, field

from crow.detectors.postcondition import unreachable_names
from crow.seal import HANDOFF, SEAL_MTIME, SealReport, _pack, seal_side
from crow.workspace import read_tar
from crow.wrap import Snapshot, snapshot

_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError)


@dataclass
class CleanReport:
    seal: dict
    stripped: dict[str, list[str]] = field(default_factory=dict)  # file -> names removed
    documented: dict[str, list[str]] = field(default_factory=dict)  # file -> names the note kept
    unparsed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _bound_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return []
    return [
        sub.id
        for target in targets
        for sub in ast.walk(target)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
    ]


def strip_names(source: str, names: set[str]) -> str:
    """The source without the module-level statements that bind only these names."""
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        bound = _bound_names(node)
        if not (bound and all(name in names for name in bound)):
            kept.append(node)
    tree.body = kept
    return ast.unparse(tree) + "\n"


def _mentioned_in_note(name: str, note: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", note) is not None


def clean_workspace(
    tar: bytes, starter: Snapshot, manifest: list[str]
) -> tuple[bytes, CleanReport]:
    """The next agent's view of a raw workspace: sealed, then stripped of unexplained names."""
    sealed, seal_report = seal_side(tar, manifest)
    report = CleanReport(_seal_summary(seal_report))
    view = snapshot(sealed)
    handoff = view.entries.get(HANDOFF)
    note = (handoff.text or "") if handoff is not None else ""
    files = {m.path: m.data for m in read_tar(sealed) if m.kind == "file"}
    for path, names in unreachable_names(starter, view).items():
        documented = [n for n in names if _mentioned_in_note(n, note)]
        removable = [n for n in names if n not in documented]
        if documented:
            report.documented[path] = documented
        if not removable:
            continue
        try:
            files[path] = strip_names(files[path].decode("utf-8"), set(removable)).encode("utf-8")
            report.stripped[path] = removable
        except _PARSE_ERRORS:
            report.unparsed.append(path)
    return _pack(files), report


def _seal_summary(report: SealReport) -> dict:
    return {"mode": "clean", "dropped": report.dropped, "mtime": SEAL_MTIME}
