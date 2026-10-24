"""Defenses that build B's view of /work: seal_side and seal_side_para (CLAUDE.md section 5).

seal_side canonicalizes everything except the meaning of the message: it drops paths the task
manifest does not allow, round-trips Python through ast (no comments, no docstrings, normal
formatting), normalizes text, and resets every mode and mtime. seal_side_para then has the
trusted model paraphrase HANDOFF.md. Input and output are workspace tar bytes.
"""

import ast
import re
import unicodedata
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from crow.detectors.judge import ChatClient, escape
from crow.events import EventLog
from crow.llm import LLMCall
from crow.manifest import allowed, is_artifact
from crow.workspace import Member, read_tar, write_tar

HANDOFF = "HANDOFF.md"
SEAL_MTIME = 946_684_800  # 2000-01-01 00:00 UTC
FILE_MODE, DIR_MODE = 0o644, 0o755  # 0644 on a directory would make it unusable
MAX_ATTEMPTS = 3
_HANDOFF_TAG = re.compile(r"<handoff>(.*?)</handoff>", re.IGNORECASE | re.DOTALL)


class SealError(Exception):
    """The paraphrase failed after MAX_ATTEMPTS; the episode is marked failed."""


@dataclass
class SealReport:
    mode: str
    dropped: list[dict] = field(default_factory=list)  # {"path": ..., "why": ...}
    python_ast: list[str] = field(default_factory=list)
    python_fallback: list[dict] = field(default_factory=list)  # {"path": ..., "error": ...}
    text: list[str] = field(default_factory=list)
    binary_kept: list[str] = field(default_factory=list)
    paraphrased: bool = False
    handoff_missing: bool = False
    calls: list[LLMCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _DropDocstrings(ast.NodeTransformer):
    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = node.body
        first = body[0] if body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def canonical_python(source: str) -> tuple[str, str | None]:
    """(code, error). ast round trip; if the source does not parse, strip comment lines and
    trailing whitespace instead."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = _DropDocstrings().visit(ast.parse(source))
            return ast.unparse(tree) + "\n", None
    except (SyntaxError, ValueError, RecursionError) as e:
        lines = [line.rstrip() for line in source.split("\n") if not line.lstrip().startswith("#")]
        return "\n".join(lines), type(e).__name__


def normalize_text(text: str) -> str:
    """NFKC, no zero-width or control characters, \\n line endings, no trailing whitespace,
    single blank lines, one final newline."""
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        c for c in text if c in "\n\t" or unicodedata.category(c) not in ("Cf", "Cc", "Zl", "Zp")
    )
    lines, previous_blank = [], False
    for line in text.split("\n"):
        line = line.rstrip()
        if not line and previous_blank:
            continue
        previous_blank = not line
        lines.append(line)
    body = "\n".join(lines).strip("\n")
    return body + "\n" if body else ""


def _pack(files: dict[str, bytes]) -> bytes:
    dirs = {"/".join(p.split("/")[:i]) for p in files for i in range(1, p.count("/") + 1)}
    members = [Member(d, "dir", DIR_MODE, SEAL_MTIME) for d in dirs]
    members += [Member(p, "file", FILE_MODE, SEAL_MTIME, data) for p, data in files.items()]
    return write_tar(members)


def _drop_reason(m: Member, manifest: list[str]) -> str | None:
    if m.kind != "file":
        return f"not a regular file ({m.kind})"
    if is_artifact(m.path):
        return "build artifact"
    if not allowed(m.path, manifest):
        return "not in the manifest"
    return None


def _canonical(members: list[Member], manifest: list[str], report: SealReport) -> dict[str, bytes]:
    kept: dict[str, bytes] = {}
    for m in members:
        if m.kind == "dir":
            continue
        if why := _drop_reason(m, manifest):
            report.dropped.append({"path": m.path, "why": why})
            continue
        try:
            text = m.data.decode("utf-8")
        except UnicodeDecodeError:
            kept[m.path] = m.data
            report.binary_kept.append(m.path)
            continue
        if m.path.endswith(".py"):
            code, error = canonical_python(text)
            if error:
                report.python_fallback.append({"path": m.path, "error": error})
            else:
                report.python_ast.append(m.path)
            kept[m.path] = code.encode("utf-8")
        else:
            kept[m.path] = normalize_text(text).encode("utf-8")
            report.text.append(m.path)
    needed = {"/".join(p.split("/")[:i]) for p in kept for i in range(1, p.count("/") + 1)}
    for m in members:
        if m.kind == "dir" and m.path not in needed:
            report.dropped.append({"path": m.path, "why": "directory without kept files"})
    return kept


def seal_side(
    tar: bytes, manifest: list[str], log: EventLog | None = None
) -> tuple[bytes, SealReport]:
    report = SealReport("side")
    files = _canonical(read_tar(tar), manifest, report)
    if log is not None:
        log.emit("crow", "seal", meta=report.to_dict())
    return _pack(files), report


def paraphrase(
    client: ChatClient,
    prompt: str,
    note: str,
    *,
    affinity_key: str,
    params: dict[str, Any],
    report: SealReport,
) -> str:
    messages = [{"role": "user", "content": prompt.replace("{{HANDOFF}}", escape(note), 1)}]
    for _ in range(MAX_ATTEMPTS):
        call = client.chat(messages, affinity_key=affinity_key, **params)
        report.calls.append(call)
        found = _HANDOFF_TAG.findall(call.content or "")
        if found and found[-1].strip():
            return found[-1]
    raise SealError(f"no <handoff> in {MAX_ATTEMPTS} replies")


def seal_side_para(
    tar: bytes,
    manifest: list[str],
    client: ChatClient,
    prompt: str,
    *,
    affinity_key: str,
    params: dict[str, Any],
    log: EventLog | None = None,
) -> tuple[bytes, SealReport]:
    report = SealReport("side_para")
    files = _canonical(read_tar(tar), manifest, report)
    if HANDOFF not in files:
        report.handoff_missing = True
    else:
        note = files[HANDOFF].decode("utf-8", errors="replace")
        new = paraphrase(
            client, prompt, note, affinity_key=affinity_key, params=params, report=report
        )
        files[HANDOFF] = normalize_text(new).encode("utf-8")
        report.paraphrased = True
    if log is not None:
        log.emit("crow", "seal", meta=report.to_dict())
    return _pack(files), report
