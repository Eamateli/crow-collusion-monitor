"""Layer 2 LLM judge, and the two baseline monitors in the same format (CLAUDE.md section 6).

One shared caller for all three. Each gets the same task spec; they differ only in which of
A's output they see:
- Crow's judge: every path with its status, the new and changed file contents, and HANDOFF.md;
- B1, the message monitor: HANDOFF.md only;
- B2, the transcript monitor: A's tool calls (tool and args), no snapshot and no results.
A reply without a parsable score is asked again; after MAX_ATTEMPTS the score is missing,
never 0 (CLAUDE.md section 6, Missing scores).
Prompt templates are passed in (they live in env/prompts.py); `{{VIEW}}` marks where the view
goes. Agent-written text always sits inside <workspace> tags, with the tag itself escaped.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import median
from typing import Any, Protocol

from crow.llm import LLMCall
from crow.wrap import Change, Snapshot

HANDOFF = "HANDOFF.md"
FILE_LIMIT, TOTAL_LIMIT, HANDOFF_LIMIT, LIST_LIMIT = 4000, 40_000, 20_000, 300
MAX_ATTEMPTS = 3
_SCORE = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)
_REASONING = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<(?=\s*/?\s*workspace)", re.IGNORECASE)


class ChatClient(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], *, affinity_key: str, **params: Any
    ) -> LLMCall: ...


@dataclass
class MonitorResult:
    name: str
    score: float | None  # in [0, 1]; None when missing
    raw_score: float | None  # 0 to 10, as the model gave it (clamped)
    reasoning: str
    calls: list[LLMCall] = field(default_factory=list)
    samples: list["MonitorResult"] = field(default_factory=list)  # set by sampled_monitor

    @property
    def missing(self) -> bool:
        return self.score is None


def escape(text: str) -> str:
    """Agent text cannot open or close our <workspace> wrapper."""
    return _TAG.sub("&lt;", text)


def block(text: str) -> str:
    return f"<workspace>\n{escape(text)}\n</workspace>"


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[cut: {len(text) - limit} more characters]"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


def _status(change: Change | None) -> str:
    if change is None:
        return "unchanged starter"
    if change.status == "renamed":
        extra = f"; changed: {', '.join(change.reasons)}" if change.reasons else ""
        return f"renamed from {change.old_path}{extra}"
    if change.status == "modified":
        return f"changed: {', '.join(change.reasons)}"
    return {"created": "new", "deleted": "deleted"}[change.status]


def task_section(spec: str) -> str:
    """The task spec, identical in all three monitors' views."""
    return f"## The task given to agent A\n\n{spec.strip()}\n\n"


def crow_view(spec: str, before: Snapshot, after: Snapshot, changes: list[Change]) -> str:
    by_path = {c.path: c for c in changes}
    paths = sorted(after.entries.keys() | {c.path for c in changes if c.status == "deleted"})
    rows = []
    for path in paths:
        e = after.entries.get(path) or before.entries[path]
        rows.append(
            f"{path} | {e.kind} | {e.size} bytes | mode {e.mode:o} | mtime {_iso(e.mtime)} "
            f"| {_status(by_path.get(path))}"
        )
    if len(rows) > LIST_LIMIT:
        rows = rows[:LIST_LIMIT] + [f"[{len(rows) - LIST_LIMIT} more paths not listed]"]
    pieces, budget = [], TOTAL_LIMIT
    written = [c.path for c in changes if c.content_changed and c.path != HANDOFF]
    for i, path in enumerate(written):
        e = after.entries[path]
        if e.kind != "file":
            continue
        body = (
            e.text if e.text is not None else f"[not shown: binary or over 200 KB, {e.size} bytes]"
        )
        piece = f"--- {path} ---\n{clip(body, FILE_LIMIT)}"
        if len(piece) > budget:
            pieces.append(f"[{len(written) - i} more files not shown]")
            break
        budget -= len(piece)
        pieces.append(piece)
    return (
        f"{task_section(spec)}"
        f"## Every path in /work after A's turn\n\n{block(chr(10).join(rows))}\n\n"
        f"## Contents of the files A created or changed\n\n"
        f"{block(chr(10).join(pieces) or '[none]')}\n\n"
        f"## HANDOFF.md, A's note for B\n\n{handoff_block(after)}"
    )


def handoff_block(after: Snapshot) -> str:
    e = after.entries.get(HANDOFF)
    if e is None:
        return block("[A left no HANDOFF.md]")
    if e.text is None:
        return block(f"[HANDOFF.md is binary or over 200 KB, {e.size} bytes]")
    return block(clip(e.text, HANDOFF_LIMIT))


def message_view(spec: str, after: Snapshot) -> str:
    """B1: the task spec and the declared message."""
    return f"{task_section(spec)}## HANDOFF.md, A's note for B\n\n{handoff_block(after)}"


def calls_block(
    commands: list[dict],
    file_limit: int = FILE_LIMIT,
    total_limit: int = TOTAL_LIMIT,
    clip_fn: Callable[[str, int], str] = clip,
) -> str:
    """Tool calls in order, tool name and arguments only, inside <workspace> tags."""
    pieces, budget = [], total_limit
    for i, call in enumerate(commands, 1):
        tool, args = call.get("tool"), call.get("args") or {}
        if tool == "bash":
            body = str(args.get("cmd", ""))
        elif tool == "write_file":
            body = f"path: {args.get('path')}\n{args.get('content', '')}"
        else:
            body = ", ".join(f"{k}: {v}" for k, v in args.items()) or "(no arguments)"
        piece = f"--- call {i}: {tool} ---\n{clip_fn(body, file_limit)}"
        if len(piece) > budget:
            pieces.append(f"[{len(commands) - i + 1} more calls not shown]")
            break
        budget -= len(piece)
        pieces.append(piece)
    return block(chr(10).join(pieces) or "[none]")


def transcript_view(spec: str, commands: list[dict]) -> str:
    """B2: the task spec and A's tool calls, tool name and arguments only."""
    return f"{task_section(spec)}## Agent A's tool calls, in order\n\n{calls_block(commands)}"


def monitor(
    name: str,
    client: ChatClient,
    template: str,
    view: str,
    *,
    affinity_key: str,
    params: dict[str, Any],
) -> MonitorResult:
    """One monitor call. A reply without a parsable score is asked again, up to MAX_ATTEMPTS;
    after that the result is missing (score None). Endpoint failures still raise."""
    messages = [{"role": "user", "content": template.replace("{{VIEW}}", view, 1)}]
    calls = []
    for _ in range(MAX_ATTEMPTS):
        call = client.chat(messages, affinity_key=affinity_key, **params)
        calls.append(call)
        content = call.content or ""
        if scores := _SCORE.findall(content):
            raw = min(10.0, max(0.0, float(scores[-1])))
            reasoning = (_REASONING.findall(content) or [""])[-1].strip()
            return MonitorResult(name, raw / 10, raw, reasoning, calls)
    return MonitorResult(name, None, None, "", calls)


def sampled_monitor(
    name: str,
    client: ChatClient,
    template: str,
    view: str,
    *,
    affinity_key: str,
    params: dict[str, Any],
    calls: int = 1,
) -> MonitorResult:
    """A monitor asked `calls` times on identical input; its score is the median of those that gave
    one, and missing only if none did. The reasoning comes from a sample nearest the median."""
    if calls == 1:
        keys = [affinity_key]
    else:
        keys = [f"{affinity_key}-{i}" for i in range(calls)]
    samples = [
        monitor(name, client, template, view, affinity_key=key, params=params) for key in keys
    ]
    every_call = [call for sample in samples for call in sample.calls]
    scored = [s for s in samples if s.raw_score is not None]
    if not scored:
        return MonitorResult(name, None, None, "", every_call, samples)
    raw = median(s.raw_score for s in scored)
    nearest = min(scored, key=lambda s: abs(s.raw_score - raw))
    return MonitorResult(name, raw / 10, raw, nearest.reasoning, every_call, samples)
