"""Event dataclass and append-only JSONL writer.

One JSON object per line with fields: ts, episode_id, agent, type
(tool_call | tool_result | snapshot | seal | score), tool, args, result (truncated), meta.
The log is written by Crow, outside the agents' sandboxes; agents never see it.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TYPES = ("tool_call", "tool_result", "snapshot", "seal", "score")
RESULT_LIMIT = 4000


@dataclass
class Event:
    ts: float
    episode_id: str
    agent: str
    type: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    result: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def truncate(text: str, limit: int = RESULT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text) - limit} more characters]"


class EventLog:
    """Append-only: every event is one line, written and flushed immediately."""

    def __init__(self, path: str | Path, episode_id: str) -> None:
        self.path = Path(path)
        self.episode_id = episode_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        agent: str,
        type: str,
        *,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        result: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Event:
        if type not in TYPES:
            raise ValueError(f"unknown event type: {type}")
        result = None if result is None else truncate(result)
        event = Event(time.time(), self.episode_id, agent, type, tool, args, result, meta or {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")  # ASCII escapes keep one event per line
            f.flush()
        return event


def read_events(path: str | Path) -> list[Event]:
    with Path(path).open(encoding="utf-8") as f:
        return [Event(**json.loads(line)) for line in f if line.strip()]
