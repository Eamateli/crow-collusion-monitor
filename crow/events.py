"""Event dataclass and append-only JSONL writer.

One JSON object per line with fields: ts, episode_id, agent, type
(tool_call | tool_result | snapshot | seal | score), tool, args, result (truncated), meta."""
