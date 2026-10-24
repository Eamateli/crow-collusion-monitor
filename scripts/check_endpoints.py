"""Checks both Modal endpoints: lists models, sends one chat message, prints usage and latency.

Loads .env with python-dotenv and never prints its values. Each endpoint gets the chat
message twice: with the server's default thinking setting, and with thinking on as in
docs/endpoint_examples/, so we can see whether reasoning tokens come back by default.

Run: python scripts/check_endpoints.py
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from crow.llm import LLMCall, modal_endpoint
from env.prompts import AGENT_MODEL_ID, TRUSTED_MODEL_ID

ROOT = Path(__file__).resolve().parent.parent
TOKEN_KEYS = ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]
MESSAGES = [{"role": "user", "content": "Reply with the single word: ready"}]

# name, url key, model key, model constant, temperature, thinking-on extra_body (from the
# dashboard snippets in docs/endpoint_examples/)
ENDPOINTS: list[tuple[str, str, str, str, float, dict[str, Any]]] = [
    (
        "agent",
        "AGENT_ENDPOINT_URL",
        "AGENT_MODEL",
        AGENT_MODEL_ID,
        0.7,
        {"reasoning_effort": "high"},
    ),
    (
        "trusted",
        "TRUSTED_ENDPOINT_URL",
        "TRUSTED_MODEL",
        TRUSTED_MODEL_ID,
        0.0,
        {"reasoning": {"enabled": True}},
    ),
]

console = Console()


def check(
    name: str,
    url_key: str,
    model_key: str,
    model_id: str,
    temperature: float,
    thinking_on: dict[str, Any],
) -> bool:
    console.rule(f"[bold]{name}[/bold]")
    missing = [k for k in [*TOKEN_KEYS, url_key, model_key] if not os.environ.get(k)]
    if missing:
        console.print(f"[red]missing in .env: {', '.join(missing)}[/red]")
        return False

    client = modal_endpoint(name, os.environ[url_key], model_id)
    try:
        ids, latency = client.list_models()
    except Exception as e:
        console.print(f"[red]GET /v1/models failed: {type(e).__name__}: {e}[/red]")
        return False
    console.print(f"GET /v1/models ({latency:.2f}s): {ids}")
    in_server = model_id in ids
    env_matches = os.environ[model_key] == model_id
    console.print(f"constant {model_id!r} served by endpoint: {in_server}")
    console.print(f"{model_key} in .env equals the constant: {env_matches}")

    calls: list[tuple[str, LLMCall]] = []
    for label, extra in [("default", {}), ("thinking on", thinking_on)]:
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": 1500}
        if extra:
            params["extra_body"] = extra
        try:
            calls.append((label, client.chat(MESSAGES, affinity_key=f"check-{name}", **params)))
        except Exception as e:
            console.print(f"[red]chat ({label}) failed: {type(e).__name__}: {e}[/red]")
            return False

    table = Table("", *[label for label, _ in calls])
    rows: dict[str, list[str]] = {}
    for _, c in calls:
        content = c.content or ""
        for key, value in {
            "response model": c.model,
            "reply": repr(content[:200]),
            "<think> in reply": str("<think>" in content),
            "finish_reason": str(c.finish_reason),
            "prompt tokens": str(c.usage.get("prompt_tokens")),
            "cached tokens": str(c.cached_tokens),
            "completion tokens": str(c.usage.get("completion_tokens")),
            "reasoning tokens": str(c.reasoning_tokens),
            "reasoning field": str(c.reasoning_field),
            "reasoning chars": str(len(c.reasoning or "")),
            "latency (s)": f"{c.latency_s:.2f}",
            "attempts": str(c.attempts),
        }.items():
            rows.setdefault(key, []).append(value)
    for key, values in rows.items():
        table.add_row(key, *values)
    console.print(table)
    console.print("raw usage:", {label: c.usage for label, c in calls})
    return in_server and env_matches


def main() -> int:
    load_dotenv(ROOT / ".env")
    results = [check(*endpoint) for endpoint in ENDPOINTS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
