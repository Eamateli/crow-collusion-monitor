"""Minimal agent loop: system prompt, tools bash/read_file/write_file/done, max 12 tool calls.

Cap and stopping rules are in CLAUDE.md section 4. AgentResult.messages holds the model's
reasoning, so it belongs in the experimenter log only; no monitor input is built from it.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from crow.llm import LLMCall
from env.prompts import (
    AGENT_PARAMS,
    MAX_TOOL_CALLS,
    NUDGE,
    TOOL_LIMIT_NOTICE,
    TOOL_OUTPUT_LIMIT,
    TOOL_SPECS,
)
from env.sandbox import EXEC_TIMEOUT_S, Sandbox, SandboxGone, resolve


class ChatClient(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], *, affinity_key: str, **params: Any
    ) -> LLMCall: ...


@dataclass
class Tool:
    spec: dict[str, Any]
    fn: Callable[..., str]
    counts: bool = True  # counts toward the cap of work tool calls
    terminal: bool = False  # ends the turn (done)


@dataclass
class AgentResult:
    messages: list[dict[str, Any]]
    calls: list[LLMCall]
    work_tool_calls: int
    nudges: int
    stop_reason: str  # done | no_tool_call | tool_limit | length | error
    error: str | None = None


def truncate(text: str, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated: {len(text) - limit} more characters]"


def sandbox_tools(sb: Sandbox) -> dict[str, Tool]:
    """The four agent tools, backed by one sandbox. Plain callables, so crow.wrap can wrap them."""

    def bash(cmd: str) -> str:
        r = sb.exec(cmd)
        status = f"timed out after {EXEC_TIMEOUT_S} s" if r.timed_out else str(r.returncode)
        return f"exit code: {status}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    def read_file(path: str) -> str:
        return sb.read_bytes(path).decode("utf-8", errors="replace")

    def write_file(path: str, content: str) -> str:
        data = content.encode("utf-8")
        sb.write_bytes(path, data)
        return f"wrote {len(data)} bytes to {resolve(path)}"

    def done() -> str:
        return "ok"

    return {
        "bash": Tool(TOOL_SPECS["bash"], bash),
        "read_file": Tool(TOOL_SPECS["read_file"], read_file),
        "write_file": Tool(TOOL_SPECS["write_file"], write_file),
        "done": Tool(TOOL_SPECS["done"], done, counts=False, terminal=True),
    }


def run_agent(
    client: ChatClient,
    system: str,
    user: str,
    tools: dict[str, Tool],
    *,
    affinity_key: str,
    params: dict[str, Any] = AGENT_PARAMS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AgentResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    specs = [t.spec for t in tools.values()]
    calls: list[LLMCall] = []
    used = nudges = 0
    final_turn = False  # the one extra turn after the cap is reached
    max_requests = 2 * max_tool_calls + 2  # runaway guard

    def stop(reason: str, error: str | None = None) -> AgentResult:
        return AgentResult(messages, calls, used, nudges, reason, error)

    while len(calls) < max_requests:
        try:
            call = client.chat(messages, affinity_key=affinity_key, tools=specs, **params)
        except Exception as e:
            return stop("error", f"llm: {type(e).__name__}: {e}")
        calls.append(call)
        messages.append(_assistant_message(call))

        if not call.tool_calls:
            if final_turn:
                return stop("tool_limit")
            if nudges == 0:
                nudges += 1
                messages.append({"role": "user", "content": NUDGE})
                continue
            return stop("length" if call.finish_reason == "length" else "no_tool_call")

        finished = False
        for tc in call.tool_calls:
            name = tc["function"]["name"]
            tool = tools.get(name)
            counts = tool is None or tool.counts  # unknown tools count, so they can't loop
            notice = ""
            if finished:
                output = "error: not run, done() was already called"
            elif counts and used >= max_tool_calls:
                output = "error: tool limit reached; this tool did not run"
            else:
                if counts:
                    used += 1
                try:
                    output = _run_tool(tool, name, tc["function"]["arguments"])
                except SandboxGone as e:
                    return stop("error", f"sandbox: {e}")
                if counts and used == max_tool_calls:
                    notice = "\n" + TOOL_LIMIT_NOTICE
                finished = tool is not None and tool.terminal
            content = truncate(output) + notice
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

        if finished:
            return stop("done")
        if final_turn:
            return stop("tool_limit")
        final_turn = used >= max_tool_calls
    return stop("error", f"runaway guard: {max_requests} model requests")


def _assistant_message(call: LLMCall) -> dict[str, Any]:
    """The full assistant message. reasoning_content goes back in, as the endpoint expects with
    tools (CLAUDE.md section 9)."""
    msg: dict[str, Any] = {"role": "assistant", "content": call.content}
    if call.tool_calls:
        msg["tool_calls"] = call.tool_calls
    if call.reasoning is not None:
        msg["reasoning_content"] = call.reasoning
    return msg


def _run_tool(tool: Tool | None, name: str, arguments: str | None) -> str:
    if tool is None:
        return f"error: unknown tool {name!r}"
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "error: arguments are not valid JSON"
    if not isinstance(args, dict):
        return "error: arguments must be a JSON object"
    try:
        return tool.fn(**args)
    except SandboxGone:
        raise
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
