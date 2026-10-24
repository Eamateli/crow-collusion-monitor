"""Agent loop rules (CLAUDE.md section 4) with a scripted fake client: no Modal, no LLM."""

import json
from typing import Any

from crow.llm import LLMCall
from env.agent import Tool, run_agent, truncate
from env.prompts import NUDGE, TOOL_LIMIT_NOTICE, TOOL_SPECS
from env.sandbox import SandboxGone


def reply(
    *tool_calls: tuple[str, dict[str, Any] | str],
    reasoning: str | None = None,
    finish_reason: str | None = None,
) -> LLMCall:
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        }
        for i, (name, args) in enumerate(tool_calls)
    ]
    return LLMCall(
        endpoint="fake",
        model="fake",
        params={},
        usage={},
        reasoning_tokens=None,
        cached_tokens=None,
        latency_s=0.0,
        attempts=1,
        content="",
        reasoning=reasoning,
        reasoning_field="reasoning_content" if reasoning else None,
        tool_calls=calls,
        finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
    )


class FakeClient:
    def __init__(self, replies: list[LLMCall | Exception]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], *, affinity_key: str, **params: Any) -> LLMCall:
        self.seen.append([dict(m) for m in messages])
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_tools(log: list[str]) -> dict[str, Tool]:
    def bash(cmd: str) -> str:
        log.append(cmd)
        return f"ran {cmd}"

    return {
        "bash": Tool(TOOL_SPECS["bash"], bash),
        "done": Tool(TOOL_SPECS["done"], lambda: "ok", counts=False, terminal=True),
        "submit": Tool(TOOL_SPECS["done"], lambda: "ok", counts=False),  # a private, free tool
    }


def run(replies: list[LLMCall | Exception], log: list[str] | None = None, **kw: Any):
    client = FakeClient(replies)
    result = run_agent(
        client,
        "sys",
        "task",
        make_tools(log if log is not None else []),
        affinity_key="k",
        params={},
        **kw,
    )
    return result, client


def tool_messages(result) -> list[str]:
    return [m["content"] for m in result.messages if m["role"] == "tool"]


def test_done_ends_turn() -> None:
    log: list[str] = []
    result, _ = run([reply(("bash", {"cmd": "ls"})), reply(("done", {}))], log)
    assert result.stop_reason == "done"
    assert log == ["ls"]
    assert result.work_tool_calls == 1
    assert result.nudges == 0


def test_reasoning_content_kept_in_history_and_omitted_when_absent() -> None:
    result, client = run([reply(("bash", {"cmd": "ls"}), reasoning="think"), reply(("done", {}))])
    second_request = client.seen[1]
    assistant = [m for m in second_request if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "think"
    last = [m for m in result.messages if m["role"] == "assistant"][-1]
    assert "reasoning_content" not in last


def test_cap_counts_work_tools_only_then_final_turn() -> None:
    log: list[str] = []
    replies = [reply(("bash", {"cmd": f"c{i}"}), ("submit", {})) for i in range(3)]
    replies += [reply(("bash", {"cmd": "over"}), ("submit", {}))]  # final turn
    result, client = run(replies, log, max_tool_calls=3)
    assert log == ["c0", "c1", "c2"]  # the 4th bash call is refused
    assert result.work_tool_calls == 3
    assert result.stop_reason == "tool_limit"
    assert len(client.seen) == 4  # exactly one extra turn after the cap
    outputs = tool_messages(result)
    assert outputs[4].endswith(TOOL_LIMIT_NOTICE)  # third bash result carries the notice
    assert outputs[6].startswith("error: tool limit reached")
    assert outputs[7] == "ok"  # free tool still runs in the final turn


def test_parallel_calls_over_the_cap_are_refused() -> None:
    log: list[str] = []
    calls = [("bash", {"cmd": f"c{i}"}) for i in range(3)]
    result, _ = run([reply(*calls), reply(("done", {}))], log, max_tool_calls=2)
    assert log == ["c0", "c1"]
    assert result.stop_reason == "done"


def test_nudge_once_then_no_tool_call() -> None:
    result, client = run([reply(), reply()])
    assert result.stop_reason == "no_tool_call"
    assert result.nudges == 1
    assert client.seen[1][-1] == {"role": "user", "content": NUDGE}


def test_nudge_then_recovers() -> None:
    result, _ = run([reply(), reply(("done", {}))])
    assert result.stop_reason == "done"
    assert result.nudges == 1


def test_length_stop() -> None:
    result, _ = run([reply(finish_reason="length"), reply(finish_reason="length")])
    assert result.stop_reason == "length"


def test_bad_json_and_unknown_tool_count_as_work_calls() -> None:
    result, _ = run([reply(("bash", "{not json"), ("nope", {})), reply(("done", {}))])
    outputs = tool_messages(result)
    assert outputs[0] == "error: arguments are not valid JSON"
    assert outputs[1] == "error: unknown tool 'nope'"
    assert result.work_tool_calls == 2


def test_tool_exception_becomes_error_result() -> None:
    result, _ = run([reply(("bash", {"wrong": 1})), reply(("done", {}))])
    assert tool_messages(result)[0].startswith("error: TypeError")
    assert result.stop_reason == "done"


def test_llm_failure_ends_with_error() -> None:
    result, _ = run([RuntimeError("endpoint down")])
    assert result.stop_reason == "error"
    assert result.error == "llm: RuntimeError: endpoint down"


def test_sandbox_gone_ends_with_error() -> None:
    def gone(cmd: str) -> str:
        raise SandboxGone("terminated")

    tools = {"bash": Tool(TOOL_SPECS["bash"], gone)}
    result = run_agent(
        FakeClient([reply(("bash", {"cmd": "ls"}))]), "s", "u", tools, affinity_key="k", params={}
    )
    assert result.stop_reason == "error"
    assert result.error == "sandbox: terminated"


def test_runaway_guard() -> None:
    result, _ = run([reply(("submit", {})) for _ in range(10)], max_tool_calls=2)
    assert result.stop_reason == "error"
    assert result.error.startswith("runaway guard")


def test_truncate() -> None:
    assert truncate("abc", limit=5) == "abc"
    assert truncate("abcdefgh", limit=5) == "abcde\n[truncated: 3 more characters]"
