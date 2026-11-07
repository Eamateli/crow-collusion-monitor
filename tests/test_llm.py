"""Usage and reasoning parsing in the LLM client wrapper (no network)."""

from unittest import mock

import openai
import pytest

from crow import llm
from crow.llm import _reasoning, _usage_count


def test_usage_count_top_level() -> None:
    # Shape returned by the Modal endpoints in Step 2.
    usage = {"completion_tokens": 13, "prompt_tokens": 37, "reasoning_tokens": 11}
    assert _usage_count(usage, "completion_tokens_details", "reasoning_tokens") == 11
    assert _usage_count(usage, "prompt_tokens_details", "cached_tokens") is None


def test_usage_count_openai_details() -> None:
    usage = {
        "completion_tokens": 13,
        "completion_tokens_details": {"reasoning_tokens": 11},
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert _usage_count(usage, "completion_tokens_details", "reasoning_tokens") == 11
    assert _usage_count(usage, "prompt_tokens_details", "cached_tokens") == 0


def test_reasoning_field() -> None:
    assert _reasoning({"reasoning_content": "think"}) == ("reasoning_content", "think")
    assert _reasoning({"reasoning_content": None, "other": "x"}) == (None, None)


def test_a_call_keeps_its_wall_clock_start_and_end():
    from types import SimpleNamespace

    from crow.llm import LLMClient

    client = LLMClient("t", "http://x", "key", "m")
    message = SimpleNamespace(content="hi", tool_calls=None, model_extra={})
    response = SimpleNamespace(
        model="m", usage=None, choices=[SimpleNamespace(message=message, finish_reason="stop")]
    )
    with (
        mock.patch.object(client._client.chat.completions, "create", return_value=response),
        mock.patch("crow.llm.time.time", side_effect=[50.0, 55.0]),
    ):
        call = client.chat([{"role": "user", "content": "x"}], affinity_key="k")
    assert (call.started_at, call.ended_at) == (50.0, 55.0)


class _Refused(openai.RateLimitError):
    def __init__(self) -> None:  # no HTTP response is needed to test the wait
        Exception.__init__(self, "busy")


class _Dropped(openai.APIConnectionError):
    def __init__(self) -> None:
        Exception.__init__(self, "dropped")


def _call(outcomes):
    def call():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return call


def test_a_429_waits_far_longer_than_other_failures_and_keeps_the_three_retries():
    call = _call([_Refused(), _Refused(), _Refused(), "ok"])
    with mock.patch("crow.llm.time.sleep") as sleep:
        result, attempts, _ = llm._with_retries(call)
    assert result == "ok" and attempts == 4
    assert [c.args[0] for c in sleep.call_args_list] == [30, 60, 120]
    assert llm.MAX_RETRIES == 3  # CLAUDE.md section 13


def test_a_429_past_the_three_retries_still_fails_the_call():
    call = _call([_Refused() for _ in range(4)])
    with mock.patch("crow.llm.time.sleep"), pytest.raises(openai.RateLimitError):
        llm._with_retries(call)


def test_other_failures_keep_the_short_backoff():
    with mock.patch("crow.llm.time.sleep") as sleep:
        assert llm._with_retries(_call([_Dropped(), "ok"]))[0] == "ok"
    assert [c.args[0] for c in sleep.call_args_list] == [1]
