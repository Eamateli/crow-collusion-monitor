"""Usage and reasoning parsing in the LLM client wrapper (no network)."""

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
    from unittest import mock

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
