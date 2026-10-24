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
