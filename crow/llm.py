"""One client wrapper for every LLM call: records model, params, usage and latency.

Talks to OpenAI-compatible endpoints (Modal Shared Endpoints). Never records the API key
or request headers.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import openai

AFFINITY_HEADER = "Modal-Routing-Affinity-Key"
MAX_RETRIES = 3
RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


@dataclass
class LLMCall:
    """One chat completion: what was asked (minus messages and headers) and what came back."""

    endpoint: str
    model: str  # model id reported in the response
    params: dict[str, Any]
    usage: dict[str, Any]  # raw usage field from the response
    reasoning_tokens: int | None  # included in completion_tokens
    cached_tokens: int | None
    latency_s: float  # successful attempt only
    attempts: int
    content: str | None
    reasoning: str | None
    reasoning_field: str | None  # server-specific message field the reasoning came from
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None


class LLMClient:
    def __init__(
        self, name: str, base_url: str, api_key: str, model: str, timeout: float = 120.0
    ) -> None:
        self.name = name
        self.model = model
        # Retries are ours (MAX_RETRIES with backoff), so the openai client must not retry.
        self._client = openai.OpenAI(
            base_url=f"{base_url.rstrip('/')}/v1", api_key=api_key, timeout=timeout, max_retries=0
        )

    def list_models(self) -> tuple[list[str], float]:
        """GET /v1/models. Returns the model ids and the latency in seconds."""
        page, _, latency = _with_retries(self._client.models.list)
        return [m.id for m in page], latency

    def chat(self, messages: list[dict[str, Any]], *, affinity_key: str, **params: Any) -> LLMCall:
        """One chat completion. params go to the API as is (temperature, max_tokens, ...)."""
        response, attempts, latency = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_headers={AFFINITY_HEADER: affinity_key},
                **params,
            )
        )
        choice = response.choices[0]
        message = choice.message
        reasoning_field, reasoning = _reasoning(message.model_extra or {})
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        return LLMCall(
            endpoint=self.name,
            model=response.model,
            params=params,
            usage=usage,
            reasoning_tokens=_usage_count(usage, "completion_tokens_details", "reasoning_tokens"),
            cached_tokens=_usage_count(usage, "prompt_tokens_details", "cached_tokens"),
            latency_s=latency,
            attempts=attempts,
            content=message.content,
            reasoning=reasoning,
            reasoning_field=reasoning_field,
            tool_calls=[tc.model_dump() for tc in message.tool_calls or []],
            finish_reason=choice.finish_reason,
        )


def modal_endpoint(name: str, url: str, model: str) -> LLMClient:
    """Client for a Modal endpoint, authenticated with the proxy token from the environment."""
    api_key = f"{os.environ['MODAL_PROXY_TOKEN_ID']}.{os.environ['MODAL_PROXY_TOKEN_SECRET']}"
    return LLMClient(name, url, api_key, model)


def _with_retries[T](fn: Callable[[], T]) -> tuple[T, int, float]:
    for attempt in range(MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            result = fn()
        except RETRYABLE:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)
            continue
        return result, attempt + 1, time.perf_counter() - start
    raise AssertionError("unreachable")


def _usage_count(usage: dict[str, Any], details: str, key: str) -> int | None:
    """A token count from the OpenAI details object, or top level (where Modal's servers put it)."""
    value = (usage.get(details) or {}).get(key)
    return value if value is not None else usage.get(key)


def _reasoning(extra: dict[str, Any]) -> tuple[str | None, str | None]:
    """Reasoning text, if the server returns it as an extra message field (name varies)."""
    for key, value in extra.items():
        if "reasoning" in key and isinstance(value, str) and value:
            return key, value
    return None, None
