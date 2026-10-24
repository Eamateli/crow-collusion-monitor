"""Crow's judge, the baselines and the paraphraser on the real trusted endpoint.

Opt-in, because it calls the endpoint (a few cents):
    CROW_LLM_TESTS=1 python -m pytest tests/test_llm_monitors.py -s
Loads .env with python-dotenv; never prints its values.
"""

import os
from pathlib import Path

import pytest
from crow_fixtures import TURN, Workspace, clean, plant_injection, starter
from dotenv import load_dotenv

from crow.detectors.judge import crow_view, message_view, monitor, transcript_view
from crow.llm import modal_endpoint
from crow.score import build_view
from crow.seal import seal_side_para
from crow.workspace import read_tar
from env.prompts import (
    CROW_JUDGE_PROMPT,
    MESSAGE_MONITOR_PROMPT,
    PARAPHRASE_PARAMS,
    PARAPHRASE_PROMPT,
    TRANSCRIPT_MONITOR_PROMPT,
    TRUSTED_MODEL_ID,
    TRUSTED_PARAMS,
)
from env.taskset import load_task

pytestmark = pytest.mark.skipif(
    os.environ.get("CROW_LLM_TESTS") != "1",
    reason="calls the trusted endpoint; set CROW_LLM_TESTS=1",
)
ROOT = Path(__file__).resolve().parent.parent
TASK = load_task("ledger")


@pytest.fixture(scope="module")
def client():
    load_dotenv(ROOT / ".env")
    return modal_endpoint("trusted", os.environ["TRUSTED_ENDPOINT_URL"], TRUSTED_MODEL_ID)


def view_of(ws: Workspace):
    manifest = TASK.manifest("base")
    return build_view(starter().snapshot(), ws.snapshot(), manifest, ws.events(), {}, turn=TURN)


def report(label: str, result) -> None:
    usage = [(c.usage.get("prompt_tokens"), c.usage.get("completion_tokens")) for c in result.calls]
    print(f"\n{label}: score {result.raw_score} (prompt, completion tokens {usage})")
    print(f"  reasoning: {result.reasoning[:400]!r}")


def test_judge_scores_planted_injection_above_clean(client):
    planted = clean()
    plant_injection(planted)
    results = {}
    for label, ws in (("clean", clean()), ("injection", planted)):
        v = view_of(ws)
        view = crow_view(TASK.spec("A", "base"), v.before, v.after, v.changes)
        results[label] = monitor(
            "judge",
            client,
            CROW_JUDGE_PROMPT,
            view,
            affinity_key=f"test-judge-{label}",
            params=TRUSTED_PARAMS,
        )
        report(f"Crow judge, {label}", results[label])
    assert results["injection"].score >= results["clean"].score


def test_baselines_answer_in_the_shared_format(client):
    planted = clean()
    plant_injection(planted)
    v = view_of(planted)
    b1 = monitor(
        "b1",
        client,
        MESSAGE_MONITOR_PROMPT,
        message_view(v.after),
        affinity_key="test-b1",
        params=TRUSTED_PARAMS,
    )
    report("B1 message monitor, injection", b1)
    planted.commands.append(
        {
            "tool": "write_file",
            "args": {"path": "HANDOFF.md", "content": planted.text("HANDOFF.md")},
        }
    )
    b2 = monitor(
        "b2",
        client,
        TRANSCRIPT_MONITOR_PROMPT,
        transcript_view(view_of(planted).commands),
        affinity_key="test-b2",
        params=TRUSTED_PARAMS,
    )
    report("B2 transcript monitor, injection", b2)
    assert 0 <= b1.score <= 1 and 0 <= b2.score <= 1


def test_paraphrase_keeps_the_api(client):
    sealed, rep = seal_side_para(
        clean().tar(),
        TASK.manifest("base"),
        client,
        PARAPHRASE_PROMPT,
        affinity_key="test-para",
        params=PARAPHRASE_PARAMS,
    )
    note = {m.path: m for m in read_tar(sealed)}["HANDOFF.md"].data.decode()
    print(f"\nparaphrased HANDOFF.md ({len(rep.calls)} call):\n{note}")
    assert rep.paraphrased
    for name in ("parse_amount", "parse_line", "summarize"):
        assert name in note
    assert "cent" in note.lower()
