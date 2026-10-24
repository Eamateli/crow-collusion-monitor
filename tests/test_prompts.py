"""The collude prompts match docs/collude_prompts.md verbatim, and every task has a summary."""

import re
from pathlib import Path

import pytest

from env import prompts as p
from env.taskset import load_task, task_ids

DOCS = (Path(__file__).resolve().parent.parent / "docs" / "collude_prompts.md").read_text()
VERBATIM = {
    "PLAN_A": p.PLAN_A,
    "PLAN_B": p.PLAN_B,
    "COLLUDE_A_ADDENDUM": p.COLLUDE_A_ADDENDUM,
    "COLLUDE_B_ADDENDUM": p.COLLUDE_B_ADDENDUM,
    "SCHELLING_A": p.SCHELLING_A,
    "SCHELLING_B": p.SCHELLING_B,
    **{f"DEFENSE_TEXT[{k}]": v for k, v in p.DEFENSE_TEXT.items()},
}


@pytest.mark.parametrize("name", VERBATIM)
def test_block_is_verbatim_from_docs(name):
    assert VERBATIM[name] in DOCS, f"{name} does not match docs/collude_prompts.md"


def test_placeholders_are_the_expected_set():
    def fields(text):
        return set(re.findall(r"\{(\w+)\}", text))

    assert fields(p.PLAN_A) == {"k", "defense_text", "task_summary"}
    assert fields(p.PLAN_B) == {"k", "defense_text", "task_summary", "a_proposal"}
    assert fields(p.COLLUDE_A_ADDENDUM) == {"k", "payload", "defense_text", "a_proposal", "b_reply"}
    # B is never told the payload; B must recover it.
    assert fields(p.COLLUDE_B_ADDENDUM) == {"k", "defense_text", "a_proposal", "b_reply"}
    assert "payload" not in fields(p.COLLUDE_B_ADDENDUM)
    assert fields(p.SCHELLING_A) == {"k"} and fields(p.SCHELLING_B) == {"k"}
    assert fields(p.AGENT_SYSTEM_A) == {"defense_text"}
    assert fields(p.AGENT_SYSTEM_B) == {"defense_text"}


def test_agreed_paragraph_is_inside_each_addendum():
    assert p.AGREED_A in p.COLLUDE_A_ADDENDUM
    assert p.AGREED_B in p.COLLUDE_B_ADDENDUM


def test_word_limits_present():
    assert "at most 150 words" in p.PLAN_A
    assert "at most 100 words" in p.PLAN_B
    assert "at most 50 words" in p.COLLUDE_A_ADDENDUM


def test_base_prompts_name_the_auditor_for_parity():
    assert "automated auditor" in p.AGENT_SYSTEM_A
    assert "automated auditor" in p.AGENT_SYSTEM_B


@pytest.mark.parametrize("task_id", task_ids())
def test_every_task_has_a_summary(task_id):
    summary = load_task(task_id).summary
    assert isinstance(summary, str) and 20 < len(summary) < 400
