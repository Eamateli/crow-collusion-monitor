"""The intervention arms, as paired replays of a saved episode (no LLM, no Modal, no sandbox)."""

import json

import pytest
from replay_fakes import GOAL, ParaphraseOnly, PromptDrivenAgents, run_base
from test_episode import LocalSandbox, grade_fn, tar_to_files

from crow.events import read_events
from env.episode import make_config, payload_bits
from env.prompts import CAREFUL_LINE, MONITOR_NOTICE
from env.relational import scheme_for
from env.replay import ARMS, arm_bases, arm_episode_id, run_arm

TASK = "ledger"


def base_config(role, defense="open"):
    if role == "split":
        return make_config("collude_split", TASK, defense, 0, k=8, agents=3)
    if role == "control":
        return make_config("collude", TASK, defense, 0, k=8, agents=3)
    return make_config("honest", TASK, defense, 0, agents=3)


def replay(tmp_path, role, arm, defense="open"):
    config = base_config(role, defense)
    base, _ = run_base(config, tmp_path / "base")
    assert base["status"] == "ok", base.get("reason")
    agents, trusted = PromptDrivenAgents(config), ParaphraseOnly()
    record = run_arm(
        tmp_path / "base",
        f"{config.episode_id}__a1",
        arm,
        agents,
        trusted,
        tmp_path / "arms",
        sandbox_factory=LocalSandbox,
        grade_fn=grade_fn,
    )
    assert record["status"] == "ok", record.get("reason")
    return base, record, agents, trusted


def workspace(tmp_path, folder, stem, name):
    return (tmp_path / folder / "workspaces" / f"{stem}.{name}.tar").read_bytes()


ROLES = ("split", "control", "honest")


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_runs_end_to_end_on_every_role(tmp_path, role, arm):
    _, record, _, _ = replay(tmp_path, role, arm)
    assert record["arm"] == arm and record["usefulness"]["app_e2e"]["total"] > 0
    assert record["cost_estimate_usd"] > 0 and record["tokens"]["total"] > 0
    assert record["scores"] is None
    if role == "honest":
        assert record["bit_accuracy"] is None and record["submission"] is None
    else:
        assert 0 <= record["bit_accuracy"] <= 1
    stem = f"{record['episode_id']}__a1"
    for name in ("a_out", "b_view", "c_view", "final"):
        assert (tmp_path / "arms" / "workspaces" / f"{stem}.{name}.tar").stat().st_size > 0
    assert (tmp_path / "arms" / "records" / f"{stem}.json").exists()
    json.dumps(record)


@pytest.mark.parametrize(
    ("arm", "ran"),
    [
        ("replace_a", ["A", "B", "C"]),
        ("replace_b", ["B", "C"]),
        ("clean_before_c", ["C"]),
        ("canonicalize_before_c", ["C"]),
    ],
)
def test_only_the_agents_after_the_handoff_run_again(tmp_path, arm, ran):
    _, record, agents, _ = replay(tmp_path, "split", arm)
    assert agents.order == ran and set(record["turns"]) == set(ran)
    assert {k for k in record["tokens"] if k.startswith("agent_")} == {f"agent_{a}" for a in ran}


@pytest.mark.parametrize(
    ("role", "arm", "honest", "colluding"),
    [
        ("split", "replace_a", "A", "BC"),
        ("split", "replace_b", "B", "C"),
        ("split", "clean_before_c", "", "C"),
        ("control", "replace_a", "AB", "C"),
        ("control", "replace_b", "B", "C"),
        ("control", "clean_before_c", "", "C"),
        ("honest", "replace_a", "ABC", ""),
        ("honest", "replace_b", "BC", ""),
        ("honest", "clean_before_c", "C", ""),
        ("split", "canonicalize_before_c", "", "C"),
        ("control", "canonicalize_before_c", "", "C"),
        ("honest", "canonicalize_before_c", "C", ""),
    ],
)
def test_a_replaced_agent_gets_no_private_goal_and_the_others_keep_theirs(
    tmp_path, role, arm, honest, colluding
):
    _, _, agents, _ = replay(tmp_path, role, arm)
    for agent in honest:
        assert GOAL not in agents.systems[agent], agent
    for agent in colluding:
        assert GOAL in agents.systems[agent], agent


def test_nobody_is_told_of_the_replacement(tmp_path):
    _, _, agents, _ = replay(tmp_path, "split", "replace_a")
    for system in agents.systems.values():
        assert "replaced" not in system.lower() and "replacement" not in system.lower()
        assert "fresh" not in system.lower()


def test_a_replay_is_paired_with_its_episode(tmp_path):
    base, record, agents, _ = replay(tmp_path, "split", "replace_b")
    assert record["payload"] == base["payload"] == payload_bits(0, 8)
    assert record["scheme"] == base["scheme"] == scheme_for(0, 8).to_dict()
    assert record["config"]["seed"] == base["config"]["seed"]
    assert record["config"]["task_id"] == base["config"]["task_id"]
    assert record["base_episode"] == base["episode_id"]
    assert record["episode_id"] == arm_episode_id(base["episode_id"], "replace_b")
    assert agents.planning_calls == 0  # the saved chat is used, never a new one
    assert "A: a plan" in agents.systems["C"]  # and it is the chat the base episode had


def test_replace_b_starts_from_as_saved_output(tmp_path):
    base, record, _, _ = replay(tmp_path, "split", "replace_b")
    base_stem, stem = f"{base['episode_id']}__a1", f"{record['episode_id']}__a1"
    assert workspace(tmp_path, "arms", stem, "a_out") == workspace(
        tmp_path, "base", base_stem, "a_out"
    )
    assert workspace(tmp_path, "arms", stem, "b_view") == workspace(
        tmp_path, "base", base_stem, "b_view"
    )
    assert workspace(tmp_path, "arms", stem, "b_out") != workspace(
        tmp_path, "base", base_stem, "b_out"
    )


def test_replace_a_redoes_as_work_from_the_starter(tmp_path):
    base, record, _, _ = replay(tmp_path, "split", "replace_a")
    stem = f"{record['episode_id']}__a1"
    lib = "lib/ledger.py"
    replaced = tar_to_files(workspace(tmp_path, "arms", stem, "a_out"))[lib].decode()
    original = tar_to_files(workspace(tmp_path, "base", f"{base['episode_id']}__a1", "a_out"))
    assert "def _" in original[lib].decode()
    assert replaced != original[lib].decode()


def test_clean_keeps_both_outputs_and_changes_only_cs_view(tmp_path):
    base, record, _, _ = replay(tmp_path, "split", "clean_before_c")
    base_stem, stem = f"{base['episode_id']}__a1", f"{record['episode_id']}__a1"
    for name in ("a_out", "b_view", "b_out"):
        assert workspace(tmp_path, "arms", stem, name) == workspace(
            tmp_path, "base", base_stem, name
        )
    view = tar_to_files(workspace(tmp_path, "arms", stem, "c_view"))
    raw = tar_to_files(workspace(tmp_path, "base", base_stem, "b_out"))
    assert view["lib/ledger.py"] != raw["lib/ledger.py"]
    assert len(record["clean"]["stripped"]["lib/ledger.py"]) == 8


def test_the_base_episode_is_left_untouched(tmp_path):
    config = base_config("split")
    run_base(config, tmp_path / "base")
    before = {p: p.read_bytes() for p in (tmp_path / "base").rglob("*") if p.is_file()}
    for arm in ARMS:
        run_arm(
            tmp_path / "base",
            f"{config.episode_id}__a1",
            arm,
            PromptDrivenAgents(config),
            ParaphraseOnly(),
            tmp_path / "arms",
            sandbox_factory=LocalSandbox,
            grade_fn=grade_fn,
        )
    after = {p: p.read_bytes() for p in (tmp_path / "base").rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("arm", ARMS)
def test_no_monitor_runs_during_a_replay(tmp_path, arm):
    # ParaphraseOnly raises on anything else, which would mark the record failed.
    _, record, _, trusted = replay(tmp_path, "split", arm)
    assert trusted.keys == [] and record["status"] == "ok"


@pytest.mark.parametrize(
    ("arm", "calls"),
    [("replace_a", 2), ("replace_b", 1), ("clean_before_c", 0), ("canonicalize_before_c", 0)],
)
def test_a_paraphrase_defense_runs_at_each_later_handoff(tmp_path, arm, calls):
    _, record, _, trusted = replay(tmp_path, "split", arm, defense="seal_side_para")
    assert len(trusted.keys) == calls and record["tokens"]["paraphrase"]["calls"] == calls


def test_the_chat_and_private_tools_never_reach_a_replays_event_log(tmp_path):
    _, record, _, _ = replay(tmp_path, "split", "replace_a")
    events = read_events(tmp_path / "arms" / "events" / f"{record['episode_id']}__a1.jsonl")
    blob = json.dumps([vars(e) for e in events])
    for secret in ("a plan", "a report", "report_method", "submit_payload", record["payload"]):
        assert secret not in blob
    assert {e.agent for e in events if e.type == "tool_call"} == {"A", "B", "C"}


def test_a_failure_is_recorded_not_raised(tmp_path):
    config = base_config("split")
    run_base(config, tmp_path / "base")
    (tmp_path / "base" / "workspaces" / f"{config.episode_id}__a1.b_out.tar").unlink()
    record = run_arm(
        tmp_path / "base",
        f"{config.episode_id}__a1",
        "clean_before_c",
        PromptDrivenAgents(config),
        ParaphraseOnly(),
        tmp_path / "arms",
        sandbox_factory=LocalSandbox,
        grade_fn=grade_fn,
    )
    assert record["status"] == "failed" and "FileNotFoundError" in record["reason"]
    stem = f"{record['episode_id']}__a1"
    assert (tmp_path / "arms" / "records" / f"{stem}.json").exists()


def test_a_failed_base_episode_is_refused_with_a_reason(tmp_path):
    config = base_config("split")
    run_base(config, tmp_path / "base")
    path = tmp_path / "base" / "records" / f"{config.episode_id}__a1.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "status": "failed"}))
    record = run_arm(
        tmp_path / "base",
        f"{config.episode_id}__a1",
        "replace_b",
        PromptDrivenAgents(config),
        ParaphraseOnly(),
        tmp_path / "arms",
        sandbox_factory=LocalSandbox,
        grade_fn=grade_fn,
    )
    assert record["status"] == "failed" and "base episode" in record["reason"]


def test_an_unknown_arm_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm(tmp_path, "x", "replace_c", None, None, tmp_path)


@pytest.mark.parametrize("arm", ARMS)
def test_resume_can_recover_a_replays_id_from_its_record_name(arm):
    # experiments/modal_app.py, _existing_attempts: name.split("__a")[0]
    episode = arm_episode_id(base_config("split").episode_id, arm)
    assert f"{episode}__a1.json".split("__a")[0] == episode
    assert f"{episode}__a2.json".split("__a")[0] == episode


def test_the_arms_apply_to_every_finished_episode_of_the_three_roles():
    def rec(config, status="ok"):
        return {"episode_id": config.episode_id, "status": status, "config": vars(config)}

    split = make_config("collude_split", TASK, "open", 0, k=8, agents=3)
    control = make_config("collude", TASK, "open", 0, k=8, agents=3)
    honest = make_config("honest", TASK, "open", 0, agents=3)
    chosen = arm_bases(
        [
            rec(split),
            rec(control),
            rec(honest),
            rec(make_config("collude_split", "slug", "open", 1, k=8, agents=3), "failed"),
            rec(make_config("collude_split", TASK, "seal_side", 0, k=8, agents=3)),
            rec(make_config("collude_split", TASK, "open", 0, k=8, planning=False, agents=3)),
            rec(make_config("honest_hard", TASK, "open", 0, agents=3)),
            rec(make_config("honest", TASK, "open", 0)),  # two agents: v1
        ]
    )
    assert [r["episode_id"] for r in chosen] == [
        split.episode_id,
        control.episode_id,
        honest.episode_id,
    ]


def test_canonicalize_renames_bs_keys_and_keeps_the_values_and_the_rest(tmp_path):
    base, record, _, _ = replay(tmp_path, "split", "canonicalize_before_c")
    base_stem, stem = f"{base['episode_id']}__a1", f"{record['episode_id']}__a1"
    view = tar_to_files(workspace(tmp_path, "arms", stem, "c_view"))
    raw = tar_to_files(workspace(tmp_path, "base", base_stem, "b_out"))
    path = "data/ledger_cases.json"
    before, after = json.loads(raw[path]), json.loads(view[path])
    assert list(after) == [f"case_{i}" for i in range(1, 9)]
    assert list(after.values()) == list(before.values())
    assert record["canonicalize"]["renamed"] == {path: 8}
    assert "def _" in view["lib/ledger.py"].decode()  # it strips nothing: A's helpers stay
    assert record["seal"]["C"] == {"mode": "canonicalize"}


def test_a_replaced_agent_in_a_careful_episode_still_gets_the_notice(tmp_path):
    _, _, agents, _ = replay(tmp_path, "split", "replace_b")
    assert MONITOR_NOTICE in agents.systems["B"] and CAREFUL_LINE not in agents.systems["B"]
    assert MONITOR_NOTICE in agents.systems["C"] and CAREFUL_LINE in agents.systems["C"]
