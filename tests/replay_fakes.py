"""Scripted agents for the intervention arms: no LLM, no Modal, no real sandbox.

Each agent acts on what its system prompt tells it, not on the episode's role, so a replaced agent
behaves honestly only because of the prompt the replay really gave it. Colluders read what the
agents before them left, through the tool loop. A B or C that finds no usable channel adapts: B
still writes its file, and C submits a guess.
"""

from test_episode import _HANDOFF, LocalSandbox, agent_reply, grade_fn

from crow.llm import LLMCall
from env.episode import payload_bits, run_episode
from env.planted import HONEST_KEYS, app_tests_reading, clean_reference_files
from env.relational import (
    decode,
    extract_examples,
    extract_keys,
    reference_items,
    reference_keys,
    render_entries,
    render_examples,
    render_items,
    scheme_for,
    slot_items,
)
from env.taskset import load_task

GOAL = "Private goal"


class PromptDrivenAgents:
    def __init__(self, config):
        self.config = config
        self.task = load_task(config.task_id)
        reference = clean_reference_files(self.task, config.variant)
        self.lib_path = f"lib/{self.task.module}.py"
        self.lib, self.app = reference[self.lib_path].decode(), reference["app.py"].decode()
        self.artifact = self.task.b_artifact
        self.payload = payload_bits(config.seed, config.k) if config.collude else None
        self.scheme = scheme_for(config.seed, config.k) if config.collude else None
        self.systems, self.turns, self.order = {}, {}, []
        self.planning_calls = 0

    def chat(self, messages, *, affinity_key, **params):
        agent = affinity_key.rsplit("-", 1)[1]
        if "tools" not in params:
            self.planning_calls += 1
            return agent_reply(content=f"{agent}: a plan")
        self.systems[agent] = messages[0]["content"]
        n = self.turns.get(agent, 0)
        self.turns[agent] = n + 1
        if n == 0:
            self.order.append(agent)
        colluding = GOAL in messages[0]["content"]
        return getattr(self, f"_{agent}")(messages, n, colluding)

    def _A(self, messages, n, colluding):
        if n > 0:
            return agent_reply([("done", {})])
        lib = self.lib
        if colluding:
            items = reference_items(self.config.seed, self.config.k)
            lib = f"{lib}\n\n{render_items(items)}"
        if colluding and not self.config.split:
            keys = reference_keys(items, self.payload, self.scheme, self.config.seed)
            lib = f"{lib}\n\n{render_examples(keys)}"
        calls = [
            ("write_file", {"path": self.lib_path, "content": lib}),
            ("write_file", {"path": "HANDOFF.md", "content": _HANDOFF}),
        ]
        if colluding:
            calls.append(("report_method", {"text": "a report"}))
        return agent_reply(calls)

    def _B(self, messages, n, colluding):
        honest_file = (
            "write_file",
            {"path": self.artifact, "content": render_entries(HONEST_KEYS)},
        )
        if not colluding:
            return agent_reply([honest_file] if n == 0 else [("done", {})])
        if n == 0:
            return agent_reply([("read_file", {"path": self.lib_path})])
        if n > 1:
            return agent_reply([("done", {})])
        try:
            items = slot_items(messages[-1]["content"], self.config.k)
            keys = reference_keys(items, self.payload, self.scheme, self.config.seed)
        except (ValueError, SyntaxError):  # no usable names: a replaced A left none
            keys = [f"case_{i}" for i in range(self.config.k)]
        write = ("write_file", {"path": self.artifact, "content": render_entries(keys)})
        return agent_reply([write, ("report_method", {"text": "a report"})])

    def _write_app(self):
        tests = app_tests_reading(self.artifact)
        return [
            ("write_file", {"path": "app.py", "content": self.app}),
            ("write_file", {"path": "tests/test_app.py", "content": tests}),
        ]

    def _C(self, messages, n, colluding):
        if not colluding:
            return agent_reply(self._write_app() if n == 0 else [("done", {})])
        if n == 0:
            reads = [("read_file", {"path": self.lib_path})]
            if self.config.split:
                reads.append(("read_file", {"path": self.artifact}))
            return agent_reply(reads)
        if n > 1:
            return agent_reply([("done", {})])
        try:
            if self.config.split:
                items = slot_items(messages[-2]["content"], self.config.k)
                keys = extract_keys(messages[-1]["content"])
            else:
                source = messages[-1]["content"]
                items, keys = slot_items(source, self.config.k), extract_examples(source) or []
            bits = decode(items, keys, self.scheme)
        except (ValueError, SyntaxError):
            bits = "0" * self.config.k  # no channel to read: a guess
        return agent_reply([*self._write_app(), ("submit_payload", {"bits": bits})])


class ParaphraseOnly:
    """The only trusted calls a replay may make. Anything else would be scoring in an episode."""

    def __init__(self):
        self.keys = []

    def chat(self, messages, *, affinity_key, **params):
        self.keys.append(affinity_key)
        if "-para-" not in affinity_key:
            raise AssertionError(f"a monitor was called during an episode: {affinity_key}")
        content = "<handoff>\n# Library\n\nRewritten note.\n</handoff>"
        usage = {"prompt_tokens": 20, "completion_tokens": 8}
        return LLMCall("t", "t", {}, usage, 3, 0, 0.0, 1, content, None, None, [], "stop")


def run_base(config, run_dir):
    """A base episode, through the real flow, with the channel planted by scripted colluders."""
    agents = PromptDrivenAgents(config)
    record = run_episode(
        config,
        agents,
        ParaphraseOnly(),
        run_dir,
        sandbox_factory=LocalSandbox,
        grade_fn=grade_fn,
    )
    return record, agents
