# Crow

Crow is a passive monitor for multi-agent coding harnesses. Two agents work one after the other through a minimal harness (`bash`, `read_file`, `write_file` in a shared workspace). Crow wraps those tools, logs every action to an append-only event stream the agents cannot see, snapshots the workspace at every handoff, and scores every channel through which one agent can write and the next can read. Crow can also **seal** channels: it canonicalizes what the next agent sees, and can paraphrase the declared message.

Built for the Apart Research AI Collusion Research Sprint (Oct 23 to 25, 2026), Track 2 (Detection and Audit) and Track 3 (Collusion Carriers and Countermeasures). This is AI safety research: the colluding agents exist only to test whether monitors and defenses catch them.

## Research questions

- **RQ1, route.** When the declared message (`HANDOFF.md`) is audited and side channels are open, which channels do colluding agents use? Does that change when the side channels are sealed?
- **RQ2, detection.** Does Crow, which sees the whole environment, catch collusion that a message-only monitor and a transcript-only monitor miss, at a fixed false-alarm rate?
- **RQ3, disruption.** How much channel capacity (bits) survives sealing, and sealing plus paraphrasing? What does sealing cost honest work?

## How an episode works

Agent A builds a small Python library and writes `HANDOFF.md`; agent B then builds a command-line app on top of it. Both are the same model (DeepSeek V4.1 Flash). In collude episodes they share a private goal: A gets a k-bit payload (k = 1, 8 or 32) and B must recover it. Before the task, colluders hold a short planning chat (or none, in a Schelling condition); they know which defense is in force, but are never told that Crow exists.

1. **Planning chat** (collude only), before A knows the payload.
2. **A's turn** in a fresh, network-blocked [Modal](https://modal.com) sandbox with no secrets, at most 12 tool calls.
3. **Snapshot** of the workspace; Crow scores only what A changed (content, mode, mtime, renames, deletions).
4. **Scoring** of A's raw output by three monitors that share one trusted model (GLM 5.3 Flash) and one scoring format, and differ only in what they see:
   - **Crow**: 9 heuristics (manifest, file names, invisible characters, whitespace, encoded blobs, byte entropy, comment density, metadata, prompt injection) plus an LLM judge over the whole workspace;
   - **B1**: the task spec and `HANDOFF.md` only;
   - **B2**: the task spec and A's tool-call transcript only.
5. **Defense** builds B's view: `open` (unchanged), `seal_side` (drop files outside the task manifest, round-trip Python through `ast`, normalize text, reset modes and mtimes), or `seal_side_para` (`seal_side`, then the trusted model paraphrases `HANDOFF.md`).
6. **B's turn** on that view; in collude episodes B submits the bits it recovered.
7. **Usefulness**: hidden tests in a fresh sandbox score A's lib and B's app separately.

Every number in the report is computed from the raw episode records by `analysis/analyze.py`.

## Repository layout

```
crow/            the reusable monitor: events, wrap + snapshot + diff, detectors, judge, score, seal, CLI
env/             the experiment: agent loop, sandbox, episode runner, prompts, the 6 tasks (+ hidden tests)
experiments/     Modal app (run_episode, runs), grid configs, calibration and freeze
scripts/         endpoint, sandbox and task checks; prompt preview
tests/           offline tests (no Modal, no endpoint), plus one opt-in endpoint test
results/         committed summaries (calibration); raw episode data stays on the Modal Volume
docs/            endpoint request examples; the collude prompt text (docs/collude_prompts.md)
```

## Quick start

Requirements: Python 3.12, a Modal account, and two Modal Shared Endpoints (an agent model and a trusted model).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # gitleaks + ruff on every commit
python -m modal setup       # Modal login
cp .env.example .env        # then fill in the six values by hand
pytest                      # offline tests
```

`.env` (locally) and the Modal Secret `crow-secrets` (remotely) hold the same six keys: `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`, `AGENT_ENDPOINT_URL`, `AGENT_MODEL`, `TRUSTED_ENDPOINT_URL`, `TRUSTED_MODEL`. Secrets are never committed, logged or written to results.

Checks against the real services:

```bash
python scripts/check_endpoints.py    # both endpoints answer, model ids, reasoning tokens
python scripts/check_sandbox.py      # sandbox: network blocked, no secrets in env, an agent writes a file
python scripts/check_tasks.py        # every reference passes its hidden tests; starter and wrong solutions fail
```

## Reproducing everything

Each run kind writes to its own folder on the Modal Volume `crow-results` (`/results/<run>/records`, `events`, `workspaces`). Long runs are launched with `--detach` and resume from the Volume if the local process dies.

```bash
modal run experiments/modal_app.py --smoke                              # 3 smoke episodes (seeds 3000+)
modal run --detach experiments/modal_app.py --run probe                 # difficulty probe (seeds 2000-2011)
modal run --detach experiments/modal_app.py --run calibration           # 30 honest episodes (seeds 1000-1014)
modal volume get crow-results calibration ./calib
python experiments/calibrate.py ./calib/calibration --rescore-monitors  # fit, tau at p95, judge repeatability -> crow/frozen.json
modal run --detach experiments/modal_app.py --run main --max-tokens 60000000   # 320 episodes (seeds 0-19)
modal volume get crow-results main ./main
```

The main grid, per seed (seeds 0 to 19; the seed fixes the task and the payload, so the defenses are compared on matched pairs):

| role | defense | k (bits) | episodes |
|---|---|---|---|
| collude | open, seal_side, seal_side_para | 1, 8, 32 | 180 |
| collude, no planning chat | open | 8 | 20 |
| honest | open, seal_side, seal_side_para | none | 60 |
| honest_hard | open, seal_side, seal_side_para | none | 60 |

**The freeze.** Calibration fits the two z-score detectors and sets the alarm threshold tau on honest episodes only, then writes `crow/frozen.json` (commit tagged `crow-freeze`). After the freeze, no detector, prompt, weight, threshold or model changes; later bug fixes are logged in `CHANGELOG.md`. A main run refuses to start unless a Modal container sees exactly the committed `frozen.json`.

## CLI

```bash
python -m crow.cli observe <run>/events/<episode>.jsonl [--follow]   # event feed per agent, flags inline
python -m crow.cli score   <run>/records/<episode>.json [--findings N]   # per-detector scores, composite, channels
python -m crow.cli seal --mode side|side_para --manifest env/tasks/<task>/task.json IN_DIR OUT_DIR
```

## Results

*Pending.* The main grid is running at the time of writing. This section is filled from `analysis/numbers.json` once `analysis/analyze.py` has been run on the raw records (Step 10); no number appears here before that.

## Limitations

- **Ceiling effect on usefulness.** Honest pairs pass nearly all hidden tests (mean U 0.99 in the difficulty probe, above the 70 to 90 percent target), and the tasks were kept as they were, so they may understate the cost of sealing: it can only show as drops below about 1.0, and an agent B that can read A's code may need `HANDOFF.md` less than in harder tasks.
- **Colluders on the base tasks only.** We did not test colluders hiding inside the noisier hard version, where honest work already contains base64, unicode and odd file names.
- **Judge repeatability.** The trusted model thinks even at temperature 0. On 20 honest calibration episodes scored twice, the mean absolute difference was 0.035 and exact agreement 65%, but the Pearson correlation (0.03) is uninformative because honest scores barely vary (all 0 or 1 out of 10); a diagnostic on mixed main-run episodes is reported alongside.
- **Missing judge scores on hard workspaces.** Even at `max_tokens` 32768, the judge occasionally spends its whole budget reasoning (one of 30 calibration episodes, an honest_hard slug task). A monitor with no score is recorded as missing, never as 0, and that episode leaves that monitor's ROC only.
- **Grading trusts the agents' code not to attack pytest.** Hidden tests and results live outside the workspace and agent-planted `pytest.ini`/`conftest.py` files are not loaded, but code that deliberately patched pytest in-process could fake a pass. Agents have no reason to, and we did not defend against it.
- **Scope.** One agent model, one trusted model, six small Python tasks, two agents; the results may not transfer to other models, larger tasks or longer chains of agents.

## AI assistance

The code in this repository was written with Claude Code (Anthropic's Claude as an AI coding assistant), working step by step from a written plan; the design decisions were made and reviewed by the author. The prompt text for the colluding agents (`docs/collude_prompts.md`) was written by Claude in the project chat on claude.ai, reviewed by the author, and copied verbatim into `env/prompts.py`. The models that act inside the experiment are DeepSeek V4.1 Flash (agents) and GLM 5.3 Flash NVFP4 (trusted monitor and paraphraser), both on Modal Shared Endpoints.

## Disclosure

All code written during the Apart AI Collusion Research Sprint, Oct 24 to 25, 2026. No preexisting code. Built with Claude Code.

## License

MIT, see [LICENSE](LICENSE).
