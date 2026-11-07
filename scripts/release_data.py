"""Build the v2 data release: every episode record without its planning chat or report_method
text, the event logs, workspaces and monitor scores as they are, and a scan showing that none of
the redacted text remains anywhere in the release.

    python scripts/release_data.py results/raw/v2 ../crow-v2-data-release

The release goes outside the repository: it holds workspace tars, which git ignores, and it is
published apart from the code. The scan also runs on the unredacted source, as a positive control
that it finds the text it looks for.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.release import build, scan, secret_texts  # noqa: E402

README = """# Crow v2 data release

Every episode of the Crow v2 experiment (three agents, distributed collusion), as the runs left it,
with one exception: each record's planning chat (`planning.proposal`, `planning.reply`,
`planning.c_reply`) and `report_method` text are removed. That text is the agents' own description
of their scheme. No monitor ever saw it, and no number in the paper uses it.

It shows working collusion schemes; it is released for monitor research.

One more change: two agents printed their sandbox environment, and in those two event-log lines
every value is replaced with `[masked]` (the variable names stay). The sandboxes hold no secrets;
the values were identifiers no reader needs, and no number reads them. `REDACTION_SCAN.json` lists
the masked lines (`environment_dumps_masked`).

## Layout

One folder per run, as on the Modal Volume `crow-results`:

| run | what | counted in results |
|---|---|---|
| `v2_main` | the main grid: 150 episodes, seeds 0 to 9 | yes |
| `v2_main_arms` | the intervention replays: 4 arms on 30 base episodes | yes |
| `v2_calibration` | 30 honest calibration episodes (seeds 1100 to 1114) | the freeze |
| `v2_smoke`, `v2_resmoke`, `v2_intervention`, `v2_resmoke_arms` | smoke and pilot runs | no |

In each: `records/` (one JSON per episode, redacted), `scores/` (every monitor's scores and saved
replies), `events/` (the tool-call log Crow kept), `workspaces/` (the workspace tars at each
handoff), and for the main run `scoring_summaries/`.

## Use

With the code at tag `crow2-freeze` or later, `python -m analysis.analyze --v2 <this folder>`
regenerates the v2 section of `analysis/numbers.json`, its tables and Figure 1. See `REPRODUCE.md`.

## Redaction scan

`REDACTION_SCAN.json` is the result of `scripts/release_data.py`: every 40-character piece of every
redacted text, whitespace collapsed, searched for in every string of every JSON file here and every
file in every workspace tar. No record holds any of it (`hits_in_records`). Elsewhere, pieces do
turn up, and they are left as they are, because they are part of what the monitors saw:
- `generic`: a piece that also appears in an honest episode, so it is task text, not a scheme;
- `colluder_text`: text the colluding agents also wrote into their commands or files (for example
  the helper names a report listed, or the check scripts careless agents ran), or a monitor's reply
  quoting it. The careless contrast puts its scheme in its commands by design.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("the release goes outside the repository")
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty")
    result = build(args.source, out)
    texts = secret_texts(
        json.loads(p.read_text()) for p in sorted(args.source.rglob("records/*.json"))
    )
    control = scan(args.source, texts)
    result["positive_control"] = {
        "what": "the same scan on the unredacted source",
        "hits": control["hits"],
        "hits_in_records": control["hits_in_records"],
    }
    result["built"] = datetime.now(UTC).isoformat(timespec="seconds")
    (out / "README.md").write_text(README)
    result.pop("hit_files")
    (out / "REDACTION_SCAN.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files_per_run"}, indent=1))
    ok = (
        result["hits_in_records"] == 0
        and result["records_still_holding_text"] == 0
        and control["hits_in_records"] > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
