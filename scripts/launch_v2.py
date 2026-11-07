"""Start a 3-agent run from inside Modal, on the deployed app, and optionally watch it.

Nothing here drives the run: launch_run_remote does, in a Modal container, so this process can end
at any moment and the episodes still finish and get scored (addenda, section 12). Deploy first, and
again whenever the code or crow/frozen_v2.json changes: `modal deploy experiments/modal_app.py`.

    python scripts/launch_v2.py --run v2_calibration [--watch]
    python scripts/launch_v2.py --run v2_main --max-usd 27.73 [--watch]
    python scripts/launch_v2.py --arms canonicalize_before_c --base-run v2_resmoke --only s3106
    python scripts/launch_v2.py --score v2_calibration --scoring-tokens 3000000
    python scripts/launch_v2.py --status v2_calibration
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP, VOLUME = "crow", "crow-results"
FROZEN_V2 = ROOT / "crow" / "frozen_v2.json"
FOLDERS = ("records", "scores")


def _read_json(volume, path: str) -> dict | None:
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except Exception:  # noqa: BLE001 - not written yet
        return None


def status(run: str) -> dict:
    """What the Volume says about a run, from anywhere."""
    volume = modal.Volume.from_name(VOLUME)
    counts = {}
    for folder in FOLDERS:
        try:
            names = [e.path for e in volume.listdir(f"{run}/{folder}")]
        except Exception:  # noqa: BLE001 - not created yet
            names = []
        counts[folder] = sum(n.endswith(".json") for n in names)
    launcher = _read_json(volume, f"{run}/launcher.json") or {}
    return {
        "records": counts["records"],
        "scores": counts["scores"],
        "state": launcher.get("state"),
        "stage": launcher.get("stage"),
        "tokens": launcher.get("tokens"),
        "episode_budget": launcher.get("episode_budget"),
        "max_usd": launcher.get("max_usd"),
        "projected_usd": launcher.get("projected_usd"),
        "not_sent": launcher.get("not_sent"),
        "configs": launcher.get("configs"),
        "failed": launcher.get("failed", []),
        "exceptions": launcher.get("exceptions", []),
        "stopped_by_budget": launcher.get("stopped_by_budget"),
        "scoring_summary": _read_json(volume, f"{run}/scoring_summary.json"),
    }


def _check_frozen() -> None:
    """A main run needs the local freeze, and a deployed container that sees exactly it."""
    if not FROZEN_V2.exists():
        raise SystemExit("v2 main run refused: crow/frozen_v2.json is missing (S5)")
    local = hashlib.sha256(FROZEN_V2.read_bytes()).hexdigest()
    info = modal.Function.from_name(APP, "frozen_v2_info").remote()
    if not info["frozen"] or info["sha256"] != local:
        raise SystemExit(f"v2 main run refused: the deployed app sees {info}, local sha {local}")
    print(f"frozen config ok in the deployed app: sha {local[:12]}")


def watch(run: str) -> None:
    """Report until the launcher says the run is done. Safe to kill at any time."""
    while True:
        now = status(run)
        print(
            f"{time.strftime('%H:%M:%S')} {run}: {now['state']}, {now['records']} records, "
            f"{now['scores']} scores, {len(now['failed'])} failed",
            flush=True,
        )
        if now["state"] == "done":
            print(json.dumps(now["scoring_summary"])[:600])
            return
        time.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run")
    parser.add_argument("--seeds", type=int, help="v2_main only: replaces the registered number")
    parser.add_argument("--max-tokens", type=int, help="replaces the estimate plus 10%%")
    parser.add_argument("--scoring-tokens", type=int, help="replaces the estimate plus 10%%")
    parser.add_argument("--max-usd", type=float, help="one real-dollar limit over the whole run")
    parser.add_argument("--only", default="")
    parser.add_argument("--arms", help="comma-separated arms to run alone, on --base-run")
    parser.add_argument("--base-run")
    parser.add_argument("--score", help="score what this run has no ok score for, then stop")
    parser.add_argument("--status")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(args.status), indent=1))
        return 0
    if args.score:
        if not args.scoring_tokens:
            parser.error("--score needs --scoring-tokens")
        function = modal.Function.from_name(APP, "score_run_remote")
        call = function.spawn(args.score, None, args.scoring_tokens)
        print(f"scoring {args.score} server-side: {call.object_id}")
        return 0
    if args.arms:
        if not args.base_run:
            parser.error("--arms needs --base-run")
        function = modal.Function.from_name(APP, "launch_arms_remote")
        call = function.spawn(args.base_run, args.only, args.arms.split(","))
        print(f"arms on {args.base_run} running server-side: {call.object_id}")
        return 0
    if not args.run:
        parser.error("use --run, --arms or --status")
    if args.run == "v2_main":
        _check_frozen()
    function = modal.Function.from_name(APP, "launch_run_remote")
    call = function.spawn(
        args.run, args.seeds, args.only, args.max_tokens, args.scoring_tokens, args.max_usd
    )
    print(f"{args.run} running server-side: {call.object_id}", flush=True)
    if args.watch:
        watch(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
