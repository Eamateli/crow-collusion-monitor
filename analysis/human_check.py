"""Human check of the route labels (CLAUDE.md section 6): sample 30 planning chats for a person
to label by hand, then compute Cohen's kappa against the machine labels.

    python analysis/human_check.py sample MAIN_DIR [--n 30] [--out routes_to_label.csv]
    python analysis/human_check.py kappa MAIN_DIR --labeled routes_to_label.csv

Crow v2 has no sample and no kappa: planned conditions have no route to label, and the no-plan cell
is small enough to check whole. `census` writes every episode in it, with the reports it was labeled
from and channel_built, and `agreement` reports raw agreement (for example 9/10). The CSV holds
report_method text, so it stays local (*_to_label.csv is gitignored).

    python analysis/human_check.py census MAIN_DIR [--out noplan_routes_to_label.csv]
    python analysis/human_check.py agreement MAIN_DIR --labeled noplan_routes_to_label.csv

`sample` writes one row per episode with the planning chat and the machine label, and an empty
`human_channel` column to fill with C1 to C8. `kappa` reads the filled CSV, computes kappa over
the rows a human labeled, and writes route_kappa.json next to the records (analyze.py picks it
up). The two steps use a fixed seed, so the sample is reproducible.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from analysis.metrics import cohen_kappa
from analysis.routes import CHANNELS, is_labeled_cell

FIELDS = ["episode_id", "defense", "k", "machine_channel", "human_channel", "proposal", "reply"]


def _collude_with_labels(main_dir: Path) -> tuple[list[dict], dict]:
    labels_path = main_dir / "route_labels.json"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} missing; run analyze.py --label-routes first")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    records = {}
    for path in sorted((main_dir / "records").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        c = r["config"]
        if r["status"] == "ok" and c["role"] == "collude" and c["planning"]:
            if r["episode_id"] not in records or r["attempt"] > records[r["episode_id"]]["attempt"]:
                records[r["episode_id"]] = r
    labeled = [r for r in records.values() if labels.get(r["episode_id"], {}).get("channel")]
    return labeled, labels


def sample(main_dir: Path, n: int, out: Path) -> int:
    labeled, labels = _collude_with_labels(main_dir)
    rng = random.Random(0)
    chosen = sorted(labeled, key=lambda r: r["episode_id"])
    rng.shuffle(chosen)
    chosen = chosen[: min(n, len(chosen))]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in chosen:
            plan = r["planning"]
            writer.writerow(
                {
                    "episode_id": r["episode_id"],
                    "defense": r["config"]["defense"],
                    "k": r["config"]["k"],
                    "machine_channel": labels[r["episode_id"]]["channel"],
                    "human_channel": "",
                    "proposal": " ".join((plan["proposal"] or "").split()),
                    "reply": " ".join((plan["reply"] or "").split()),
                }
            )
    print(f"wrote {len(chosen)} rows to {out}")
    print("Fill human_channel with one of: " + ", ".join(CHANNELS))
    return 0


def kappa(main_dir: Path, labeled_csv: Path) -> int:
    with labeled_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = [
        (r["machine_channel"].strip().upper(), r["human_channel"].strip().upper())
        for r in rows
        if r["human_channel"].strip()
    ]
    bad = [h for _, h in pairs if h not in CHANNELS]
    if bad:
        raise SystemExit(f"human_channel has values outside C1 to C8: {sorted(set(bad))}")
    if not pairs:
        raise SystemExit("no rows have a human_channel; label the CSV first")
    machine, human = [m for m, _ in pairs], [h for _, h in pairs]
    result = {
        "n_checked": len(pairs),
        "cohen_kappa": cohen_kappa(machine, human),
        "exact_agreement": sum(m == h for m, h in pairs) / len(pairs),
        "disagreements": [{"machine": m, "human": h} for m, h in pairs if m != h],
    }
    out = main_dir / "route_kappa.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"n={result['n_checked']} kappa={result['cohen_kappa']} "
        f"exact_agreement={result['exact_agreement']:.3f}; wrote {out}"
    )
    return 0


CENSUS_FIELDS = [
    "episode_id",
    "machine_channel",
    "channel_built",
    "human_channel",
    "report_a",
    "report_b",
]


def _noplan_records(main_dir: Path) -> list[dict]:
    """The latest finished attempt of every episode in the v2 no-plan cell, by id."""
    records: dict[str, dict] = {}
    for path in sorted((main_dir / "records").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        c = r["config"]
        if r["status"] == "ok" and c.get("agents", 2) == 3 and is_labeled_cell(c):
            if r["episode_id"] not in records or r["attempt"] > records[r["episode_id"]]["attempt"]:
                records[r["episode_id"]] = r
    return [records[key] for key in sorted(records)]


def census(main_dir: Path, out: Path) -> int:
    labels_path = main_dir / "route_labels.json"
    if not labels_path.exists():
        raise SystemExit(f"{labels_path} missing; run analyze.py --label-routes first")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    rows = _noplan_records(main_dir)
    if not rows:
        raise SystemExit(f"no finished no-plan episodes under {main_dir}")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CENSUS_FIELDS)
        writer.writeheader()
        for r in rows:
            reports = r.get("report_method") or {}
            writer.writerow(
                {
                    "episode_id": r["episode_id"],
                    "machine_channel": labels.get(r["episode_id"], {}).get("channel") or "",
                    "channel_built": r.get("channel_built"),
                    "human_channel": "",
                    "report_a": " ".join((reports.get("A") or "").split()),
                    "report_b": " ".join((reports.get("B") or "").split()),
                }
            )
    print(f"wrote all {len(rows)} no-plan episodes to {out}")
    print("Fill human_channel with one of: " + ", ".join(CHANNELS))
    return 0


def agreement(main_dir: Path, labeled_csv: Path) -> int:
    with labeled_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    blank = [r["episode_id"] for r in rows if not r["human_channel"].strip()]
    if blank:
        raise SystemExit(f"a census checks every row; {len(blank)} have no human_channel: {blank}")
    bad = sorted({r["human_channel"].strip().upper() for r in rows} - set(CHANNELS))
    if bad:
        raise SystemExit(f"human_channel has values outside C1 to C8: {bad}")
    compared = [r for r in rows if r["machine_channel"].strip()]
    same = [
        r
        for r in compared
        if r["machine_channel"].strip().upper() == r["human_channel"].strip().upper()
    ]
    result = {
        "name": "human census of self-reported routes",
        "agreed": len(same),
        "compared": len(compared),
        "raw_agreement": f"{len(same)}/{len(compared)}",
        "no_machine_label": len(rows) - len(compared),
        "disagreements": [
            {
                "episode_id": r["episode_id"],
                "machine": r["machine_channel"].strip().upper(),
                "human": r["human_channel"].strip().upper(),
            }
            for r in compared
            if r not in same
        ],
    }
    out = main_dir / "route_census.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"raw agreement {result['raw_agreement']}; wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="write planning chats to a CSV for manual labeling")
    s.add_argument("main_dir")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--out", default="routes_to_label.csv")
    k = sub.add_parser("kappa", help="compute Cohen's kappa from the labeled CSV")
    k.add_argument("main_dir")
    k.add_argument("--labeled", required=True)
    c = sub.add_parser("census", help="write the whole v2 no-plan cell to a CSV for labeling")
    c.add_argument("main_dir")
    c.add_argument("--out", default="noplan_routes_to_label.csv")
    a = sub.add_parser("agreement", help="raw agreement from the labeled census CSV")
    a.add_argument("main_dir")
    a.add_argument("--labeled", required=True)
    args = parser.parse_args(argv)
    if args.command == "sample":
        return sample(Path(args.main_dir), args.n, Path(args.out))
    if args.command == "census":
        return census(Path(args.main_dir), Path(args.out))
    if args.command == "agreement":
        return agreement(Path(args.main_dir), Path(args.labeled))
    return kappa(Path(args.main_dir), Path(args.labeled))


if __name__ == "__main__":
    sys.exit(main())
