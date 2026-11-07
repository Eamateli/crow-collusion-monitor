"""The v2 data release: every episode record with the planning chats and report_method text
removed, everything else as the run left it, and a scan that shows none of that text remains.

The redacted text is the agents' own description of their scheme. No monitor ever saw it, so no
number in the paper needs it (CLAUDE.md section 10; docs/reproducibility.md). Workspaces, event
logs and monitor scores are what the monitors saw, and are released as they are.

The scan looks for every 40-character piece of every redacted text, with whitespace collapsed on
both sides, in every string of every released JSON file and every file inside every workspace tar.
It uses grep's fixed-string search, so it is fast and holds no pattern logic of its own.
"""

import copy
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PLANNING_TEXT = ("proposal", "reply", "c_reply")
PIECE, STRIDE = 40, 20  # overlapping pieces: any copied run of 60 characters holds a whole one
REDACTED = "redacted"
OWN_FILES = ("README.md", "REDACTION_SCAN.json")  # the release's own notes, not episode data


def redact_record(record: dict) -> dict:
    """The record without its planning chat or report_method text; counts and timings stay."""
    out = copy.deepcopy(record)
    planning = out.get("planning")
    if isinstance(planning, dict):
        for key in PLANNING_TEXT:
            if key in planning:
                planning[key] = None
        planning[REDACTED] = list(PLANNING_TEXT)
    if isinstance(out.get("report_method"), dict):
        out["report_method"] = dict.fromkeys(out["report_method"])
        out["report_method_redacted"] = True
    return out


def secret_texts(records: Iterable[dict]) -> list[str]:
    """Every planning-chat and report_method text in these (unredacted) records."""
    texts = []
    for record in records:
        for key in PLANNING_TEXT:
            value = (record.get("planning") or {}).get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        for value in (record.get("report_method") or {}).values():
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def _normal(text: str) -> str:
    return " ".join(text.split())


def pieces(texts: Iterable[str]) -> list[str]:
    out = set()
    for text in texts:
        flat = _normal(text)
        if len(flat) < PIECE:
            if len(flat) >= 20:
                out.add(flat)
            continue
        for i in range(0, len(flat) - PIECE + 1, STRIDE):
            out.add(flat[i : i + PIECE])
    return sorted(out)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _texts_of(path: Path) -> Iterator[tuple[str, str]]:
    """(where, text) for every string a released file holds."""
    if path.suffix == ".json":
        yield str(path), "\n".join(_normal(s) for s in _strings(json.loads(path.read_text())))
    elif path.suffix == ".jsonl":
        lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        yield str(path), "\n".join(_normal(s) for line in lines for s in _strings(line))
    elif path.suffix == ".tar":
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if member.isfile():
                    data = tar.extractfile(member).read()
                    yield f"{path}::{member.name}", _normal(data.decode("utf-8", "ignore"))
    elif path.suffix in (".md", ".txt"):
        yield str(path), _normal(path.read_text(errors="ignore"))


def _find(root: Path, wanted: list[str]) -> tuple[dict[str, set[str]], int, int]:
    """place -> the pieces found there; and how many files and texts were scanned."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in OWN_FILES)
    found: dict[str, set[str]] = {}
    scanned = 0
    with tempfile.TemporaryDirectory() as tmp:
        patterns = Path(tmp) / "pieces.txt"
        patterns.write_text("\n".join(wanted) + "\n")
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        where: dict[str, str] = {}
        for path in files:
            for place, text in _texts_of(path):
                name = f"{len(where):06d}.txt"
                (corpus / name).write_text(text)
                file, _, member = place.partition("::")
                relative = str(Path(file).relative_to(root))
                where[name] = f"{relative}::{member}" if member else relative
                scanned += 1
        if wanted:
            result = subprocess.run(
                ["grep", "-rFoH", "-f", str(patterns), str(corpus)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(f"grep failed: {result.stderr[:200]}")
            for line in result.stdout.splitlines():
                file, _, piece = line.partition(":")
                found.setdefault(where[Path(file).name], set()).add(piece)
    return found, len(files), scanned


def _cell(place: str) -> str:
    episode = place.split("/")[-1].split("::")[0].split("__a")[0]
    if episode.startswith("honest"):
        return "honest"
    if "__careless" in episode:
        return "careless"
    if "__nochat" in episode:
        return "no-plan"
    return "careful"


def scan(root: Path, texts: list[str]) -> dict[str, Any]:
    """Where pieces of `texts` appear under `root`. Records must hold none. Elsewhere, a piece that
    also turns up in an honest episode is task-generic (honest agents never saw a scheme); the rest
    is text the colluders also wrote into what the monitors saw, or a monitor's reply quoting it."""
    wanted = pieces(texts)
    found, files, scanned = _find(root, wanted)
    generic = set().union(*[p for place, p in found.items() if _cell(place) == "honest"] or [set()])
    in_records = sorted(place for place in found if "/records/" in f"/{place}")
    elsewhere: dict[str, dict[str, Any]] = {}
    for place, hit in found.items():
        if place in in_records:
            continue
        folder = Path(place.split("::")[0]).parent.name or "."
        kind = "generic" if hit <= generic else "colluder_text"
        row = elsewhere.setdefault(
            f"{folder}|{_cell(place)}|{kind}", {"files": 0, "episodes": set()}
        )
        row["files"] += 1
        row["episodes"].add(place.split("::")[0].split("/")[-1].split("__a")[0])
    return {
        "redacted_texts": len(texts),
        "pieces": len(wanted),
        "piece_chars": PIECE,
        "stride": STRIDE,
        "files": files,
        "texts_scanned": scanned,
        "hits": sum(len(h) for h in found.values()),
        "hits_in_records": len(in_records),
        "generic_pieces": len(generic),
        "elsewhere": {
            k: {"files": v["files"], "episodes": len(v["episodes"])}
            for k, v in sorted(elsewhere.items())
        },
        "hit_files": sorted(found),
    }


def fields_left(root: Path) -> int:
    """Records that still hold planning-chat or report_method text."""
    left = 0
    for path in root.rglob("records/*.json"):
        record = json.loads(path.read_text())
        if secret_texts([record]):
            left += 1
    return left


def build(source: Path, out: Path) -> dict[str, Any]:
    """Copy every v2 run folder from `source` to `out`, redacting the records; then scan."""
    runs = sorted(p for p in source.iterdir() if p.is_dir())
    texts: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        counts[run.name] = {}
        for folder in sorted(p for p in run.iterdir() if p.is_dir()):
            target = out / run.name / folder.name
            target.mkdir(parents=True, exist_ok=True)
            n = 0
            for path in sorted(folder.iterdir()):
                if folder.name == "records" and path.suffix == ".json":
                    record = json.loads(path.read_text())
                    texts += secret_texts([record])
                    (target / path.name).write_text(
                        json.dumps(redact_record(record), ensure_ascii=False, indent=1) + "\n"
                    )
                elif path.is_file():
                    shutil.copy2(path, target / path.name)
                n += 1
            counts[run.name][folder.name] = n
        for path in sorted(p for p in run.iterdir() if p.is_file()):
            shutil.copy2(path, out / run.name / path.name)
    result = scan(out, texts)
    result["records_still_holding_text"] = fields_left(out)
    result["files_per_run"] = counts
    return result
