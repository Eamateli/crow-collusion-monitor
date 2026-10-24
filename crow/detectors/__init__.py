"""Detector registry: the 9 layer 1 heuristics and the layer 2 judge."""

from crow.detectors.heuristics import HEURISTICS, DetectorResult, View

DETECTORS = [*HEURISTICS, "judge"]


def run_heuristics(view: View) -> dict[str, DetectorResult]:
    return {name: detect(view) for name, detect in HEURISTICS.items()}
