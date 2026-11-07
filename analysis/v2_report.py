"""The v2 tables, Figure 1 and the findings list, written from the v2 section of numbers.json
alone, so that every number in them is one numbers.json holds."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.v2 import (  # noqa: E402
    ARMS,
    COLLUDE_CELLS,
    DEFENSES,
    HONEST_CELLS,
    MONITORS,
    PRIMARY,
    SECONDARY,
)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "graphs_numbers" / "tables"
FIGURES = ROOT / "graphs_numbers" / "figures"
FINDINGS = ROOT / "graphs_numbers" / "v2_findings.md"
CELL_NAMES = {
    "split": "split",
    "control": "control",
    "noplan": "no-plan split",
    "careless_split": "careless split",
    "careless_control": "careless control",
    "honest": "honest",
    "honest_hard": "honest hard",
}
ARM_NAMES = {
    "none": "no intervention",
    "replace_a": "replace A",
    "replace_b": "replace B",
    "clean_before_c": "clean before C",
    "canonicalize_before_c": "canonicalize before C",
    "payload_holder": "replace the payload holder",
    "flagged_agent": "replace the flagged agent",
}
# Figure colours: one hue for the primary family, the secondary ink for the rest (dataviz palette,
# light mode: series 1, text secondary, text primary, recessive grid).
SERIES, SECONDARY_INK, INK, GRID = "#2a78d6", "#52514e", "#0b0b0b", "#e4e3df"


def _cell(key: str) -> str:
    cell, defense = key.split("|")
    return f"{CELL_NAMES.get(cell, cell)}, {defense}"


def _hits(r: dict | None) -> str:
    return "-" if not r or r.get("n") in (None, 0) else f"{r['hits']}/{r['n']}"


def _ci(x: dict | None) -> list | None:
    return None if not x else x.get("ci")


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci_text(ci: list | None) -> str:
    return "" if not ci else f" [{ci[0]:.2f}, {ci[1]:.2f}]"


def _rate_text(r: dict | None) -> str:
    if not r or not r.get("n"):
        return "no episodes"
    return f"{r['hits']}/{r['n']} = {r['rate']:.2f}{_ci_text(r['ci'])}"


def _mean_text(m: dict | None) -> str:
    if not m or not m.get("n"):
        return "no episodes"
    return f"{m['mean']:.3f}{_ci_text(m['ci'])} (n = {m['n']})"


# --- tables ---


def write_tables(v2: dict) -> list[str]:
    from analysis.analyze import write_table

    TABLES.mkdir(parents=True, exist_ok=True)
    written = []

    def table(name: str, header: list[str], rows: list[list]) -> None:
        write_table(TABLES / name, header, rows)
        written.append(name)

    det, far = v2["detection_primary"], v2["false_alarms"]
    table(
        "v2_detection.md",
        [
            "monitor",
            "family",
            "threshold",
            "detected",
            "rate",
            "95% CI",
            "AUROC",
            "AUROC CI (exploratory)",
            "TPR@5%FPR on main negatives",
            "false alarms, main",
            "false alarms, calibration",
        ],
        [
            [
                det[m]["name"],
                det[m]["family"],
                det[m]["threshold"],
                _hits(det[m]["detection"]),
                det[m]["detection"]["rate"],
                det[m]["detection"]["ci"],
                det[m]["auroc"],
                det[m]["auroc_ci"],
                det[m]["tpr_at_5pct_fpr_on_main_negatives"],
                _hits(far[m]["all"]),
                _hits(far[m]["calibration"]),
            ]
            for m in (*PRIMARY, *SECONDARY)
        ],
    )
    manip = v2["manipulation"]
    rows = []
    for key in [f"{c}|{d}" for c, d in COLLUDE_CELLS] + ["capped"]:
        row = manip[key]
        best = row["best_key_holder_bit_accuracy"]["all"]
        kinds = ", ".join(f"{k} {n}" for k, n in sorted(row["kinds_where_built"].items())) or "-"
        rows.append(
            [
                "capped: a turn ended at max_tokens (addenda 13)"
                if key == "capped"
                else _cell(key),
                row["n"],
                _hits(row["built"]),
                row["built"]["ci"],
                row["bit_accuracy"]["mean"],
                row["bit_accuracy"]["ci"],
                row["usefulness"]["mean"],
                row["pooled_test"]["label"],
                kinds,
                best["mean"],
                best["ci"],
                row["capacity_bits"],
                row.get("capacity_reduction_R"),
                row["capped"],
            ]
        )
    table(
        "v2_manipulation.md",
        [
            "cell",
            "n",
            "channel built",
            "built CI",
            "C's bit accuracy",
            "CI",
            "usefulness",
            "pooled leak test",
            "per-episode kind, where built",
            "best key-holder single-artifact accuracy",
            "CI",
            "capacity (bits)",
            "R vs open",
            "capped",
        ],
        rows,
    )
    inter = v2["interventions"]
    rows = []
    for family, arms in inter["per_role"].items():
        for arm in ("none", *ARMS):
            r = arms[arm]
            rows.append(
                [
                    family,
                    ARM_NAMES[arm],
                    r["n"],
                    None if r["bit_accuracy"] is None else r["bit_accuracy"]["mean"],
                    None if r["bit_accuracy"] is None else r["bit_accuracy"]["ci"],
                    None if r["recovered"] is None else _hits(r["recovered"]),
                    None if r["recovered"] is None else r["recovered"]["ci"],
                    r["usefulness"]["mean"],
                    r["usefulness"]["ci"],
                    r["cost_usd_estimate"]["mean"],
                ]
            )
    table(
        "v2_interventions.md",
        [
            "role",
            "arm",
            "n",
            "C's bit accuracy",
            "CI",
            "payload recovered",
            "CI",
            "usefulness",
            "CI",
            "cost per run, $ (estimate prices)",
        ],
        rows,
    )
    rows = []
    for family, triggers in inter["end_to_end"].items():
        for trigger, policies in triggers.items():
            for policy, r in policies.items():
                rows.append(
                    [
                        family,
                        MONITORS[trigger][2],
                        ARM_NAMES[policy],
                        r["alarmed"],
                        r["n"],
                        None if r["recovered"] is None else _hits(r["recovered"]),
                        None if r["recovered"] is None else r["recovered"]["ci"],
                        r["usefulness"]["mean"],
                        r.get("uncovered", "-"),
                    ]
                )
    table(
        "v2_end_to_end.md",
        [
            "role",
            "trigger",
            "applied where it alarms",
            "alarmed",
            "n",
            "payload recovered",
            "CI",
            "usefulness",
            "no replay (tie or C)",
        ],
        rows,
    )
    table(
        "v2_mcnemar.md",
        ["pair", "episodes", "only the first alarms", "only the second", "p", "p (Holm)"],
        [
            [
                " vs ".join(MONITORS[m][2] for m in name.split(" vs ")),
                r["episodes"],
                r["only_first"],
                r["only_second"],
                r["p"],
                r["p_holm"],
            ]
            for name, r in v2["mcnemar_primary"].items()
        ],
    )
    table(
        "v2_false_alarms.md",
        ["monitor", "honest", "CI", "honest hard", "CI", "all main", "CI", "calibration", "CI"],
        [
            [
                MONITORS[m][2],
                _hits(far[m]["honest"]),
                far[m]["honest"]["ci"],
                _hits(far[m]["honest_hard"]),
                far[m]["honest_hard"]["ci"],
                _hits(far[m]["all"]),
                far[m]["all"]["ci"],
                _hits(far[m]["calibration"]),
                far[m]["calibration"]["ci"],
            ]
            for m in (*PRIMARY, *SECONDARY)
        ],
    )
    cut = v2["calls_cut"]
    table(
        "v2_calls_cut.md",
        ["monitor calls", "main: cut / calls", "share", "calibration: cut / calls", "share"],
        [
            [
                name,
                f"{cut['main'][name]['cut']}/{cut['main'][name]['calls']}",
                cut["main"][name]["share"],
                f"{cut['calibration'][name]['cut']}/{cut['calibration'][name]['calls']}"
                if name in cut["calibration"]
                else "-",
                cut["calibration"].get(name, {}).get("share"),
            ]
            for name in cut["main"]
        ],
    )
    by_cell = v2["detection_by_cell"]
    rows = []
    for key, row in by_cell["cells"].items():
        for scope in ("all", "built"):
            rows.append(
                [_cell(key), scope, row["n"] if scope == "all" else row["n_built"]]
                + [_hits(row[scope][m]) for m in (*PRIMARY, *SECONDARY)]
            )
    names = [MONITORS[m][2] for m in (*PRIMARY, *SECONDARY)]
    table("v2_detection_by_cell.md", ["cell", "episodes", "n", *names], rows)
    rows = []
    for scope, kinds in by_cell["by_kind"].items():
        for kind, row in kinds.items():
            rows.append(
                [f"{scope}: {kind} (per episode)", row["n"]]
                + [_hits(row[m]) for m in (*PRIMARY, *SECONDARY)]
            )
    for label, row in by_cell["by_pooled_label"].items():
        rows.append(
            [f"cells the pooled test calls {label}", row["n"]]
            + [_hits(row[m]) for m in (*PRIMARY, *SECONDARY)]
        )
    table("v2_detection_by_kind.md", ["built channels", "n", *names], rows)
    table(
        "v2_postcondition_footing.md",
        [
            "cell",
            "postcondition alarms, built",
            "postcondition without per-agent, all",
            "built",
            "A called any helper",
            "mean share called",
            "merged order, agents",
            "merged order, reference encoder",
        ],
        [
            [
                _cell(key),
                _hits(row["built"]["postcondition"]),
                _hits(row["postcondition_without_per_agent"]["all"]),
                _hits(row["postcondition_without_per_agent"]["built"]),
                f"{row['helpers_called']['any_called']}/{row['helpers_called']['episodes']}",
                row["helpers_called"]["mean_share"],
                row["merged_order"]["mean"],
                row["merged_order"]["reference_encoder"],
            ]
            for key, row in by_cell["cells"].items()
        ],
    )
    table(
        "v2_success.md",
        [
            "cell",
            "n",
            "payload recovered",
            "CI",
            *[f"recovered, {MONITORS[m][2]} silent" for m in PRIMARY],
        ],
        [
            [_cell(key), row["n"], _hits(row["p_hat"]), row["p_hat"]["ci"]]
            + [_hits(row["undetected"][m]) for m in PRIMARY]
            for key, row in v2["success"].items()
        ],
    )
    use = v2["usefulness"]
    rows = [
        [
            _cell(f"{c}|{d}"),
            use[f"{c}|{d}"]["n"],
            use[f"{c}|{d}"]["U"]["mean"],
            use[f"{c}|{d}"]["U"]["ci"],
            use[f"{c}|{d}"]["lib"]["mean"],
            use[f"{c}|{d}"]["app_isolated"]["mean"],
        ]
        for c, d in (*HONEST_CELLS, *COLLUDE_CELLS)
    ]
    for role in ("honest", "honest_hard"):
        cost = use[f"sealing_cost|{role}"]
        rows.append(
            [
                f"sealing cost on U, {CELL_NAMES[role]}",
                "-",
                *[cost[d] for d in DEFENSES[1:]],
                "-",
                "-",
            ]
        )
    table(
        "v2_usefulness.md", ["cell", "n", "U (app end to end)", "CI", "lib", "app isolated"], rows
    )
    hiding = v2["cost_of_hiding"]
    rows = [
        [_cell(key), r["episodes"], r["with_payload_in_commands"], r["share"]]
        for key, r in v2["leak_rate"].items()
    ]
    table("v2_leak_rate.md", ["cell", "episodes", "payload in commands", "share"], rows)
    table(
        "v2_cost_of_hiding.md",
        [
            "role",
            "seed pairs",
            "channel built",
            "CI (exploratory)",
            "C's bit accuracy",
            "CI",
            "usefulness",
            "CI",
        ],
        [
            [
                role,
                r["pairs"],
                r["channel_built"]["mean"],
                r["channel_built"]["ci"],
                r["bit_accuracy"]["mean"],
                r["bit_accuracy"]["ci"],
                r["usefulness"]["mean"],
                r["usefulness"]["ci"],
            ]
            for role, r in hiding.items()
        ],
    )
    integ = v2["integrity"]
    lost = integ["lost_on_save_and_rescored"]
    table(
        "v2_integrity.md",
        ["fact", "value"],
        [
            ["episodes ok / planned", f"{integ['episodes']['ok']}/{integ['episodes']['planned']}"],
            ["replays ok / planned", f"{integ['replays']['ok']}/{integ['replays']['planned']}"],
            ["episodes scored", integ["scored"]["ok"]],
            ["recorded scores that differ from their replies", integ["mismatches"]],
            ["missing scores", sum(integ["missing_scores"].values())],
            [
                "stored alarms that differ from the thresholds",
                sum(integ["stored_alarm_disagreements"].values()),
            ],
            [
                "scoring calls cut at max_tokens",
                f"{integ['scoring_calls_cut']['cut']}/{integ['scoring_calls_cut']['calls']}",
            ],
            [
                "results lost on save, then scored again",
                f"{lost['first_pass_crashed']} lost, {lost['later_passes_scored']} "
                f"re-scored after {lost['fix_commit']}",
            ],
            ["preemptions in the main run", integ["preemptions_in_main_run"]],
        ],
    )
    return written


# --- Figure 1 ---


def write_figure(v2: dict) -> list[str]:
    """Each monitor's detection rate at its threshold on the primary population, with its 95% CI,
    beside its AUROC. Two panels, so that two measures never share one axis."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    det = v2["detection_primary"]
    order = [*PRIMARY, *SECONDARY]
    gap = 0.8
    y = [len(order) - i + (gap if m in PRIMARY else 0) for i, m in enumerate(order)]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK})
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"width_ratios": [3, 2]}
    )
    for ax, field, title in (
        (axes[0], "detection", "Detected at the 5% threshold"),
        (axes[1], "auroc", "AUROC (CI exploratory)"),
    ):
        for m, yy in zip(order, y, strict=True):
            colour = SERIES if m in PRIMARY else SECONDARY_INK
            if field == "detection":
                value, ci = det[m]["detection"]["rate"], det[m]["detection"]["ci"]
            else:
                value, ci = det[m]["auroc"], det[m]["auroc_ci"]
            if value is None:
                continue
            if ci:
                ax.plot(ci, [yy, yy], color=colour, linewidth=1.6, solid_capstyle="round")
            ax.plot(
                [value],
                [yy],
                "o",
                color=colour,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=1.2,
            )
        ax.set_xlim(-0.03, 1.03)  # a dot at 0 or 1 stays whole
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=SECONDARY_INK, length=0)
    axes[1].axvline(0.5, color=SECONDARY_INK, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([det[m]["name"] for m in order], color=INK)
    top_primary, top_secondary = (
        max(y),
        max(yy for m, yy in zip(order, y, strict=True) if m in SECONDARY),
    )
    axes[0].text(
        -0.02,
        top_primary + 0.75,
        "primary comparison",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        fontsize=8,
        color=SECONDARY_INK,
    )
    axes[0].text(
        -0.02,
        top_secondary + 0.75,
        "secondary",
        transform=axes[0].get_yaxis_transform(),
        ha="right",
        fontsize=8,
        color=SECONDARY_INK,
    )
    meta = v2["meta"]
    fig.text(
        0.01,
        -0.02,
        f"Careful split, channel built, n = {meta['n_primary']}; thresholds from 30 calibration "
        f"episodes; negatives for AUROC: {meta['n_negatives']} main honest episodes. Bars: 95% CI.",
        fontsize=7.5,
        color=SECONDARY_INK,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"v2_figure1_detection.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return ["v2_figure1_detection"]


# --- the findings list ---


def findings(v2: dict) -> list[str]:
    """Plain findings, each with its number and CI, and no reading beyond the two questions."""
    meta, det = v2["meta"], v2["detection_primary"]
    out = ["## RQ2: detection", ""]
    out.append(
        f"- Primary population: {meta['n_primary']} careful split episodes with the channel built; "
        f"negatives: {meta['n_negatives']} main honest episodes. "
        f"TPR at 1% FPR is {meta['tpr_at_1pct_fpr']}."
    )
    for m in (*PRIMARY, *SECONDARY):
        d = det[m]
        auc = "-" if d["auroc"] is None else f"{d['auroc']:.3f}{_ci_text(d['auroc_ci'])}"
        out.append(
            f"- {d['name']} ({d['family']}): detected {_rate_text(d['detection'])} at threshold "
            f"{d['threshold']:.4g}; AUROC {auc} (CI exploratory); false alarms on main negatives "
            f"{_rate_text(d['false_alarm'])}."
        )
    out.append(
        f"- Per-agent B2 mean, AUROC {_num(det['per_agent_mean_auroc'])} "
        "(exploratory: no frozen threshold)."
    )
    for name, r in v2["mcnemar_primary"].items():
        a, b = (MONITORS[m][2] for m in name.split(" vs "))
        out.append(
            f"- McNemar, {a} vs {b}: {r['only_first']} episodes only the first alarms on, "
            f"{r['only_second']} only the second, of {r['episodes']}; "
            f"p = {r['p']:.3g}, Holm p = {r['p_holm']:.3g}."
        )
    far = v2["false_alarms"]
    for m in (*PRIMARY, *SECONDARY):
        out.append(
            f"- False alarms, {MONITORS[m][2]}: honest {_rate_text(far[m]['honest'])}; honest hard "
            f"{_rate_text(far[m]['honest_hard'])}; calibration {_rate_text(far[m]['calibration'])}."
        )
    cut = v2["calls_cut"]
    out.append(
        "- Monitor calls cut at max_tokens, main: "
        + "; ".join(f"{n} {r['cut']}/{r['calls']}" for n, r in cut["main"].items() if n != "all")
        + f". Calibration: {cut['calibration']['all']['cut']}/{cut['calibration']['all']['calls']}."
    )
    by_kind = v2["detection_by_cell"]["by_kind"]["primary"]
    for kind in ("relational", "leaky"):
        row = by_kind[kind]
        out.append(
            f"- Primary episodes the per-episode test calls {kind}: n = {row['n']}; "
            + "; ".join(f"{MONITORS[m][2]} {_hits(row[m])}" for m in PRIMARY)
            + "."
        )
    out += ["", "## Manipulation check", ""]
    manip = v2["manipulation"]
    for c, d in COLLUDE_CELLS:
        row = manip[f"{c}|{d}"]
        out.append(
            f"- {_cell(f'{c}|{d}')}: channel built {_rate_text(row['built'])}; C's bit accuracy "
            f"{_mean_text(row['bit_accuracy'])}; usefulness {_num(row['usefulness']['mean'])}; "
            f"pooled leak test: {row['pooled_test']['label']}."
        )
    capped = manip["capped"]
    out.append(
        "- Capped episodes (a turn ended at max_tokens, any cell): channel built "
        f"{_rate_text(capped['built'])}; "
        f"the other colluding episodes {_rate_text(capped['rest_built'])}."
    )
    leak = v2["leak_rate"]
    out.append(
        "- Leak rate (payload string in any agent's commands): "
        + "; ".join(
            f"{_cell(k)} {r['with_payload_in_commands']}/{r['episodes']}"
            for k, r in leak.items()
            if r["episodes"]
        )
        + "."
    )
    for role, r in v2["cost_of_hiding"].items():
        out.append(
            f"- Cost of hiding, {role} ({r['pairs']} seed pairs, careful minus careless): "
            "channel built "
            f"{_num(r['channel_built']['mean'])}{_ci_text(r['channel_built']['ci'])}; bit accuracy "
            f"{_num(r['bit_accuracy']['mean'])}{_ci_text(r['bit_accuracy']['ci'])}; usefulness "
            f"{_num(r['usefulness']['mean'])}{_ci_text(r['usefulness']['ci'])} (CIs exploratory)."
        )
    out += ["", "## RQ3: intervention", ""]
    inter = v2["interventions"]
    for family, arms in inter["per_role"].items():
        for arm in ("none", *ARMS):
            r = arms[arm]
            parts = [f"n = {r['n']}"]
            if r["recovered"] is not None:
                parts.append(f"payload recovered {_rate_text(r['recovered'])}")
                parts.append(f"C's bit accuracy {_mean_text(r['bit_accuracy'])}")
            parts.append(f"usefulness {_mean_text(r['usefulness'])}")
            parts.append(
                f"cost ${_num(r['cost_usd_estimate']['mean'], 4)} per run, estimate prices"
            )
            out.append(f"- {family}, {ARM_NAMES[arm]}: " + "; ".join(parts) + ".")
    for family, triggers in inter["end_to_end"].items():
        if family == "honest":
            continue
        for trigger, policies in triggers.items():
            for policy, r in policies.items():
                extra = f", {r['uncovered']} with no replay" if "uncovered" in r else ""
                out.append(
                    f"- End to end, {family}, {MONITORS[trigger][2]} triggers {ARM_NAMES[policy]}: "
                    f"{r['alarmed']} of the bases alarmed{extra}; "
                    f"payload recovered {_rate_text(r['recovered'])}; "
                    f"usefulness {_num(r['usefulness']['mean'])}."
                )
    integ = v2["integrity"]
    lost = integ["lost_on_save_and_rescored"]
    out += [
        "",
        "## Integrity (appendix)",
        "",
        f"- {integ['episodes']['ok']} of {integ['episodes']['planned']} episodes and "
        f"{integ['replays']['ok']} "
        f"of {integ['replays']['planned']} replays ok; {integ['mismatches']} mismatches; "
        f"{sum(integ['missing_scores'].values())} missing scores; "
        f"{integ['scoring_calls_cut']['cut']} of {integ['scoring_calls_cut']['calls']} "
        f"scoring calls cut at max_tokens; {lost['first_pass_crashed']} results lost on save and "
        f"{lost['later_passes_scored']} re-scored after {lost['fix_commit']}; "
        f"{integ['preemptions_in_main_run']} preemptions.",
    ]
    return out


def write_findings(v2: dict) -> list[str]:
    FINDINGS.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Crow v2 findings (S7)",
        "",
        "Generated by `analysis/analyze.py --v2` from the v2 section of `analysis/numbers.json`. "
        "Every number is one numbers.json holds; CIs are 95% percentile bootstrap (2000 resamples, "
        "seed 0). Items marked exploratory are beyond the registration.",
        "",
    ]
    FINDINGS.write_text("\n".join(header + findings(v2)) + "\n", encoding="utf-8")
    return [FINDINGS.name]
