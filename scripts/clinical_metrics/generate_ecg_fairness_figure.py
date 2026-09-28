#!/usr/bin/env python3
"""Generate an auditable ECG fairness figure from locked seed-level results.

The plot reports sex-specific ectopy recall alongside the per-seed equal-
opportunity (EO) gaps for the binary MIT-BIH ectopy protocol. The input summary
is produced by ``aggregate_mitbih_multiseed_classification.py`` and stores the
sex-specific recalls used to calculate each selected EO value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use Times New Roman so figure typography matches the IEEEtran document body
# (the `ptm` family). Register all four variants so bold/italic labels use real,
# matching glyphs instead of matplotlib's synthesized faux weights. A local
# `fonts/` copy is preferred when present; otherwise the system msttcorefonts
# install is used.
from matplotlib import font_manager

_TNR_DIR_CANDIDATES = (
    Path(__file__).resolve().parent / "fonts",
    Path("/usr/share/fonts/truetype/msttcorefonts"),
)
_TNR_FILES = (
    "Times_New_Roman.ttf",
    "Times_New_Roman_Bold.ttf",
    "Times_New_Roman_Italic.ttf",
    "Times_New_Roman_Bold_Italic.ttf",
)
for _dir in _TNR_DIR_CANDIDATES:
    if all((_dir / _f).exists() for _f in _TNR_FILES):
        for _f in _TNR_FILES:
            font_manager.fontManager.addfont(str(_dir / _f))
        break

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_INPUT = Path("experiments/mitbih_binary_ectopy_five_seed/multiseed_per_seed_summary.csv")
DEFAULT_OUTPUT = Path("fairness_article/figures/generated/fig_ecg_fairness_summary")
VARIANT_NAMES = {
    "teacher": "Teacher",
    "student": "Student (no KD)",
    "KD": "Standard KD",
    "T1": "EBTD",
    "O2": "GCOA",
    "T1+O2": "EBTD+GCOA",
}
METHODS = list(VARIANT_NAMES.values())
DISPLAY_LABELS = [
    "Teacher", "Student\n(no KD)", "Standard\nKD", "EBTD", "GCOA", "EBTD+\nGCOA"
]
GROUP_COLORS = {"Female": "#006795", "Male": "#306627"}
SEED_COLOR = "#737373"
MEAN_COLOR = "#DD7432"
GRID_COLOR = "#E0E0E0"
ERROR_COLOR = "#737373"
AXIS_COLOR = "#404040"
PAIR_LINE_COLOR = "#BFBFBF"


def style_fig3_axis(axis: plt.Axes) -> None:
    """Apply Figure 3's neutral grid and arrow-ended axis treatment."""
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axis.tick_params(axis="both", length=0, width=0, pad=3, labelsize=7.5, colors=AXIS_COLOR)
    axis.grid(True, axis="both", color=GRID_COLOR, linewidth=0.4)
    axis.set_axisbelow(True)
    arrow = dict(
        arrowstyle="->", color=AXIS_COLOR, linewidth=0.8,
        mutation_scale=8, shrinkA=0, shrinkB=0,
    )
    axis.annotate("", xy=(1.015, 0), xytext=(0, 0), xycoords="axes fraction",
                  arrowprops=arrow, annotation_clip=False)
    axis.annotate("", xy=(0, 1.025), xytext=(0, 0), xycoords="axes fraction",
                  arrowprops=arrow, annotation_clip=False)


def save_axis_pdf(figure: plt.Figure, axis: plt.Axes, path: Path) -> None:
    """Save one panel, including its labels and legend, as a vector PDF."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = axis.get_tightbbox(renderer).expanded(1.04, 1.04)
    figure.savefig(path, bbox_inches=bounds.transformed(figure.dpi_scale_trans.inverted()))


def load_results(path: Path) -> pd.DataFrame:
    """Extract per-seed sex recall and EO values from the locked summary."""
    rows: list[dict[str, float | int | str]] = []
    source = pd.read_csv(path)
    for _, record in source.iterrows():
        if record["variant"] not in VARIANT_NAMES:
            continue
        details = json.loads(record["eo_details"])
        ectopy = details["1"]
        for sex_code, label in (("F", "Female"), ("M", "Male")):
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "method": VARIANT_NAMES[record["variant"]],
                    "sex": label,
                    "ectopy_support": int(ectopy["supports"][sex_code]),
                    "ectopy_recall": float(ectopy["recalls"][sex_code]),
                    "raw_eo_gap": float(record["selected_eo"]),
                }
            )
    output = pd.DataFrame(rows)
    expected = len(METHODS) * 5 * 2
    if len(output) != expected:
        raise ValueError(f"Expected {expected} sex-specific rows, found {len(output)}")
    return output


def plot_fairness_summary(results: pd.DataFrame, output_prefix: Path) -> pd.DataFrame:
    """Plot absolute sex-specific recall and matched EO gaps across seeds."""
    summary = (
        results.groupby(["method", "sex"], observed=True)["ectopy_recall"]
        .agg(mean="mean", sd=lambda values: values.std(ddof=1))
        .reset_index()
    )
    eo_by_seed = (
        results[["seed", "method", "raw_eo_gap"]]
        .drop_duplicates()
        .pivot(index="seed", columns="method", values="raw_eo_gap")
        .reindex(columns=METHODS)
    )

    figure, (recall_axis, eo_axis) = plt.subplots(2, 1, figsize=(3.5, 5.25), constrained_layout=True)
    method_positions = np.arange(len(METHODS))
    offsets = {"Female": -0.18, "Male": 0.18}
    for sex in GROUP_COLORS:
        subset = summary[summary["sex"] == sex].set_index("method").loc[METHODS]
        positions = method_positions + offsets[sex]
        recall_axis.bar(
            positions,
            subset["mean"],
            width=0.32,
            yerr=subset["sd"],
            capsize=2.5,
            color=GROUP_COLORS[sex],
            edgecolor=AXIS_COLOR,
            linewidth=0.6,
            error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": ERROR_COLOR},
            label=sex,
        )
        points = results[results["sex"] == sex]
        for index, method in enumerate(METHODS):
            values = points.loc[points["method"] == method, "ectopy_recall"].to_numpy()
            recall_axis.scatter(
                np.full(values.size, positions[index]),
                values,
                s=18,
                color=SEED_COLOR,
                alpha=1.0,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )

    recall_axis.set_xticks(method_positions, DISPLAY_LABELS, fontsize=6.5)
    recall_axis.set_ylabel("Ectopy recall (higher better)", fontsize=8)
    recall_axis.set_ylim(0, 1)
    recall_axis.set_yticks(np.arange(0, 1.01, 0.2))
    style_fig3_axis(recall_axis)
    recall_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5,
        frameon=False, handlelength=1.1, handletextpad=0.45,
        columnspacing=1.0, borderaxespad=0,
    )

    eo_positions = np.arange(len(METHODS))
    for _, values in eo_by_seed.iterrows():
        eo_axis.plot(eo_positions, values.to_numpy(), color=PAIR_LINE_COLOR, linewidth=0.55, alpha=1.0, zorder=1)
        eo_axis.scatter(eo_positions, values.to_numpy(), color=SEED_COLOR, s=18, edgecolors="white", linewidths=0.6, zorder=2)
    means = eo_by_seed.mean(axis=0).to_numpy()
    sds = eo_by_seed.std(axis=0, ddof=1).to_numpy()
    eo_axis.errorbar(
        eo_positions,
        means,
        yerr=sds,
        fmt="D",
        markersize=5.5,
        markerfacecolor=MEAN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.6,
        color=MEAN_COLOR,
        ecolor=ERROR_COLOR,
        elinewidth=0.55,
        capsize=2.5,
        capthick=0.55,
        linestyle="none",
        zorder=3,
        label="Mean ± SD",
    )
    eo_axis.set_xticks(eo_positions, DISPLAY_LABELS, fontsize=6.5)
    eo_axis.set_ylabel("Female/male EO gap (lower better)", fontsize=8)
    eo_axis.set_ylim(0, 0.28)
    eo_axis.set_yticks(np.arange(0, 0.281, 0.05))
    style_fig3_axis(eo_axis)
    eo_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5,
        frameon=False, handlelength=1.1, handletextpad=0.45, borderaxespad=0,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".eps"), format="eps", bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    save_axis_pdf(
        figure,
        recall_axis,
        output_prefix.parent / "fig_ecg_group_ectopy_detection.pdf",
    )
    save_axis_pdf(
        figure,
        eo_axis,
        output_prefix.parent / "fig_ecg_ectopy_eo_gap.pdf",
    )
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a locked-result ECG fairness figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = load_results(args.input)
    summary = plot_fairness_summary(results, args.output_prefix)
    results.to_csv(args.output_prefix.parent / "ecg_ectopy_recall_by_seed.csv", index=False)
    summary.to_csv(args.output_prefix.parent / "ecg_ectopy_recall_summary.csv", index=False)
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
