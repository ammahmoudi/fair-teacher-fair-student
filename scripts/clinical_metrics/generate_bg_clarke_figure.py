#!/usr/bin/env python3
"""Generate auditable BG clinical-accuracy and group-recall figures.

Inputs are the matched, five-seed OhioT1DM prediction files used by the primary
manuscript table. Zone percentages are calculated by py-agata's tested
``clarke`` implementation (Clarke et al., Diabetes Care, 1987). The local
vectorized zone labels are used only to color a deterministic plotting sample;
the script asserts that their aggregate percentages equal py-agata's output.
It also verifies that the paired prediction files reproduce the locked primary
RMSE summary after the canonical per-seed three-decimal rounding.

Each standalone CEG shows 45-minute forecast targets on the x-axis and forecasts
on the y-axis. Scatter points are deterministic, display-only samples from every
valid test prediction across all five seeds. A separate two-panel fairness chart
reports sex-stratified hypoglycemia detection recall and the paired raw EO gaps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from py_agata.error import clarke

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fairness.utils.analyzer_utils import get_ohiot1dm_default_data

PIPELINE_ROOT = Path(
    "distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17"
    "/phase_3_distillation"
)
LOCKED_RESULTS = PIPELINE_ROOT.parent / "multiseed_robustness_results.csv"
SEEDS = ("238822", "247659", "427368", "809906", "831363")
METHODS = {
    "Standard KD": "bert_to_bert-tiny_all_patients_seed{seed}",
    "EBTD": "bert_to_bert-tiny_all_patients_fair_teacher_seed{seed}",
    "GCOA": "bert_to_bert-tiny_all_patients_o2_gender_seed{seed}",
    "EBTD+GCOA": "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed{seed}",
}
REFERENCE_METHODS = ("Teacher", "Student (no KD)")
FAIRNESS_METHODS = (*REFERENCE_METHODS, *METHODS)
TEACHER_STUDENT_SUITE = Path("teacher_student_multiseed")
TEACHER_STUDENT_PHASES = {
    "Teacher": "phase_1_teacher",
    "Student (no KD)": "phase_2_student",
}
LOCKED_METHOD_NAMES = {
    "Standard KD": "Baseline KD (no fairness)",
    "EBTD": "Distilled from Fair Teacher (T1)",
    "GCOA": "Standalone O2 Calibration Head",
    "EBTD+GCOA": "Distilled from Fair Teacher + O2 Calibration Head",
}
ZONE_NAMES = np.array(["A", "B", "C", "D", "E"])
ZONE_COLORS = np.array(["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#9D0208"])
GENDER = {patient: details["gender"] for patient, details in get_ohiot1dm_default_data().items()}
HYPOGLYCEMIA_THRESHOLD = 70.0
GROUP_COLORS = {"Female": "#006795", "Male": "#306627"}
SEED_COLOR = "#737373"
MEAN_COLOR = "#DD7432"
GRID_COLOR = "#E0E0E0"
ERROR_COLOR = "#737373"
AXIS_COLOR = "#404040"
PAIR_LINE_COLOR = "#BFBFBF"


def load_reference_group_tprs(suite_dir: Path) -> pd.DataFrame:
    """Load teacher and standalone-student recalls from their five-seed suite."""
    rows: list[dict[str, float | int | str]] = []
    inference_subpath = Path("per_patient_inference/time_llm_per_patient_inference_ohiot1dm")
    for method, phase in TEACHER_STUDENT_PHASES.items():
        for seed in SEEDS:
            inference_root = suite_dir / f"seed_{seed}" / phase / inference_subpath
            paths = sorted(inference_root.glob("**/inference_results_reformatted.csv"))
            patients = {next(part for part in path.parts if part.startswith("patient_")) for path in paths}
            if len(paths) != 12 or len(patients) != 12:
                raise ValueError(
                    f"{method}, seed {seed}: expected 12 patient inference files, found {len(paths)}"
                )
            by_group: dict[str, dict[str, list[np.ndarray]]] = {
                group: {"reference": [], "forecast": []} for group in GROUP_COLORS
            }
            for path in paths:
                reference, forecast = paired_values(path)
                patient = next(
                    part.removeprefix("patient_") for part in path.parts if part.startswith("patient_")
                )
                group = GENDER.get(patient)
                if group not in by_group:
                    raise ValueError(f"{method}, seed {seed}: unknown patient group for {patient}")
                by_group[group]["reference"].append(reference)
                by_group[group]["forecast"].append(forecast)
            for group, values in by_group.items():
                reference = np.concatenate(values["reference"])
                forecast = np.concatenate(values["forecast"])
                positives = reference < HYPOGLYCEMIA_THRESHOLD
                tpr = float(np.mean(forecast[positives] < HYPOGLYCEMIA_THRESHOLD))
                rows.append(
                    {"method": method, "seed": seed, "sex": group, "hypoglycemia_tpr": tpr}
                )
    return pd.DataFrame(rows)


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


METHOD_FILE_STEMS = {
    "Standard KD": "standard_kd",
    "EBTD": "ebtd_t1",
    "GCOA": "gcoa",
    "EBTD+GCOA": "ebtd_gcoa",
}


def clarke_zone_indices(reference: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Return Clarke zones as zero-based A--E indices for finite input pairs.

    The ordered conditions reproduce py-agata.error.clarke. This local function
    is deliberately validated against py-agata before its labels are used only
    for scatter colors and zone-level descriptive summaries.
    """
    zone = np.full(reference.size, 1, dtype=np.int8)  # Zone B default.
    # Assign low-priority conditions first to reproduce py-agata's A, E, C, D
    # if/elif ordering at boundary intersections.
    zone[
        ((reference >= 240) & (forecast >= 70) & (forecast <= 180))
        | ((reference <= 175 / 3) & (forecast <= 180) & (forecast >= 70))
        | ((reference >= 175 / 3) & (reference <= 70) & (forecast >= 1.2 * reference))
    ] = 3
    zone[
        ((reference >= 70) & (reference <= 290) & (forecast >= reference + 110))
        | ((reference >= 130) & (reference <= 180) & (forecast <= 1.4 * reference - 182))
    ] = 2
    zone[(reference >= 180) & (forecast <= 70)] = 4
    zone[(reference <= 70) & (forecast >= 180)] = 4
    zone[(forecast <= 70) & (reference <= 70)] = 0
    zone[(forecast <= 1.2 * reference) & (forecast >= 0.8 * reference)] = 0
    return zone


def paired_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Flatten all paired forecast horizons from one patient inference CSV."""
    frame = pd.read_csv(path)
    references: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    for true_column in frame.columns:
        if not true_column.endswith("_true"):
            continue
        forecast_column = true_column.removesuffix("_true") + "_pred"
        if forecast_column not in frame:
            continue
        reference = frame[true_column].to_numpy(dtype=float)
        forecast = frame[forecast_column].to_numpy(dtype=float)
        valid = np.isfinite(reference) & np.isfinite(forecast)
        references.append(reference[valid])
        forecasts.append(forecast[valid])
    if not references:
        raise ValueError(f"No paired true/prediction columns in {path}")
    return np.concatenate(references), np.concatenate(forecasts)


def py_agata_percentages(reference: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Calculate A--E percentages via the package's validated CEGA function."""
    times = np.arange(reference.size)
    observed = pd.DataFrame({"t": times, "glucose": reference})
    predicted = pd.DataFrame({"t": times, "glucose": forecast})
    values = clarke(observed, predicted)
    return np.array([values[key] for key in ("a", "b", "c", "d", "e")])


def load_method(
    method: str, template: str, pipeline_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load matched forecasts, CEG summaries, RMSE audits, and group TPRs."""
    sample_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    audits: list[dict[str, float | int | str]] = []
    group_tprs: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        paths = sorted((pipeline_root / template.format(seed=seed)).glob("**/inference_results_reformatted.csv"))
        patients = {next(part for part in path.parts if part.startswith("patient_")) for path in paths}
        if len(paths) != 12 or len(patients) != 12:
            raise ValueError(f"{method}, seed {seed}: expected 12 patient inference files, found {len(paths)}")

        seed_reference: list[np.ndarray] = []
        seed_forecast: list[np.ndarray] = []
        by_group: dict[str, dict[str, list[np.ndarray]]] = {
            group: {"reference": [], "forecast": []} for group in GROUP_COLORS
        }
        for path in paths:
            reference, forecast = paired_values(path)
            seed_reference.append(reference)
            seed_forecast.append(forecast)
            patient = next(part.removeprefix("patient_") for part in path.parts if part.startswith("patient_"))
            group = GENDER.get(patient)
            if group not in by_group:
                raise ValueError(f"{method}, seed {seed}: unknown patient group for {patient}")
            by_group[group]["reference"].append(reference)
            by_group[group]["forecast"].append(forecast)
        reference = np.concatenate(seed_reference)
        forecast = np.concatenate(seed_forecast)
        zone = clarke_zone_indices(reference, forecast)
        zone_percentages = np.bincount(zone, minlength=5) / reference.size * 100
        validated_percentages = py_agata_percentages(reference, forecast)
        if not np.allclose(zone_percentages, validated_percentages, atol=1e-10):
            raise AssertionError(
                f"{method}, seed {seed}: plot-zone labels disagree with py-agata: "
                f"{zone_percentages} vs {validated_percentages}"
            )

        summaries.append({"method": method, "seed": seed, **dict(zip(ZONE_NAMES, validated_percentages))})
        audits.append(
            {
                "method": method,
                "seed": seed,
                "n_forecasts": reference.size,
                "rmse": float(np.sqrt(np.mean((forecast - reference) ** 2))),
            }
        )
        for group, values in by_group.items():
            group_reference = np.concatenate(values["reference"])
            group_forecast = np.concatenate(values["forecast"])
            hypoglycemia = group_reference < HYPOGLYCEMIA_THRESHOLD
            group_tprs.append(
                {
                    "method": method,
                    "seed": seed,
                    "sex": group,
                    "hypoglycemia_support": int(hypoglycemia.sum()),
                    "hypoglycemia_tpr": float((group_forecast[hypoglycemia] < HYPOGLYCEMIA_THRESHOLD).mean()),
                }
            )
        sample_frames.append(
            pd.DataFrame({"method": method, "seed": seed, "reference": reference, "forecast": forecast, "zone": ZONE_NAMES[zone]})
        )

    return (
        pd.concat(sample_frames, ignore_index=True),
        pd.DataFrame(summaries),
        pd.DataFrame(audits),
        pd.DataFrame(group_tprs),
    )


def verify_locked_rmse(audits: pd.DataFrame, locked_results: Path) -> pd.DataFrame:
    """Confirm that the CEG source files reproduce the locked primary RMSE rows."""
    if not locked_results.exists():
        raise FileNotFoundError(f"Locked primary results not found: {locked_results}")
    locked = pd.read_csv(locked_results).set_index("method")
    checks: list[dict[str, float | str]] = []
    for method, locked_name in LOCKED_METHOD_NAMES.items():
        method_audit = audits.loc[audits["method"] == method]
        per_seed_rmse = method_audit["rmse"].round(3)
        observed_mean = per_seed_rmse.mean()
        observed_std = per_seed_rmse.std(ddof=0)
        expected = locked.loc[locked_name]
        expected_mean = float(expected["rmse_mean"])
        expected_std = float(expected["rmse_std"])
        if not np.isclose(observed_mean, expected_mean, atol=5e-7) or not np.isclose(
            observed_std, expected_std, atol=5e-7
        ):
            raise AssertionError(
                f"{method}: CEG inputs do not reproduce locked RMSE "
                f"({observed_mean:.6f} +/- {observed_std:.6f} versus "
                f"{expected_mean:.6f} +/- {expected_std:.6f})"
            )
        checks.append(
            {
                "method": method,
                "locked_method": locked_name,
                "rmse_mean_from_ceg_inputs": observed_mean,
                "rmse_std_from_ceg_inputs": observed_std,
                "locked_rmse_mean": expected_mean,
                "locked_rmse_std": expected_std,
            }
        )
    return pd.DataFrame(checks)


def draw_clarke_boundaries(axis: plt.Axes) -> None:
    """Draw the standard Clarke-grid decision boundaries over the scatter panel."""
    line = {"color": "#343A40", "linewidth": 0.8, "alpha": 0.8, "zorder": 3}
    x = np.linspace(0, 400, 401)
    axis.plot(x, x, linestyle="--", **line)
    axis.plot(x, 0.8 * x, linestyle=":", **line)
    axis.plot(x, 1.2 * x, linestyle=":", **line)
    axis.plot([70, 70], [0, 400], **line)
    axis.plot([180, 180], [0, 70], **line)
    axis.plot([0, 400], [70, 70], **line)
    axis.plot([240, 240], [70, 180], **line)
    c_x = np.linspace(70, 290, 221)
    axis.plot(c_x, c_x + 110, **line)
    c_x = np.linspace(130, 180, 51)
    axis.plot(c_x, 1.4 * c_x - 182, **line)
    d_x = np.linspace(175 / 3, 70, 50)
    axis.plot(d_x, 1.2 * d_x, **line)
    for label, xy in {"A": (265, 275), "B": (300, 180), "C": (145, 330), "D": (45, 160), "E": (245, 35)}.items():
        axis.text(*xy, label, color="#343A40", fontsize=7, fontweight="bold", ha="center", va="center", zorder=4)


def plot_clarke_figure(predictions: pd.DataFrame, method: str, output_prefix: Path, sample_size: int) -> None:
    """Create one clean, standalone CEG scatter plot for a single method."""
    rng = np.random.default_rng(20260726 + list(METHODS).index(method))
    figure, axis = plt.subplots(figsize=(4.15, 4.05), constrained_layout=True)
    cmap = ListedColormap(ZONE_COLORS)
    zone_to_index = {zone: index for index, zone in enumerate(ZONE_NAMES)}

    panel = predictions[predictions["method"] == method]
    n_sample = min(sample_size, len(panel))
    sampled = panel.iloc[rng.choice(len(panel), size=n_sample, replace=False)]
    zone_values = sampled["zone"].map(zone_to_index).to_numpy()
    axis.scatter(
        sampled["reference"], sampled["forecast"], c=zone_values, cmap=cmap,
        s=2.0, alpha=0.24, linewidths=0, rasterized=True, zorder=2,
    )
    draw_clarke_boundaries(axis)
    axis.set_xlim(0, 400)
    axis.set_ylim(0, 400)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(method, fontsize=8, fontweight="bold", pad=5)
    axis.set_xlabel("Reference glucose (mg/dL)", fontsize=7)
    axis.set_ylabel("Forecast glucose (mg/dL)", fontsize=7)
    axis.tick_params(labelsize=6.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=color, label=f"Zone {zone}")
               for zone, color in zip(ZONE_NAMES, ZONE_COLORS)]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
        ncol=5, frameon=True, fontsize=5.5, handletextpad=0.3,
        columnspacing=0.65, borderpad=0.35,
    )

    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)


def plot_fairness_summary(group_tprs: pd.DataFrame, output_prefix: Path) -> pd.DataFrame:
    """Plot absolute sex-specific TPRs alongside the paired raw EO gaps."""
    summary = (
        group_tprs.groupby(["method", "sex"], observed=True)["hypoglycemia_tpr"]
        .agg(mean="mean", sd=lambda values: values.std(ddof=0))
        .reset_index()
    )
    eo_by_seed = pd.DataFrame(index=pd.Index(SEEDS, name="seed"))
    for method in FAIRNESS_METHODS:
        sex_tprs = (
            group_tprs[group_tprs["method"] == method]
            .pivot(index="seed", columns="sex", values="hypoglycemia_tpr")
            .reindex(SEEDS)
        )
        eo_by_seed[method] = (sex_tprs["Female"] - sex_tprs["Male"]).abs()

    figure, (tpr_axis, eo_axis) = plt.subplots(2, 1, figsize=(3.5, 5.25), constrained_layout=True)
    method_positions = np.arange(len(FAIRNESS_METHODS))
    offsets = {"Female": -0.18, "Male": 0.18}
    for sex in GROUP_COLORS:
        subset = summary[summary["sex"] == sex].set_index("method").loc[list(FAIRNESS_METHODS)]
        positions = method_positions + offsets[sex]
        tpr_axis.bar(
            positions, subset["mean"], width=0.32,
            yerr=subset["sd"], capsize=2.5, color=GROUP_COLORS[sex],
            edgecolor=AXIS_COLOR, linewidth=0.6,
            error_kw={"elinewidth": 0.55, "capthick": 0.55, "ecolor": ERROR_COLOR}, label=sex,
        )
        values = group_tprs[group_tprs["sex"] == sex].set_index("method").loc[list(FAIRNESS_METHODS)]
        for method_index, method in enumerate(FAIRNESS_METHODS):
            seed_values = values.loc[method]
            if isinstance(seed_values, pd.Series):
                tpr_axis.scatter(
                    np.full(len(seed_values), positions[method_index]),
                    seed_values["hypoglycemia_tpr"], s=18, color=SEED_COLOR, alpha=1.0,
                    edgecolors="white", linewidths=0.6, zorder=4,
                )

    display_labels = [
        "Teacher", "Student\n(no KD)", "Standard\nKD", "EBTD", "GCOA", "EBTD+\nGCOA"
    ]
    tpr_axis.set_xticks(method_positions, display_labels, fontsize=6.5)
    tpr_axis.set_ylabel("Hypoglycemia TPR (higher better)", fontsize=8)
    tpr_axis.set_ylim(0, 1)
    tpr_axis.set_yticks(np.arange(0, 1.01, 0.2))
    style_fig3_axis(tpr_axis)
    tpr_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5,
        frameon=False, handlelength=1.1, handletextpad=0.45,
        columnspacing=1.0, borderaxespad=0,
    )

    eo_positions = np.arange(len(FAIRNESS_METHODS))
    for _, values in eo_by_seed.iterrows():
        eo_axis.plot(eo_positions, values.to_numpy(), color=PAIR_LINE_COLOR, linewidth=0.55, alpha=1.0, zorder=1)
        eo_axis.scatter(eo_positions, values.to_numpy(), color=SEED_COLOR, s=18, edgecolors="white", linewidths=0.6, zorder=2)
    eo_means = eo_by_seed.mean(axis=0).to_numpy()
    eo_sds = eo_by_seed.std(axis=0, ddof=0).to_numpy()
    eo_axis.errorbar(
        eo_positions, eo_means, yerr=eo_sds, fmt="D", markersize=5.5,
        markerfacecolor=MEAN_COLOR, markeredgecolor="white", markeredgewidth=0.6,
        color=MEAN_COLOR, ecolor=ERROR_COLOR, elinewidth=0.55,
        capsize=2.5, capthick=0.55, linestyle="none", zorder=3,
        label="Mean ± SD",
    )
    eo_axis.set_xticks(eo_positions, display_labels, fontsize=6.5)
    eo_axis.set_ylabel("Raw EO gap (lower better)", fontsize=8)
    eo_axis.set_ylim(0, 0.28)
    eo_axis.set_yticks(np.arange(0, 0.281, 0.05))
    style_fig3_axis(eo_axis)
    eo_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7.5,
        frameon=False, handlelength=1.1, handletextpad=0.45, borderaxespad=0,
    )
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".eps"), format="eps", bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    save_axis_pdf(
        figure,
        tpr_axis,
        output_prefix.parent / "fig_bg_group_hypoglycemia_detection.pdf",
    )
    save_axis_pdf(
        figure,
        eo_axis,
        output_prefix.parent / "fig_bg_hypoglycemia_eo_gap.pdf",
    )
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a validated five-seed BG Clarke Error Grid figure.")
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument("--teacher-student-suite", type=Path, default=TEACHER_STUDENT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=Path("fairness_article/figures/generated"))
    parser.add_argument("--sample-size", type=int, default=25000, help="Display-only points per method panel.")
    args = parser.parse_args()

    prediction_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    group_tpr_frames: list[pd.DataFrame] = []
    for method, template in METHODS.items():
        predictions, summary, audit, group_tprs = load_method(method, template, args.pipeline_root)
        prediction_frames.append(predictions)
        summary_frames.append(summary)
        audit_frames.append(audit)
        group_tpr_frames.append(group_tprs)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)
    group_tpr_frames.insert(0, load_reference_group_tprs(args.teacher_student_suite))
    group_tprs = pd.concat(group_tpr_frames, ignore_index=True)
    rmse_checks = verify_locked_rmse(audits, args.pipeline_root.parent / "multiseed_robustness_results.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output_dir / "bg_ceg_zone_summary_by_seed.csv", index=False)
    audits.to_csv(args.output_dir / "bg_ceg_rmse_audit_by_seed.csv", index=False)
    rmse_checks.to_csv(args.output_dir / "bg_ceg_locked_rmse_check.csv", index=False)
    group_tprs.to_csv(args.output_dir / "bg_hypoglycemia_tpr_by_seed.csv", index=False)
    all_predictions.groupby(["method", "seed", "zone"], observed=True).size().rename("n_forecasts").reset_index().to_csv(
        args.output_dir / "bg_ceg_zone_counts_by_seed.csv", index=False
    )
    for method in METHODS:
        plot_clarke_figure(
            all_predictions,
            method,
            args.output_dir / f"fig_bg_clarke_{METHOD_FILE_STEMS[method]}",
            args.sample_size,
        )
    group_summary = plot_fairness_summary(group_tprs, args.output_dir / "fig_bg_group_hypoglycemia_tpr")
    group_summary.to_csv(args.output_dir / "bg_hypoglycemia_tpr_summary.csv", index=False)

    print("Validated CEG zone percentages (mean +/- SD across five seeds):")
    for method in METHODS:
        subset = summaries[summaries["method"] == method]
        values = [f"{zone}={subset[zone].mean():.3f}+/-{subset[zone].std(ddof=1):.3f}%" for zone in ZONE_NAMES]
        print(f"{method}: " + ", ".join(values))
    print("Verified CEG source RMSE against multiseed_robustness_results.csv")
    print(f"Wrote standalone CEG figures and {args.output_dir / 'fig_bg_group_hypoglycemia_tpr.pdf'}")


if __name__ == "__main__":
    main()
