"""Plot duration-stratified baseline performance and width errors."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULT_DIR = ANALYSIS_DIR / "results" / "baseline_by_duration"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
OUTPUT_DIR = ANALYSIS_DIR / "figures"


def load_summary() -> list[dict[str, float | str]]:
    with SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_keys = [key for key in rows[0] if key != "duration_bin"]
    converted: list[dict[str, float | str]] = []
    for row in rows:
        converted.append(
            {
                key: value if key == "duration_bin" else float(value)
                for key, value in row.items()
            }
        )
    return converted


def wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return 100 * (center - half_width), 100 * (center + half_width)


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.85)
    axis.set_axisbelow(True)


def main() -> None:
    rows = load_summary()
    grouped = [row for row in rows if row["duration_bin"] != "All"]
    labels = ["<5", "5–<10", "10–<20", "≥20"]
    counts = [int(row["queries"]) for row in grouped]
    positions = np.arange(len(grouped))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, (recall_axis, error_axis) = plt.subplots(
        1, 2, figsize=(7.2, 3.05), constrained_layout=True
    )

    # Panel (a): Top-1 recall, with 95% Wilson confidence intervals.
    metrics = [
        ("r1_at_0.5_percent", "R1@0.5", "#0072B2"),
        ("r1_at_0.7_percent", "R1@0.7", "#D55E00"),
    ]
    bar_width = 0.36
    for metric_index, (key, legend_label, color) in enumerate(metrics):
        values = np.asarray([float(row[key]) for row in grouped])
        errors_low: list[float] = []
        errors_high: list[float] = []
        for value, count in zip(values, counts):
            successes = value / 100 * count
            lower, upper = wilson_interval(successes, count)
            errors_low.append(value - lower)
            errors_high.append(upper - value)
        offset = (metric_index - 0.5) * bar_width
        bars = recall_axis.bar(
            positions + offset,
            values,
            width=bar_width,
            color=color,
            label=legend_label,
            yerr=np.vstack([errors_low, errors_high]),
            error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
        )
        recall_axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in values],
            padding=5,
            fontsize=7.2,
            color=color,
        )

    recall_axis.set_xticks(
        positions, [f"{label}\n($n$={count})" for label, count in zip(labels, counts)]
    )
    recall_axis.set_xlabel("Ground-truth duration bin (s)")
    recall_axis.set_ylabel("Recall at one (%)")
    recall_axis.set_ylim(0, 43)
    recall_axis.set_title("(a) Top-1 localization performance", loc="left")
    recall_axis.legend(frameon=False, loc="upper left", ncol=2)
    style_axes(recall_axis)

    # Panel (b): median signed width error, with interquartile range.
    medians = np.asarray([float(row["width_error_median"]) for row in grouped])
    quartile_1 = np.asarray([float(row["width_error_p25"]) for row in grouped])
    quartile_3 = np.asarray([float(row["width_error_p75"]) for row in grouped])
    asymmetric_error = np.vstack([medians - quartile_1, quartile_3 - medians])
    error_axis.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    error_axis.errorbar(
        positions,
        medians,
        yerr=asymmetric_error,
        fmt="o",
        color="#009E73",
        markerfacecolor="#009E73",
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=7,
        elinewidth=2.2,
        capsize=5,
        capthick=1.2,
        label="Median and IQR",
    )
    for position, value in zip(positions, medians):
        vertical_offset = 3.0 if value >= 0 else -5.0
        error_axis.text(
            position,
            value + vertical_offset,
            f"{value:+.0f} s",
            ha="center",
            va="center",
            color="#007A59",
            fontsize=8,
        )
    error_axis.set_xticks(positions, labels)
    error_axis.set_xlabel("Ground-truth duration bin (s)")
    error_axis.set_ylabel("Predicted width − ground-truth width (s)")
    error_axis.set_ylim(-53, 22)
    error_axis.set_title("(b) Top-1 width error", loc="left")
    error_axis.legend(frameon=False, loc="lower left")
    style_axes(error_axis)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "baseline_performance_by_duration.pdf"
    png_path = OUTPUT_DIR / "baseline_performance_by_duration.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved vector figure: {pdf_path}")
    print(f"Saved 600-DPI figure: {png_path}")


if __name__ == "__main__":
    main()
