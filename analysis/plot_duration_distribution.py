"""Create a publication-ready duration-distribution figure.

The figure compares only the training splits: CASTELLA (real recordings) and
Clotho-Moment (synthetic recordings). It is exported as vector PDF and as a
600-DPI PNG for convenient use in LaTeX and presentation software.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

DATASETS = {
    "CASTELLA (real)": DATA_DIR / "castella_train_release.jsonl",
    "Clotho-Moment (synthetic)": DATA_DIR
    / "clotho_moment_train_release.jsonl",
}

COLORS = {
    "CASTELLA (real)": "#0072B2",
    "Clotho-Moment (synthetic)": "#D55E00",
}


def load_widths(path: Path) -> np.ndarray:
    widths: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            for start, end in item["relevant_windows"]:
                width = float(end) - float(start)
                if width <= 0:
                    raise ValueError(f"Non-positive moment duration in {path}")
                widths.append(width)
    return np.asarray(widths, dtype=float)


def bin_percentages(widths: np.ndarray) -> np.ndarray:
    counts = np.asarray(
        [
            np.count_nonzero(widths < 5),
            np.count_nonzero((widths >= 5) & (widths < 10)),
            np.count_nonzero((widths >= 10) & (widths < 20)),
            np.count_nonzero(widths >= 20),
        ]
    )
    return counts / len(widths) * 100


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    widths_by_dataset = {
        name: load_widths(path) for name, path in DATASETS.items()
    }

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

    figure, (ecdf_axis, bar_axis) = plt.subplots(
        1, 2, figsize=(7.2, 3.05), constrained_layout=True
    )

    # Panel (a): empirical cumulative distribution on a logarithmic x-axis.
    for name, widths in widths_by_dataset.items():
        ordered = np.sort(widths)
        cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
        ecdf_axis.step(
            ordered,
            cumulative,
            where="post",
            color=COLORS[name],
            linewidth=2.0,
            label=name,
        )

    ecdf_axis.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    ecdf_axis.set_xscale("log")
    ecdf_axis.set_xlim(0.8, 320)
    ecdf_axis.set_ylim(0, 1.01)
    ticks = [1, 2, 5, 10, 20, 50, 100, 300]
    ecdf_axis.xaxis.set_major_locator(FixedLocator(ticks))
    ecdf_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ecdf_axis.set_xlabel("Moment duration (s, log scale)")
    ecdf_axis.set_ylabel("Cumulative proportion")
    ecdf_axis.set_title("(a) Empirical cumulative distribution", loc="left")
    ecdf_axis.legend(frameon=False, loc="lower right")
    style_axes(ecdf_axis)

    medians = {
        name: float(np.median(widths)) for name, widths in widths_by_dataset.items()
    }
    for name, median in medians.items():
        ecdf_axis.scatter(
            [median], [0.5], s=24, color=COLORS[name], edgecolor="white", zorder=3
        )
    ecdf_axis.annotate(
        f"Median = {medians['CASTELLA (real)']:g} s",
        xy=(medians["CASTELLA (real)"], 0.5),
        xytext=(1.4, 0.61),
        color=COLORS["CASTELLA (real)"],
        arrowprops={"arrowstyle": "-", "color": COLORS["CASTELLA (real)"], "lw": 0.8},
    )
    ecdf_axis.annotate(
        f"Median = {medians['Clotho-Moment (synthetic)']:g} s",
        xy=(medians["Clotho-Moment (synthetic)"], 0.5),
        xytext=(24, 0.39),
        color=COLORS["Clotho-Moment (synthetic)"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["Clotho-Moment (synthetic)"],
            "lw": 0.8,
        },
    )

    # Panel (b): bins chosen to match the planned short-moment evaluation.
    labels = ["<5", "5–<10", "10–<20", "≥20"]
    positions = np.arange(len(labels))
    bar_width = 0.37
    offsets = [-bar_width / 2, bar_width / 2]

    for offset, (name, widths) in zip(offsets, widths_by_dataset.items()):
        percentages = bin_percentages(widths)
        bars = bar_axis.bar(
            positions + offset,
            percentages,
            width=bar_width,
            color=COLORS[name],
            label=name,
        )
        bar_axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in percentages],
            padding=2,
            fontsize=7.2,
            color=COLORS[name],
        )

    bar_axis.set_xticks(positions, labels)
    bar_axis.set_xlabel("Duration bin (s)")
    bar_axis.set_ylabel("Annotated moments (%)")
    bar_axis.set_ylim(0, 62)
    bar_axis.set_title("(b) Proportion in duration bins", loc="left")
    style_axes(bar_axis)

    pdf_path = OUTPUT_DIR / "train_duration_distribution.pdf"
    png_path = OUTPUT_DIR / "train_duration_distribution.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved vector figure: {pdf_path}")
    print(f"Saved 600-DPI figure: {png_path}")


if __name__ == "__main__":
    main()
