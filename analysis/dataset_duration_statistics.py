"""Compare annotated moment durations in CASTELLA and Clotho-Moment.

This script uses only the JSONL annotations shipped with the baseline repository.
It deliberately has no third-party dependencies, so it can be run before the
audio features or the PyTorch environment are ready.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

DATASETS = {
    "CASTELLA train (real)": DATA_DIR / "castella_train_release.jsonl",
    "CASTELLA val (real)": DATA_DIR / "castella_val_release.jsonl",
    "Clotho-Moment train (synthetic)": DATA_DIR
    / "clotho_moment_train_release.jsonl",
    "Clotho-Moment val (synthetic)": DATA_DIR
    / "clotho_moment_val_release.jsonl",
}


@dataclass(frozen=True)
class Moment:
    dataset: str
    qid: str
    audio_duration: float
    start: float
    end: float

    @property
    def width(self) -> float:
        return self.end - self.start

    @property
    def relative_width(self) -> float:
        return self.width / self.audio_duration


def load_moments(name: str, path: Path) -> tuple[int, list[Moment], int]:
    query_count = 0
    rounded_boundary_count = 0
    moments: list[Moment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            query_count += 1
            duration = float(item["duration"])
            if duration <= 0:
                raise ValueError(f"{path}:{line_number}: non-positive duration")
            for start, end in item["relevant_windows"]:
                start, end = float(start), float(end)
                if not (0 <= start < end):
                    raise ValueError(
                        f"{path}:{line_number}: invalid window [{start}, {end}] "
                        f"for duration {duration}"
                    )
                # CASTELLA uses one-second annotation boundaries. A small number
                # of end points are rounded to the next integer (for example,
                # 300 s for an audio duration stored as 299 s). Keep these valid
                # annotations but count them so the data-quality detail is visible.
                if end > duration:
                    if end - duration > 1.0 + 1e-6:
                        raise ValueError(
                            f"{path}:{line_number}: window end {end} exceeds "
                            f"duration {duration} by more than one second"
                        )
                    rounded_boundary_count += 1
                moments.append(Moment(name, str(item["qid"]), duration, start, end))
    return query_count, moments, rounded_boundary_count


def percentile(values: list[float], probability: float) -> float:
    """Linear interpolation, equivalent to NumPy's default percentile method."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def duration_bin(width: float) -> str:
    if width < 5:
        return "<5 s"
    if width < 10:
        return "5-<10 s"
    if width < 20:
        return "10-<20 s"
    return ">=20 s"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[int, list[Moment], int]] = {}
    for name, path in DATASETS.items():
        loaded[name] = load_moments(name, path)

    summary_rows: list[dict[str, object]] = []
    bin_rows: list[dict[str, object]] = []

    for name, (query_count, moments, rounded_boundary_count) in loaded.items():
        widths = [moment.width for moment in moments]
        relative_widths = [moment.relative_width for moment in moments]
        width_summary = summarize(widths)
        relative_summary = summarize(relative_widths)
        summary_rows.append(
            {
                "dataset": name,
                "queries": query_count,
                "moments": len(moments),
                "rounded_end_boundaries": rounded_boundary_count,
                **{f"width_seconds_{key}": value for key, value in width_summary.items()},
                **{
                    f"relative_width_{key}": value
                    for key, value in relative_summary.items()
                },
            }
        )

        for label in ("<5 s", "5-<10 s", "10-<20 s", ">=20 s"):
            count = sum(duration_bin(width) == label for width in widths)
            bin_rows.append(
                {
                    "dataset": name,
                    "duration_bin": label,
                    "count": count,
                    "percentage": 100 * count / len(widths),
                }
            )

    summary_path = OUTPUT_DIR / "duration_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    bins_path = OUTPUT_DIR / "duration_bins.csv"
    with bins_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bin_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bin_rows)

    print("\nMoment-duration summary")
    print("=" * 92)
    for row in summary_rows:
        print(
            f"{row['dataset']:<38} "
            f"queries={row['queries']:>5}  moments={row['moments']:>5}  "
            f"median={row['width_seconds_median']:>5.1f}s  "
            f"mean={row['width_seconds_mean']:>5.1f}s  "
            f"relative median={100 * row['relative_width_median']:>5.1f}%"
        )

    rounded_total = sum(int(row["rounded_end_boundaries"]) for row in summary_rows)
    if rounded_total:
        print(
            f"\nData note: retained {rounded_total} window(s) whose integer end "
            "boundary exceeds the stored audio duration by at most one second."
        )

    print("\nAbsolute-duration bins")
    print("=" * 92)
    for name in DATASETS:
        formatted = [
            f"{row['duration_bin']}: {row['percentage']:.1f}%"
            for row in bin_rows
            if row["dataset"] == name
        ]
        print(f"{name:<38} " + " | ".join(formatted))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {bins_path}")


if __name__ == "__main__":
    main()
