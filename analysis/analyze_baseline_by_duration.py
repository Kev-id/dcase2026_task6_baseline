"""Analyze CASTELLA validation predictions by ground-truth moment duration.

The Top-1 matching rule follows ``src/standalone_eval/eval.py``: the highest
scoring predicted window is matched to the ground-truth window with maximum
temporal IoU. For a duration bin, only ground-truth windows in that bin are
eligible for matching. Consequently, a query with ground-truth windows in
multiple bins can appear once in each applicable bin, as in the official
duration-filtering logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO_ROOT / "data" / "castella_val_release.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "results" / "submission.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "baseline_by_duration",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def temporal_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0 else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_records(label: str, records: list[dict]) -> dict[str, object]:
    ious = [float(record["top1_iou"]) for record in records]
    gt_widths = [float(record["matched_gt_width"]) for record in records]
    pred_widths = [float(record["pred_width"]) for record in records]
    width_errors = [float(record["width_error"]) for record in records]
    width_ratios = [float(record["width_ratio"]) for record in records]
    abs_center_errors = [float(record["abs_center_error"]) for record in records]

    return {
        "duration_bin": label,
        "queries": len(records),
        "mean_iou": statistics.fmean(ious),
        "median_iou": statistics.median(ious),
        "r1_at_0.3_percent": 100 * sum(value >= 0.3 for value in ious) / len(ious),
        "r1_at_0.5_percent": 100 * sum(value >= 0.5 for value in ious) / len(ious),
        "r1_at_0.7_percent": 100 * sum(value >= 0.7 for value in ious) / len(ious),
        "gt_width_mean": statistics.fmean(gt_widths),
        "gt_width_median": statistics.median(gt_widths),
        "pred_width_mean": statistics.fmean(pred_widths),
        "pred_width_median": statistics.median(pred_widths),
        "pred_width_p25": percentile(pred_widths, 0.25),
        "pred_width_p75": percentile(pred_widths, 0.75),
        "width_error_mean": statistics.fmean(width_errors),
        "width_error_median": statistics.median(width_errors),
        "width_error_p25": percentile(width_errors, 0.25),
        "width_error_p75": percentile(width_errors, 0.75),
        "width_ratio_median": statistics.median(width_ratios),
        "overwide_percent": 100 * sum(value > 0 for value in width_errors) / len(records),
        "at_least_twice_gt_width_percent": 100
        * sum(value >= 2 for value in width_ratios)
        / len(records),
        "abs_center_error_median": statistics.median(abs_center_errors),
    }


def build_record(
    label: str,
    ground_truth: dict,
    prediction: dict,
    eligible_windows: list[list[float]],
) -> dict[str, object]:
    top_prediction = [float(value) for value in prediction["pred_relevant_windows"][0][:2]]
    top_score = (
        float(prediction["pred_relevant_windows"][0][2])
        if len(prediction["pred_relevant_windows"][0]) >= 3
        else float("nan")
    )
    ious = [temporal_iou(top_prediction, window) for window in eligible_windows]
    matched_index = max(range(len(ious)), key=ious.__getitem__)
    matched_gt = [float(value) for value in eligible_windows[matched_index]]

    pred_width = top_prediction[1] - top_prediction[0]
    gt_width = matched_gt[1] - matched_gt[0]
    pred_center = (top_prediction[0] + top_prediction[1]) / 2
    gt_center = (matched_gt[0] + matched_gt[1]) / 2

    return {
        "duration_bin": label,
        "qid": ground_truth["qid"],
        "query": ground_truth["query"],
        "eligible_gt_windows": len(eligible_windows),
        "matched_gt_start": matched_gt[0],
        "matched_gt_end": matched_gt[1],
        "matched_gt_width": gt_width,
        "pred_start": top_prediction[0],
        "pred_end": top_prediction[1],
        "pred_width": pred_width,
        "top1_score": top_score,
        "top1_iou": ious[matched_index],
        "start_error": top_prediction[0] - matched_gt[0],
        "end_error": top_prediction[1] - matched_gt[1],
        "width_error": pred_width - gt_width,
        "abs_width_error": abs(pred_width - gt_width),
        "width_ratio": pred_width / gt_width,
        "center_error": pred_center - gt_center,
        "abs_center_error": abs(pred_center - gt_center),
    }


def main() -> None:
    args = parse_args()
    ground_truth = load_jsonl(args.ground_truth)
    predictions = load_jsonl(args.predictions)
    prediction_by_qid = {item["qid"]: item for item in predictions}
    ground_truth_qids = {item["qid"] for item in ground_truth}

    missing = ground_truth_qids - prediction_by_qid.keys()
    extra = prediction_by_qid.keys() - ground_truth_qids
    if missing or extra:
        raise ValueError(
            f"Prediction/ground-truth mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    bins: list[tuple[str, Callable[[float], bool]]] = [
        ("<5 s", lambda width: width < 5),
        ("5-<10 s", lambda width: 5 <= width < 10),
        ("10-<20 s", lambda width: 10 <= width < 20),
        (">=20 s", lambda width: width >= 20),
    ]

    records_by_label: dict[str, list[dict]] = {label: [] for label, _ in bins}
    records_by_label["All"] = []

    for item in ground_truth:
        prediction = prediction_by_qid[item["qid"]]
        all_windows = [[float(start), float(end)] for start, end in item["relevant_windows"]]
        records_by_label["All"].append(
            build_record("All", item, prediction, all_windows)
        )

        for label, belongs in bins:
            eligible = [
                window
                for window in all_windows
                if belongs(window[1] - window[0])
            ]
            if eligible:
                records_by_label[label].append(
                    build_record(label, item, prediction, eligible)
                )

    all_records = [
        record
        for label, _ in bins
        for record in records_by_label[label]
    ]
    summaries = [summarize_records("All", records_by_label["All"])] + [
        summarize_records(label, records_by_label[label]) for label, _ in bins
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "per_query.csv"
    summary_path = args.output_dir / "summary.csv"
    metadata_path = args.output_dir / "metadata.json"

    with details_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "ground_truth": str(args.ground_truth.resolve()),
        "predictions": str(args.predictions.resolve()),
        "ground_truth_queries": len(ground_truth),
        "prediction_queries": len(predictions),
        "note": (
            "A query can occur in multiple duration bins when its ground-truth "
            "windows span multiple bins. Matching follows the official Top-1 rule."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Predictions: {args.predictions}")
    print("=" * 120)
    print(
        f"{'Bin':<10} {'N':>4} {'Mean IoU':>9} {'R1@.5':>8} {'R1@.7':>8} "
        f"{'GT med':>8} {'Pred med':>9} {'Width err':>10} {'Overwide':>9} {'>=2x':>7}"
    )
    for row in summaries:
        print(
            f"{row['duration_bin']:<10} {row['queries']:>4} "
            f"{row['mean_iou']:>9.3f} {row['r1_at_0.5_percent']:>7.1f}% "
            f"{row['r1_at_0.7_percent']:>7.1f}% {row['gt_width_median']:>7.1f}s "
            f"{row['pred_width_median']:>8.1f}s {row['width_error_median']:>+9.1f}s "
            f"{row['overwide_percent']:>8.1f}% {row['at_least_twice_gt_width_percent']:>6.1f}%"
        )
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {details_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
