"""Validate the realized duration distribution of the M1 sampler."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from duration_sampling import (  # noqa: E402
    build_target_matched_sampler,
    duration_bin_index,
    load_jsonl,
)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "data" / "clotho_moment_train_release.jsonl",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / "data" / "castella_train_release.jsonl",
    )
    parser.add_argument("--bin-edges", type=float, nargs="+", default=[5, 10, 20])
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis" / "results" / "m1_sampling_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_jsonl(args.source)
    sampler, report = build_target_matched_sampler(
        source,
        args.target,
        args.bin_edges,
        args.seed,
        args.power,
    )
    sampler.set_epoch(args.epoch)
    sampled_indices = list(sampler)
    realized_counts: Counter = Counter()
    for index in sampled_indices:
        start, end = source[index]["relevant_windows"][0]
        realized_counts[
            duration_bin_index(float(end) - float(start), args.bin_edges)
        ] += 1

    for index, row in enumerate(report["bins"]):
        row["realized_count"] = realized_counts[index]
        row["realized_probability"] = realized_counts[index] / len(sampled_indices)
    report["validated_epoch"] = args.epoch
    report["source_path"] = portable_path(args.source)
    report["target_path"] = portable_path(args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("M1 duration-sampling validation")
    print(f"samples={len(sampled_indices)}, seed={args.seed}, epoch={args.epoch}")
    for row in report["bins"]:
        print(
            f"{row['label']:>8}: source={row['source_probability']:.4%}, "
            f"target={row['target_probability']:.4%}, "
            f"expected={row['expected_sample_probability']:.4%}, "
            f"realized={row['realized_probability']:.4%}, "
            f"weight={row['importance_weight']:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
