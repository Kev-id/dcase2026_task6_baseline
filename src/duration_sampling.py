"""Duration-aware sampling utilities for controlled pretraining experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Sampler


def duration_bin_index(width: float, bin_edges: Sequence[float]) -> int:
    """Return the index of the half-open duration bin containing ``width``."""
    if width <= 0:
        raise ValueError(f"Moment width must be positive, got {width}")
    for index, edge in enumerate(bin_edges):
        if width < edge:
            return index
    return len(bin_edges)


def duration_bin_labels(bin_edges: Sequence[float]) -> list[str]:
    labels: list[str] = []
    lower = 0.0
    for edge in bin_edges:
        if lower == 0:
            labels.append(f"<{edge:g}s")
        else:
            labels.append(f"{lower:g}-<{edge:g}s")
        lower = edge
    labels.append(f">={lower:g}s")
    return labels


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def count_window_bins(records: Iterable[dict], bin_edges: Sequence[float]) -> Counter:
    counts: Counter = Counter()
    for item in records:
        windows = item.get("relevant_windows", [])
        if not windows:
            raise ValueError(f"Query {item.get('qid')} has no relevant window")
        for start, end in windows:
            counts[duration_bin_index(float(end) - float(start), bin_edges)] += 1
    return counts


class EpochWeightedSampler(Sampler[int]):
    """Weighted sampler whose draws are deterministic for a given epoch.

    The epoch-specific seed makes the sample sequence independent of how many
    earlier epochs ran in the current process. This is important when a cloud
    training job resumes after an automatic shutdown.
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        num_samples: int,
        seed: int,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or len(self.weights) == 0:
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not torch.isfinite(self.weights).all() or (self.weights <= 0).any():
            raise ValueError("all sampling weights must be finite and positive")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_target_matched_sampler(
    source_records: Sequence[dict],
    target_path: str | Path,
    bin_edges: Sequence[float],
    seed: int,
    power: float = 1.0,
) -> tuple[EpochWeightedSampler, dict]:
    """Create an importance sampler matching target window-duration bins.

    M1 uses one annotated moment per Clotho-Moment query. Enforcing that
    invariant keeps the probability calculation transparent: each query's
    weight is ``(p_target(bin) / p_source(bin)) ** power``. Target probabilities
    may be estimated from data with multiple windows per query, as in CASTELLA.
    ``power=1`` performs exact bin matching in expectation.
    """
    if not 0 < power <= 1:
        raise ValueError(f"power must be in (0, 1], got {power}")
    edges = [float(edge) for edge in bin_edges]
    if edges != sorted(edges) or len(set(edges)) != len(edges):
        raise ValueError(f"bin_edges must be strictly increasing, got {bin_edges}")

    source_bins: list[int] = []
    for item in source_records:
        windows = item.get("relevant_windows", [])
        if len(windows) != 1:
            raise ValueError(
                "Target-matched M1 sampling requires exactly one relevant window "
                f"per source query; query {item.get('qid')} has {len(windows)}"
            )
        start, end = windows[0]
        source_bins.append(duration_bin_index(float(end) - float(start), edges))

    target_records = load_jsonl(target_path)
    source_counts = Counter(source_bins)
    target_counts = count_window_bins(target_records, edges)
    n_bins = len(edges) + 1
    missing_source = [index for index in range(n_bins) if source_counts[index] == 0]
    missing_target = [index for index in range(n_bins) if target_counts[index] == 0]
    if missing_source or missing_target:
        raise ValueError(
            "Every duration bin must occur in both datasets; "
            f"missing_source={missing_source}, missing_target={missing_target}"
        )

    source_total = sum(source_counts.values())
    target_total = sum(target_counts.values())
    source_probs = [source_counts[index] / source_total for index in range(n_bins)]
    target_probs = [target_counts[index] / target_total for index in range(n_bins)]
    bin_weights = [
        (target_probs[index] / source_probs[index]) ** power
        for index in range(n_bins)
    ]
    sample_weights = [bin_weights[index] for index in source_bins]
    sampler = EpochWeightedSampler(sample_weights, len(source_records), seed)

    weighted_mass = [source_probs[index] * bin_weights[index] for index in range(n_bins)]
    mass_total = sum(weighted_mass)
    expected_probs = [mass / mass_total for mass in weighted_mass]
    labels = duration_bin_labels(edges)
    report = {
        "method": "target_duration_importance_sampling",
        "target_path": str(target_path),
        "bin_edges_seconds": edges,
        "power": power,
        "seed": int(seed),
        "num_samples_per_epoch": len(source_records),
        "source_query_count": len(source_records),
        "target_query_count": len(target_records),
        "bins": [
            {
                "label": labels[index],
                "source_count": source_counts[index],
                "source_probability": source_probs[index],
                "target_window_count": target_counts[index],
                "target_probability": target_probs[index],
                "importance_weight": bin_weights[index],
                "expected_sample_probability": expected_probs[index],
            }
            for index in range(n_bins)
        ],
    }
    return sampler, report
