import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from duration_sampling import (  # noqa: E402
    EpochWeightedSampler,
    build_target_matched_sampler,
    duration_bin_index,
)


def record(qid, width):
    return {
        "qid": str(qid),
        "duration": 30,
        "relevant_windows": [[0, width]],
    }


class DurationSamplingTest(unittest.TestCase):
    def test_duration_bins_are_half_open(self):
        edges = [5, 10, 20]
        self.assertEqual(duration_bin_index(4.999, edges), 0)
        self.assertEqual(duration_bin_index(5, edges), 1)
        self.assertEqual(duration_bin_index(10, edges), 2)
        self.assertEqual(duration_bin_index(20, edges), 3)

    def test_sampler_is_deterministic_per_epoch(self):
        first = EpochWeightedSampler([1, 2, 3], num_samples=20, seed=7)
        second = EpochWeightedSampler([1, 2, 3], num_samples=20, seed=7)
        first.set_epoch(11)
        second.set_epoch(11)
        self.assertEqual(list(first), list(second))
        second.set_epoch(12)
        self.assertNotEqual(list(first), list(second))

    def test_full_importance_ratio_matches_target_in_expectation(self):
        source = [record("short", 2), record("long-1", 12), record("long-2", 12)]
        target = [record("short-1", 2), record("short-2", 2), record("long", 12)]
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "target.jsonl"
            with target_path.open("w", encoding="utf-8") as handle:
                for item in target:
                    handle.write(json.dumps(item) + "\n")
            _, report = build_target_matched_sampler(
                source,
                target_path,
                bin_edges=[5],
                seed=2023,
                power=1.0,
            )
        expected = [row["expected_sample_probability"] for row in report["bins"]]
        observed_target = [row["target_probability"] for row in report["bins"]]
        self.assertEqual(expected, observed_target)


if __name__ == "__main__":
    unittest.main()
