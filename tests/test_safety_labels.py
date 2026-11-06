"""Offline safety-label semantics; no torch or simulator dependency."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "safety_labels", Path(__file__).resolve().parents[1] / "scripts/research/safety_labels.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def episode(name="a", events=None, complete=True, success=True):
    return dict(episode_id=name, task="insert", complete=complete,
                task_success=success, events=events if events is not None else [[], []])


class SafetyLabelsTest(unittest.TestCase):
    def test_censoring_and_future_alignment(self):
        result = module.analyze([episode(events=[[], ["collision"], []])], [1, 2, 4])
        labels = result["labels"]
        self.assertEqual(labels[0]["future_violation"], {"1": 0, "2": 1, "4": 1})
        self.assertEqual(labels[2]["future_violation"], {"1": 0, "2": None, "4": None})

    def test_success_is_not_safe_success(self):
        result = module.analyze([episode(events=[["collision"]]), episode("b")], [1])
        metrics = result["per_task"]["insert"]
        self.assertEqual(metrics["success_rate"], 1)
        self.assertEqual(metrics["safe_success_rate"], 0.5)
        self.assertEqual(metrics["violation_episode_rate"], 0.5)

    def test_no_cross_episode_future(self):
        result = module.analyze([episode(events=[[]]), episode("b", [["collision"]])], [2])
        self.assertIsNone(result["labels"][0]["future_violation"]["2"])

    def test_incomplete_is_not_success_or_safe_negative(self):
        result = module.analyze([episode(complete=False, success=None)], [4])
        self.assertIsNone(result["per_task"]["insert"]["success_rate"])
        self.assertEqual(result["per_task"]["insert"]["completion_coverage"], 0)
        self.assertTrue(all(r["future_violation"]["4"] is None for r in result["labels"]))

    def test_reject_invalid_contract(self):
        for episodes, horizons in [([], [1]), ([episode()], [0]),
                                   ([episode(), episode()], [1]),
                                   ([episode(events=[None])], [1]),
                                   ([episode(complete=False)], [1])]:
            with self.subTest(episodes=episodes, horizons=horizons):
                with self.assertRaises(ValueError):
                    module.analyze(episodes, horizons)


if __name__ == "__main__":
    unittest.main()
