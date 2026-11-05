import gc
import os
import unittest
from unittest import mock

from xiconwam.engine import runtime_tuning
from xiconwam.engine.runtime_tuning import (
    apply_deepspeed_runtime_settings,
    enable_expandable_segments,
    maybe_freeze_gc,
)


def _zero2():
    return {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
        }
    }


class RuntimeTuningTest(unittest.TestCase):
    def tearDown(self):
        runtime_tuning._logged_gc_freeze = False

    def test_zero2_overlaps_without_replacing_an_explicit_disable(self):
        with mock.patch.dict(os.environ, {"XICONWAM_FAST_RUNTIME": "1"}):
            config = _zero2()
            apply_deepspeed_runtime_settings(config)
            self.assertTrue(config["zero_optimization"]["overlap_comm"])

            disabled = _zero2()
            disabled["zero_optimization"]["overlap_comm"] = False
            apply_deepspeed_runtime_settings(disabled)
            self.assertFalse(disabled["zero_optimization"]["overlap_comm"])

    def test_offload_and_other_stages_keep_their_communication_path(self):
        with mock.patch.dict(os.environ, {"XICONWAM_FAST_RUNTIME": "1"}):
            offload = _zero2()
            offload["zero_optimization"]["offload_optimizer"]["device"] = "cpu"
            apply_deepspeed_runtime_settings(offload)
            self.assertNotIn("overlap_comm", offload["zero_optimization"])

            stage3 = {"zero_optimization": {"stage": 3, "overlap_comm": True}}
            apply_deepspeed_runtime_settings(stage3)
            self.assertTrue(stage3["zero_optimization"]["overlap_comm"])

            stage1 = {"zero_optimization": {"stage": 1}}
            apply_deepspeed_runtime_settings(stage1)
            self.assertNotIn("overlap_comm", stage1["zero_optimization"])

    def test_opt_out_leaves_config_and_allocator_unchanged(self):
        with mock.patch.dict(os.environ, {"XICONWAM_FAST_RUNTIME": "0"}, clear=False):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            config = _zero2()
            apply_deepspeed_runtime_settings(config)
            enable_expandable_segments()
            self.assertNotIn("overlap_comm", config["zero_optimization"])
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", os.environ)

    def test_allocator_appends_without_overriding_other_settings(self):
        with mock.patch.dict(
            os.environ,
            {
                "XICONWAM_FAST_RUNTIME": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            },
        ):
            enable_expandable_segments()
            self.assertEqual(
                os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                "max_split_size_mb:128,expandable_segments:True",
            )
            enable_expandable_segments()
            self.assertEqual(
                os.environ["PYTORCH_CUDA_ALLOC_CONF"].count("expandable_segments"),
                1,
            )

    def test_gc_freeze_runs_only_on_early_steps(self):
        with mock.patch.dict(os.environ, {"XICONWAM_FAST_RUNTIME": "1"}):
            with (
                mock.patch.object(gc, "collect") as collect,
                mock.patch.object(gc, "freeze") as freeze,
            ):
                for step in (0, 1, 2, 3, 4, 5):
                    maybe_freeze_gc(step)
                self.assertEqual(collect.call_count, 3)
                self.assertEqual(freeze.call_count, 3)

        runtime_tuning._logged_gc_freeze = False
        with mock.patch.dict(os.environ, {"XICONWAM_FAST_RUNTIME": "0"}):
            with (
                mock.patch.object(gc, "collect") as collect,
                mock.patch.object(gc, "freeze") as freeze,
            ):
                maybe_freeze_gc(1)
                collect.assert_not_called()
                freeze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
