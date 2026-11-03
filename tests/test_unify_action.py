import unittest

import numpy as np
import torch

from xiconwam.data.lerobot.transforms.action_state_merger import ConcatLeftAlign
from xiconwam.data.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)


class UnifyActionTest(unittest.TestCase):
    def test_width_and_robodojo_roundtrip(self):
        self.assertEqual(UNIFY_DIM, 48)
        dst = parse_unify_spec(["0-13"], unify_dim=UNIFY_DIM)
        self.assertEqual(list(dst), list(range(14)))
        raw = np.arange(14, dtype=np.float32)
        unified, dim_valid = map_to_unify(raw, dst, unify_dim=UNIFY_DIM)
        self.assertEqual(unified.shape, (48,))
        self.assertTrue(dim_valid[:14].all())
        self.assertFalse(dim_valid[14:].any())
        np.testing.assert_array_equal(unmap_from_unify(unified, dst), raw)

    def test_merger_scatters_and_pads(self):
        merger = ConcatLeftAlign(
            action_target_dim=UNIFY_DIM,
            state_target_dim=UNIFY_DIM,
            unify_action_map=["0-13"],
            unify_state_map=None,
        )
        merger.set_shape_meta(
            {
                "action": [{"key": "default", "shape": 14}],
                "state": [{"key": "default", "shape": 14}],
            }
        )
        batch = {
            "action": {"default": torch.arange(14, dtype=torch.float32).unsqueeze(0)},
            "state": {"default": torch.ones(1, 14)},
        }
        out = merger.forward(batch)
        self.assertEqual(tuple(out["action"].shape), (1, 48))
        self.assertEqual(int((~out["action_dim_is_pad"]).sum()), 14)
        self.assertTrue(out["action_dim_is_pad"][14:].all())
        back = merger.backward(
            {
                "action": out["action"].unsqueeze(0),
                "state": out["state"].unsqueeze(0),
            }
        )
        self.assertEqual(tuple(back["action"]["default"].shape), (1, 1, 14))


if __name__ == "__main__":
    unittest.main()
