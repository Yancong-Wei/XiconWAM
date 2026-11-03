import unittest

import numpy as np

from eval.robodojo_io import action_vector_to_xpolicylab, pack_joint_state
from xiconwam.data.lerobot.robot_video_dataset import trained_window


class EvalIoTest(unittest.TestCase):
    def test_joint_vector_roundtrip(self):
        vector = np.arange(14, dtype=np.float32)
        packed = action_vector_to_xpolicylab(vector)
        restored = pack_joint_state({"state": packed})
        np.testing.assert_allclose(restored, vector)

    def test_training_window_is_nine_frames_and_thirty_two_actions(self):
        video_frames, action_steps = trained_window(33, 4)
        self.assertEqual(video_frames, 9)
        self.assertEqual(action_steps, 32)
        self.assertEqual(list(range(0, 33, 4)), [0, 4, 8, 12, 16, 20, 24, 28, 32])


if __name__ == "__main__":
    unittest.main()
