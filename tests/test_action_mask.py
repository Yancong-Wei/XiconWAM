import unittest

import torch

from xiconwam.actor.helpers.action_mask import masked_action_mse_per_sample


class ActionMaskTest(unittest.TestCase):
    def test_dim_pad_excluded_from_loss(self):
        pred = torch.zeros(2, 4, 8)
        target = torch.zeros(2, 4, 8)
        pred[..., 0] = 1.0
        target[..., 0] = 0.0
        dim_pad = torch.ones(2, 8, dtype=torch.bool)
        dim_pad[:, 0] = False
        loss = masked_action_mse_per_sample(
            pred, target, action_is_pad=None, action_dim_is_pad=dim_pad
        )
        self.assertTrue(torch.allclose(loss, torch.ones(2)))

        dim_pad_all = torch.ones(2, 8, dtype=torch.bool)
        dim_pad_all[:, 1:] = False
        loss_pad0 = masked_action_mse_per_sample(
            pred, target, action_dim_is_pad=dim_pad_all
        )
        self.assertTrue(torch.allclose(loss_pad0, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
