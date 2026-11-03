import unittest

import torch
from torch import nn

from xiconwam.actor.action_expert import ActionExpert
from xiconwam.actor.hide import HidePolicy
from xiconwam.actor.idm import IDMPolicy
from xiconwam.actor.mot import MoTPolicy
from xiconwam.actor.mixture import StreamMixer
from xiconwam.vision.wan.wan_video_dit import WanVideoDiT


class _LatentCodec(nn.Module):
    temporal_downsample_factor = 4

    def encode(self, video, device=None):
        del device
        batch, _, frames, height, width = video.shape
        latent_frames = (frames - 1) // self.temporal_downsample_factor + 1
        return torch.zeros(
            batch,
            16,
            latent_frames,
            height // 8,
            width // 8,
            device=video.device,
            dtype=video.dtype,
        )


def _tiny(policy_cls, **kwargs):
    video = WanVideoDiT(
        hidden_dim=32,
        in_dim=16,
        ffn_dim=64,
        out_dim=16,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        attn_head_dim=8,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        attention_backend="sdpa",
    )
    action = ActionExpert(
        hidden_dim=32,
        action_dim=14,
        ffn_dim=64,
        text_dim=32,
        freq_dim=16,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=8,
        num_layers=1,
        projector_hidden_dim=16,
        attention_backend="sdpa",
    )
    return policy_cls(
        video_expert=video,
        action_expert=action,
        streams=StreamMixer({"video": video, "action": action}),
        vae=_LatentCodec(),
        text_dim=32,
        state_dim=14,
        state_position="sequence",
        projector_hidden_dim=16,
        device="cpu",
        torch_dtype=torch.float32,
        **kwargs,
    )


def _sample():
    return {
        "video": torch.randn(1, 3, 5, 32, 32),
        "action": torch.randn(1, 4, 14),
        "proprio": torch.randn(1, 4, 14),
        "context": torch.randn(1, 3, 32),
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
    }


class StepTest(unittest.TestCase):
    def test_coupled_backward(self):
        torch.manual_seed(0)
        loss, metrics = _tiny(MoTPolicy)(_sample())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_action", metrics)
        loss.backward()

    def test_inverse_backward(self):
        torch.manual_seed(0)
        loss, metrics = _tiny(IDMPolicy, video_cond_noise_prob=0.0)(_sample())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_action", metrics)
        loss.backward()

    def test_hide_backward_does_not_train_video_from_action(self):
        torch.manual_seed(0)
        video = _tiny(MoTPolicy).video_expert
        policy = HidePolicy(
            video_dit=video,
            action_dit=ActionExpert(
                hidden_dim=32,
                action_dim=14,
                ffn_dim=64,
                text_dim=32,
                freq_dim=16,
                eps=1e-6,
                num_heads=4,
                attn_head_dim=8,
                num_layers=1,
                projector_hidden_dim=16,
                attention_backend="sdpa",
            ),
            vae=_LatentCodec(),
            state_dim=14,
            state_position="sequence",
            text_dim=32,
            video_hidden_layer=0,
            feature_timestep=1.0,
            loss_lambda_video=0.0,
            device="cpu",
            torch_dtype=torch.float32,
        )
        loss, metrics = policy(_sample())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_action", metrics)
        loss.backward()
        video_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in video.parameters()
            if parameter.grad is not None
        )
        action_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in policy.action_dit.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(video_grad, 0.0)
        self.assertGreater(action_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
