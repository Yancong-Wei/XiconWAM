import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from xiconwam.data.lerobot.processors.window_packer import WindowPacker
from xiconwam.data.lerobot.transforms.action_state_merger import (
    DEFAULT_ACTION_DIM,
    ConcatLeftAlign,
)
from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.launch import validation_dataset_config
from xiconwam.prepare_action_expert import action_expert_spec
from xiconwam.precompute_text import dataset_text_settings

ROOT = Path(__file__).resolve().parents[1]
register_default_resolvers()


class ComposeTest(unittest.TestCase):
    def _compose(self, config_name="task/robodojo_a800", *overrides):
        with initialize_config_dir(
            config_dir=str(ROOT / "configs"), version_base="1.3"
        ):
            return compose(
                config_name=config_name,
                overrides=["output_dir=./runs/probe", *overrides],
            )

    def test_unspecified_action_width_is_48(self):
        self.assertEqual(DEFAULT_ACTION_DIM, 48)
        merger = ConcatLeftAlign()
        self.assertEqual(merger.action_target_dim, 48)
        self.assertEqual(merger.state_target_dim, 48)
        packer = WindowPacker(
            shape_meta={"images": [], "action": [], "state": []}, num_obs_steps=1
        )
        self.assertEqual(packer.action_output_dim, 48)
        self.assertEqual(packer.proprio_output_dim, 48)

    def test_task_holds_the_training_settings(self):
        cfg = self._compose()
        self.assertEqual(cfg.model._target_, "xiconwam.launch.build_model")
        self.assertEqual(cfg.model.architecture, "mot")
        self.assertEqual(cfg.model.backbone.name, "wan22")
        self.assertEqual(cfg.data.train.shape_meta.action[0].shape, 14)
        self.assertEqual(list(cfg.data.train.unify_action_map), ["0-13"])
        self.assertEqual(cfg.data.train.processor.action_output_dim, 48)
        self.assertEqual(cfg.data.train.processor.proprio_output_dim, 48)
        self.assertEqual(
            cfg.data.train.processor.action_state_merger.action_target_dim, 48
        )
        self.assertEqual(
            list(cfg.data.train.processor.action_state_merger.unify_action_map),
            ["0-13"],
        )
        self.assertEqual(cfg.model.state_dim, 48)
        self.assertIsNone(cfg.data.get("val"))
        self.assertEqual(cfg.batch_size * cfg.gradient_accumulation_steps, 20)
        self.assertEqual(cfg.max_steps, 200000)
        self.assertEqual(cfg.weight_decay, 1.0e-2)
        self.assertEqual(cfg.warmup_ratio, 0.05)
        self.assertEqual(cfg.seed, 42)
        self.assertFalse(cfg.model.lora.enable)
        self.assertFalse(cfg.model.backbone.use_gradient_checkpointing)
        self.assertEqual(int(cfg.zero), 2)
        self.assertFalse(cfg.zero_offload)
        expert = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
        self.assertEqual(expert["preset"], "standard")
        self.assertEqual(expert["action_dim"], 48)
        self.assertEqual(expert["hidden_dim"], 1024)
        self.assertEqual(expert["ffn_dim"], 4096)
        self.assertEqual(expert["num_heads"], 24)
        self.assertEqual(expert["num_layers"], 30)
        self.assertIn("ActionDiT_Wan22_5b.pt", cfg.model.action_dit_pretrained_path)
        self.assertEqual(cfg.model.loss.lambda_video, 1.0)

    def test_models_are_separate_configs(self):
        idm = self._compose("task/robodojo_wan", "model=idm")
        self.assertEqual(idm.model.architecture, "idm")
        self.assertEqual(idm.model.video_cond_noise_prob, 0.5)
        self.assertEqual(idm.max_steps, 30000)
        hide = self._compose("task/robodojo_cosmos", "model=hide")
        resolved = OmegaConf.to_container(hide.model, resolve=True)
        self.assertEqual(resolved["architecture"], "hide")
        self.assertEqual(resolved["extract_layer"], 16)
        self.assertEqual(resolved["feature_timestep"], 1.0)
        self.assertEqual(resolved["action_dit_config"]["hidden_dim"], 1024)
        self.assertEqual(resolved["action_dit_config"]["num_layers"], 28)
        self.assertEqual(resolved["action_dit_config"]["num_heads"], 16)
        self.assertTrue(resolved["action_dit_config"]["cross_attention"])
        self.assertIn(
            "ActionDiT_Cosmos25_2b.pt", resolved["action_dit_pretrained_path"]
        )
        self.assertEqual(resolved["group_learning_rate"]["video"], 1.0e-5)
        self.assertEqual(resolved["group_learning_rate"]["action"], 1.0e-4)
        self.assertNotIn("action_head", resolved)
        self.assertNotIn("detach_video_hidden", resolved)

    def test_flux_action_expert_matches_the_staged_tower(self):
        cfg = self._compose("task/robodojo_flux")
        expert = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
        self.assertEqual(expert["preset"], "staged")
        self.assertFalse(expert["cross_attention"])
        self.assertEqual(expert["num_layers_double"], 5)
        self.assertEqual(expert["num_layers_single"], 20)
        self.assertEqual(cfg.model.loss.lambda_video, 0.5)

    def test_val_reuses_train_and_cameras_are_not_repeated(self):
        cfg = self._compose("task/robodojo_wan")
        processor = instantiate(cfg.data.train.processor)
        self.assertEqual(processor.num_output_cameras, 3)
        self.assertEqual(
            [type(op).__name__ for op in processor.train_transforms["cam_high"]],
            ["ToTensor", "Resize", "ColorJitter"],
        )
        self.assertEqual(
            list(processor.train_transforms["cam_high"][1].size), [256, 320]
        )
        val = validation_dataset_config(cfg.data)
        resolved = OmegaConf.to_container(val, resolve=True)
        self.assertFalse(resolved["is_training_set"])
        self.assertIn("wan22", resolved["text_embedding_cache_dir"])

    def test_hide_uses_the_same_action_expert(self):
        mot = self._compose("task/robodojo_wan")
        backbone, action = action_expert_spec(mot)
        self.assertEqual(backbone["name"], "wan22")
        self.assertEqual(action["hidden_dim"], 1024)
        self.assertEqual(action["num_layers"], 30)
        dirs, cache_dir, context_len = dataset_text_settings(mot)
        self.assertTrue(dirs)
        self.assertEqual(context_len, 128)
        self.assertIn("wan22", str(cache_dir))
        hide = self._compose("task/robodojo_wan", "model=hide")
        hide_backbone, hide_action = action_expert_spec(hide)
        self.assertEqual(hide_backbone["name"], backbone["name"])
        self.assertEqual(hide_action["hidden_dim"], action["hidden_dim"])
        self.assertEqual(hide_action["num_layers"], action["num_layers"])
        self.assertEqual(
            hide.model.action_dit_pretrained_path, mot.model.action_dit_pretrained_path
        )

    def test_mixture_data_composes(self):
        cfg = self._compose("task/robodojo_wan", "data=mixture")
        OmegaConf.resolve(cfg)
        self.assertFalse(cfg.data.train._recursive_)
        self.assertEqual(
            cfg.data.train._target_, "xiconwam.data.mixture.build_mixture_dataset"
        )
        self.assertTrue(cfg.data.train.sources.robodojo.enabled)
        self.assertEqual(
            list(cfg.data.train.sources.robodojo.config.unify_action_map), ["0-13"]
        )
        self.assertEqual(
            cfg.data.train.sources.robodojo.config.processor.action_output_dim, 48
        )

    def test_layout(self):
        self.assertEqual(
            sorted(path.stem for path in (ROOT / "configs/model").glob("*.yaml")),
            ["hide", "idm", "mot"],
        )
        self.assertEqual(
            sorted(path.stem for path in (ROOT / "configs/data").glob("*.yaml")),
            ["mixture", "robodojo"],
        )
        self.assertFalse((ROOT / "configs/config.yaml").exists())
        self.assertFalse((ROOT / "configs/train").exists())
        self.assertTrue((ROOT / "configs/deepspeed/ds_zero2_config.json").is_file())
        self.assertFalse((ROOT / "scripts/train.py").exists())
        self.assertTrue((ROOT / "scripts/prepare_action_expert.sh").is_file())
        self.assertTrue((ROOT / "scripts/precompute_text_embeds.sh").is_file())
        self.assertFalse((ROOT / "scripts/prepare_action_expert.py").exists())
        self.assertFalse((ROOT / "scripts/precompute_text_embeds.py").exists())
        self.assertFalse((ROOT / "src/xiconwam/actor/dit4dit_head.py").exists())
        self.assertTrue((ROOT / "src/xiconwam/data/unify_action.py").is_file())
        self.assertTrue((ROOT / "src/xiconwam/data/mixture.py").is_file())


if __name__ == "__main__":
    unittest.main()
