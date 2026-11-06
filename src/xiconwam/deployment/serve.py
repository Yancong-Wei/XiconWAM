"""Run `python -m xiconwam.deployment.serve --help`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xicon_engine import PolicyInfo, serve

from .policy import HoldPolicy, XiconWAMPolicy


def load_policy(
    config_path, checkpoint, contract_path, *, device="cuda", inference_steps=20
):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from xiconwam.actor.helpers.inference import configure_model_execution
    from xiconwam.data.lerobot.utils.normalizer import load_dataset_stats_from_json
    from xiconwam.kit.config_resolvers import register_default_resolvers
    from xiconwam.kit.pytorch_utils import mixed_precision_to_dtype

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    register_default_resolvers()
    cfg = (
        OmegaConf.create(config_path)
        if OmegaConf.is_config(config_path)
        else OmegaConf.load(config_path)
    )
    OmegaConf.resolve(cfg)
    data = cfg.data.train
    # Fail closed for unsupported preprocessing instead of silently changing training semantics.
    if data.concat_multi_camera != "robotwin" or list(data.video_size) != [384, 320]:
        raise ValueError(
            "initial deployment supports robotwin video_size=[384,320] only"
        )
    horizon = int(data.num_frames) - 1
    ratio = int(data.action_video_freq_ratio)
    if ratio <= 0 or horizon <= 0 or horizon % ratio or (horizon // ratio) % 4:
        raise ValueError("invalid training video/action window")
    info = PolicyInfo.from_payload(json.loads(Path(contract_path).read_text()))
    if info.horizon != horizon:
        raise ValueError("contract horizon must equal training num_frames-1")
    processor = instantiate(data.processor)
    if processor.drop_high_level_prob != 1.0:
        raise ValueError(
            "deployment requires deterministic low-level instruction selection"
        )
    if list(data.camera_layout) != list(info.camera_keys):
        raise ValueError("camera_layout must match contract and processor camera order")
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(data.pretrained_norm_stats)
    )
    # Construct/validate adapter before loading multi-GB weights.
    policy = XiconWAMPolicy(
        None,
        processor,
        info,
        num_video_frames=horizon // ratio + 1,
        inference_steps=inference_steps,
        override_instruction=data.get("override_instruction"),
    )
    if not cfg.model.backbone.get("load_text_encoder", False):
        from functools import lru_cache

        from xiconwam.data.lerobot.text_embedding_cache import (
            load_text_embedding_cache,
            prompt_hash,
            text_embedding_cache_filename,
        )

        dtype = mixed_precision_to_dtype(cfg.mixed_precision)
        cache_dir = Path(data.text_embedding_cache_dir)

        @lru_cache(maxsize=16)
        def text_context(prompt):
            filename = text_embedding_cache_filename(
                prompt, int(data.context_len), str(data.text_encoder_id), dtype=dtype
            )
            return load_text_embedding_cache(
                cache_dir / filename,
                int(data.context_len),
                str(data.text_encoder_id),
                prompt_hash(prompt),
                expected_dtype=dtype,
            )

        policy.text_context = text_context
    model = instantiate(
        cfg.model,
        model_dtype=mixed_precision_to_dtype(cfg.mixed_precision),
        device=device,
    )
    model.load_checkpoint(str(checkpoint))
    model = configure_model_execution(
        model,
        vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
        inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse", True),
    )
    model.eval()
    policy.model = model
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="XiconWAM model-owned Xicon Engine server"
    )
    parser.add_argument(
        "--contract",
        help="Reviewed PolicyInfo JSON; exact robot joint/camera layout",
    )
    parser.add_argument("--config", help="Resolved training run config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--bundle", help="Verify and load a portable deployment bundle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--hold-fixture",
        action="store_true",
        help="Transport test; does NOT load a model",
    )
    args = parser.parse_args(argv)
    if args.inference_steps <= 0:
        parser.error("--inference-steps must be positive")
    if args.bundle:
        if args.config or args.checkpoint or args.contract or args.hold_fixture:
            parser.error(
                "--bundle cannot be combined with --config/--checkpoint/--contract/--hold-fixture"
            )
        from .bundle import load_bundle_config

        cfg, checkpoint, contract = load_bundle_config(args.bundle)
        policy = load_policy(
            cfg,
            checkpoint,
            contract,
            device=args.device,
            inference_steps=args.inference_steps,
        )
    elif args.hold_fixture:
        if args.config or args.checkpoint:
            parser.error("--hold-fixture cannot be combined with model inputs")
        if not args.contract:
            parser.error("--hold-fixture requires --contract")
        policy = HoldPolicy(
            PolicyInfo.from_payload(json.loads(Path(args.contract).read_text()))
        )
    else:
        if not args.config or not args.checkpoint or not args.contract:
            parser.error("model serving requires --config, --checkpoint and --contract")
        policy = load_policy(
            args.config,
            args.checkpoint,
            args.contract,
            device=args.device,
            inference_steps=args.inference_steps,
        )
    serve(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
