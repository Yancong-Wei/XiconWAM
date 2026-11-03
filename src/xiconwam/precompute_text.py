"""Cache frozen text-encoder outputs for the closed RoboDojo prompt set.

Training reads these files and does not run the text encoder.
"""

import argparse
import os
import uuid
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from xiconwam.data.lerobot.lerobot.datasets.utils import load_info
from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT
from xiconwam.data.lerobot.text_embedding_cache import (
    build_text_embedding_payload,
    prompt_hash,
    text_embedding_cache_filename,
)
from xiconwam.kit.config_resolvers import register_default_resolvers
from xiconwam.kit.logging_config import get_logger, setup_logging
from xiconwam.kit.pytorch_utils import mixed_precision_to_dtype
from xiconwam.vision.registry import load_text_encoder, normalize_spec

register_default_resolvers()
logger = get_logger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compose_task(config_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(project_root() / "configs"), version_base="1.3"
    ):
        return compose(
            config_name=config_name,
            overrides=["output_dir=./runs/text", *overrides],
        )


def dataset_text_settings(cfg):
    data = cfg.data.train
    cache_dir = data.get("text_embedding_cache_dir")
    if cache_dir is None or not str(cache_dir).strip():
        raise ValueError("data.train.text_embedding_cache_dir is required.")
    context_len = data.get("context_len")
    if context_len is None:
        raise ValueError("data.train.context_len is required.")
    dirs = [str(path) for path in data.dataset_dirs]
    return dirs, Path(str(cache_dir)).expanduser(), int(context_len)


def read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for ds_dir in dataset_dirs:
        root = Path(ds_dir)
        version = str(load_info(root).get("codebase_version"))
        if version != "v3.0":
            raise ValueError(f"Unsupported dataset version {version!r} at {root}.")
        table = pq.ParquetFile(root / "meta/tasks.parquet")
        names = table.schema_arrow.names
        if "task" not in names:
            raise ValueError(f"Missing task column in {root / 'meta/tasks.parquet'}.")
        for batch in table.iter_batches(columns=["task"]):
            for task in batch.column(0).to_pylist():
                prompt = DEFAULT_PROMPT.format(task=str(task))
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    return prompts


def _distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def encode_prompts(cfg, prompts: list[str], *, overwrite: bool) -> None:
    setup_logging()
    is_distributed, rank, world_size, local_rank = _distributed()
    dataset_dirs, cache_dir, context_len = dataset_text_settings(cfg)
    del dataset_dirs
    backbone = normalize_spec(OmegaConf.to_container(cfg.model.backbone, resolve=True))
    encoder_id = str(backbone["text_encoder_id"])
    dtype = mixed_precision_to_dtype(cfg.mixed_precision)
    local = prompts[rank::world_size] if is_distributed else prompts
    pending = []
    skipped = 0
    for prompt in local:
        filename = text_embedding_cache_filename(
            prompt_hash(prompt), context_len, encoder_id, is_hash=True, dtype=dtype
        )
        if (cache_dir / filename).is_file() and not overwrite:
            skipped += 1
        else:
            pending.append(prompt)
    if not pending:
        logger.info("Text cache is complete under %s. skipped=%d", cache_dir, skipped)
        return
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    encoder = load_text_encoder(
        backbone, device=device, torch_dtype=dtype, context_len=context_len
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for start in tqdm(
        range(0, len(pending), encoder.batch_size),
        desc="text embeddings",
        disable=is_distributed and rank != 0,
    ):
        batch = pending[start : start + encoder.batch_size]
        with torch.no_grad():
            encoded = encoder.encode_batch(batch)
        for prompt, context, mask in zip(batch, encoded.context, encoded.mask):
            context = context.detach().to(device="cpu", dtype=dtype)
            mask = mask.detach().to(device="cpu", dtype=torch.bool)
            context[~mask] = 0
            digest = prompt_hash(prompt)
            payload = build_text_embedding_payload(
                context=context,
                mask=mask,
                context_len=context_len,
                encoder_id=encoder_id,
                prompt_digest=digest,
                dtype=dtype,
            )
            filename = text_embedding_cache_filename(
                digest, context_len, encoder_id, is_hash=True, dtype=dtype
            )
            _save(payload, cache_dir / filename)
    logger.info("Wrote text embeddings to %s.", cache_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Precompute frozen text embeddings for a task."
    )
    parser.add_argument("--config-name", default="task/robodojo_wan")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    cfg = compose_task(args.config_name, list(args.overrides))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Composed task config must be a DictConfig.")
    dirs, _, _ = dataset_text_settings(cfg)
    prompts = read_unique_prompts(dirs)
    if not prompts:
        raise ValueError("No task prompts were found.")
    logger.info("Encoding %d unique prompts.", len(prompts))
    encode_prompts(cfg, prompts, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
