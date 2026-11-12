import json
import os
import re
import shutil
from pathlib import Path
import time

from xiconwam.kit.fs import ensure_dir
from xiconwam.kit.logging_config import get_logger

logger = get_logger(__name__)


def _drop_file_cache(path: str) -> None:
    """Release a checkpoint file's page cache back to the memory cgroup."""
    if not path or not os.path.isfile(path):
        return
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        os.close(fd)


class CheckpointMixin:
    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path), optimizer=None
        )
        logger.warning(
            "Loaded model weights only; optimizer/scheduler/step were not restored under ZeRO2."
        )

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        suffix = getattr(model, "weights_suffix", ".pt")
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}{suffix}")
        temporary_path = os.path.join(self.weights_dir, f"{step_tag}.tmp{suffix}")
        started_at = time.perf_counter()
        logger.info("[ckpt] saving weights: %s", ckpt_path)
        if getattr(self, "dit_train_mode", "joint") == "action_only":
            model.save_action_checkpoint(temporary_path, step=self.global_step)
        else:
            model.save_checkpoint(temporary_path, optimizer=None, step=self.global_step)
        os.replace(temporary_path, ckpt_path)
        size_gib = os.path.getsize(ckpt_path) / (1024**3)
        logger.info(
            "[ckpt] saved weights: path=%s size=%.3fGiB elapsed=%.2fs",
            ckpt_path,
            size_gib,
            time.perf_counter() - started_at,
        )
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        temporary = state_file + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        os.replace(temporary, state_file)

    @staticmethod
    def _prune_step_artifacts(directory: str, max_to_keep: int, *, suffix: str = ""):
        pattern = re.compile(rf"^step_(\d+){re.escape(suffix)}$")
        artifacts = []
        for path in Path(directory).iterdir():
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            if suffix:
                if not path.is_file():
                    continue
            elif not path.is_dir() or path.is_symlink():
                continue
            artifacts.append((int(match.group(1)), path))

        artifacts.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        for _, path in artifacts[max_to_keep:]:
            if suffix:
                path.unlink()
            else:
                shutil.rmtree(path)
            logger.info("[ckpt] removed old artifact: %s", path)

    def _prune_saved_checkpoints(self):
        if self.checkpoint_save_limit is None:
            return
        model = self.accelerator.unwrap_model(self.model)
        self._prune_step_artifacts(
            self.weights_dir,
            self.checkpoint_save_limit,
            suffix=getattr(model, "weights_suffix", ".pt"),
        )
        self._prune_step_artifacts(
            self.state_dir,
            self.checkpoint_save_limit,
        )

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if self.accelerator.is_main_process:
            # A failed rewrite of this step must not leave an old completion marker.
            (Path(state_path) / "trainer_state.json").unlink(missing_ok=True)
        self.accelerator.wait_for_everyone()
        state_started_at = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info("[ckpt] saving distributed state: %s", state_path)
        restore_zero_save = self._stagger_zero_checkpoint_writes()
        try:
            self.accelerator.save_state(output_dir=state_path)
        finally:
            if restore_zero_save is not None:
                restore_zero_save()
        # Publish the completion marker after every rank finishes its state shard.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            logger.info(
                "[ckpt] saved distributed state: path=%s elapsed=%.2fs",
                state_path,
                time.perf_counter() - state_started_at,
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._prune_saved_checkpoints()
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def _stagger_zero_checkpoint_writes(self):
        """Write one ZeRO optimizer shard at a time, then drop its cache."""
        import torch.distributed as dist

        engine = self.model
        original = getattr(engine, "_save_zero_checkpoint", None)
        if (
            not callable(original)
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return None

        def staggered(save_path, tag):
            rank = dist.get_rank()
            for active in range(dist.get_world_size()):
                if rank == active:
                    original(save_path, tag)
                    _drop_file_cache(engine._get_zero_ckpt_name(save_path, tag))
                dist.barrier()

        engine._save_zero_checkpoint = staggered

        def restore():
            engine._save_zero_checkpoint = original

        return restore

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                # An exhausted restored iterator yields nothing, so Accelerate
                # does not advance its epoch. Resume directly at the next one.
                batches_per_epoch = (
                    self.dataset_size
                    + self.batch_size * self.accelerator.num_processes
                    - 1
                ) // (self.batch_size * self.accelerator.num_processes)
                completed_epochs, self.batch_in_epoch = divmod(
                    self.batch_in_epoch, batches_per_epoch
                )
                self.epoch += completed_epochs
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch
                    * self.batch_size
                    * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info(
            "Loaded accelerate training state from %s at step=%d",
            state_dir,
            self.global_step,
        )
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )
