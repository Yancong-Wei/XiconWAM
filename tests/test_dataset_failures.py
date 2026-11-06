import errno
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from xiconwam.data.lerobot.robot_video_dataset import EpisodeWindowDataset


def dataset(training=True):
    result = EpisodeWindowDataset.__new__(EpisodeWindowDataset)
    result.is_training_set = training
    result.max_read_retries = 2
    result.read_error_counts = Counter()
    result.lerobot_dataset = SimpleNamespace(dataset_dirs=["test-source"])
    return result


@pytest.mark.parametrize(
    "error", [ValueError("bad schema"), FileNotFoundError("missing cache")]
)
def test_configuration_and_missing_files_never_replace_samples(error):
    ds, calls = dataset(), []

    def get(index):
        calls.append(index)
        raise error

    ds._get = get
    with pytest.raises(RuntimeError, match="index=7.*test-source") as caught:
        ds[7]
    assert caught.value.__cause__ is error
    assert calls == [7]
    assert ds.read_error_counts[type(error).__name__] == 1


def test_only_transient_reads_retry_the_same_index(monkeypatch):
    ds, calls = dataset(), []
    monkeypatch.setattr(
        "xiconwam.data.lerobot.robot_video_dataset.time.sleep", lambda seconds: None
    )

    def get(index):
        calls.append(index)
        if len(calls) < 3:
            raise OSError(errno.EIO, "temporary failure")
        return {"original_index": index}

    ds._get = get
    assert ds[9] == {"original_index": 9}
    assert calls == [9, 9, 9]
    assert ds.read_error_counts["OSError"] == 2


def test_validation_does_not_retry_or_change_index():
    ds, calls = dataset(False), []

    def get(index):
        calls.append(index)
        raise OSError(errno.EIO, "failure")

    ds._get = get
    with pytest.raises(RuntimeError, match="training=False.*attempts=1"):
        ds[3]
    assert calls == [3]


def test_online_text_cannot_allocate_an_encoder_in_worker(monkeypatch):
    ds = dataset()
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="DataLoader workers"):
        ds._ensure_online_text_encoder()


def test_preflight_checks_instruction_set_before_sample_reads():
    ds = dataset()
    ds.lerobot_dataset.processor = SimpleNamespace(
        drop_high_level_prob=1.0, augment_instruction=lambda d: d["task"]
    )
    ds.lerobot_dataset.multi_dataset = SimpleNamespace(
        _datasets=[SimpleNamespace(meta=SimpleNamespace(tasks={0: "pick", 1: "place"}))]
    )
    ds.override_instruction = None
    ds.text_embedding_cache_dir = None
    ds.context_len, ds.text_encoder_id, ds.text_embedding_dtype = (
        128,
        "test",
        torch.float32,
    )
    ds.load_text_encoder_on_miss = False
    seen = []

    def load(prompt):
        seen.append(prompt)
        raise FileNotFoundError("precompute required")

    ds._get_cached_text_context = load
    with pytest.raises(FileNotFoundError, match="precompute required"):
        ds._preflight_text_cache()
    assert len(seen) == 1
    assert "robot" in seen[0]


def test_preflight_uses_override_verbatim():
    from xiconwam.data.lerobot.prompts import DEFAULT_PROMPT

    ds = dataset()
    ds.override_instruction = "抓取@pick"
    ds.lerobot_dataset.processor = SimpleNamespace(drop_high_level_prob=0.5)
    ds.text_embedding_cache_dir = None
    ds.context_len, ds.text_encoder_id, ds.text_embedding_dtype = (
        128,
        "test",
        torch.float32,
    )
    ds.load_text_encoder_on_miss = False
    seen = []
    ds._get_cached_text_context = seen.append
    ds._preflight_text_cache()
    assert seen == [DEFAULT_PROMPT.format(task=ds.override_instruction)]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("transient", [False, True])
def test_nested_lerobot_reader_preserves_requested_index(
    training, transient, monkeypatch
):
    from xiconwam.data.lerobot.base_lerobot_dataset import BaseLerobotDataset

    calls = []
    error = (
        OSError(errno.EIO, "temporary read") if transient else ValueError("bad schema")
    )

    class Storage:
        num_frames = 20

        def __getitem__(self, index):
            calls.append(index)
            raise error

    inner = BaseLerobotDataset.__new__(BaseLerobotDataset)
    inner.multi_dataset = Storage()
    outer = dataset(training)
    outer._get = inner.__getitem__
    monkeypatch.setattr(
        "xiconwam.data.lerobot.robot_video_dataset.time.sleep", lambda _: None
    )
    with pytest.raises(RuntimeError, match="index=7") as caught:
        outer[7]
    assert caught.value.__cause__ is error
    assert calls == [7] * (3 if training and transient else 1)
