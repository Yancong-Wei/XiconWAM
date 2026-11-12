"""Small synthetic checkpoints exercise conversion without allocating 5B parameters."""
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

TOOLS = Path(__file__).resolve().parents[1] / "scripts/integrations/openwam"
sys.path.insert(0, str(TOOLS))
convert = importlib.import_module("convert_checkpoint")
check = importlib.import_module("check_checkpoint")
from scripts.split_dit_checkpoint import export_video

SCHEMAS = {"dit": {"weight": (2, 2)}, "vae": {"model.weight": (2,)}, "text_encoder": {"weight": (3,)}}


def fixture(tmp_path):
    source = tmp_path / "openwam"
    source.mkdir()
    tokenizer = source / "tokenizer/google/umt5-xxl"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    (tokenizer / "tokenizer.json").write_text("{}")
    tensors = {
        "video_backbone.dit.weight": torch.ones(2, 2),
        "video_backbone.vae.model.weight": torch.full((2,), 2.0),
        "video_backbone.text_encoder.weight": torch.full((3,), 3.0),
        "action_backbone.weight": torch.tensor([4.0]),
    }
    path = source / "checkpoint.safetensors"
    save_file(tensors, path)
    return path, tensors


def test_convert_and_validate_complete_model(tmp_path):
    source, tensors = fixture(tmp_path)
    out = tmp_path / "model"
    convert.convert(source, out, expected_schemas=SCHEMAS)
    report = check.check_checkpoint(out, expected_schemas=SCHEMAS, finite=True)
    assert report["valid"] and all(report[k]["valid"] for k in SCHEMAS)
    torch.testing.assert_close(load_file(out / "diffusion_pytorch_model.safetensors")["weight"], tensors["video_backbone.dit.weight"])
    vae = torch.load(out / "Wan2.2_VAE.pth", weights_only=True)
    torch.testing.assert_close(vae["weight"], tensors["video_backbone.vae.model.weight"])
    text = torch.load(out / "models_t5_umt5-xxl-enc-bf16.pth", weights_only=True)
    torch.testing.assert_close(text["weight"], tensors["video_backbone.text_encoder.weight"])
    assert (out / "google/umt5-xxl/tokenizer.json").is_file()
    assert json.loads((out / "xiconwam_source.json").read_text())["source"] == "OpenWAM"
    (out / "Wan2.2_VAE.pth").unlink()
    assert not check.check_checkpoint(out, expected_schemas=SCHEMAS)["valid"]


def test_missing_or_mismatched_component_never_creates_output(tmp_path):
    source, _ = fixture(tmp_path)
    bad_schema = {**SCHEMAS, "text_encoder": {"weight": (5,)}}
    out = tmp_path / "bad-model"
    with pytest.raises(ValueError, match="shape_mismatches"):
        convert.convert(source, out, expected_schemas=bad_schema)
    assert not out.exists()


def test_export_trained_video_reuses_base_encoders(tmp_path):
    source, _ = fixture(tmp_path)
    base = tmp_path / "base"
    convert.convert(source, base, expected_schemas=SCHEMAS)
    trained = tmp_path / "trained.safetensors"
    save_file({"weight": torch.full((2, 2), 9.0)}, trained, metadata={"backbone_name": "wan22"})
    out = tmp_path / "trained-model"
    export_video({"name": "wan22"}, load_file(trained), {"backbone_name": "wan22"}, out, base_model=base, expected_schema=SCHEMAS["dit"])
    assert check.check_checkpoint(out, expected_schemas=SCHEMAS)["valid"]
    torch.testing.assert_close(load_file(out / "diffusion_pytorch_model.safetensors")["weight"], torch.full((2, 2), 9.0))
    assert (out / "Wan2.2_VAE.pth").read_bytes() == (base / "Wan2.2_VAE.pth").read_bytes()
    assert (out / "models_t5_umt5-xxl-enc-bf16.pth").read_bytes() == (base / "models_t5_umt5-xxl-enc-bf16.pth").read_bytes()
