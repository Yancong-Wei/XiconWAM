"""Generate Stage-2 pseudo-videos and train a Wan2.2-TI2V-5B Video DiT on RGB."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VPP2 = ROOT / "third_party/VPP2"
sys.path.insert(0, str(ROOT / "src"))

from download_teacher import STAGE2, WAN_DIR

from xiconwam.data.video_manifest import load_manifest, read_frames, resize_image


def write_manifest(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args):
    rows = load_manifest(args.real_manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    for row in rows:
        if not row["prompt"].strip() or any(
            token in row["prompt"] for token in ("@@", "\r", "\n")
        ):
            raise ValueError("VPP2 prompts must be nonempty single lines without @@")
        if row.get("condition_image") or row.get("source") == "vpp2_stage2":
            raise ValueError("Prepare the teacher from real RGB clips")
    work = args.work_dir.resolve()
    if work.exists():
        raise FileExistsError(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work.parent, prefix=".vpp2-prep-") as scratch:
        prepared = Path(scratch) / "prepared"
        (prepared / "images").mkdir(parents=True)
        lines = []
        for index, row in enumerate(rows, 1):
            frames = read_frames(
                row["video"],
                start=int(row.get("start_frame", 0)),
                stride=int(row.get("frame_stride", 1)),
                count=17,
            )
            image = Path("images") / f"{index:06d}.png"
            resize_image(frames[0], 416, 256).save(prepared / image)
            lines.append(f"{row['prompt'].strip()}@@{work / image}\n")
        write_manifest(prepared / "real.jsonl", rows)
        (prepared / "teacher_inputs.txt").write_text("".join(lines), encoding="utf-8")
        prepared.rename(work)


def generate(args):
    work = args.work_dir.resolve()
    assets = args.teacher_dir.resolve()
    command = [
        "bash",
        str(VPP2 / "scripts/infer_video.sh"),
        "--checkpoint",
        str(assets / STAGE2),
        "--wan-model-dir",
        str(assets / WAN_DIR),
        "--prompt-file",
        str(work / "teacher_inputs.txt"),
        "--output-dir",
        str(work / "teacher"),
        "--num-frames",
        "17",
        "--sampling-steps",
        "30",
        "--guidance-scale",
        "4",
        "--sigma-shift",
        "3",
        "--height",
        "256",
        "--width",
        "416",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, env={**os.environ, "PYTHON_BIN": args.python}, check=True)


def collect(args):
    work = args.work_dir.resolve()
    real = load_manifest(work / "real.jsonl")
    teacher = work / "teacher"
    run = json.loads((teacher / "run.json").read_text())
    if any(
        run["settings"].get(key) != value
        for key, value in (
            ("num_frames", 17),
            ("height", 256),
            ("width", 416),
            ("caption_prompt", False),
        )
    ):
        raise ValueError("Expected raw 17-frame VPP2 outputs at 256×416")
    generated = run["videos"]
    if len(real) != len(generated):
        raise ValueError("Teacher run does not cover every prepared clip")
    mixed = []
    for index, (row, entry) in enumerate(zip(real, generated, strict=True), 1):
        image = work / "images" / f"{index:06d}.png"
        output = f"{index:06d}.mp4"
        if (Path(entry["input"]).resolve(), entry["prompt"], entry["output"]) != (
            image,
            row["prompt"].strip(),
            output,
        ):
            raise ValueError(f"Teacher output does not match prepared clip {index}")
        video = teacher / output
        frames = read_frames(video, start=0, stride=1, count=17)
        if any(frame.size != (416, 256) for frame in frames):
            raise ValueError(f"Unexpected frame dimensions in teacher output {video}")
        synthetic = {
            "video": str(video),
            "prompt": row["prompt"],
            "condition_image": str(image),
            "source": "vpp2_stage2",
            "real_video": row["video"],
            "teacher_seed": entry["seed"],
        }
        if "episode_id" in row:
            synthetic["episode_id"] = row["episode_id"]
        mixed.extend([row, synthetic])
    write_manifest(work / "student.jsonl", mixed)
    print(
        f"Prepared {len(real)} real and {len(real)} teacher clips: {work / 'student.jsonl'}"
    )


def train(args, overrides):
    work = args.work_dir.resolve()
    manifest = work / "student.jsonl"
    rows = load_manifest(manifest)
    wan = args.wan_dir.resolve()
    command = [
        "bash",
        str(ROOT / "scripts/train_from_yaml.sh"),
        str(ROOT / "configs/task/wan_video.yaml"),
        f"data.train.manifest={manifest}",
        f"model.video_dit.model_id={wan}",
    ]
    if args.val_manifest is not None:
        val_path = args.val_manifest.resolve()
        val = load_manifest(val_path)
        training_videos = {row.get("real_video", row["video"]) for row in rows}
        training_episodes = {row["episode_id"] for row in rows if "episode_id" in row}
        for row in val:
            if row.get("condition_image") or row.get("source") == "vpp2_stage2":
                raise ValueError("Validation must use real clips")
            if (
                row["video"] in training_videos
                or row.get("episode_id") in training_episodes
            ):
                raise ValueError("Training and validation share a video or episode")
        command.extend(
            [
                "data.val={_target_:xiconwam.data.video_manifest.VideoManifestDataset}",
                f"+data.val.manifest={val_path}",
                "eval_every=250",
            ]
        )
    subprocess.run([*command, *overrides], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser(
        "prepare", help="Extract condition images and make VPP2 prompts"
    )
    prep.add_argument("--real-manifest", type=Path, required=True)
    prep.add_argument("--limit", type=int)
    teacher = subparsers.add_parser("generate", help="Run the upstream VPP2 teacher")
    teacher.add_argument("--teacher-dir", type=Path, required=True)
    teacher.add_argument("--device", default="cuda:0")
    teacher.add_argument("--python", default=sys.executable)
    teacher.add_argument("--seed", type=int, default=3402)
    teacher.add_argument("--dry-run", action="store_true")
    collection = subparsers.add_parser(
        "collect", help="Check videos and create a 50/50 student manifest"
    )
    student = subparsers.add_parser(
        "train", help="Train the 5B student with XiconWAM; accepts Hydra overrides"
    )
    student.add_argument("--wan-dir", type=Path, required=True)
    student.add_argument("--val-manifest", type=Path)
    for command in (prep, teacher, collection, student):
        command.add_argument("--work-dir", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    if args.command == "train":
        train(args, overrides)
    else:
        if overrides:
            parser.error(f"Unrecognized arguments: {' '.join(overrides)}")
        {"prepare": prepare, "generate": generate, "collect": collect}[args.command](
            args
        )


if __name__ == "__main__":
    main()
