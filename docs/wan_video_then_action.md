# 先训练 Wan2.2 Video DiT，再训练 MoT Action DiT

从 XiconWAM 根目录运行。训练使用 XiconWAM 环境。VPP2 教师推理可用独立环境，通过 `--python /path/to/vpp2/bin/python` 指定。

配置职责见[配置说明](configuration.md)。主要入口如下。

- `configs/model/video_dit/{wan22,cosmos25,flux2}.yaml` 管理视频骨干。
- `configs/model/action_dit/{wan22,cosmos25,flux2}.yaml` 管理对应 Action DiT 的结构及权重路径；与 `video_dit` 在 `model` 下并列，默认随 Video DiT 自动选择。
- `configs/model/{mot,idm,hide,video}.yaml` 定义组件连接、损失和 `training` 冻结策略。`video` 只选择视频组件。
- `configs/training/default.yaml` 管理共享训练、数据加载、验证和日志默认值。
- `configs/task/*.yaml` 选择模型和数据，并配置训练步数、批量大小以及 DeepSpeed 的 `zero`、`zero_offload`。

Wan/Cosmos 的 Action 使用 standard 结构，Flux 使用 staged 结构。视频骨干的张量结构统一写在自己的 `dit_config` 中。Action 的 heads、head dimension、
层数由对应的 `model.action_dit.dit_config` 定义，构建前会检查几何兼容性。
HIDE 使用带 cross-attention 的 Action，支持 Wan/Cosmos。Flux 的纯视觉任务预测当前帧与未来端点。

## 1. 准备或选择 Video DiT

```bash
export WAN=/absolute/path/to/Wan2.2-TI2V-5B
export OPENWAM=/absolute/path/to/checkpoint_step_154000.safetensors
export VIDEO_MODEL=/absolute/path/to/OpenWAM-Wan22
export OW=scripts/integrations/openwam

# 默认从 checkpoint 同目录的 tokenizer/google/umt5-xxl 读取 tokenizer；
# 若不在此处，可附加 --tokenizer-dir /path/to/umt5-xxl。
python "$OW/convert_checkpoint.py" "$OPENWAM" --output "$VIDEO_MODEL"
python "$OW/check_checkpoint.py" "$VIDEO_MODEL"
python "$OW/check_checkpoint.py" "$WAN"
```

OpenWAM-α 使用 [公开 Foundation checkpoint](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model)。
转换严格核对并导出 OpenWAM 的 Video DiT、VAE 和文本编码器，并复制其 tokenizer，形成完整 Wan2.2 模型目录；不会把 Action 权重放入模型。转换仅验证结构和数值格式，不保证纯视频推理与 OpenWAM 原始联合推理效果一致。

通过不同的 `--wan-dir` 对比原始 Wan 和 OpenWAM 模型，图像、提示和 seed 保持一致：

```bash
python "$OW/infer_video.py" --wan-dir "$VIDEO_MODEL" \
  --image /absolute/path/to/frame.png --prompt "Pick up the cup." \
  --seed 42 --output /absolute/path/to/openwam.mp4
```

## 2. 独立训练 Video DiT

每行写一个视频窗口。路径可相对于 manifest。真实窗口的有效采样帧率应与教师生成帧率接近。视频需要覆盖第 `start_frame + (frames - 1) * frame_stride` 帧。
如果同一 episode 被拆为多个文件，分割训练和验证时使用同一 `episode_id`。

```json
{"video":"clips/episode01.mp4","prompt":"Pick up the cup.","episode_id":"episode01","start_frame":0,"frame_stride":2}
```

```bash
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_from_yaml.sh configs/task/wan_video.yaml \
  data.train.manifest=/absolute/path/to/train.jsonl \
  "model.video_dit.model_id=$VIDEO_MODEL" max_steps=1000
```

改为 `model.video_dit.model_id=$WAN` 即从原始 Wan 初始化。训练产生的 `checkpoints/weights/step_*.safetensors` 只含 Video DiT；使用训练时对应的模型目录补齐 VAE、文本编码器和 tokenizer：

```bash
export TRAINED_VIDEO=/absolute/path/to/trained-video-model
python scripts/split_dit_checkpoint.py /absolute/path/to/run/checkpoints/weights/step_001000.safetensors \
  --backbone-config /absolute/path/to/run/config.yaml --base-model "$VIDEO_MODEL" \
  --video-model "$TRAINED_VIDEO"
python "$OW/check_checkpoint.py" "$TRAINED_VIDEO"
```

`--base-model` 必须是这次训练所使用的编码器来源；若训练从原始 Wan 开始，则改为 `$WAN`。导出会复制大文件并需要额外磁盘空间。

### VPP2 Stage-2 教师

```bash
bash scripts/integrations/setup.sh  # VPP2 教师生成需要 upstream 仓库
export VPP=third_party/VPP2/xiconwam_tools
export TEACHER=/absolute/path/to/vpp2-stage2
export WORK=/absolute/path/to/distillation
python "$VPP/download_teacher.py" --output "$TEACHER"
python "$VPP/distill.py" prepare --real-manifest /absolute/path/to/train.jsonl \
  --work-dir "$WORK" --limit 8
python "$VPP/distill.py" generate --teacher-dir "$TEACHER" --work-dir "$WORK" --dry-run
python "$VPP/distill.py" generate --teacher-dir "$TEACHER" --work-dir "$WORK"
python "$VPP/distill.py" collect --work-dir "$WORK"
GPU_IDS=0,1,2,3,4,5,6,7 python "$VPP/distill.py" train --wan-dir "$WAN" \
  --work-dir "$WORK" --val-manifest /absolute/path/to/val.jsonl max_steps=250
```

先用 8 个窗口检查显存、吞吐和质量。正式数据使用新的 `WORK` 并去掉 `--limit`。
教师使用 [VPP2 公开 Stage-2 17 帧权重](https://huggingface.co/Haodong082399/VPP2/tree/main/checkpoints_video)
及共享 VAE、CLIP、UMT5。教师采样 30 步，CFG 4，shift 3，师生共用 256×416 的窗口尺寸。
学生数据中每个真实窗口对应一个教师 RGB 窗口，二者各占一半。教师窗口的第一帧使用原始条件图，
全片使用学生的 Wan2.2 VAE 编码。训练结果是可独立加载的 5B Video DiT，结构与上一节相同。

## 3. 冻结视频，训练 Action DiT

已有的联合训练权重可以用一个命令同时拆出视频模型和 Action DiT：

```bash
python scripts/split_dit_checkpoint.py /absolute/path/to/run/checkpoints/weights/step_001000.pt \
  --backbone-config /absolute/path/to/run/config.yaml \
  --base-model /absolute/path/to/the-base-Wan-model-used-for-training \
  --video-model /absolute/path/to/split-video-model \
  --action-output /absolute/path/to/action.pt
```

原联合权重通常没有冻结的 VAE、文本编码器和 tokenizer，因此 `--base-model` 必须指向该次训练实际使用的模型目录；不可自动从 DiT 权重推知。只需要 Action 时可省略 `--backbone-config`、`--base-model`、`--video-model`。输入应为模型权重文件，而非 DeepSpeed 续训状态目录或 LoRA-only 权重；续训直接使用 `resume`。

RoboDojo 训练默认读取预计算文本嵌入。更换到 OpenWAM 编码器后，请使用独立的编码器标识重新生成缓存，避免误用原始 Wan 的旧缓存。以下 `model_id` 和 `text_encoder_id` 必须在缓存生成及训练时保持一致：

```bash
bash scripts/precompute_text_embeds.sh configs/task/robodojo_wan.yaml \
  "model.video_dit.model_id=$TRAINED_VIDEO" model.video_dit.text_encoder_id=openwam-umt5
```

```bash
export ACTION=/absolute/path/to/action.pt
PYTHONPATH=src python scripts/split_dit_checkpoint.py /absolute/path/to/joint.pt \
  --action-output "$ACTION"

GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml \
  model.training.mode=action_only \
  "model.video_dit.model_id=$TRAINED_VIDEO" model.video_dit.text_encoder_id=openwam-umt5 \
  "model.action_dit.checkpoint=$ACTION" batch_size=1 max_steps=250
```

此阶段 MoT 继续从冻结的视频流获取条件。优化器只更新 Action DiT 和状态编码器。
`model.action_dit.checkpoint` 可独立替换；省略时沿用原 Action 初始化，另设
`model.action_dit.pretrained_path=null` 可从随机 Action 权重起步。
Action-only 的 `checkpoints/weights/step_*.pt` 保存完整 Action DiT 和状态编码器。
在 MoT 配置中指定完整视频模型的 `model.video_dit.model_id` 和
`model.action_dit.checkpoint` 即可组装。完整训练续跑只需提供保存状态的路径，配置会自动从原 run 恢复：

```bash
bash scripts/train_from_yaml.sh resume=/absolute/path/to/run
# 指定某一步也可以
bash scripts/train_from_yaml.sh resume=/absolute/path/to/run/checkpoints/state/step_003000
```

完整训练状态会恢复两个 DiT、优化器、学习率调度器、RNG 与 dataloader 位置。
run 目录形式自动选择最高步数的完整状态，输出写入新的 run。
初始分开加载的两个 checkpoint 在完整续训时由训练状态接替。`resume=权重文件` 只恢复模型权重。

联合训练下也可设置 `model.training.freeze_video=true` 或 `model.training.freeze_action=true`，
按原任务配置继续计算联合目标。拆分脚本接受完整 MoT/IDM/HIDE 权重。


## 4. 使用同一配置进行独立视频推理

```bash
PYTHONPATH=src python deployment/infer_video.py \
  --config-name task/wan_video "model.video_dit.model_id=$TRAINED_VIDEO" \
  --image /absolute/path/to/frame.png --prompt "Pick up the cup." \
  --output /absolute/path/to/prediction.mp4
```

也可用 `--run-config /absolute/path/to/run/config.yaml` 读取已保存的结构。如果该配置用于完整续训而设置了 `from_scratch=true`，同时设置 `model.video_dit.from_scratch=false` 和导出目录的 `model.video_dit.model_id`。
通过 `model/video_dit=cosmos25` 或 `=flux2` 更换视觉骨干。
该推理入口会加载选中骨干自己的文本编码器和 VAE。Wan/Cosmos 默认输出 17 帧，Flux 默认输出两个端点。
`deployment/model_server.py` 与部署服务均读取同一份配置中的 `model.video_dit` 和 `model.action_dit`。

### Cosmos / Flux 的训练权重

同一个脚本也可以处理这两种骨干的联合权重或独立 Video 权重：

```bash
python scripts/split_dit_checkpoint.py /path/to/weights.pt \
  --backbone-config /path/to/run/config.yaml --base-model /path/to/original-cosmos \
  --video-model /path/to/trained-cosmos --action-output /path/to/action.pt
# model.video_dit.model_id=/path/to/trained-cosmos

python scripts/split_dit_checkpoint.py /path/to/weights.pt \
  --backbone-config /path/to/run/config.yaml \
  --video-model /path/to/trained-flux.safetensors --action-output /path/to/action.pt
# model.video_dit.model_path=/path/to/trained-flux.safetensors
```

Cosmos 输出包含 Video DiT 和原模型的 `tokenizer.pth`，文本编码器仍使用配置中的 `reason_model_id`。Flux 输出 Video DiT 文件，VAE 和文本编码器仍使用 `ae_model_path` 与 `qwen3_model_spec`。如果输入的是独立 Video 权重，则省略 `--action-output`。`--backbone-config` 也可以使用与训练结构完全一致的 `configs/model/video_dit/*.yaml`。

### 在服务器提取需要哪些文件

提取 XiconWAM 自己的联合模型，只需要本项目当前版本的 `scripts/`、`src/`、相应配置和 Python 环境，以及：

- `run/checkpoints/weights/step_*.pt`：联合模型权重；不是 `checkpoints/state/step_*` 续训目录。
- 同一个 run 的 `config.yaml`：记录当时的视频结构。
- 视频训练时使用的基础模型目录：为 Wan 导出补齐 VAE、文本编码器和 tokenizer；只提取 Action 时不需要。

单独提取不需要数据集、8 张 GPU、DeepSpeed 多卡启动，也不需要 `third_party` 仓库；命令运行在 CPU 上，但读写大权重需要足够内存和磁盘空间。结果目录需要是不存在的新路径。`--base-model` 与权重对应关系需要自行核实。

如果转换 OpenWAM，直接运行本仓库 `scripts/integrations/openwam/` 下的脚本即可，也不需要克隆 OpenWAM 代码仓库；但必须准备 OpenWAM 的 checkpoint 和匹配 tokenizer。VPP2 生成教师视频是另一条流程，需要 VPP2 仓库和相应模型。

本地未提交的修改不会出现在服务器的普通 `git clone` 中。先将当前版本同步到服务器独立目录，再在已有训练环境运行；若原训练仍在进行，不要覆盖它正在使用的工作目录。部署时确认服务器的代码版本包含本次新增脚本及配置。
