# 配置分层与参数检查

当前入口是 `configs/task/*.yaml`。配置先组合为一份 Hydra 配置，再写入 run 的 `config.yaml`。模型结构、数据契约和训练调度分别维护。

## 模型：视频、动作和连接方式

- `configs/model/video_dit/{wan22,cosmos25,flux2}.yaml` → `model.video_dit`：视频模型来源、结构、文本／VAE 依赖、视频 scheduler 和视频 LoRA。
- `configs/model/action_dit/{wan22,cosmos25,flux2}.yaml` → `model.action_dit`：动作模型结构、初始化、训练后权重和 MoT / IDM 动作 scheduler。默认随 Video DiT 自动选择，也可显式选择，但必须满足几何兼容性。
- `configs/model/{mot,idm,hide,video}.yaml` → 外层 `model`：组件如何连接、状态条件、损失和更新策略。`video` 不创建 Action DiT。

`video_dit` 是视频组件入口，仍包含该模型配套的 VAE 与文本编码器来源。它们在训练中冻结；把字段改名并不意味着只加载一个 DiT 文件。

Video DiT 的 `dit_config` 是结构的主要定义处。外层 `num_heads`、`num_layers`、`hidden_dim`、`text_dim` 等是给配对和数据缓存使用的元信息；能从结构推导的值使用插值。修改结构时编辑 `dit_config`，不要单独改元信息。Action 的 heads、head width、层数和文本维度需满足当前连接协议；Action 自身的 hidden width 不必等于 Video hidden width。

`from_scratch` 控制是否加载 DiT 初始权重，不会免除 VAE／文本资源依赖。Action 的 `checkpoint` 优先于 `pretrained_path`；`from_scratch=true` 与非空 Action checkpoint 冲突。完整续训由保存的训练状态接管两者，不能借续训覆盖模型结构。

`model.state_encoder_hidden_dim` 是状态编码器 MLP 的中间宽度；Action `dit_config.projector_hidden_dim` 属于动作模型自己的投影，两者不是同一个参数。状态／动作维度默认 48，RoboDojo 原始 joint14 映射到槽位 0–13。维度一致不能证明动作语义一致。

`model.training` 管理 `joint` / `action_only` / `video_only` 和冻结策略。Action-only 更新 Action DiT 与状态编码器；VAE、文本编码器不参与更新。`model.loss` 是联合策略的两路损失权重；纯视频模型使用自己的视频目标。

### 架构专属设置

IDM 的 `video_cond_noise_prob=0.5` 控制训练时视频条件加噪概率，允许范围为 `[0,1]`。

HIDE 专属参数留在 `model/hide.yaml`：

- `feature_layer`：取视频特征的零基 block 下标。`auto` 保留 Wan 的 17、Cosmos 的 16；缩小或更换结构时应显式指定并检查范围。
- `feature_timestep=1.0`：读取特征时的视频噪声位置，1 表示纯噪声端，允许 `[0,1]`。
- `repeated_diffusion_steps=1`：对同一样本重复计算动作损失，每次使用独立噪声；不是推理去噪步数。
- `action_inference_steps=4`：HIDE 动作 Euler 采样步数。
- `noise_beta_alpha=1.5`、`noise_beta_beta=1.0`、`noise_s=0.999`：训练动作噪声时间为 `clamp(Beta(alpha,beta)/noise_s,0,1)`。
- `learning_rate_scale.video=0.1`、`.action=1.0`：乘以 task 的 `learning_rate` 得到优化器初始分组学习率；action 组也包含状态编码器。调度末值遵循 scheduler 的定义。

HIDE 不支持 FLUX staged Action，也不使用通用 Action scheduler；其 `action_dit.scheduler` 显式置空。Wan/Cosmos 的 Action scheduler 在 MoT / IDM 中有效。HIDE 的视频 scheduler 仍有效。

上述数值保留现有实验设置；它们经过配置／小模型功能检查，不代表效果最优。HIDE 的分组学习率现在随 task 学习率变化；默认 `1e-4` 仍得到视频 `1e-5`、动作 `1e-4`。历史 `group_learning_rate` 绝对值配置仍兼容。

## 数据：原始观测和模型输入契约

`configs/data/robodojo.yaml` 管理数据目录、相机、原始动作／状态、统一槽位映射、窗口、归一化、划分和增强。它引用模型提供的文本编码器信息及目标维度，避免各维护一份常量。

- `window_steps=33` 是原始控制时序中的 33 个位置；`action_video_freq_ratio=4` 表示每 4 个位置取一帧视频。结果为 9 帧视频与 32 步动作，不是 33 帧视频。
- `global_sample_stride` 是从原数据取样的步长，影响物理时间间隔；不是视频与动作的频率比。部署 fps 必须相应调整。
- `shape_meta.images[].shape` 决定每个相机的 resize；`camera_layout` 决定顺序；`concat_multi_camera` 决定拼接方法；`video_size` 是最终画布大小。RoboDojo 使用三相机 L 形画布 384×320。
- `shape_meta.action/state` 描述磁盘原始宽度；`unify_action_map` / `unify_state_map` 描述进入模型的槽位，不能通过更改 shape 把不同物理量当成相同数据。
- `val_set_proportion=0.01` 是 episode 留出比例；验证复用训练统计量和 resize，不使用训练颜色增强。这个比例是否足够，需要结合实际 episode 和任务覆盖率判断。
- `pretrained_norm_stats` 指向 XiconWAM 统计格式，不能直接换成 LeRobot `meta/stats.json`。
- `preflight_text_cache=true` 提前检查文本缓存；`load_text_encoder_on_miss=false` 让缓存缺失明确失败。开启在线回退需要 `data.train.video_dit`，并可能增加 worker 的内存／模型加载成本。
- `max_read_retries=2` 是有限重读策略；异常数据不能作为零误差验证样本。

删除了未被使用的 `camera_key: null` 配置。保留了确实控制处理器行为的 `stack_images`、`include_gt_action`、归一化模式、槽位映射和颜色增强设置。

`data=mixture` 是分源组合入口，默认只有 RoboDojo；`seed` 引用根训练 seed，不再固定为另一份 42。`weight_strategy=proportional` 按数据量分配，`uniform` 均匀分配，只有 `manual` 才使用各 source 的 `weight`。默认配置移除了不生效的 `weight: 1.0`；手动加权时显式添加该字段。不能仅凭配置文件名声称完成多源预训练。

`data=video` 使用 JSONL manifest；`frames` 是实际 RGB 帧数，与 RoboDojo 的 `window_steps` 含义不同。该数据没有动作与状态，只适用于 `model=video`。训练 manifest 必填，验证 manifest 需要显式提供；默认 `eval_every=0`。

## 训练与 task

`configs/training/default.yaml` 提供 trainer、DataLoader、优化器、验证、checkpoint、日志和运行默认值。它通过 Hydra 的全局包组合，继续支持 `batch_size=...`、`learning_rate=...` 等根级 CLI 覆盖。

task 选择 data、model、Video DiT 和训练默认值，并写实验差异：Wan 90k steps；Cosmos / FLUX 30k；A800 recipe 200k、batch 10、累积 2；视频训练 1k、batch 1、学习率 `1e-5`。这些训练预算不代表已收敛，显存与吞吐仍需目标 GPU 实测。

参数之间有以下关系：

- 有效 batch 为 `batch_size × world_size × gradient_accumulation_steps`；`batch_size` 是每个进程的大小。
- `max_epochs` 非空时优先于 `max_steps`；`save_every_epochs` 非空时优先于 `save_every`。A800 recipe 默认每 epoch 保存，单独改 `save_every` 不会改变这个优先级；可同时设 `save_every_epochs=null`。
- `checkpoint_save_limit=3` 限制每种 step 产物的保留数量；`null` 不裁剪。
- `eval_num_samples=4` 是每个 source 跨所有 rank 的固定窗口数，0 表示全量；`eval_every=0` 关闭周期验证。`eval_seed` 单独固定，便于比较同一批窗口。
- `num_workers` 是每 rank 的 worker 数，prefetch 也按 worker 累计；`num_workers=0` 时 persistent/prefetch 不生效。不要把多卡 recipe 的 worker 数直接当成本机通用设置。
- `max_grad_norm` 在 DeepSpeed prepare 前绑定到 `gradient_clipping`，避免只改 trainer 却仍按 JSON 的固定值裁剪。DeepSpeed 的 constant scheduler 保留各组独立学习率上限。
- `wandb.entity` 对应 W&B entity；历史 `workspace` 继续兼容。关闭 W&B 时仍保留完整字段，方便直接启用。
- `zero` / `zero_offload` 选择 DeepSpeed JSON，优化器／批量相关 auto 字段由训练运行时绑定。`vae_micro_batch_size`、`inference_cross_kv_reuse` 属于运行内存和推理设置。

## 使用与兼容

```bash
# 只检查组合和值；不加载权重、不训练。
bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml --dry-run
bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml --dry-run \
  model=hide model/video_dit=cosmos25 model.feature_layer=16

# 视频组件和动作组件分别覆盖。
bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml \
  model.video_dit.model_id=/data/trained-video \
  model.action_dit.checkpoint=/data/action.pt model.training.mode=action_only

# 原有 run 读取其保存的完整配置。
bash scripts/train_from_yaml.sh resume=/data/run
```

当前 CLI 使用 `model/video_dit=...` 和 `model.video_dit.*`；旧的 `model/backbone=...` CLI group 已移除。旧 run 中的 `backbone`、`video_backbone`、嵌套 Action 和 `action_backbone` 仍可读取；旧数据字段 `num_frames` 与 `backbone` 也保留读取兼容。新 run 只使用当前命名。

启动前检查架构、Action 几何、HIDE 特征层、冻结冲突、常见数值范围、窗口对齐和未知模型／视频字段。dry-run 不能证明文件存在、权重效果、GPU 显存或多卡稳定性；完整验收范围见[可靠性说明](reliability.md)。
