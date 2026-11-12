# 可迁移的部署模型包

模型包将一次部署需要的配置、策略权重、预训练资源、归一化统计量、文本缓存和动作契约绑定在同一目录中。导出不会实例化模型或访问机器人。导出和运行都使用已安装的 XiconWAM 与 Engine 包。

## 导出、检查和运行

在训练所用工作目录执行；配置必须使用 run 中已经解析的 `config.yaml`：

```bash
xiconwam-bundle export \
  --config /data/run/config.yaml \
  --checkpoint /data/run/checkpoints/weights/step_003000.pt \
  --contract /data/deployment/robodojo.json \
  --base-dir /path/to/XiconWAM \
  --instruction 'pick the object' \
  --output /data/releases/robodojo-step003000

xiconwam-bundle verify /data/releases/robodojo-step003000
xiconwam-serve --bundle /data/releases/robodojo-step003000 \
  --device cuda --inference-steps 20 --host 127.0.0.1 --port 8765
```

未重新安装 entrypoint 时，可以用 `python -m xiconwam.deployment.bundle` 和 `python -m xiconwam.deployment.serve`。

`--base-dir` 解释训练配置中原有的相对资源路径，默认当前目录。`--config`、`--checkpoint`、`--contract` 本身按当前目录解析。输出目录必须不存在，已有目录不会被覆盖。可以重复 `--instruction`，显式要求部署任务的文本 embedding 已存在；不指定时仅校验所选编码器、长度和 dtype 对应的缓存，不假定任意自然语言都有缓存。

多源训练必须通过 `--data-source SOURCE` 明确选择一个已启用的训练源。服务仍使用该源的相机、统计量、动作映射和契约；一个导出包不自动声明跨机型泛化能力。若原数据的 `meta/info.json` 仍可访问，导出会校验契约 fps 与 `dataset_fps / global_sample_stride` 一致；原数据不可访问时 fps 由提供的契约负责。

## 包内容与可复现信息

```text
manifest.json         # 格式版本、入口、资源绑定、文件 SHA-256/字节数
config.yaml           # 部署配置，活跃资源路径相对于包目录
training_config.yaml  # 原训练配置快照，仅用于溯源
checkpoint.pt         # 选定的策略权重，不包含训练状态目录
contract.json         # PolicyInfo：schema、轴名/单位、表示、相机、fps、horizon
assets/               # 本地预训练资源、动作初始化权重、统计量、文本缓存
```

manifest 同时记录 UTC 导出时间、所选数据源、文本 prompt hash/缓存索引、可选指令列表、Python/依赖版本、Git commit、工作树是否有未提交改动，以及项目 Python 源码的内容 hash。包不会包含数据集、优化器状态或凭据。代码 hash 用于溯源；运行时仍需安装对应代码和依赖，模型包不是 Python 环境镜像。

导出校验文件与资源绑定；实际加载完整策略时还会严格检查缺失、额外键和张量形状。独立视频权重不能当作完整策略载入，分阶段训练应显式使用组件接口。文件校验不能证明模型效果合格，也不能替代实际权重加载。

预训练目录会实际复制，可能很大；相同资源及其子路径会复用已经复制的目录。顶层资源软链接和文件软链接会物化为内容，内部目录软链接需要先物化。开启在线文本编码器时，其模型必须提供为本地路径；导出不会自动下载远程模型。训练时使用 LoRA 的配置及基础权重一起保留。

## 校验与发布语义

导出先写同一父目录下的临时目录，生成清单并校验后才重命名为最终目录。失败会清理临时产物。运行 `--bundle` 会在分配模型前检查：

- 格式版本、文件清单、每个文件的大小和 SHA-256。
- 入口是否纳入清单，配置与资源绑定是否一致，manifest 与动作契约是否一致。
- 资源路径是否越出包目录、是否存在软链接。

加载时将资源绑定为该包位置下的绝对路径，因此移动目录、更换工作目录、删除原始资源后仍可加载。环境依赖版本、安装代码内容差异只警告，不强制阻止研究迭代。`--bundle` 与旧的三路径加载参数互斥，旧入口继续可用。

SHA-256 清单用于完整性检查，没有数字签名；应从可信来源获得整个包。不要在导出后修改包内配置或资源，应重新导出新版本。真实大权重加载、CUDA、性能与硬件验收需要单独进行。

## 固定验证集和数据错误策略

训练验证的 `eval_seed` 默认 42，`eval_num_samples` 默认 4，表示每个源在所有 rank 合计评估 4 个固定窗口；设为 0 可评估该源所有窗口。选择不随训练 step 或 rank 数变化，多源验证直接遍历各源的留出集，不经过训练混合权重抽样。各 rank 保持相同前向次数，补齐的重复工作不计入指标。

多源统计量分别保存在运行目录 `dataset_stats/<source>.json`。自动计算仅使用训练集；验证集复用对应源的统计量。运行 `config.yaml` 会记录这些实际路径，供模型包导出使用。没有有效视频目标帧时，视频指标保持缺省，训练日志仍可正常记录其他指标。

每次验证生成 `eval/step_XXXXXX.json`，记录选定窗口索引、各指标分母和按源指标。时间 padding、动作空槽和图像 padding 会被排除；动作先逆映射、反归一化，再在声明的物理动作字段上统计。`action_l1` 为绝对误差均值，兼容字段 `action_l2` 为**平方误差均值（MSE）**，不是 RMSE。另有逐字段/逐轴指标；不同单位和不同源的总平均仅作兼容摘要，分析时应优先比较按源、按轴指标。验证会恢复随机数状态及训练模式。

训练前默认验证指令集合的文本缓存，不在缓存缺失时偷偷加载文本编码器。显式开启 `load_text_encoder_on_miss` 时，仅支持 `num_workers=0`；worker 内在线编码会明确报错。旧配置若有非确定性的高/低层指令混合，需要先统一指令预计算约定，或显式关闭 `preflight_text_cache`。

读取失败不再随机替换样本。训练只对明确的临时 I/O errno 做有限次数同索引重试（默认 `max_read_retries=2`），记录进程内错误计数和源路径；配置错误、缓存缺失、损坏数据会保留异常原因并报错。验证读取失败立即报错。元数据和文本缓存预检不等于完整视频扫描，实际坏视频仍会在读取时暴露。

混合采样按源内固定种子的排列跨 epoch 轮转，再全局混排，低权重源可以覆盖整个源。项目 sampler 在恢复训练时传递 epoch offset；共享 epoch 通知持久 worker 重建本地索引。固定源顺序、权重、长度、seed 和 epoch 时可复现。
