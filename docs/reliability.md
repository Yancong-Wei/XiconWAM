# 可靠性检查与验收范围

检查日期：2026-11-12。当前结论是：本地回归覆盖的训练、数据和通信路径可以继续用于研究开发；Linux / CUDA 多卡长训和真实策略效果仍需单独验收。

## 本轮清理

- 删除无调用的 `vision/wan/spec.py`、仅作别名的 `vision/wan/video_training.py` 和无调用的 `data/lerobot/transforms/misc.py`。
- 删除 `launch.run_inference` 旧入口；独立视频推理使用 `PYTHONPATH=src python deployment/infer_video.py`，策略服务继续使用现有部署入口。
- `build_mot`、`build_idm`、`build_hide` 合并到同一构建实现；保留历史 Hydra 关键字目标和旧 PEFT LoRA 参数。`build_actor` 保留原 actor 类型及 checkpoint 标识。旧 Python 位置参数调用不在兼容范围内。
- 删除 checkpoint 模块中重复的训练辅助函数和无用导入，移除无调用的直接依赖 `boto3`、`termcolor` 并更新锁文件。
- 清除 `src/` 下 228 个生成的 `.pyc` 文件和 18 个空缓存目录。
- Transformers 从 4.57.0 调整到补丁版 4.57.1；上游修复说明见 [4.57.1 发布记录](https://github.com/huggingface/transformers/releases/tag/v4.57.1)。
- V2 与安全研究材料集中到 `docs/research/`，操作指南从 [文档导航](README.md) 进入。

源码 Python 文件从 116 个降为 113 个。本轮保留了有调用的可选骨干、架构、数据处理和部署模块。

## 修复的问题

### 断点恢复

DataLoader 创建迭代器时也会抽取随机种子；原先会改变刚恢复的模型 RNG。现在使用独立 generator，避免这次额外抽样改变模型噪声。

如果恰好在一个 epoch 的最后一个 batch 保存，恢复时先规范化到下一 epoch，避免空迭代器造成 Accelerate 与采样器的 epoch 不一致。

回归使用真实 CPU `TrainSession`、Accelerate、AdamW、梯度累积和随机训练目标。在 epoch 内、epoch 边界和下一 epoch 中断后恢复，比较模型、优化器、调度器、数据进度和后续随机数，要求逐值一致。覆盖完整 batch 和末尾不足一个 batch 的情况。

历史的平铺 Action 配置使用 `backbone` 时，恢复路径现读取正确字段，并有针对性回归。

这项证明限于单进程、`num_workers=0`、确定性样本读取和相同训练计划。多 worker 随机增强、GPU 非确定算子、多卡 ZeRO 和中途变更训练计划的逐值一致性尚未证明。

### Checkpoint 完整性

重写同一步训练状态前，先撤销旧的 `trainer_state.json` 完成标记；分片写入失败后不会保留旧标记供自动续训误选。

MoT / IDM / HIDE 的完整原生策略权重检查缺键、多键、形状和状态编码器是否匹配。视频组件权重必须通过 `load_video_checkpoint()` 或原生视频模型导出流程显式加载，不能再被完整策略接口默默当作完整权重接受。Action 组件仍通过显式格式及其加载逻辑组装。

新保存的 LoRA 权重附带 trainable-key 清单，加载时核对张量清单和 adapter 根模块。历史 LoRA 文件仍可加载，但没有清单时会警告：无法完整核验缺失的可训练张量。清单与 strict loading 都不能证明权重效果或底座来源正确，部署仍需绑定原始配置与资源。

## 本地检查结果

- 全量测试：160 项通过、1 项跳过，另有 5 个 subtest 通过；包含配置重构后的模型构建、旧 run 和模型包兼容回归。
- 配置与启动 dry-run：Wan / Cosmos / FLUX × MoT / IDM / HIDE / Video 的 12 组组合中，11 组通过，HIDE + FLUX 按预期拒绝。
- 测试覆盖缩小模型的前向与反向、冻结策略、权重往返与损坏拒绝、数据与验证指标、模型包和 Engine 本机通信。
- 当前配置使用并列的 `model.video_dit` / `model.action_dit`；参数职责、HIDE 分组学习率、数据窗口及 task 覆盖关系见[配置说明](configuration.md)。DeepSpeed scheduler 的按组学习率接口核对了 [0.16.9 源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.9/deepspeed/runtime/lr_schedules.py)，但本地回归未启动 GPU ZeRO 引擎。
- `uv lock --check --offline` 通过；Python 语法、脚本 `bash -n`、未定义名称／重复定义和文档本地链接检查通过。

测试环境是 macOS arm64、Python 3.11、PyTorch 2.7.1，测试时使用 Accelerate 1.5.2 和 Transformers 4.57.1。它不是 README 中 Linux / CUDA 的完整锁定环境。本轮 Engine 测试使用本机独立 `xicon_engine` 源码路径，未验证新构建 wheel；跨仓库 `robot_control` 测试因外部模块未安装而跳过。本轮没有执行真实机器人运动。

在已安装项目和可选 Engine 测试依赖的环境中运行：

```bash
python -m pytest -q tests
uv lock --check
bash scripts/train_from_yaml.sh configs/task/robodojo_wan.yaml --dry-run
```

## 仍需验证

本轮没有加载完整预训练骨干开展 GPU 训练，没有 Isaac / RoboDojo 闭环成绩，也没有真机策略效果证据。因此，测试通过不能作为生产稳定性或任务成功率证明。

下一次目标 GPU 验收应固定一个实际配置和数据切片，完成短训、定期验证、权重导出与重新加载、中途保存及恢复；记录 loss、梯度、吞吐、峰值显存和 checkpoint 开销。随后验证实际使用的多卡 ZeRO 模式和持续训练。其他骨干与架构分别验收，不从默认路径外推。

`docs/research/` 中的 V2、安全预测与恢复方案继续作为研究提案保留，不计入已验证能力。
