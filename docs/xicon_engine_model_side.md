# XiconWAM 模型端接入 Xicon Engine

当前优先级是**训练后的开环验证**。现阶段尚无训练完成的 checkpoint；本文把可运行的通信/评估基础设施和未来的模型效果验收分开。

## 分工与两条部署路线

- XiconWAM：加载训练 run 配置、权重、文本缓存；处理三相机/状态；模型预测；解码为物理单位的动作。
- Xicon Engine：独立安装的轻量 Python 包；WebSocket、MessagePack、请求/reset、chunk 调度、LeRobot 开环评估和画图。不携带 Torch、模型权重、机器人驱动。
- robot_control：采集图像/关节状态、检查模型契约、执行时钟、限幅、停止及录制。模型进程不导入 robot_control。
- RoboDojo 打榜：保留原 XPolicyLab 路线；本次没有替换它的协议或评测入口。Engine 开环误差不能替代 RoboDojo 成功率。
- 真机：后续由 `deployment.engine_client` 通过 Engine 接入。机型可以扩展，但 checkpoint 必须训练过相应动作/观测语义，不能仅凭维度相同就部署。

Engine 是外部发行包；不把另一仓库源码复制进 XiconWAM，也不在运行时拼接另一仓库路径。

## 安装：先用可复现 wheel

先在 Engine 仓库执行 `python scripts/build.py`，然后在已有的 XiconWAM 模型环境中安装产物：

```bash
python -m pip install '/path/to/xicon_engine-0.1.0.dev0-py3-none-any.whl[validation]'
python -m pip install -e '.[engine-validation]' --extra-index-url https://download.pytorch.org/whl/cu128
```

目前 optional extra 精确锁定 `0.1.0.dev0`，不假定 PyPI 已存在该包。两端使用同一已验收 wheel；发布版本后再一起更新锁定。机器人环境只需安装不带 validation 的 Engine wheel。

使用 uv 时，`uv sync --locked` 安装锁定的训练/评测依赖，然后用 `uv pip install '/path/to/xicon_engine-0.1.0.dev0-py3-none-any.whl[validation]'` 单独安装 Engine。项目通过 uv 的 `exclude-dependencies` 将这个外部 wheel 排除在公共索引求解之外；`uv sync --extra engine` 也不会自动安装它。后续同步若需保留该 wheel，使用 `uv sync --locked --inexact`。

只想看无权重演示时，不必安装整个 CUDA 训练栈。在装有 Engine validation extra 的轻量环境，从本仓库根目录执行：

```bash
PYTHONPATH=src python -m xiconwam.deployment.smoke --output outputs/my-engine-demo
```

已安装 XiconWAM 时对应命令为 `xiconwam-engine-smoke --output outputs/my-engine-demo`。

演示生成 80 帧 LeRobot v3 合成记录（三相机、14 维），启动真实回环 WebSocket，使用 hold 替身策略，调用 **Engine 自己的**评估器输出曲线。它不加载模型、不导入机器人驱动；`DEMO_ONLY.json` 记录证据性质。输出目录必须为空。

## 训练后：同一模型服务做开环验证

从 XiconWAM 仓库根目录运行。路径按当前工作目录解析，与训练配置中的相对路径一致；移动到服务器时应保持这些资源路径或改为绝对路径。

```bash
xiconwam-serve \
  --config /data/run/config.yaml \
  --checkpoint /data/run/checkpoints/weights/step_003000.pt \
  --contract /data/deployment/robodojo.json \
  --device cuda --inference-steps 20 --host 127.0.0.1 --port 8765
```

`--contract` 是完整 `PolicyInfo` JSON，字段包括 name、horizon、action_dim、state_dim、fps、schema_id、camera_keys、action_names、action_units 和 representation。当前 RoboDojo recipe 是：

- action_dim/state_dim 为 **14**；内部 48 维头由训练 `unify_action_map` 映射及逆映射。
- `num_frames=33`、ratio=4 对应 **9 个视频帧、32 步动作 horizon**。不是观测 33 帧，也不从返回动作再裁掉 32 行。
- camera_keys 按训练顺序为 `cam_high, cam_left_wrist, cam_right_wrist`。图像经训练 processor 的验证变换缩放，拼成 384×320，归一化到 [-1,1]。
- fps 必须来自数据采样频率；关节/夹爪 names 和 units 必须由具体数据/机器人协议确认。不要把夹爪直接声明成 rad，也不要把示例 fixture schema 用于真机。
- representation 为 `absolute_joint_position`。支持无动作变换或 `RelativeJointTransform`；相对关节输出以本次原始物理状态还原。其他姿态/旋转变换当前会拒绝启动。

模型状态使用训练统计量归一化，预测先逆映射、反归一化，再恢复相对动作。推理保持完整 chunk；没有复用会裁历史行的 `WindowPacker.postprocess()`。

训练默认 `load_text_encoder=false`，服务按指令读取同一 backbone、dtype、context_len 的文本缓存；缺失时明确报错。先用已有 precompute 流程准备任务指令缓存。若 run 配置显式加载文本编码器，可在线编码。多机型、多布局、EE/旋转动作仍需相应训练配置和适配；当前实现先支持 RoboDojo 三相机 joint 配方。

终端二直接运行 Engine 的开环评估：

```bash
xicon-engine validate \
  --url ws://127.0.0.1:8765 \
  --dataset /data/robodojo --episode 0 \
  --mode sync --execute-steps 10 --chunks 8 --timeout 300 \
  --image-map cam_high=observation.images.cam_high \
              cam_left_wrist=observation.images.cam_left_wrist \
              cam_right_wrist=observation.images.cam_right_wrist \
  --output outputs/robodojo-openloop-step003000-ep0
```

按真实 Parquet 列名调整 image-map、state-key、action-key。当前 Engine 读取 **LeRobot v3**；robot_control 常见的 v2.1 录制不能直接交给它，需要先转换或另加读取适配。GT 必须和模型解码输出具有同样的顺序、物理单位与动作时间对齐，不能把归一化 GT 与物理动作直接比较。

输出：`actions_pred_vs_gt.png`、`actions_error.png`、`cameras.png`、`open_loop_timeline.png`、`predictions.npz`、`summary.json`、`event_trace.jsonl`。

曲线横轴是“各次请求的 horizon 拼接行”；有重叠预测时保留重叠。`valid_samples` 统计有效预测行，可能大于 episode 长度；尾部 padding 不计入 MAE/RMSE。这是 teacher-forced 开环：每次读录制的真实观测，不能推导闭环成功率、安全性或视频预测质量。它只比较动作；WAM 视频生成效果仍走原模型评估。

先固定 checkpoint、episode、seed 和 sync 参数比较误差。模型慢也能完成 sync 开环；先测真实推理延迟，再决定真机是否有足够异步余量。async 的剩余旧 chunk 时间约 `(H-K)/fps`，不足则 Engine 会 underrun；开环时序是把测得 RPC 耗时投影到虚拟执行时钟。

## 后续 robot_control 配合

机器人环境使用相同 Engine wheel，启动已有入口：

```bash
python -m deployment.engine_client \
  --robot /path/to/reviewed_robot.yaml --runtime /path/to/simulation_runtime.yaml \
  --policy-url ws://MODEL_HOST:8765 --schema-id REVIEWED_SCHEMA \
  --task 'task matching training' --execute-fps 20 \
  --mode sync --execute-steps 10 --action-horizon 32 --chunk-num 2 --dry-run
```

实际 fps 以训练数据为准。robot_control 会检查 schema、维度、action_names、相机集合、fps；建议同时传 `--action-units` 做精确单位断言。`cam_head` 与 `cam_high` 需要在配置/数据约定中统一。当前一次只有一个活动客户端，离线评估和机器人应依次使用服务。默认监听回环；跨机部署才显式选择受控网络地址。

本次无真机运动授权，也没有接触硬件。CLI 的 hold fixture 只是通信工具，不是经过训练的机器人策略。

## 检查结果与发布判断（2026-11-03）

本机 macOS 验证：

- Engine lint、format、55 项测试通过；覆盖独立进程通信、协议/reset、超时、单请求串行、调度、开环及错误情况。
- sdist/wheel 构建及 `twine check --strict` 通过；干净临时环境安装 wheel 后，独立服务/客户端握手、sync/Future、reset 和关闭通过。
- robot_control 现有 7 项集成测试通过，包含 mock adapter、仿真 adapter 和真实回环 WebSocket。
- 新增 XiconWAM 适配测试覆盖状态裁剪、非连续 unify slots、反归一化、相对动作的原始参考、完整 chunk、文本 context、相机契约及真实 RPC 开环报告。
- 新增 XiconWAM hold fixture → 真实 WebSocket → robot_control mock 执行循环，sync/async 均通过。
- CPU 适配测试使用 Torch 2.7.1、Torchvision 0.22.1；尚无 Linux CUDA 权重推理、真实 RoboDojo 数据效果、XPolicyLab 打榜或硬件验收。

结论：**足以作为这两条基础链路的早期依赖，并制作固定 wheel 试用；目前的证据不足以称作稳定正式版。** Engine 本次仅调整多轴曲线图的标题留白，避免 14 轴图标题与第一行重叠；其原有未提交工作已保留。

建议先做内部 wheel，再在具备账户/包名权限后发布 TestPyPI 预发布版，验收后再发布 PyPI。公开发布不是稳定性的来源；包版本、协议契约和验证证据才是。正式版前至少补齐 Linux/Python 3.10 的干净安装与全测、长时重连/故障压力、真实 checkpoint/留出数据开环，以及真实部署网络下的延迟分位数。真机验收单独进行。

Engine 默认单客户端、单在途推理，不能当多租户 GPU 服务。XiconWAM 现有 checkpoint loader 部分分支允许 non-strict 权重载入；正式模型上线前还应收紧完整性校验，防止把缺权重的研究 checkpoint 当成完整策略。

本次只构建和本机验证，没有上传 TestPyPI/PyPI，也没有确认包名归属。模型端和机器人端应先固定同一 artifact；上线后根据实际日志而非 Echo 的低延迟调整 H/K 和超时。
