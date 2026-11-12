# XiconWAM 文档导航

安装、RoboDojo 联合训练与闭环评测从[项目 README](../README.md)开始。默认基线是 Wan2.2 + MoT + RoboDojo joint14，内部动作／状态接口为 48D。

## 操作指南

- [配置分层与参数检查](configuration.md)：Video DiT / Action DiT / 架构、数据契约、共享训练默认值及旧配置兼容。
- [数据发现](dataset_discovery.md)：嵌套 LeRobot v3 数据集、分源处理和统计量约束。
- [视频与动作分阶段训练](wan_video_then_action.md)：视频训练、组件权重、OpenWAM / VPP2 工具和独立视频推理。
- [部署模型包](deployment_bundle.md)：资源绑定、迁移校验、验证指标和数据错误策略。
- [Xicon Engine 接入](xicon_engine_model_side.md)：通信演示、开环验证和机器人端契约。
- [可靠性与验收范围](reliability.md)：本轮实际检查结果、兼容行为和未验证项。

## 源码职责

- `src/xiconwam/launch.py` 统一构建模型、数据与训练会话；`train.py` 是 Hydra 训练入口。旧构建名称只为历史 Hydra 配置保留，不再维护独立实现。
- `actor/` 管理策略、动作模型、训练目标、采样及组件权重。`vision/` 管理 Wan / Cosmos / FLUX 的加载和骨干实现。
- `model_config.py` 解释视频／动作配对；`engine/configuration.py` 检查参数并恢复运行配置；`engine/launch_config.py` 生成启动配置。三者职责不同。
- `data/` 管理数据发现、48D 映射和采样。`data/lerobot/lerobot/` 是现有 LeRobot 读写实现；外层处理器仍有实际调用，不能机械删除或移动导入路径。
- `engine/` 管理优化器、训练循环、断点和留出集验证；`kit/` 存放共用工具。
- `src/xiconwam/deployment/` 对接外部 Xicon Engine 包；根目录 `deployment/` 是 ZMQ 服务；`eval/XPolicyLab_core/` 是 RoboDojo 评测入口。

Cosmos / FLUX、IDM / HIDE、多源混合和纯视频训练是可选路径。配置可组合不等于每种组合都有真实权重、GPU 训练或闭环成绩。

## 研究提案

以下材料属于未实现或未验收的研究设计，不是当前框架功能或效果承诺：

- [V2 设计](research/xiconwam_v2_design.md)
- [V2 实验与投稿分析](research/xiconwam_v2_ral_review.md)
- [预测避险与任务恢复](research/safety_research_plan.md)，对应离线工具 `scripts/research/safety_labels.py`

研究材料保留原调研日期；其外部论文、版本和投稿规则没有在本轮重新核验。
