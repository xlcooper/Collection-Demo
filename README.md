# Collection-Demo

面向陌生场景的第一阶段自动化对象记忆 Demo：使用 SAM3 发现并分割物体，使用 MLLM 判断对象新颖性、生成结构化标签，并把同一具体物体的多次观测持续归并到显式记忆库。

- **当前版本**：v0.1.0
- **当前阶段**：M0——服务器环境先验
- **当前状态**：环境自检脚本已提供，等待 auto DL 服务器生成并回传报告

## 第一阶段范围

```text
场景图像 → SAM3 候选与 mask → MLLM 新颖性判断
        → 新对象自动标注 / 已知对象观测归并 → 对象记忆库
```

本阶段只实现图像批处理、对象发现、身份判断、自动标注、存储、去重和持续扩充。不实现认知完整度判断、主动视角规划、机械臂控制、空间地图或完整世界模型。

## 模型基线

### 视觉分割

- 使用 Meta 官方开源 [SAM3](https://github.com/facebookresearch/sam3) 代码与权重。
- 首选与官方主分支兼容的最新 SAM 3.x checkpoint；实际 checkpoint 在服务器环境和访问权限确认后锁定。
- SAM3 是可提示分割模型，“无提示发现所有独立物体”的效果必须先通过服务器冒烟实验验证。MVP 只保留一种确定的全量候选策略，不并行维护多套发现方案。

### MLLM

- **首选**：[Qwen3-VL-8B-Instruct](https://github.com/QwenLM/Qwen3-VL)。用于多图对象对比、细粒度识别、属性描述和受约束 JSON 输出。
- **低资源降级**：Qwen3-VL-4B-Instruct。
- **质量升级**：Qwen3-VL-32B-Instruct，仅在 8B 实测不足且服务器资源允许时采用。
- **对照备选**：[InternVL3.5-8B-HF](https://github.com/OpenGVLab/InternVL)，只用于选型对比，不在 MVP 同时维护。
- MVP 使用 Instruct 版，不使用 Thinking 版；当前任务更重视稳定结构化输出、延迟和显存占用。

最终型号、精度与量化方式必须依据 `environment/server_env_report.json` 和固定 Demo 实测确定。

## 导航

| 路径 | 用途 | Git 状态 |
|---|---|---|
| `README.md` | 项目入口、长期准则和导航 | 跟踪 |
| `CHANGELOG.md` | 每次 push 的版本与进度摘要 | 跟踪 |
| `scripts/check_server_env.py` | auto DL 服务器环境自检 | 跟踪 |
| `environment/server_env_report.json` | 服务器生成的环境事实 | 生成后跟踪 |
| `docs/00_项目纲要.md` | 完整项目背景 | 仅本地，忽略 |
| `docs/01_第一阶段项目设计规划书.md` | 第一阶段技术规划 | 仅本地，忽略 |

## 服务器环境自检

所有脚本只在 auto DL 服务器运行。服务器 pull 后执行：

```bash
python scripts/check_server_env.py
```

默认输出：

```text
environment/server_env_report.json
```

脚本会记录 GPU、驱动、CUDA、Conda、Python、PyTorch、SAM3/MLLM 依赖、Git 和磁盘等必要事实，并输出 `ready`、`warning` 或 `blocked`。即使发现阻塞项，也会先写完报告。报告不采集令牌值、用户名或主机 IP。

报告生成后，需要同步更新 `CHANGELOG.md`，提交并 push；本地 pull 后再据此确定 `environment.yml`。

## 长期开发准则

本节是所有新对话和后续开发必须优先读取、持续遵守的项目规则。准则可以扩展：当用户在对话中明确要求把新的规范写入 README 时，应将其加入本节或相应长期章节，并在 CHANGELOG 中记录；未经用户明确要求，不把临时讨论自动升级为长期规则。

1. **MVP 与奥卡姆剃刀**：优先完成满足当前目标的最小方案，避免额外扩展、过早抽象和过度设计。
2. **聚焦第一阶段**：当前只开发自动化对象记忆 Demo；主动采集、视角规划、机械臂和世界模型均不提前实施。
3. **Friendly 代码**：模块、脚本和逻辑必须职责清楚、命名直接、流程可读；优先短函数、显式类型、结构化模型输出和可操作错误信息。
4. **本地/服务器分工**：本地目录只用于人与 AI 协作、修改、审查、提交和 push；不在本地安装深度学习环境或运行脚本/测试。所有脚本、测试与 Demo 只在 auto DL 服务器执行。
5. **Conda 管理**：项目运行环境统一由 Conda 管理；依赖以 `environment.yml` 为事实源，不把临时安装命令当作最终配置，不长期使用 `base` 环境运行项目。
6. **Git 与环境先验**：所有可共享开发资产通过 Git 管理。每个完成的本地修改批次都必须更新 CHANGELOG、提交并 push；若 push 失败必须明确报告。功能开发前先由服务器运行环境自检，提交其报告，本地 pull 后再基于真实环境继续设计。

## 版本与 Push 规则

- 每次 push 都必须在 `CHANGELOG.md` 顶部新增一个版本条目，包含语义化版本号、北京时间时间戳和简短进度摘要。
- 默认递增补丁版本；阶段目标或兼容性发生明显变化时再递增次版本或主版本。
- 一个对话回合中的关联修改作为一个版本批次提交，避免为同一件事制造大量无意义版本。
- push 前确认工作区无遗漏；push 后确认远端分支已更新。

## 下一步

1. auto DL 服务器 pull 当前版本并运行环境自检。
2. 将 `environment/server_env_report.json` 连同 CHANGELOG 更新提交并 push。
3. 本地 pull 报告，确定 SAM3 checkpoint、Qwen3-VL 规模和 `environment.yml`。
4. 开始 M1：配置、数据模型、SQLite schema 与 CLI 骨架。
