# Collection-Demo

第一阶段自动化对象记忆 Demo：SAM3 发现并分割场景物体，MLLM 判断对象新颖性、生成结构化标签，并把同一具体物体的多次观测归并至显式记忆库。

- **当前版本**：v0.2.3
- **当前阶段**：M0——服务器环境整改
- **当前状态**：环境声明与模型冒烟脚本已就绪，等待服务器执行

## 当前范围

```text
场景图像 → SAM3 候选、mask 与 crop
        → MLLM 新颖性判断和自动标注
        → 新建对象 / 归并已知对象 → 显式对象记忆库
```

本阶段不实现认知完整度、主动视角规划、机械臂控制、空间地图或完整世界模型。

## 当前技术决策

- **视觉模型**：官方 `facebook/sam3` 图像 checkpoint。
- **服务器权重**：`/root/autodl-tmp/Collection-Demo/weights/sam3/sam3.pt`，仅服务器保留并由 Git 忽略。
- **MLLM 首选**：`Qwen/Qwen3-VL-8B-Instruct-FP8`。
- **MLLM 回退**：`Qwen/Qwen3-VL-4B-Instruct`。
- **运行方式**：先批量运行 SAM3 并保存中间结果，释放模型后再运行 MLLM；两个模型不同时驻留显存。
- **存储方式**：文件系统保存图像资产，SQLite 保存结构化索引。
- **环境基线**：Python 3.12、PyTorch 2.10.0+cu128、Transformers 4.57.6。

## 环境关键摘要

| 项目 | 当前结论 |
|---|---|
| GPU | RTX 4090 24 GiB，可用于顺序运行两个模型 |
| PyTorch | CUDA 可用，当前 base 为 PyTorch 2.12.1+cu126 |
| Python | 3.10.8，不满足 SAM3；项目环境必须使用 3.12 |
| CUDA 工具链 | PyTorch 为 cu126，系统 nvcc 为 11.8；MVP 暂不编译可选 CUDA 扩展 |
| 模型依赖 | SAM3、Transformers 和 Qwen 工具尚未安装 |
| 磁盘 | 报告后已清理，用户确认约 44.1 GiB 可用；待下次自检固化 |
| SAM3 权重 | 已认证并放置于 `weights/sam3/sam3.pt`，不进入 Git |

完整判断与执行顺序见 `docs/02_服务器环境分析.md`。

## 导航

| 路径 | 用途 | Git 状态 |
|---|---|---|
| `README.md` | 项目摘要、长期准则和导航 | 跟踪 |
| `CHANGELOG.md` | 每次 push 的版本与进度摘要 | 跟踪 |
| `environment/server_env_report.json` | 服务器环境原始事实 | 跟踪 |
| `docs/02_服务器环境分析.md` | 环境详细分析、模型决策和下一步 | 跟踪 |
| `docs/03_服务器环境与冒烟测试指南.md` | 服务器环境创建与执行命令 | 跟踪 |
| `environment.yml` | Conda 基础环境事实源 | 跟踪 |
| `requirements.txt` | PyTorch、Qwen 与普通 Python 依赖 | 跟踪 |
| `requirements-sam3.txt` | SAM3 固定提交的独立安装入口 | 跟踪 |
| `scripts/check_server_env.py` | 服务器环境自检 | 跟踪 |
| `docs/00_项目纲要.md` | 完整项目背景 | 仅本地，忽略 |
| `docs/01_第一阶段项目设计规划书.md` | 第一阶段技术规划 | 仅本地，忽略 |

## 下一步

1. 服务器 pull 当前版本。
2. 依次使用 `environment.yml`、`requirements.txt` 和 `requirements-sam3.txt` 配置环境。
3. 运行环境自检并准备一张不进入 Git 的测试图像。
4. 运行 `scripts/run_server_smoke_tests.sh` 顺序测试 SAM3 与 Qwen3-VL。
5. push 三份生成报告；本地 pull 分析后决定是否进入 M1。

详细命令见 `docs/03_服务器环境与冒烟测试指南.md`。

## 长期开发准则

本节是所有新对话和后续开发必须优先读取并持续遵守的规则。准则可以扩展：用户明确要求把新规范写入 README 时，应加入本节或相应长期章节并记录到 CHANGELOG；未经明确要求，不把临时讨论升级为长期规则。

1. **MVP 与奥卡姆剃刀**：优先完成满足当前目标的最小方案，避免额外扩展、过早抽象和过度设计。
2. **聚焦第一阶段**：当前只开发自动化对象记忆 Demo；主动采集、视角规划、机械臂和世界模型均不提前实施。
3. **Friendly 代码**：模块、脚本和逻辑必须职责清楚、命名直接、流程可读；优先短函数、显式类型、结构化模型输出和可操作错误信息。
4. **本地/服务器分工**：本地只用于人与 AI 协作、修改、审查、提交和 push；不在本地安装深度学习环境或运行脚本/测试。所有脚本、测试与 Demo 只在 auto DL 服务器执行。
5. **Conda 与依赖管理**：项目运行环境统一由 Conda 管理；Conda 基础环境以 `environment.yml` 为事实源，Python 依赖以 `requirements*.txt` 为事实源，不把临时安装命令当作最终配置，不长期使用 `base` 环境运行项目。环境配置必须按可独立重跑的步骤拆分，不使用单个脚本串联全量安装。
6. **Git 与环境先验**：收到服务器已 push 的通知后，本地首先执行 pull，并只依据拉取到的文件分析。每个完成的本地修改批次都必须更新 CHANGELOG、提交并 push；若 push 失败必须明确报告。

## 版本规则

- 每次 push 在 CHANGELOG 顶部增加语义化版本号、北京时间时间戳和简短摘要。
- 默认递增补丁版本；阶段目标或兼容性明显变化时再递增次版本或主版本。
- 同一对话回合中的关联修改作为一个版本批次提交。
