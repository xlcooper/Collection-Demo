# `data/` 运行数据目录说明

`data/` 是 Collection-Demo 在服务器上的运行工作区，用于保存输入图片、对象记忆库和实验产物。除本说明文件外，目录内容均由 Git 忽略。

## Git 管理边界

| 内容 | 是否进入 Git | 原因 |
|---|---|---|
| `data/README.md` | 是 | 说明目录用途、命名和保留策略 |
| 输入图片 | 否 | 可能包含隐私，且二进制文件会扩大仓库历史 |
| `memory.sqlite` | 否 | 运行中持续变化的二进制数据库，不能可靠 diff 或 merge |
| crop、mask、overlay、对象参考图 | 否 | 数量多、可由输入和代码重新生成 |
| Qwen 原始响应与内部 run report | 否 | 保留在对应记忆库中供本机审计 |
| `environment/*.json` 精选验收报告 | 是 | 体积小、机器可读，是跨机器同步的验收证据 |

Git 即使删除大文件仍会保留历史对象，因此不适合直接管理整个 `data/`。项目采用三层策略：

1. Git 跟踪代码、配置、文档和目录说明。
2. `environment/` 跟踪经过选择的服务器报告。
3. `data/` 保留真实输入、数据库和可再生产物。

## 当前服务器目录

下表解释项目迄今使用过的 `data/` 一级目录。目录存在只代表曾创建或开始运行，不代表对应实验已经通过；完成状态必须依据 `environment/*.json` 和 `PROGRESS.md`。

| 目录 | 类型 | 用途 | 建议保留策略 |
|---|---|---|---|
| `memory/` | 默认记忆库 | `config/default.yaml` 的默认持久化根目录，适合正式 Demo 或长期保留的对象记忆 | 与临时实验隔离；确认不再需要前不要删除 |
| `smoke/` | 输入 | SAM3/Qwen 环境冒烟使用的最小测试图片 | 体积通常很小，可长期保留在服务器 |
| `m5_input/` | 输入 | M5 多视角端到端验收图片目录 | 验收和报告复核完成前保留 |
| `m5_qwen_gate_input/` | 历史输入 | M5 单场景 Qwen gate 使用的一张隔离场景图 | 对应报告已保存且无需复现后可清理 |
| `m5_qwen_gate_memory_01/` | 历史实验记忆库 | 第一轮逐候选 Qwen gate；计数通过但出现真实物体/阴影语义交换 | 对应报告为 `environment/m5_qwen_gate_report.json` |
| `m5_qwen_gate_memory_02/` | 历史实验记忆库 | mask 隔离 crop 后的第二轮逐候选 Qwen gate | 对应报告为 `environment/m5_qwen_gate_report_v2.json` |
| `m5_image_batch_gate_memory_01/` | 历史实验记忆库 | 同一源图7个候选一次 Qwen 批量调用的 gate | 对应报告为 `environment/m5_image_batch_gate_report.json` |
| `m5_validation_memory_01/` | 当前验收记忆库 | 多视角 M5-B 的独立对象记忆库；包含完整 SQLite 和运行资产 | 报告拉取、真值复核和验收结论完成前保留 |
| `temp/` | 未标准化临时目录 | 不是当前配置或正式流程要求的标准目录；仅凭名称无法判断内容 | 先检查内容和来源，不自动删除，也不作为验收证据 |

如果服务器上的目录名被界面截断，`m5_image_batch_gate_memor...` 对应当前命名规范下的 `m5_image_batch_gate_memory_01/`。

## 记忆库根目录结构

`memory/`、`m5_qwen_gate_memory_01/`、`m5_validation_memory_01/` 等目录都采用相同结构。它们不是代码副本，而是彼此隔离的对象记忆数据库。

```text
<memory_root>/
├─ memory.sqlite
├─ sources/
├─ proposals/
├─ objects/
├─ raw_responses/
└─ run_reports/
```

| 路径 | 内容 | 是否可再生成 |
|---|---|---|
| `memory.sqlite` | `runs`、`source_images`、`proposals`、`objects`、`observations`、`decisions` 六类结构化索引和状态 | 不应单独重建；它是该记忆库的主索引 |
| `sources/` | 按 SHA-256 保存的规范源图副本 | 可从原始输入重新生成，但用于当前库审计 |
| `proposals/` | 按 run/proposal 保存的 `crop.png`、`mask.png`、`overlay.jpg` | 可重新推理生成 |
| `objects/` | 按 object/observation 保存的新建或归并对象观测资产 | 可重新运行生成，但与 SQLite 对象 ID 配套 |
| `raw_responses/` | Qwen 每次源图批量调用的原始文本、token、耗时和解析结果 | 可重新推理生成，但模型输出可能不完全一致 |
| `run_reports/` | 该记忆库内部保存的逐次运行报告 | 可由对应运行生成；精选报告另存到 `environment/` 进入 Git |

不要只复制或删除记忆库中的某一个子目录。`memory.sqlite` 与图像资产通过相对路径关联，迁移、归档或清理时应把整个 `<memory_root>/` 视为一个整体。

## 命名约定

| 模式 | 含义 | 示例 |
|---|---|---|
| `<purpose>_input/` | 某次验证的只读输入图片 | `m5_input/`、`m5_qwen_gate_input/` |
| `<purpose>_memory_<NN>/` | 某次独立实验的完整记忆库 | `m5_validation_memory_01/` |
| `memory/` | 默认或正式 Demo 记忆库 | `data/memory/` |

同一批图片需要重跑时，应使用新的编号记忆库，例如从 `_01` 改为 `_02`。复用旧库会触发 SHA-256 幂等门，使已经完成的图片直接跳过。

不要新建含义不明的 `temp/`、`test2/` 或 `new/` 作为正式实验目录。临时数据优先放到仓库根目录的 `runs/`，需要持久化的实验按上述命名。

## 保留与清理原则

- **正式记忆库**：按整体备份和迁移，不拆分内部文件。
- **当前验收库与输入**：报告完成、真值复核和验收结论形成前保留。
- **历史 gate 记忆库**：对应 `environment/*.json` 已进入 Git，且确认不再需要复现后，可以整目录清理以释放空间。
- **未知目录**：先查看内容、大小和生成时间，再决定；不得仅凭名称删除。
- **模型权重**：位于 `weights/`，不属于 `data/`，不要与实验产物一起清理。

本文件只描述目录用途。当前阶段、实验是否通过和下一动作以根目录 `PROGRESS.md` 为准。
