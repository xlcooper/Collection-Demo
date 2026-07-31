# Collection-Demo

面向开放场景图片的自动对象发现、语义标注与持续对象记忆 Demo。

Collection-Demo 接收一批不断增加的场景图片，在不要求用户提供物体类别的前提下提出候选对象、生成结构化标签、建立对象档案，并尝试把同一具体物体在不同图片中的观测持续归并到同一档案。

**愿景：**构建能够从持续视觉观测中发现、识别并长期维护真实对象身份与知识，最终为主动感知、规划和机器人交互提供可审计对象记忆的智能系统。

## 系统架构

```mermaid
flowchart LR
    A["场景图片"] --> B["SHA-256 幂等门"]
    B -->|"新图片"| C["SAM3 自动点网格候选"]
    B -->|"相同哈希"| Z["跳过"]
    C --> D["过滤与 mask / crop / overlay"]
    D --> E["Qwen3-VL 源图级批量判断"]
    M["已有对象卡片与参考图"] --> E
    E --> F{"对象决策"}
    F -->|"new"| G["创建对象"]
    F -->|"existing"| H["追加观测"]
    F -->|"ignored"| I["记录并忽略"]
    F -->|"uncertain"| J["保留 pending"]
    G --> K["SQLite + 文件资产"]
    H --> K
    I --> K
    J --> K
    K --> L["对象记忆与运行报告"]
```

SAM3 与 Qwen3-VL 在单张 GPU 上顺序驻留：先完成全批次候选生成并释放 SAM3，再加载 Qwen 进行语义和身份判断，避免两个视觉模型同时占用显存。

## 当前实现

- 递归发现 JPG、JPEG、PNG 和 WebP，并根据文件原始字节计算 SHA-256；文件名不同但内容完全相同的图片只处理一次。
- 使用 `16 × 16` 自动点网格驱动 SAM3 生成候选，不要求使用者输入 `cup`、`mouse` 等类别词。
- 依据置信度、面积、IoU、包含关系和数量上限过滤候选，并为保留项生成 `mask.png`、隔离背景的 `crop.png` 和场景 `overlay.jpg`。
- 每张源图把全部保留候选、当前对象卡片和最近参考图放入一次 Qwen3-VL 调用，生成候选有效性、结构化标签和实例身份判断。
- 使用 `new`、`existing`、`ignored`、`uncertain` 四类决策创建对象、追加观测、排除无效候选或保留待定项。
- 通过 Pydantic schema、候选覆盖检查、对象 ID 范围校验、SQLite 事务和失败回滚保证数据边界可追踪。

这里的“无类别提示”表示使用者无需提供物体类别；SAM3 仍由项目生成的内部几何点提示驱动，不代表系统可以绝对穷举任意场景中的所有物体。

## Workflow

```text
读取输入目录
→ 计算 SHA-256 并跳过已完成的相同图片
→ 加载 SAM3，为所有新图生成并过滤候选
→ 保存 mask、crop 和 overlay
→ 释放 SAM3
→ 加载 Qwen3-VL
→ 逐图读取已有对象卡片和参考图
→ 批量判断候选有效性、标签与具体实例身份
→ 校验结构化输出
→ 事务化写入对象、观测、候选和决策
→ 输出运行报告
```

对象记忆在每张图片处理前重新查询，因此较早图片创建或更新的对象会参与后续图片的身份比较。

## 模型与实验环境

| 环节 | 选型 | 用途 |
|---|---|---|
| 候选分割 | SAM3 官方图像 checkpoint | 自动几何点提示下的对象 mask 候选 |
| 视觉推理 | `Qwen/Qwen3-VL-8B-Instruct-FP8` | 候选有效性、结构化标签和实例身份比较 |
| 推理框架 | Hugging Face Transformers | 单机本地推理 |
| 图像处理 | Pillow、NumPy | crop、mask、overlay 与几何后处理 |
| 数据校验 | Pydantic | 模型输出和持久化边界 |
| 持久化 | SQLite + 本地文件系统 | 结构化索引与图像资产 |

参考实验环境为 Linux、Python 3.12、PyTorch 2.10、CUDA 12.8 和单张 NVIDIA RTX 4090 24 GiB；项目使用 Conda 管理环境，模型权重和缓存保存在项目数据盘且不进入 Git。

## 数据与存储

仓库保留一套固定输入和对应的完整对象记忆输出：

```text
data/
├─ input/                  # 固定 Demo 输入
└─ memory/                 # 当前完整输出与增量记忆
   ├─ memory.sqlite
   ├─ sources/
   ├─ proposals/
   ├─ objects/
   ├─ raw_responses/
   └─ run_reports/
```

`memory.sqlite` 保存运行、源图、候选、对象、观测和决策的结构化索引；图像资产使用相对路径与数据库关联，因此整个 `data/memory/` 可以作为一个完整单元迁移和审计。

当前 `data/memory/` 已包含一次端到端运行结果。继续加入新图片时可直接增量运行；若要使用同一输入从零比较新逻辑，应指定一个新的空记忆目录，避免已有哈希使图片被跳过。

## 运行方式

运行前需要：

1. 根据 `environment.yml`、`requirements.txt` 和 `requirements-sam3.txt` 建立项目 Conda 环境。
2. 将 SAM3 checkpoint 放在 `weights/sam3/sam3.pt`。
3. 准备 Qwen3-VL 本地缓存，并让 `HF_HOME`、`HF_HUB_CACHE` 指向该缓存。

在现有记忆上处理新增图片：

```bash
export HF_HOME="$PWD/weights/qwen"
export HF_HUB_CACHE="$HF_HOME/hub"

python scripts/run_object_memory.py \
  --input-dir data/input \
  --report environment/run_report.json
```

使用相同输入从空记忆重新运行：

```bash
python scripts/run_object_memory.py \
  --input-dir data/input \
  --memory-root data/memory_fresh \
  --report environment/run_report_fresh.json
```

`scripts/check_server_env.py` 可用于检查服务器依赖、模型路径、GPU 和缓存配置。

## 项目结构

```text
Collection-Demo/
├─ config/                    # 当前业务配置
├─ data/
│  ├─ input/                  # 固定输入图片
│  ├─ memory/                 # 完整对象记忆输出
│  └─ README.md               # 数据目录契约
├─ docs/                      # 环境与业务运行说明
├─ environment/               # 当前服务器和 Demo 机器报告
├─ scripts/
│  ├─ run_object_memory.py    # 正式端到端入口
│  └─ check_server_env.py     # 服务器环境检查
├─ src/object_memory/         # 对象记忆实现
├─ tests/                     # 当前功能回归测试
├─ AGENTS.md                  # AI 协作与项目管理规范
├─ PROGRESS.md                # 当前执行状态与验收事实
├─ CHANGELOG.md               # 版本历史
├─ environment.yml
├─ requirements.txt
├─ requirements-sam3.txt
└─ pyproject.toml
```
