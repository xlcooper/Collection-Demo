# `data/` Demo 数据目录

`data/` 只保留当前 Demo 的固定输入和一套完整对象记忆输出。历史 gate、smoke、阶段候选与重复暂存目录已从当前工作树删除，可在 Git 历史中追溯。

## 最终结构

```text
data/
├─ input/
└─ memory/
   ├─ memory.sqlite
   ├─ sources/
   ├─ proposals/
   ├─ objects/
   ├─ raw_responses/
   └─ run_reports/
```

### `input/`

固定 Demo 输入集，共28个文件：

- 14张唯一图片；
- 14个逐字节完全相同的副本；
- 文件名采用无语义编号，不向模型暗示物体类别或分组；
- 副本用于验证 SHA-256 幂等门，与普通冗余文件不同。

### `memory/`

当前端到端运行形成的完整对象记忆，也是 `config/default.yaml` 的默认记忆根目录。

| 路径 | 内容 |
|---|---|
| `memory.sqlite` | 运行、源图、候选、对象、观测与决策索引 |
| `sources/` | 按 SHA-256 保存的规范源图副本 |
| `proposals/` | 候选的 `crop.png`、`mask.png` 与 `overlay.jpg` |
| `objects/` | 每个对象的累计观测资产 |
| `raw_responses/` | Qwen 原始响应、解析结果、token 与耗时 |
| `run_reports/` | 记忆库内部运行报告 |

数据库中的资产路径均相对于 `data/memory/`，因此数据库和上述目录必须作为一个整体移动、版本化或清理，不能单独删除某类图片。

当前输出由目录整理前的一次服务器运行生成。原始报告中的 `input_path` 仍记录当时的 `data/m5_input/` 路径；报告内容保持原样以保留执行事实，现有输入文件已原样迁入 `data/input/`，文件哈希未改变。

## 使用规则

- 在现有 `data/memory/` 上运行时，已完成的相同哈希图片会直接跳过，新增图片会继续写入同一对象记忆。
- 使用同一输入从零比较新逻辑时，必须指定一个新的空记忆根，例如 `data/memory_fresh/`。
- 运行产生的新记忆根应完整检查后再决定是否替换当前 `data/memory/`。
- 模型权重、Hugging Face 缓存、Conda 环境和根目录隐藏真值不进入 Git。
- 编辑器生成的 `.ipynb_checkpoints/` 不属于 Demo 数据。

实验效果、当前问题和下一步以根目录 `PROGRESS.md` 为准。
