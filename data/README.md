# `data/` Demo 数据目录

`data/` 保存固定 Demo 输入和运行时生成的完整对象记忆。当前仓库保留同一输入以及三次独立正式实验的记忆根；每个记忆根都是不可拆分的审计单元，实验结论见根目录 `PROGRESS.md`。更早的已清理产物只能从 Git 历史恢复，不能把历史摘要复制成当前结果。

## 目录结构

```text
data/
├─ input/
├─ memory/                  # v0.16.1 失败实验，默认记忆根
├─ memory_v017_01/          # v0.17.0 失败实验，独立比较根
└─ memory_v017_02/          # v0.17.4 失败实验，独立比较根
```

三个记忆根内部都包含 `memory.sqlite`、`sources/`、`proposals/`、`objects/`、`raw_responses/` 和 `run_reports/`。

### `input/`

固定 Demo 输入集，共28个文件：

- 14张内容唯一的图片；
- 14个逐字节完全相同的副本；
- 文件名不向模型暗示类别或分组；
- 副本用于验证 SHA-256 幂等门。

### 记忆根

`config/default.yaml` 默认使用 `memory/`；`memory_v017_01/` 和 `memory_v017_02/` 是通过 `--memory-root` 选择的独立实验根。任意空记忆根首次运行时都会创建以下内容：

| 路径 | 内容 |
|---|---|
| `memory.sqlite` | 运行、源图、候选、对象、观测和决策索引 |
| `sources/` | 按 SHA-256 保存的规范源图副本 |
| `proposals/` | 候选的 `crop.png`、`mask.png` 和 `overlay.jpg` |
| `objects/` | 每个对象的累计观测资产 |
| `raw_responses/` | 两轮 Qwen 的原始响应、解析结果、token 和耗时 |
| `run_reports/` | 记忆库内部运行报告 |

数据库中的资产路径均相对记忆根，因此数据库和上述目录必须作为一个整体移动、版本化或清理，不能只删除数据库或某一类图片。

## 使用规则

- 当前 `memory/`、`memory_v017_01/` 与 `memory_v017_02/` 都是已保留的失败实验与审计证据，不得局部删除、覆盖或直接当作干净实验起点。
- 全部相关 source 已完成的记忆根才适合增量运行：相同哈希的已完成图片会跳过，新图片继续写入同一对象记忆。
- failed 或中断的 source 只能在原 run 内恢复；新 run 不能接管其他 run 的未完成哈希。
- 重做、修复后复验或并排比较时，使用独立空记忆根，例如 `data/memory_compare/`。
- 新记忆必须完整审计后，才能作为当前正式结果保留。
- 模型权重、Hugging Face 缓存、Conda 环境和隐藏真值不进入 Git。
- `.ipynb_checkpoints/` 不属于 Demo 数据。

实验事实与审计结论见根目录 `PROGRESS.md`；当前协作方向见 `AGENTS.md`。
