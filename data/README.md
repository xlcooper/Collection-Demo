# `data/` Demo 数据目录

`data/` 保存固定 Demo 输入和运行时生成的一套完整对象记忆。原有运行记录已按用户授权清理；当前仓库只保留 `input/`，`memory/` 会在下一次运行时重新创建。需要恢复旧产物时应使用 Git 历史，不能把历史摘要复制成当前结果。

## 目录结构

```text
data/
├─ input/
└─ memory/                  # 运行后生成
   ├─ memory.sqlite
   ├─ sources/
   ├─ proposals/
   ├─ objects/
   ├─ raw_responses/
   └─ run_reports/
```

### `input/`

固定 Demo 输入集，共28个文件：

- 14张内容唯一的图片；
- 14个逐字节完全相同的副本；
- 文件名不向模型暗示类别或分组；
- 副本用于验证 SHA-256 幂等门。

### `memory/`

`config/default.yaml` 默认使用的记忆根。首次运行会创建以下内容：

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

- 当前默认记忆根没有历史哈希；下一次运行会处理14张唯一图片，并在模型调用前跳过14张逐字节副本。
- 成功运行后，相同哈希的已完成图片会直接跳过，新图片会继续写入同一对象记忆。
- 并排比较另一组逻辑时，使用独立空记忆根，例如 `data/memory_compare/`。
- 新记忆必须完整审计后，才能作为当前正式结果保留。
- 模型权重、Hugging Face 缓存、Conda 环境和隐藏真值不进入 Git。
- `.ipynb_checkpoints/` 不属于 Demo 数据。

实验事实与审计结论见根目录 `PROGRESS.md`；当前协作方向见 `AGENTS.md`。
