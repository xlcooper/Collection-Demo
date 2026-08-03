# `data/` Demo 数据目录

`data/` 保存固定输入和运行时生成的完整对象记忆。当前只保留固定输入；下一次实验会从空白状态创建默认记忆库。

## 当前结构

```text
data/
└─ input/                  # 固定 Demo 输入，28个文件
```

`input/` 中有14张内容唯一的图片和14个字节完全相同的副本。副本用于验证 SHA-256 幂等门，文件名不会向模型提示类别或分组。

Web 实验台直接管理这个目录：上传会先检查文件名、大小、像素数、扩展名与真实图片格式，并拒绝覆盖同名文件；删除只移除明确选择的 `data/input/` 图片，不会回删已经形成的对象记忆。端到端实验运行期间，上传和删除会被锁定。

默认运行后会创建：

```text
data/memory/
├─ memory.sqlite
├─ sources/
├─ proposals/
├─ objects/
├─ raw_responses/
└─ run_reports/
```

| 路径 | 内容 |
|---|---|
| `memory.sqlite` | 运行、源图、候选、对象、观测和决定的索引 |
| `sources/` | 按 SHA-256 保存的处理用原图 |
| `proposals/` | 候选的 `crop.png`、`mask.png` 和原色场景加定位框的 `overlay.jpg` |
| `objects/` | 每个长期对象的观测图片；observation描述当次视图，对象卡保存稳定累计信息 |
| `raw_responses/` | 两轮 Qwen 的原始 JSON 回答、token 和耗时 |
| `run_reports/` | 记忆库内部运行报告 |

## 使用规则

- SQLite 中保存的是相对路径，所以数据库和上述目录必须作为一个整体移动、提交或清理。
- 默认 `data/memory/` 不存在时，程序会自动新建；全部相关图片成功完成后，才适合继续增量加入新图。
- 失败实验要重做时，应清理整个记忆根，或改用另一个全新的记忆根，不能只删除数据库或一类图片。
- 模型权重、Hugging Face 缓存、Conda 环境和隐藏人工参照不进入 Git。
- Web 的进度事件、进程状态和日志位于本地忽略的 `temp/web_runs/`；它们用于页面刷新后的运行状态恢复，不属于可迁移对象记忆。正式结果仍以完整记忆根和 `environment/run_report.json` 为准。

之前三次实验的记忆根已在 v0.18.0 按用户授权删除。删除前完整快照可从 Git 提交 `f413654` 恢复；实验摘要见根目录 `PROGRESS.md`。
