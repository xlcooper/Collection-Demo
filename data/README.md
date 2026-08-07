# `data/` 数据契约

`data/` 只负责保存 Demo 输入和运行时生成的完整对象记忆。当前实验是否已经运行、效果如何，统一见根目录 [PROGRESS.md](../PROGRESS.md)。

## 目录结构

```text
data/
├── input/                       # Web 与 CLI 共用的场景图片
├── memory/                      # CLI 默认记忆库（按需创建）
└── memory_<内部编号>/           # Web 创建的独立记忆库，可保留多份
    ├── memory_info.json         # Web 显示名称与服务器内部编号
    ├── memory.sqlite
    ├── sources/
    ├── proposals/
    ├── objects/
    ├── raw_responses/
    └── run_reports/
```

`input/` 支持 JPG、JPEG、PNG 和 WebP。程序按文件原始字节计算 SHA-256；文件名不同但内容完全相同的图片只处理一次。Web 上传会检查文件名、大小、像素数、扩展名和真实格式，并拒绝覆盖同名文件；可以逐张删除或一键清空当前可识别的输入图片，均不会回删已经形成的对象记忆。实验运行期间不允许修改输入。

## 记忆库内容

| 路径 | 内容 |
|---|---|
| `memory.sqlite` | run、源图、候选、对象、观测和决策的结构化索引 |
| `sources/` | 按 SHA-256 保存的处理用原图 |
| `proposals/` | 候选的 `crop.png`、`mask.png`、`overlay.jpg` 和 `fingerprint.npz` |
| `objects/` | 保留的目录边界；新 schema 不再复制候选图片到对象目录 |
| `raw_responses/` | 每张唯一新图唯一一次 Qwen 回答、token 与耗时 |
| `run_reports/` | 该记忆库内每次运行的报告 |

SQLite 保存相对路径，因此数据库与上述资产目录必须作为一个整体移动、提交或清理。`environment/run_report.json` 是最近一次 Web/CLI 运行的外部报告副本，不替代记忆库内的报告。

对象表只保存一份当前结构化文字摘要。每条 observation 引用原 proposal 的 crop、mask 和视觉指纹，不重复保存图片或逐视角文字描述；页面通过这些引用展示跨视角时间线，历史自动匹配只读取 `fingerprint.npz`。

## 状态与管理规则

- 默认 `data/memory/` 不存在时，CLI 会自动创建；Web 新建库使用服务器生成的安全内部编号，页面名称不会成为服务器路径。
- 当前 DINOv3 工作流使用 SQLite schema v3。旧 schema v2 记忆库保留原样并在 Web 中只读展示，不自动迁移；新实验应从页面新建空白库。
- 只有相关 run、source 和候选都已成功完成的记忆库才允许继续加入新图片；失败、不完整或不可读的库在 Web 中只供复核和整库删除。
- 空白库的首次 Web 运行可以执行全新 Demo 覆盖检查；已有库的增量运行不重复套用只适用于首次实验的覆盖门槛。
- Web 删除以整库为单位，先隔离目标库并提交选择状态，再清理文件；写盘失败时恢复原库。不能只删除数据库或某一类图片。
- Web 的短期进度、进程状态和日志位于 Git 忽略的 `temp/web_runs/`，不属于可迁移对象记忆。
- 模型权重、Hugging Face 缓存、Conda 环境和隐藏人工参照不进入 Git；正式实验的输入、完整记忆库和对应报告进入 Git。

运行入口、参数和增量实验方式见[业务代码运行与验证指南](../docs/04_业务代码运行与验证指南.md)。
