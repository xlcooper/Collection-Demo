# `runs/` 验证产物目录说明

`runs/` 保存环境冒烟、SAM3 候选校准和分阶段验证产生的过程资产。它与 `data/<memory_root>/` 的正式对象记忆库不同：前者主要服务单项复现和问题定位，后者保存端到端对象记忆状态。

当前 Demo 数据规模有限，本目录全量进入 Git。这样本地可以检查服务器实际生成的候选图、mask、overlay、模型响应和历史实验目录，并依据代码引用判断哪些仍是最终 Demo 所需依赖、哪些只是可以清理的过往测试遗迹。

已知代码入口包括：

| 路径模式 | 生成入口 | 主要内容 |
|---|---|---|
| `runs/smoke/` | `scripts/run_server_smoke_tests.sh`、`smoke_sam3.py`、`smoke_qwen.py` | SAM3/Qwen 冒烟输出、响应和可视化 |
| `runs/m2/` | `scripts/verify_m2.py` | 候选 crop、mask、overlay，供 M2 检查并可被 M3 复用 |
| `runs/m3/` | `scripts/verify_m3.py` | 空记忆与同视图身份验证的 Qwen 原始响应 |
| `runs/m5_auto_candidates_v2/` | `scripts/verify_m2.py` 的 M5-A 校准命令 | 无类别点网格候选及可视化 |

以上只依据当前代码入口说明目录类型，不代表服务器一定只有这些目录，也不代表所有历史内容都应保留。服务器完成首次全量提交后，应进一步建立实际文件清单，并按以下四类审计：

1. 最终 Demo 运行所需；
2. 验收或问题解释所需；
3. 可由代码稳定再生、且不再需要的历史产物；
4. 来源或依赖暂不能确认。

在首次快照完成前不删除任何未知目录。当前验收状态仍以 `PROGRESS.md` 和 `environment/*.json` 为准，不能依据 `runs/` 是否存在判断通过。
