# 第一阶段进度看板

> **维护规则（强制）**：本文件是可反复查看的开发进度事实源。每次收到服务器结果、推进里程碑或改变实现决策后，本地 AI 必须同步更新状态、证据、完成条件和下一动作。没有报告或用户明确确认时不得标记为已通过；聊天结论与已回传报告冲突时以报告为准。过时状态必须替换，不以追加说明掩盖。

## 当前快照

- **版本**：v0.3.2
- **当前关卡**：M0——模型冒烟验证
- **刚完成**：SAM3 使用与测试图像匹配的具体单数类别提示完成真实推理
- **下一动作**：运行 Qwen3-VL 单模型冒烟测试
- **进入 M1 的剩余条件**：Qwen 冒烟通过、双模型顺序运行通过、最新报告回传

状态说明：✅ 已完成且有依据；🟡 正在进行或执行通过但证据待回传；⬜ 尚未开始；⛔ 被阻塞。

## 里程碑

| 里程碑 | 状态 | 最小交付物 / 完成条件 | 当前证据 | 下一动作 |
|---|---|---|---|---|
| P0 项目基线 | ✅ | 第一阶段范围、开发准则、Git 与环境流程固化 | `README.md`、环境声明和执行指南 | 持续维护，不扩展阶段范围 |
| M0-A 服务器环境 | ✅ | Conda、CUDA、SAM3 及 Qwen 依赖可用，0 个阻塞项 | `environment/server_env_report.json`；最新报告为 `warning`，0 blocker | 模型冒烟后重新自检磁盘余量 |
| M0-B SAM3 单模型冒烟 | 🟡 | 真实加载 checkpoint，输出至少一个 proposal 和 mask，并记录耗时/显存 | 用户已明确确认具体类别提示运行通过；通过报告尚未 push | 将 `environment/sam3_smoke_report.json` 回传 Git |
| M0-C Qwen 单模型冒烟 | 🟡 | 8B FP8 加载成功，输出可解析的约定 JSON，并记录耗时/显存 | 权重已下载，尚无推理报告 | 运行 `scripts/smoke_qwen.py` |
| M0-D 双模型顺序运行 | ⬜ | 同一环境中先后完成 SAM3、Qwen，且无显存或磁盘阻塞 | 单模型尚未全部通过 | M0-B、M0-C 固化后运行顺序测试 |
| M1 单图对象流水线 | ⬜ | 输入一张场景图，保存 SAM3 候选、mask、crop，并为候选生成结构化标注 | 尚未开始 | M0 全部通过后实现最短端到端路径 |
| M2 显式对象记忆 | ⬜ | 图像资产落盘，SQLite 建立对象与观测条目，可新增和读取 | 已确定“文件系统 + SQLite” | 在 M1 输出稳定后实现最小 schema |
| M3 去重与持续归并 | ⬜ | 新观测可判定为新对象或归并至已有对象，不重复建条目 | 尚未开始 | 先以可解释候选检索 + MLLM 判断实现 MVP |
| M4 批量自动化 | ⬜ | 连续处理多张图，单个对象失败不破坏已有记忆，并生成运行摘要 | 尚未开始 | M3 通过后串联现有模块 |
| M5 第一阶段验收 | ⬜ | 无人工标注地完成“发现—标注—存储—去重—扩充”的可演示闭环 | 尚未开始 | 用固定小型场景集做服务器验收 |

## 当前关键判断

1. SAM3 的模型加载、BF16 推理和 mask 输出链路已经跑通；通过报告仍需回传，才能固化精确提示、proposal 数、耗时和显存证据。
2. 通用提示 `object` 在当前测试图像上返回 0 proposals，不能据此宣称“全量发现物体”。M1 必须先确定一个最小、可复现的全量候选发现策略。
3. Qwen3-VL-8B-Instruct-FP8 是否能在 RTX 4090 上按当前 Transformers 方案稳定加载和输出 JSON，仍是进入 M1 前的主要未知项。
4. 系统 `nvcc` 11.8 只影响可选源码扩展；当前 MVP 不编译这些扩展，因此不是阻塞项。

## 现在执行

服务器拉取 v0.3.2 后，直接运行 Qwen 冒烟：

```bash
git pull origin main
conda activate "$PWD/.conda/envs/object-memory-demo"
export OMP_NUM_THREADS=8
export HF_HOME="$PWD/weights/qwen"
export HF_HUB_CACHE="$HF_HOME/hub"

python scripts/smoke_qwen.py \
  --image data/smoke/scene.jpg \
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
  --output-dir runs/smoke/qwen \
  --report environment/qwen_smoke_report.json
```

通过后再运行双模型顺序测试；命令和报告回传方式见 `docs/03_服务器环境与冒烟测试指南.md`。

## 第一阶段边界

本看板不加入认知完整度、缺失视角分析、主动视角规划、机械臂控制、空间地图、因果推理或完整世界模型；这些均不属于第一阶段验收。
