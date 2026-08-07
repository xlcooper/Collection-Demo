# Collection Demo 自动候选与视觉聚类优化方案

> 本文取代已删除的 `DEMO优化方案.md`，是当前“自动候选 → 视觉聚类 → 聚类语义审查”实现与后续实验的设计依据。代码实现不等于服务器验证；真实运行事实仍以 `PROGRESS.md` 为准。

面向读者的当前架构总览见 [README 架构图](README.md#架构)；本文保留配置、算法约束和验收顺序等实现级细节。

## 0. 开始运行前先配置什么

### 0.1 服务器路径

```text
仓库：/root/autodl-tmp/Collection-Demo
Conda：/root/autodl-tmp/Collection-Demo/.conda/envs/object-memory-demo
SAM3：weights/sam3/sam3.pt
Qwen缓存：weights/qwen
DINOv3：weights/dinov3/dinov3-vitb16-pretrain-lvd1689m
```

进入环境：

```bash
cd /root/autodl-tmp/Collection-Demo
conda activate "$PWD/.conda/envs/object-memory-demo"
export HF_HOME="$PWD/weights/qwen"
export HF_HUB_CACHE="$HF_HOME/hub"
```

### 0.2 DINOv3

当前固定模型与 revision：

```text
facebook/dinov3-vitb16-pretrain-lvd1689m
5931719e67bbdb9737e363e781fb0c67687896bc
```

用户已经按既定要求完成下载，不应在每次运行前重复下载。只需确认：

```bash
test -f weights/dinov3/dinov3-vitb16-pretrain-lvd1689m/config.json
```

若换新服务器且模型不存在，先在 Hugging Face 接受 Meta 访问条件并登录，再执行：

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir weights/dinov3/dinov3-vitb16-pretrain-lvd1689m
```

运行时不联网补 DINOv3 文件，不下载备用尺寸，不引入向量数据库。

### 0.3 当前配置入口

所有可调参数统一在 `config/default.yaml`：

```yaml
sam3_pipeline:
  points_per_side: 16
  points_per_batch: 32
  confidence_threshold: 0.88
  max_candidates_per_image: 24

mllm_pipeline:
  prompt_version: object-memory-cluster-review-v1
  max_clusters_per_batch: 8

visual_fingerprint:
  input_size: 512
  cluster_global_similarity_threshold: 0.75
  match_threshold: 0.75
  ambiguity_margin: 0.05
  max_cluster_representatives: 4
```

这些阈值是首轮实验起点，不是已证明最优值。修改效果参数时必须使用新记忆库并保存对应报告。

## 1. 为什么改变流程

上一版让 Qwen 先看每张图、发现目标并生成 SAM3 文本提示。真实实验暴露了两个根本问题：

1. 首轮 Qwen 漏掉或叫错的物体不会进入 SAM3；`drink cup` 等类别错误会直接变成分割召回错误。
2. Qwen 调用数量随图片数增长，且大量时间花在“告诉 SAM 要找什么”，没有充分利用视觉编码器对跨视角相似性的处理能力。

历史自动点网格实验说明 SAM3 可以无需类别词产生较高召回的候选，但也会生成大量背景、固定结构和物体零件。新方案不把这些候选直接建档，而是让脚本和 DINOv3 先整理，再让 Qwen 只审查少量视觉聚类。

## 2. 新方案的大白话版本

1. SAM3 先把所有新图片都扫一遍，尽量提出可能的物体区域，不需要 Qwen 提示类别。
2. 脚本先删掉明显重复、太小、太大或被完整区域包含的 mask。
3. DINOv3 给每个剩余区域生成“视觉指纹”，把不同图片中看起来像同一个东西的区域分组。
4. 每组挑少量代表视角，做成左边隔离物体、右边原图位置的接触表。
5. Qwen 最后看这些组，判断它是完整物体、背景/零件，还是证据不够；确认物体时再命名、描述类内特征和建立对象档案。
6. 如果记忆库已有对象，再用 DINOv3 的全局和局部视觉证据确认是否真是同一实例。Qwen 和视觉证据冲突时不强行合并。
7. 聚类中的每个有效视角都保存为 observation。视觉聚类减少的是 Qwen 审查单元，不是删除观测证据。

## 3. 总体技术流程

```text
输入文件
  │
  ├─ SHA-256 内容去重
  ▼
SAM3 自动点网格
  │  每图 16×16 正点提示，每点保留最高分 mask
  ▼
确定性候选过滤
  │  分数 / 面积 / IoU / 包含关系 / 每图上限
  ▼
DINOv3 指纹
  │  CLS 全局向量 + mask 内 patch 局部向量
  ▼
受约束跨图聚类
  │  CLS 阈值；同一 source 的两个候选禁止进同一组
  ▼
聚类解释资产
  │  成员、来源、相似度、代表视角接触表
  ▼
历史对象视觉匹配
  │  CLS 粗排 + 局部 patch 对应 + 歧义差距
  ▼
Qwen 聚类批次审查
  │  接触表 + 历史视觉证据 + 对象文字卡
  │  object / ignore / uncertain + 对象摘要 + 身份假设
  ▼
最终决定
  │  new / existing / uncertain / ignored
  ▼
SQLite + 指纹 + 接触表 + 原始响应 + 报告 + Web
```

模型顺序驻留：SAM3 完成全部图片后释放；DINOv3 完成全部候选和聚类准备后释放；Qwen 最后加载。三者不同时占用显存。

## 4. 阶段一：SAM3 自动点网格

### 4.1 输入与输出

输入是一张完整 RGB 场景图。SAM3 只接收正点提示，不接收类别词。默认网格为 `16×16`，位置落在每个网格单元中心。

每个点启用多 mask 输出，并保留质量分最高的 mask。原始候选记录：

- `raw_candidate_id`：如 `grid_point_000123`；
- `prompt`：固定为 `automatic_point_grid`，仅表示候选来源；
- 点提示质量分；
- mask 与 bbox。

### 4.2 脚本过滤

按质量分从高到低处理：

- 质量分 `< 0.88`：`low_confidence`；
- mask 为空或形状异常：明确过滤原因；
- 面积比例 `< 0.0005` 或 `> 0.5`：过小或过大；
- 与更高分 mask 的 IoU `≥ 0.9`：重复 mask；
- 自身 `≥ 90%` 被更大、更高分 mask 包含：包含候选；
- 超过每图24个：`candidate_limit`。

该过滤只依赖几何和分数，不判断“是不是物体”。背景、零件和完整物体的语义判断留给 Qwen。

## 5. 阶段二：DINOv3 指纹

### 5.1 两级视觉表示

每个候选保存：

- 全局 CLS：表示当前候选视角的整体外观，用于本轮所有候选的快速聚类和历史对象粗排；
- mask 内 patch：表示局部形状、部件、纹理和细节，用于与少量历史候选做更细的实例比较。

二者都属于视角级视觉语义，不取代对象级文字摘要。第一版不建立对象原型向量，直接保留每个历史 observation 的指纹。

### 5.2 持久化

每个 `fingerprint.npz` 保存归一化 CLS、mask 内有效 patch 和 patch 坐标。SQLite 保存路径、SHA-256、模型 ID、revision、特征层、输入尺寸、dtype 和维度，确保结果可审计。

## 6. 阶段三：跨图候选聚类

### 6.1 聚类目标

聚类不是最终身份结论，而是给 Qwen 减少重复审查的视觉假设。它回答“这些跨图候选是否值得一起看”，不直接回答“它们一定是同一实例”。

### 6.2 算法

1. 计算不同 source 候选之间的 CLS 余弦相似度；
2. 保留 `≥ 0.75` 的边并按相似度降序处理；
3. 合并两个集合前检查 source 集合是否相交；
4. 若相交则拒绝合并，防止同图相似物体自动归为一个实例；
5. 形成单成员或多成员聚类；
6. 聚类 ID 由成员 proposal ID 排序后哈希生成，便于重复审计。

### 6.3 代表视角

按成员对组内其他成员的平均相似度排序，再参考 SAM3 分数，最多选择4个代表。接触表每行保存：

- 左：中性背景上的隔离 crop；
- 右：完整原图上的候选定位；
- 标签：proposal ID、source ID 与 SAM3 分数。

接触表解决两个解释需求：隔离 crop 让 Qwen 看细节，原图上下文帮助判断它是完整对象、附属零件还是背景。

## 7. 阶段四：Qwen 聚类语义审查

### 7.1 调用单位

Qwen 不按图调用，也不按候选逐个调用，而是按最多8个聚类组成一个批次。若5张图片最后形成6个聚类，正常情况下只需要1次 Qwen；若形成17个聚类，则需要3次。

### 7.2 输入

- 聚类接触表；
- 成员 proposal/source ID 与 CLS 相似度范围；
- 该聚类对历史对象的 DINOv3 匹配证据；
- 当前活跃对象的文字摘要。

不把历史 crop、mask 或视觉向量直接放进文本上下文。历史视觉匹配由代码完成，Qwen只接收结构化分数与对象 ID。

### 7.3 输出

每个聚类必须覆盖一次且只能覆盖一次：

```text
cluster_id
verdict: object | ignore | uncertain
identity_hypothesis: new | existing | uncertain
matched_object_id
short_reason
object_summary
```

`object_summary` 只在 `verdict=object` 时存在，重点描述类内区别：对称性、轮廓比例、部件布局、材质纹理、磨损和清晰可读的品牌/型号。颜色和材质附着在具体部件上，不使用多个同名扁平“颜色”标签。

### 7.4 完整性判断

Qwen 必须过滤：背景、桌面/墙面等固定结构、阴影、反射、纹理、已有完整物体的零件和重复碎片。聚类内混入不同物体、只看到无法确认的局部或物体边界含糊时使用 `uncertain`。

## 8. 阶段五：历史身份与记忆提交

### 8.1 历史匹配

聚类中的每个成员分别和历史 observation 指纹比较：

```text
visual_score = 0.5 × CLS相似度 + 0.5 × 局部patch匹配比例
```

若一个聚类的不同成员明确匹配不同对象，整组标为歧义，不允许选择其中一个强行归档。

### 8.2 决定规则

- `object + new` 且历史无匹配：新建一个对象；
- `object + existing A` 且视觉也明确匹配 A：更新 A；
- `ignore`：全部成员记录为过滤；
- Qwen 不确定、视觉歧义或二者冲突：全部成员记录为 `uncertain`；
- 不因类别、颜色或材质相同而直接合并。

### 8.3 一组多视角如何写入

一个新聚类只创建一个 object。第一条 proposal 的 decision 记为 `new`；其余视角记为指向同一新对象的 `existing`，reason code 为 `cluster_member`。每个 proposal 都生成独立 observation，并引用自己的指纹和展示资产。

对象只保留一份当前文字摘要。crop 和 mask 不复制到对象目录，Web 通过 observation → proposal 引用展示。

## 9. Web 与最终可解释表达

Web 不再重点展示“Qwen 给 SAM 的文字提示”，因为该步骤已经不存在。页面必须能回答以下问题：

1. 输入为什么是这些图片，哪些只是内容重复；
2. SAM3 在每张图提出了多少候选，脚本为什么过滤某些候选；
3. 某个视觉聚类包含哪些 source/proposal，相似度是多少；
4. Qwen 为什么认为它是完整对象、背景/零件或不确定；
5. 它对历史对象的全局、局部和综合视觉证据是什么；
6. 最终形成哪个 object，保存了哪些 observation；
7. 原始 Qwen 回答和正式报告在哪里。

四个中间结果页对应输入整理、自动分割、视觉聚类和语义审查。最终对象卡保留单一对象摘要与观测时间线；候选血缘连接 `source → proposal → cluster → decision → object/observation`。

自动分割页按源图组织默认折叠的结果块，摘要只显示原始、保留和脚本过滤数量，展开后展示该图在SAM阶段保留并送入DINOv3的全部候选。该阶段是不可被后续结果改写的过程快照：Qwen后续将某个候选判为忽略、接受或不确定，都不能改变02页的候选数量、卡片或状态标签。视觉聚类页在顶部统一解释同图约束、`clu_…`、`prop_…`和CLS最低/平均/最高值，各聚类卡不重复相同原理说明；Qwen及最终状态只在后续阶段解释。

旧 Web 中“5张输入显示10行”的问题来自进度事件按 `source_id`、报告按 `input_path` 建键。新页面用两者的别名表合并同一图片。

## 10. 代码模块职责

| 模块 | 职责 |
|---|---|
| `sam3_adapter.py` | 自动点网格、分批调用 `predict_inst`、输出 CPU 候选 |
| `sam3_postprocess.py` | 几何过滤、同图去重和候选展示资产 |
| `dinov3_adapter.py` | CLS/patch 指纹提取、落盘和历史单视角匹配 |
| `candidate_clustering.py` | 跨 source 约束聚类、聚类证据、接触表 |
| `cluster_review.py` | Qwen 聚类批次消息、schema 校验和对象卡约束 |
| `pipeline.py` | 三模型顺序阶段、进度、报告和最终身份协调 |
| `memory_loop.py` / `memory_store.py` | 聚类事务、对象/观测/决策持久化 |
| `web_service.py` / `web_static/` | 运行管理、报告读取和解释展示 |

旧的 `scene_guidance.py` 和 `identity_decision.py` 已退出并删除。当前流程不应重新引入 Qwen→SAM 文本提示，除非后续实验明确证明自动候选路线不可用并由用户重新决策。

## 11. 报告与审计契约

报告 schema 为8。除每图候选外，必须包含 `clusters[]`：

```text
cluster_id
member_proposal_ids / source_ids
representative_proposal_ids
global_similarity.min / mean / max
contact_sheet
historical_visual_evidence
qwen_review
raw_response
final_decision
object_id
error
```

Qwen 原始响应按 `raw_responses/<run>/<cluster_batch>/response.json` 保存，包含本批次全部聚类输入、对象卡 ID、结构化解析结果、原始文本、token、耗时和错误。

结构 `passed` 只表示运行完成且引用完整，不表示语义效果通过。

## 12. 实验验收顺序

### 12.1 服务器回归

```bash
PYTHONPATH="$PWD/src" python -m unittest discover -s tests -v
```

### 12.2 固定输入结构实验

新建空白记忆库，检查：

- 三模型确实顺序加载和释放；
- 原始点网格、过滤候选、指纹、聚类和接触表完整；
- Qwen 调用数等于聚类批次数，而不是图片数；
- 正确对象是否形成、观测是否覆盖多个视角；
- 背景和零件是否被过滤；
- Web、SQLite、报告和全部资产引用完整。

### 12.3 同类实例实验

后续数据至少包含：同一实例多视角、两个以上外观相近的同类不同实例、对称与人体工学鼠标等结构差异、品牌清晰/不可见视角、遮挡和低信息视角。

重点统计：

- SAM3 真实物体候选召回；
- DINOv3 聚类的误合并与误拆分；
- Qwen 完整对象/零件/背景判断；
- 历史对象匹配的同实例召回与异实例误合并；
- `uncertain` 是否出现在真实证据不足处；
- 聚类批处理相对旧逐图 Qwen 的耗时变化。

只有完成这组实验，才能调整并声明默认聚类与历史匹配阈值。

## 13. 当前明确不做

- 区域级空间语义、场景关系和变化环境；
- 主动视角规划、机械臂控制和世界模型；
- DINOv3 微调、度量学习或对象原型向量；
- 文本向量、向量数据库和跨模态投影器；
- 自动重试、备用模型、静默降级和第二次 Qwen 复核；
- 用类别、颜色或材质替代具体实例证据。
