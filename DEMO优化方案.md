# Collection Demo 优化方案

> 本文是 v0.21.0 实现依据与后续实验验收书。单次 Qwen、DINOv3 视觉指纹、新对象摘要和 Web 展示已进入代码；当前正式实验事实仍以 `PROGRESS.md` 为准，上述新能力尚未在服务器回归测试或正式实验中验证。

## 0. 开始开发前先准备什么

### 0.1 先确认服务器与磁盘

服务器验证仍在现有 AutoDL 项目环境中完成：

```text
仓库：/root/autodl-tmp/Collection-Demo
Conda：/root/autodl-tmp/Collection-Demo/.conda/envs/object-memory-demo
GPU：RTX 4090 24 GiB
```

最近一次服务器报告只剩约 23.41 GiB 数据盘空间。下载前先执行：

```bash
cd /root/autodl-tmp/Collection-Demo
conda activate "$PWD/.conda/envs/object-memory-demo"
df -h .
```

本方案只选择一个 DINOv3 模型，不同时下载多个尺寸做备用。模型权重继续放在 `weights/`，不进入 Git。

### 0.2 申请并下载 DINOv3

第一版使用 Meta 官方的 DINOv3 ViT-B/16：

```text
模型：facebook/dinov3-vitb16-pretrain-lvd1689m
revision：5931719e67bbdb9737e363e781fb0c67687896bc
目录：weights/dinov3/dinov3-vitb16-pretrain-lvd1689m
```

选择 ViT-B/16 是为了先在 24 GiB 单卡和当前磁盘条件下完成最小验证，不提前引入 ViT-L、ViT-H+ 或 7B 版本。该模型需要在 Hugging Face 页面接受 Meta 的访问条件：

- [DINOv3 ViT-B/16 模型页](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
- [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)
- [DINOv3 论文](https://arxiv.org/abs/2508.10104)

取得访问权限后在服务器执行：

```bash
hf auth login

mkdir -p weights/dinov3

hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --revision 5931719e67bbdb9737e363e781fb0c67687896bc \
  --local-dir weights/dinov3/dinov3-vitb16-pretrain-lvd1689m
```

当前环境已有 `transformers==4.57.6`、PyTorch、Pillow 和 NumPy，官方 DINOv3 已由该版本 Transformers 支持。第一版先复用现有依赖，不新增训练框架、向量数据库或服务端推理框架。

### 0.3 当前已加入的配置

`config/default.yaml` 已加入以下明确配置，路径由配置读取，不散落在代码里：

```yaml
models:
  dinov3_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
  dinov3_model_path: weights/dinov3/dinov3-vitb16-pretrain-lvd1689m
  dinov3_revision: 5931719e67bbdb9737e363e781fb0c67687896bc

visual_fingerprint:
  input_size: 512
  storage_dtype: float16
  similarity_metric: cosine
  global_feature: cls_token
  local_feature: patch_tokens
```

`input_size: 512` 是首轮实验起点，不是已经验证的最优值。后续只能依据同类实例实验调整。

### 0.4 开发开始前还要准备的数据

现有固定输入只有鼠标、水瓶、饮料杯三个具体实例，不能验证类内身份辨识。用户后续采集的新验证集至少需要：

- 同一物体的多个视角；
- 两个或以上外观相近的同类不同实例；
- 相近颜色、材质但结构不同的物体，例如对称鼠标与人体工学鼠标；
- 品牌或标识清晰可见和不可见的视角；
- 遮挡、背面和低信息视角。

这些图片进入新实验前应建立隐藏真值，明确每张图中的具体实例 ID。当前先完成方案和实现，不据现有三实例数据宣称类内辨识有效。

## 1. 方案概述

### 1.1 优化前基线

本次优化替换的旧流程是：

```text
Qwen 首轮发现目标并生成 SAM3 文本提示
→ SAM3 分割
→ Qwen 第二轮查看 crop、上下文和对象卡，决定 new/existing/uncertain
→ 写入对象卡和逐视角描述
```

最近一次正式实验中，Qwen 首轮 4 次调用，第二轮 11 次调用，共 15 次。当前对象身份主要依靠第二轮 Qwen 的视觉比较，描述偏类别共有特征，尚未验证相似同类实例。

### 1.2 已实现的核心思路

用大白话说：

- Qwen 负责发现、命名、描述和迭代对象的文字档案；
- SAM3 负责把目标从场景里分割出来；
- DINOv3 负责为每个视角生成视觉指纹并比较是否为同一实例；
- 每张唯一新图只调用一次 Qwen，不再进行第二轮 Qwen；
- 每个对象卡只保留一份持续更新的对象级文本，不再展示多条重复观测描述；
- crop 和 mask 只在生成视觉指纹时使用一次，之后只作为对象卡和审计页面的视觉证据；
- 区域级空间语义、变化环境、三维关系和主动采集不在本次优化范围内。

当前代码流程是：

```text
图片哈希去重
→ Qwen 查看当前完整图片和已有对象文字摘要（每图一次）
→ 输出目标、SAM3 指令、当前可见事实、身份假设和候选汇总文本
→ SAM3 分割完整对象
→ DINOv3 生成全局与局部视觉指纹
→ 视觉指纹与历史视角比较
→ VLM 假设和视觉证据一致时提交 new/existing
→ 冲突或证据不足时写入 uncertain，不再调用 Qwen
→ 更新 SQLite、对象摘要、视觉指纹和展示资产
```

## 2. 本次优化的目标与边界

### 2.1 必须解决

1. 对象描述从类别共有特征扩展到类内身份特征。
2. 颜色、材质等属性必须与具体部件关联，不能只保存扁平颜色列表。
3. 每个对象只维护一份对象级汇总文本，并随确认的新视角迭代。
4. 使用 DINOv3 视觉指纹完成跨视角图像比较，不再让历史 crop 参与自动匹配。
5. 删除第二轮 Qwen；每张新图只允许一次 Qwen 调用。
6. 同类不同实例证据不足时使用 `uncertain`，不得仅凭类别、颜色或材质归并。
7. 候选、指纹、决定、对象和资产仍保持可追踪。

### 2.2 本次明确不做

- 区域级空间语义及固定位置关系；
- 变化环境和物体位置变化建模；
- 三维地图、相机位姿、深度和场景图；
- 主动视角规划、机械臂控制和世界模型；
- DINOv3 微调、度量学习或自建训练集训练；
- 文本向量持久化和向量数据库；
- 对象原型向量；第一版直接比较对象的历史视角指纹；
- 自动重试、拆单救援、备用模型、静默降级或第二次 Qwen 复核。

## 3. 两类长期记忆表示

本次只实现“对象级文本语义”和“视角级视觉语义”。二者互补，不相互替代。

### 3.1 对象级文本语义

每个对象只保存一份当前汇总档案。建议结构如下：

```json
{
  "object_name_zh": "人体工学无线鼠标",
  "coarse_category": "电子设备",
  "fine_category": "鼠标",
  "stable_description": "银灰色人体工学鼠标，整体非左右对称，右侧轮廓隆起，中央有滚轮和左右按键区域。",
  "stable_identity_features": [
    "整体非左右对称",
    "右侧轮廓隆起",
    "中央滚轮两侧有黑色分隔线"
  ],
  "brand_or_markings": [],
  "part_appearance": [
    {
      "part": "外壳",
      "color": ["银灰色"],
      "material": ["塑料"]
    },
    {
      "part": "滚轮",
      "color": ["黑色"],
      "material": ["橡胶"]
    }
  ],
  "summary_confidence": 0.0
}
```

设计要求：

- 类别只是档案的一部分，不能用“有滚轮和左右按键”替代类内特征；
- 品牌、型号和文字只有清晰可见时才记录，不凭外形猜测；
- 稳定身份特征优先描述轮廓、非对称结构、比例、部件布局、纹理、标识和独特损伤；
- `part_appearance` 解决当前“透明、白色”无法说明分别属于瓶身和瓶盖的问题；
- 不写入桌面、邻近物体、位置、对象 ID、匹配结论或上下文框颜色；
- 当前视角看不见某个特征，不代表该特征不存在；
- 新证据与旧摘要冲突时不直接覆盖为新事实，最终决定应为 `uncertain` 或保留旧摘要。

第一版只存结构化文本，不持久化文本嵌入。未来如果增加自然语言检索，再从当前对象档案生成文本向量。

### 3.2 视角级视觉语义

每条有效观测保存一份 DINOv3 视觉指纹：

1. **全局指纹**：使用归一化 CLS token，表示当前视角的整体实例外观，用于快速排列最相似的历史视角。
2. **局部指纹**：保存 mask 内有效 patch token，用于比较轮廓、部件布局、纹理和局部结构。

第一版不建立对象原型向量。对象得分直接由当前候选与该对象所有历史视角的比较结果得到，避免平均向量抹掉特定视角的区分特征。

DINOv3 不负责输出类别名称、品牌文字或最终对象描述。它只提供图像对图像的视觉证据。DINOv3 官方论文虽包含实例检索和跨视角局部对应实验，但没有证明相似日用品的具体实例识别，因此必须用用户后续采集的数据验证。

## 4. 单次 Qwen 的文本流程

### 4.1 为什么改成每张图一次

当前首轮每批最多处理 4 张图片，但对象摘要只有在一张图片完成身份确认后，才能作为下一张图片的输入。为了让第二个视角真正承接第一个视角的结果，优化后必须按唯一源图顺序处理：

```text
第1张图：Qwen → SAM3 → DINOv3 → 提交对象摘要
第2张图：Qwen读取第1张图后的对象摘要 → SAM3 → DINOv3 → 再提交
……
```

因此未来配置中的 `scene_batch_size` 应从 4 改为 1。这个变化不是错误救援，而是状态依赖带来的正式流程调整。每张图仍只调用一次 Qwen，不重试。

### 4.2 Qwen 的输入

每次调用只提供：

- 当前完整源图；
- 当前记忆库中每个活动对象的 `object_id`、类别和对象级汇总文本；
- 输出 schema 和完整物体选择规则。

不提供：

- DINOv3 向量或 patch token；
- 历史 crop、mask、overlay；
- 区域空间关系；
- 自动生成的相似对象结论。

当前记忆库很小，第一版可以提供全部活动对象的紧凑文本档案，不提前设计文本向量检索和复杂卡片裁剪。

### 4.3 Qwen 的输出

每个可见独立物体实例必须有一个独立 target。即使同图出现两个鼠标，也要输出两个 target；相同 `sam_text_prompt` 可以在 SAM3 阶段去重执行。

建议输出结构：

```json
{
  "source_id": "src_...",
  "targets": [
    {
      "target_id": "target_001",
      "object_name_zh": "银灰色人体工学鼠标",
      "sam_text_prompt": "computer mouse",
      "current_view_facts": {
        "category": "鼠标",
        "visible_identity_features": [
          "整体非左右对称",
          "右侧轮廓隆起"
        ],
        "brand_or_markings": [],
        "part_appearance": []
      },
      "identity_hypothesis": "existing",
      "matched_object_id": "obj_...",
      "identity_short_reason": "当前可见非对称轮廓与已有档案一致",
      "proposed_object_summary": {},
      "temporary_target_anchor": [0.0, 0.0, 1.0, 1.0]
    }
  ]
}
```

字段职责：

- `sam_text_prompt`：只给 SAM3，不作为对象事实；
- `current_view_facts`：本次图片中确实可见的事实，只保存在原始响应和运行报告中，不作为对象卡的第二份长期描述；
- `identity_hypothesis`：Qwen 根据当前图和历史文字做出的候选假设，不是最终决定；
- `proposed_object_summary`：假设成立时要提交的新对象摘要；
- `temporary_target_anchor`：只用于把同类多实例的 Qwen target 与 SAM3 mask 对齐，完成本轮后不作为区域语义写入长期记忆。

`temporary_target_anchor` 是临时工程关联，不属于本方案已经延后的区域级空间语义。

### 4.4 对象摘要如何迭代

第一视角、记忆为空时：

- Qwen 输出 `new` 假设和第一版对象摘要；
- DINOv3 确认没有历史视觉匹配后创建对象。

后续视角进入时：

- Qwen 读取旧对象摘要；
- 输出当前可见事实和合并后的候选摘要；
- DINOv3 确认同一实例后才提交该摘要；
- 若 Qwen 与 DINOv3 结论冲突，保持旧摘要不变并记录 `uncertain`。

这样对象卡始终只有一份最新文本。旧视角事实仍可从 Qwen 原始响应和运行报告追踪，但不再作为七八条重复描述出现在对象卡或 observation 主字段中。

## 5. SAM3 与同类多实例关联

### 5.1 SAM3 保持的职责

SAM3 仍接收 Qwen 生成的简短英文完整物体类别，当前阈值和几何后处理先保持不变。相同图片中相同的 `sam_text_prompt` 只执行一次查询，SAM3 可以返回多个实例 mask。

### 5.2 target 与 mask 的一对一关联

为支持同类多实例：

1. Qwen 为每个可见实例输出独立 target 和临时框；
2. SAM3 为同一类别返回多个 proposal；
3. 使用临时框与 proposal bbox 的重合度进行一对一匹配；
4. 无法形成唯一对应时，该 proposal 进入 `uncertain`，不重新调用 Qwen。

临时框不进入对象卡，不用于长期位置记忆。

### 5.3 删除第二轮 Qwen 后的边界

旧流程中的第二轮 Qwen 还承担候选完整性和附着部件判断。删除后，这部分由以下证据共同承担：

- 首轮 Qwen 只提出值得建档的完整独立物体；
- SAM3 文本指令使用完整物体基础类别；
- 现有面积、重复、包含关系和每图数量过滤继续保留；
- target 与 proposal 必须完成明确关联。

第一版不增加另一个 VLM、分类器或自动救援来替代第二轮 Qwen。真实实验若证明某类错误重新出现，再根据证据决定下一步。

## 6. DINOv3 视觉指纹实现

### 6.1 输入

DINOv3 使用当前 proposal 的 `crop.png` 和 `mask.png` 生成一次视觉指纹：

- crop 保留物体原始像素；
- mask 外继续使用固定中性灰背景；
- mask 缩放到 patch 网格后，只保存达到有效覆盖比例的 patch token；
- 输入预处理、尺寸和归一化必须固定并写入配置与报告。

完成指纹生成后，历史自动匹配只读取指纹文件，不再次读取 crop 或 mask。

### 6.2 输出与存储格式

每个有效 proposal 生成一个 `.npz` 文件，建议包含：

```text
global_embedding     float16 [D]
local_embeddings     float16 [N, D]
local_patch_indices  int32   [N, 2]
```

SQLite 或伴随元数据还要记录：

- 模型 ID 与 revision；
- 特征层；
- 输入尺寸；
- dtype 和维度；
- 是否 L2 归一化；
- 指纹文件相对路径与 SHA-256。

第一版不保存文本向量，不保存对象原型向量，不引入专用向量数据库。

### 6.3 匹配计算

对每个新 proposal：

1. 在处理当前图片的候选前固定一次历史指纹快照；同图内刚提交的候选不进入该快照；
2. 用全局指纹与快照中的所有历史视角计算余弦相似度；
3. 按对象聚合，取每个对象的最佳历史视角作为粗排结果；
4. 对排名靠前的对象计算局部 patch 对应分数；
5. 结合全局分数、局部分数、第一名与第二名差距，得到视觉身份结论；
6. 阈值只从后续同类实例数据确定，不用现有三实例结果假装完成标定。

建议第一版使用简单可解释的加权式：

```text
visual_score = global_weight × global_similarity
             + local_weight  × local_match_ratio
```

权重、匹配阈值和歧义差距全部放入配置。代码不自动搜索参数，也不根据单张图片临时改变阈值。

## 7. 最终身份决定与写入规则

最终决定由 Qwen 的文本假设和 DINOv3 的视觉证据共同确定：

| Qwen 假设 | DINOv3 结果 | 最终决定 | 对象摘要 |
|---|---|---|---|
| existing A | 明确匹配 A | existing A | 提交候选汇总文本 |
| new | 所有历史对象均低于匹配阈值 | new | 用候选汇总文本建档 |
| uncertain | 任意 | uncertain | 不更新 |
| existing A | 最佳视觉匹配为 B | uncertain | 不更新 |
| existing A | 视觉证据不足 | uncertain | 不更新 |
| new | 视觉上明确匹配已有对象 | uncertain | 不更新 |
| 任意 | 第一、第二视觉候选差距不足 | uncertain | 不更新 |

这里的 `uncertain` 是正常终态，不触发第二次 Qwen、重试或静默归并。

## 8. 记忆库与资产调整

### 8.1 对象主档案

`objects` 保存一份当前对象级文本档案和更新时间。扁平的 `material_json`、`color_json`、`shape`、`description` 应收敛为新的结构化对象摘要，Web 不再渲染多个同名“颜色”标签。

### 8.2 观测

`observations` 保存：

- `object_id`、`proposal_id`、`source_image_id`；
- 全局与局部视觉指纹引用；
- 指纹模型和处理元数据；
- 创建时间。

不再把 `description` 作为对象卡中的逐视角长期文本。当前视角事实保留在 Qwen 原始响应与运行报告中。

### 8.3 crop、mask与overlay

- `crop.png`、`mask.png` 在 proposal 审计层只保存一份；
- observation 通过 proposal 引用，不再复制三份图片到对象目录；
- crop 和 mask 可以在对象卡中展示；
- overlay 只用于候选血缘审计，不进入对象长期观测展示；
- 自动身份匹配只使用视觉指纹，不读取历史图片资产。

### 8.4 旧记忆库边界

新实现使用新的空白记忆库进行实验，不从现有两轮Qwen历史记忆库猜测生成 DINOv3 指纹或新对象摘要。当前完整记忆库继续作为历史实验依据和 Web 展示证据。数据库 schema 已升级并明确标记旧库只读，不编写静默回填或自动重建路径。

## 9. 单卡执行与模型驻留

优化后每张图都依赖上一张图提交后的对象摘要，因此不能继续“全部 Qwen → 全部 SAM3 → 全部 Qwen”的三大阶段批处理。

目标执行方式是：

```text
加载 Qwen、SAM3、DINOv3
→ 对每张唯一新图依次执行 Qwen一次、SAM3、DINOv3、提交
→ 下一张图读取最新记忆
→ 全部完成后释放模型
```

现有实测峰值为 Qwen 约 11.8 GiB、SAM3 约 5.1 GiB，DINOv3 ViT-B/16 尚未实测。实现 DINOv3 adapter 后，服务器必须先进行一次三模型联合驻留显存检查：

- 如果峰值在 RTX 4090 24 GiB 内，正式采用单次加载和逐图交替推理；
- 如果超出显存，直接暴露结果并暂停该编排，由用户重新决定；
- 不预写自动 CPU offload、反复加载、备用模型或隐式降级路径。

## 10. Web 页面调整

### 10.1 对象卡

对象卡展示：

- 对象名称、类别和唯一当前汇总文本；
- 稳定身份特征；
- 品牌或标识；
- 按部件组织的颜色与材质；
- 观测次数；
- 每个视角的 crop 与 mask。

不再展示 7 条内容重复的鼠标描述，也不把颜色数组渲染为多个重复“颜色”字段。

### 10.2 观测时间线

观测时间线保留图片、来源和时间，不再把每条观测描述当作长期对象知识。需要审计文字时，从对应 Qwen 原始响应查看 `current_view_facts`。

### 10.3 候选血缘

候选血缘增加：

```text
Qwen target / identity_hypothesis
→ SAM3 proposal
→ DINOv3 global/local score
→ final decision
→ object / observation
```

页面必须区分 Qwen 的身份假设、视觉匹配证据和最终决定，不能把其中任一项单独显示为已确认事实。

## 11. 实施顺序

后续 AI 按以下顺序开发，每一步完成后再进入下一步：

### 第1步：配置和 schema

- 增加 DINOv3 配置模型；
- 定义对象摘要、单次 Qwen 输出、视觉指纹和决定 schema；
- 将首轮批次设计改为按图状态更新；
- 明确数据库和报告 schema 新版本。

### 第2步：DINOv3 adapter

- 本地路径加载官方模型；
- 固定预处理；
- 提取 CLS 与有效 patch token；
- 写入 `.npz` 与元数据；
- 实现余弦相似度和局部匹配；
- 单元测试使用小型固定 tensor 或桩模型，不下载真实权重。

### 第3步：合并 Qwen 两阶段职责

- 重写 `scene_guidance.py` 的提示和输出；
- 输入对象文字摘要；
- 输出当前视角事实、身份假设和候选汇总摘要；
- 删除 `identity.py` 中第二轮 Qwen 调用路径及无效配置；
- 保留一次调用、原始响应写盘和 schema 校验。

### 第4步：逐图编排

- 将 `pipeline.py` 改为单图闭环；
- 每张图完成 Qwen、SAM3、DINOv3和事务提交后再处理下一张；
- 实现同提示去重、target与proposal关联；
- 实现表格中的最终决定规则；
- 失败直接记录并停止受影响 source，不自动重试。

### 第5步：存储和资产引用

- 更新 `memory_store.py` 与数据库 schema；
- observation 引用 proposal 资产，删除重复复制；
- 保存视觉指纹及哈希；
- 对象表只保留一份当前摘要；
- 原始 Qwen 当前视角事实继续写入 raw response 和报告。

### 第6步：Web与报告

- 对象卡按新摘要结构展示；
- 观测时间线只展示视角证据；
- 候选血缘显示视觉分数与决定依据；
- 报告增加 DINOv3 模型、revision、显存、耗时、指纹数量和匹配分布；
- 删除第二轮 Qwen 统计并明确每张唯一图一次 Qwen。

### 第7步：服务器验证

- 激活项目 Conda 环境运行全部单元测试；
- 检查三模型联合驻留显存；
- 先用新的空白记忆库运行现有固定输入，验证结构和回归边界；
- 再用用户采集的同类实例数据验证类内身份辨识；
- 每次实验使用独立记忆库和报告，不覆盖现有正式证据。

## 12. 测试与验收

### 12.1 代码与结构验收

- 每张唯一新图 Qwen 调用次数恰好为 1；
- 不存在第二轮 Qwen、自动重试或单图救援；
- 每个正式 proposal 都有一份完整视觉指纹或明确失败记录；
- 指纹模型、revision、预处理和哈希可追踪；
- `new/existing` 都同时有 Qwen 假设和 DINOv3 证据；
- 冲突结果为 `uncertain`，对象摘要不被覆盖；
- 对象卡只有一份当前汇总文本；
- observation 不复制 proposal 图片；
- SQLite 完整性、外键、指纹和资产引用全部通过。

### 12.2 当前固定输入验证

当前固定输入只用于检查：

- 首轮 v5 目标与 SAM3 召回；
- 单次 Qwen schema 和逐图摘要迭代；
- 3个已知对象是否仍能形成完整记忆；
- 视觉指纹是否完整生成；
- 当前对象卡污染是否复现。

它不能证明类内不同实例已经通过。

### 12.3 同类实例验证

用户新数据到达后重点统计：

- 同一实例跨视角的 DINOv3 Top-1 对象检索正确率；
- 不同同类实例的分数分布与第一、第二名差距；
- 错误合并数；
- 错误拆档数；
- 正确 `uncertain` 数与错误 `uncertain` 数；
- 全局指纹单独使用与“全局+局部指纹”的对比；
- 对象摘要是否增加了真实类内特征；
- 品牌不可见时是否出现臆测；
- 部件颜色和材质对应是否正确。

最重要的身份边界是不得把两个同类不同实例错误合并。具体数值门槛在数据集冻结并明确样本量后确定，不能提前用现有三实例结果设置。

## 13. 需要修改的主要文件

| 文件 | 预计职责变化 |
|---|---|
| `config/default.yaml`、`src/object_memory/config.py` | DINOv3、指纹和单图流程配置 |
| `src/object_memory/schemas.py` | 对象摘要、单次Qwen输出、视觉证据和新决定结构 |
| `src/object_memory/scene_guidance.py` | 合并发现、文字身份假设和摘要迭代 |
| `src/object_memory/identity.py` | 删除第二轮Qwen调用；仅保留可复用的解析逻辑时才继续存在 |
| `src/object_memory/dinov3_adapter.py` | 新增DINOv3加载、特征提取和指标 |
| `src/object_memory/pipeline.py` | 改为每图一次Qwen的单图闭环 |
| `src/object_memory/memory_store.py` | 新对象摘要、指纹引用和资产去重 |
| `src/object_memory/memory_loop.py` | 新写入和摘要提交规则 |
| `src/object_memory/web_service.py`、`web_static/` | 新对象卡、视觉证据和血缘展示 |
| `tests/` | schema、指纹、匹配、单次调用、存储和Web回归 |
| `data/README.md`、`docs/03_服务器环境搭建指南.md`、`docs/04_业务代码运行与验证指南.md` | 实现完成后同步稳定契约和运行方式 |

## 14. 开发原则

1. 先完成最小可验证闭环，不同时引入多个视觉编码器。
2. 不预设自动重试、第二次 Qwen、拆单救援、模型降级或备用阈值。
3. 原始响应、视觉分数、指纹元数据和最终决定直接落盘。
4. 失败是实验事实，不静默修补。
5. 不用类别、颜色或材质单独证明同一具体实例。
6. 不把 DINOv3 论文中的实例检索结果扩大为当前日用品类内识别已经有效。
7. 没有新实验时不改写 `PROGRESS.md` 的效果结论。
8. 本方案实施后必须新建空白记忆库验证，不接管当前正式实验库。

## 15. 设计参考

- [Affordance RAG：Multi-Level Representation](https://arxiv.org/abs/2512.18987)：参考其文本语义与视觉语义互补表示，但本次不实现区域语义和层级空间节点。
- [DINOv3](https://arxiv.org/abs/2508.10104)：参考 CLS 实例检索与稠密 patch 特征；同类日用品身份辨识仍需本项目自行验证。
- [DINOv3 官方实现](https://github.com/facebookresearch/dinov3)：模型和权重来源。
