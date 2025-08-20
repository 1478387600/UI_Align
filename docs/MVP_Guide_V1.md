# 项目总览

**项目名称**：移动 App 截图—文本对齐与页面分类（MVP）

**定位**：课程作业级别，目标是熟悉大模型相关训练流水线与多模态检索/分类的基本方法。无需论文产出或工程交付，但强调规范的实验与复现。

**核心任务**：

1. **模态对齐**（Contrastive）：使 *截图图像* 与 *一句话摘要* 在同一嵌入空间对齐（检索可用）。
2. **页面分类**（Classification）：根据截图图像预测页面类别（课程作业可先做单层 10–20 类）。

**模型路线（MVP）**：

* 使用 **SigLIP ViT-B/16** 双塔作为基础（图像塔 + 文本塔）。
* 文本输入来自 **Qwen‑VL** 批量生成的“页面一句话摘要”；**编码**时仍由 SigLIP 的文本塔完成（保持塔间对齐）。
* **两阶段训练**：

  * **Stage‑1 预适配**：在 **RICO + Screen2Words** 上仅做对齐训练。
  * **Stage‑2 微调**：在自建小数据上做“对齐 + 分类”的多任务训练。
* **显存控制**：QLoRA（4-bit）+ bf16/fp16 + 梯度累积（单卡 \~48GB 可跑）。

**不做内容（MVP）**：不做 layout 分支、不做 Agent、不做端到端大模型微调。

---

## 1. 数据与标注

### 1.1 公开数据（预适配用）

* **RICO**：移动 App UI 截图与视图层级信息（本项目仅用截图）。
* **Screen2Words**：针对 UI 截图的一句话摘要（caption）。

> 使用方式：对齐 `image` 与 `caption`，形成 `(img, cap)` 配对样本，不涉及页面真实类别。用于 Stage‑1 训练对齐能力。

### 1.2 自建小数据（微调用）

* **目标 App**：麦当劳、瑞幸咖啡、航旅纵横（或其他你熟悉的 App）。
* **采集数量**：每个 App 约 200–500 张截图。收集典型页面：首页、点单/下单、筛选、支付、订单详情、个人中心等。
* **页面标签**：课程级可定义单层 10–20 类（如 `mcd_order`, `luckin_order`, `ctrip_flight_search` 等）。确保每类 ≥50 张，便于训练与验证。
* **文本摘要**：使用 **Qwen‑VL** 为每张截图生成一句话摘要（caption）。

#### 1.2.1 Qwen‑VL 生成摘要

* **模板化 Prompt（中文）**：

  ```text
  你是一名移动应用界面理解助手。请用一句中文简洁总结这张手机App截图的页面类型和主要目的。
  要求：不超过30个字，避免无关修饰词；尽量包含品牌/功能关键词。
  输出：仅一句话。
  ```
* **示例输出**：

  * “麦当劳点单页，展示套餐与加购按钮。”
  * “瑞幸下单页，列出热卖咖啡并可加入购物车。”
  * “航旅纵横机票搜索页，包含日期与城市选择。”
* **生成策略**：

  * 控制长度（12–30 字）；
  * 可对重复、口水话进行正则清洗；
  * 同一页面多次生成时，选择最贴合标签、最稳定的描述。

#### 1.2.2 标注规范

* **唯一主标签**：每张图仅一个主类别（课程作业简化）。
* **类名建议**：`app_abbrev + '_' + page_key`（统一小写、下划线）。
* **类目表**：`label_map.json` 统一映射到 `0..C-1`。

### 1.3 数据质量检查

* Caption 是否与图片一致（抽样 5–10% 人工验收）。
* 类别分布是否极端不平衡（如某类 < 20 样本则合并或补采）。
* 去重：根据感知哈希（pHash）或简单 hash 去除高度重复截图。

---

## 2. 目录结构与数据格式

```
project/
  data/
    rico_screen2words/
      images/*.jpg
      captions.jsonl           # {"image":"xxx.jpg","caption":"..."}
    custom_app/
      images/
        mcdonalds/*.jpg
        luckin/*.jpg
        ctrip/*.jpg
      captions.jsonl           # 自建数据对应 caption（Qwen‑VL 生成）
      labels.jsonl             # {"image":"mcd_001.jpg","label":"mcd_order"}
      label_map.json           # {"mcd_order":0, "mcd_home":1, ...}
  src/
    datasets.py
    model_siglip_lora.py
    losses.py
    train_stage1_align.py
    train_stage2_align_cls.py
    eval_retrieval.py
    eval_cls.py
    infer_demo.py
  configs/
    stage1.yaml
    stage2.yaml
    accelerate_config.yaml
  outputs/
  requirements.txt
  README.md
```

### 2.1 `captions.jsonl`（行级 JSON）

```json
{"image":"mcd_001.jpg","caption":"麦当劳点单页，展示套餐和加购按钮"}
```

### 2.2 `labels.jsonl`

```json
{"image":"mcd_001.jpg","label":"mcd_order"}
```

### 2.3 `label_map.json`

```json
{"mcd_order":0, "mcd_home":1, "luckin_order":2, "ctrip_flight_search":3}
```

---

## 3. 环境与依赖

**硬件**：单卡 GPU（≈48GB 显存），CPU 任意，存储≥100GB（含数据与模型 cache）。

**推荐软件栈**：

* Python 3.10/3.11
* PyTorch ≥ 2.1（支持 bf16）
* transformers ≥ 4.41
* accelerate ≥ 0.30
* peft ≥ 0.11
* bitsandbytes ≥ 0.43（QLoRA 4-bit 量化）
* datasets, pillow, opencv-python, scikit-learn, numpy, matplotlib
* (可选) wandb 或 tensorboard 日志

**`requirements.txt` 示例**：

```
accelerate>=0.30.0
bitsandbytes>=0.43.0
datasets>=2.19.0
huggingface_hub
numpy
opencv-python
peft>=0.11.0
Pillow
scikit-learn
torch>=2.1.0
transformers>=4.41.0
matplotlib
```

**Accelerate 配置**（`configs/accelerate_config.yaml` 示例）：

```yaml
compute_environment: LOCAL_MACHINE
device_fp16: true
frac_gpu_memory: 1.0
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false

distributed_type: NO
```

> 若 bf16 不可用，退回 `mixed_precision: fp16`。

---

## 4. 模型设计（MVP）

### 4.1 基础模型：SigLIP ViT‑B/16 双塔

* **图像塔**：ViT‑B/16（224×224 输入）。
* **文本塔**：SigLIP 自带文本编码器。
* **投影层**：分别将图像/文本特征投影到同一维度（通常 D=512/768）。
* **logit\_scale**：可学习温度参数（初始化约 1/0.07 的 log）。

> 实际实现随权重仓库不同，API 细节略有区别（HuggingFace / open\_clip 均可）。本项目以 HF 形式为主。

### 4.2 QLoRA（4-bit）与冻结策略

* **量化**：加载模型为 4-bit（nf4 或 fp4），显存友好。
* **LoRA 作用层**：建议对图像/文本塔的**最后若干 Transformer Block**与**投影层**加 LoRA；避免全网注入。
* **冻结**：前 50–70% 的层完全冻结，仅训练后部与投影层（配合 LoRA）。
* **梯度检查点**：开启以换显存。

### 4.3 分类头

* 在**图像塔**的 pooled/CLS 特征上接 `Linear(d, num_classes)`，训练于自建数据（Stage‑2）。
* 若检索特征已归一化，可另取投影前中间层或复制一份未归一化特征用于分类（视实现而定）。

---

## 5. 训练任务与损失

### 5.1 对比对齐（Stage‑1 / Stage‑2）

* **InfoNCE（对称）**：

  * `logits_per_image = s * (img @ txt.T)`
  * `logits_per_text = logits_per_image.T`
  * 目标：图像与其对应 caption 的 index 匹配；in-batch negatives。
* **超参**：

  * 学习温度 `s = exp(logit_scale)`；
  * 文本最大长度 64；
  * 图像尺寸 224；
  * 基本增广：随机裁剪、水平翻转、轻度模糊、随机贴片（模拟弹窗）。

### 5.2 页面分类（Stage‑2）

* **输入**：图像特征（来自图像塔）。
* **损失**：交叉熵 CE。
* **联合损失**：

  * 预适配：`L = L_align`；
  * 微调：`L = λ * L_align + (1-λ) * L_cls`，建议 `λ=0.5` 起步。

---

## 6. 训练流程

### 6.1 Stage‑1：RICO+Screen2Words 预适配

* **目的**：学到“UI 风格下的图-文对齐”。
* **数据**：`data/rico_screen2words/images/` 与 `captions.jsonl`。
* **策略**：

  * 混精度：bf16（或 fp16）。
  * QLoRA：4-bit（nf4），LoRA `r=16`，`alpha=16`。
  * 冻结前 8–9 层 Transformer block，仅调后部 & 投影层。
  * Batch：per‑device 16，梯度累计 16，**全局 batch≈256**。
  * LR：2e‑5 ～ 5e‑5（仅更新 LoRA/后几层/投影层），cosine + warmup 3%。
  * Epoch：3–5。
* **命令**：

  ```bash
  accelerate launch src/train_stage1_align.py \
    --model_name google/siglip-base-patch16-224 \
    --train_images data/rico_screen2words/images \
    --train_captions data/rico_screen2words/captions.jsonl \
    --output_dir outputs/stage1 \
    --precision bf16 --load_in_4bit --lora_r 16 --lora_alpha 16 \
    --freeze_ratio 0.6 --per_device_batch_size 16 --grad_accum 16 \
    --lr 2e-5 --epochs 5 --warmup_ratio 0.03
  ```

### 6.2 Stage‑2：自建数据微调（对齐 + 分类）

* **目的**：适配目标 App 的语义与类别体系。
* **数据**：`data/custom_app/images/`、`captions.jsonl`、`labels.jsonl`、`label_map.json`。
* **策略**：

  * 延续 Stage‑1 的 LoRA/冻结方案；
  * 新增分类头（学习率可单独设 1e‑3）；
  * 联合损失权重 `λ=0.5`；
  * Epoch：5–10，早停（验证集 3 次无提升停止）。
* **命令**：

  ```bash
  accelerate launch src/train_stage2_align_cls.py \
    --resume_from outputs/stage1 \
    --train_images data/custom_app/images \
    --train_captions data/custom_app/captions.jsonl \
    --train_labels data/custom_app/labels.jsonl \
    --label_map data/custom_app/label_map.json \
    --output_dir outputs/stage2 \
    --precision bf16 --load_in_4bit --lora_r 16 --lora_alpha 16 \
    --freeze_ratio 0.6 --per_device_batch_size 16 --grad_accum 16 \
    --lr 1e-5 --head_lr 1e-3 --epochs 8 --warmup_ratio 0.03 --lambda_align 0.5
  ```

---

## 7. 关键代码骨架

> 代码仅示意，具体以所用 SigLIP 权重的 API 为准。

### 7.1 数据集（对齐）

```python
# src/datasets.py
import json, os
from PIL import Image
from torch.utils.data import Dataset

class PairCaptionDataset(Dataset):
    def __init__(self, root, jsonl, image_processor, tokenizer, max_len=64):
        self.root = root
        self.items = [json.loads(l) for l in open(jsonl, 'r', encoding='utf-8')]
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(os.path.join(self.root, 'images', it['image'])).convert('RGB')
        pixel = self.image_processor(images=img, return_tensors='pt')['pixel_values'][0]
        tok = self.tokenizer(
            it['caption'], max_length=self.max_len, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        return {
            'pixel_values': pixel,
            'input_ids': tok['input_ids'][0],
            'attention_mask': tok['attention_mask'][0],
        }
```

### 7.2 数据集（分类）

```python
class CaptionClsDataset(PairCaptionDataset):
    def __init__(self, root, jsonl_cap, jsonl_lbl, label_map_path, *args, **kwargs):
        super().__init__(root, jsonl_cap, *args, **kwargs)
        self.labels = {j['image']: j['label'] for j in map(json.loads, open(jsonl_lbl, 'r', encoding='utf-8'))}
        self.label2id = json.load(open(label_map_path, 'r', encoding='utf-8'))

    def __getitem__(self, i):
        item = super().__getitem__(i)
        img_name = self.items[i]['image']
        lab = self.label2id[self.labels[img_name]]
        item['label'] = lab
        return item
```

### 7.3 模型载入 + QLoRA 注入

```python
# src/model_siglip_lora.py
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

class SigLipDualEncoder(nn.Module):
    def __init__(self, name='google/siglip-base-patch16-224', load_in_4bit=True, lora_r=16, lora_alpha=16):
        super().__init__()
        quant = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModel.from_pretrained(
            name, torch_dtype=torch.bfloat16, quantization_config=quant, device_map='auto'
        )
        # 假设 self.model 输出包含 image_embeds / text_embeds（不同实现需适配）
        self.logit_scale = nn.Parameter(torch.ones([]) * (1/0.07)).log()

        # 仅对最后若干层与投影层加 LoRA（示意：按模块名筛选）
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=['q_proj','k_proj','v_proj','out_proj','proj'])
        self.model = get_peft_model(self.model, lora)

    def forward(self, pixel_values, input_ids, attention_mask):
        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        img, txt = out.image_embeds, out.text_embeds
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        return img, txt, self.logit_scale.exp()
```

### 7.4 损失与训练步

```python
# src/losses.py
import torch, torch.nn.functional as F

def contrastive_loss(img, txt, logit_scale):
    logits_i = logit_scale * img @ txt.t()
    logits_t = logits_i.t()
    y = torch.arange(img.size(0), device=img.device)
    return (F.cross_entropy(logits_i, y) + F.cross_entropy(logits_t, y)) / 2
```

### 7.5 分类头（Stage‑2）

```python
class ImageClassifierHead(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
```

### 7.6 评测（检索 Recall\@K）

```python
# src/eval_retrieval.py（核心思路）
import torch

def recall_at_k(img_feat, txt_feat, ks=(1,5,10)):
    sim = img_feat @ txt_feat.t()  # 假设已 L2 归一化
    gt = torch.arange(sim.size(0), device=sim.device)
    ranks_txt = (sim.argsort(dim=1, descending=True) == gt[:,None]).nonzero()[:,1]
    res = {}
    N = sim.size(0)
    for k in ks:
        res[f'R@{k} (i->t)'] = (ranks_txt < k).float().mean().item()
    # 反向 t->i
    sim_t = sim.t()
    ranks_img = (sim_t.argsort(dim=1, descending=True) == gt[:,None]).nonzero()[:,1]
    for k in ks:
        res[f'R@{k} (t->i)'] = (ranks_img < k).float().mean().item()
    return res
```

---

## 8. 超参数与默认配置

**通用**：

* 图像尺寸：224；文本最大长度：64；
* 优化器：AdamW；weight decay: 0.05；
* 调度：cosine；warmup\_ratio=0.03；
* 随机种子：42；`torch.backends.cudnn.benchmark = True`。

**Stage‑1**：

* precision：bf16（或 fp16）
* QLoRA：load\_in\_4bit（nf4）；LoRA `r=16, alpha=16`
* 冻结：前 60% 层
* per\_device\_batch\_size=16；grad\_accum=16（全局≈256）
* lr=2e‑5（仅更新后层 + 投影 + LoRA）；epochs=3–5

**Stage‑2**：

* 同 Stage‑1；新增分类头
* lr\_base=1e‑5；head\_lr=1e‑3
* λ=0.5；epochs=5–10；early stop=3

---

## 9. 评测与报告

### 9.1 数据切分

* 自建数据按 8:1:1 划分 train/val/test；确保每类都有验证与测试样本。

### 9.2 指标

* **对齐**：`Recall@1/5/10`（i→t 与 t→i）。
* **分类**：Top‑1 / Top‑3 Acc；混淆矩阵（了解最易混类别）。

### 9.3 报告建议结构

1. 任务与数据说明（含类目表）。
2. 训练设置（硬件、超参、冻结/QLoRA 策略）。
3. 结果表格：

   * 表1：Stage‑0（原始 SigLIP） vs Stage‑1 vs Stage‑2 的检索 Recall\@K。
   * 表2：分类 Top‑1/Top‑3；以及仅训分类头 vs 联合损失的对比。
4. 可视化：

   * 文本查询的 Top‑5 检索结果图；
   * 分类混淆矩阵；
   * 错误样例分析（典型混淆对）。

---

## 10. 推理与 Demo（可选）

### 10.1 CLI 推理脚本

* 输入：单张截图
* 输出：预测类别 + Top‑5 文本检索相似度（可选）

```bash
python src/infer_demo.py \
  --ckpt outputs/stage2/best.pt \
  --image path/to/test.jpg \
  --label_map data/custom_app/label_map.json
```

### 10.2 简易可视化（可选）

* 使用 `matplotlib` 高亮预测类别与相似度；
* 或使用 `streamlit` 做简易界面（课程加分项）。

---

## 11. 常见问题与规避

1. **Qwen‑VL 文本漂移**：摘要尽量模板化；长度 12–30 字；同类页面保持风格一致。
2. **类别不平衡**：采集时保证每类 ≥50；不足则合并近似页面或使用 class-balanced sampler。
3. **显存爆**：确保仅对后几层与投影层做 LoRA；启用 gradient checkpointing；减少 per‑device batch。
4. **评测混淆**：同一 App 的“首页/点单/购物车”易混；适当加入“标题栏裁剪”与“弹窗贴片”增广。
5. **实现差异**：不同 SigLIP 开源实现暴露的中间特征与投影层命名不同；需要按 `print(model)` 核对模块名再注入 LoRA。

---

## 12. 扩展方向（非 MVP）

* **Layout‑Lite**：

  1. OCR‑Grid Token：将 OCR 文本附粗粒度位置 token（ROW/COL），与文本一同编码，几乎不增显存；
  2. 控件计数特征：统计按钮/输入框/底部导航数量，将统计向量拼到分类头。
* **跨 App 零样本**：留出某个 App 仅用于测试，观察泛化。
* **蒸馏**：用 Qwen‑VL 问答作为软标签，做多任务联合训练（QA 蒸馏头）。

---

## 13. 实施里程碑（建议）

* **第 1 周**：环境搭建；下载 RICO/Screen2Words；编写数据解析；跑通 Stage‑1 小规模对齐训练。
* **第 2 周**：自建数据采集与清洗；Qwen‑VL 批量生成 caption；完成 Stage‑2 训练与基本评测。
* **第 3 周**：结果可视化与报告撰写；错误分析；（可选）做简易推理 Demo。

---

## 14. 复现实验清单（Checklist）

* [ ] 固定随机种子（42），记录软件版本。
* [ ] 保存训练/验证日志与配置（YAML/JSON）。
* [ ] 保存最佳权重（按验证指标）。
* [ ] 导出 `requirements.txt` 与 `accelerate_config.yaml`。
* [ ] 在 README.md 中记录运行命令与数据来源说明。

---

### 附录 A：Qwen‑VL 批量摘要伪代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, glob, os
from PIL import Image

model_id = "Qwen/Qwen2-VL-7B-Instruct"  # 任选可用的 Qwen-VL 指令模型

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype='auto')
tok = AutoTokenizer.from_pretrained(model_id)

prompt = "你是一名移动应用界面理解助手。请用一句中文简洁总结这张手机App截图的页面类型和主要目的。要求：不超过30个字，避免无关修饰词；尽量包含品牌/功能关键词。输出：仅一句话。"

records = []
for p in glob.glob('data/custom_app/images/**/*.jpg', recursive=True):
    img = Image.open(p).convert('RGB')
    msgs = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image"}]}
    ]
    out = model.chat(tokenizer=tok, messages=msgs, images=[img])  # 具体API按模型实现调整
    cap = out.strip().split('\n')[0][:40]
    records.append({"image": os.path.basename(p), "caption": cap})

with open('data/custom_app/captions.jsonl','w',encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

> 说明：不同 Qwen‑VL 版本 API 不同，请根据所用仓库的 `.chat()` 或 `generate()` 接口适配；必要时先将图像转为符合模型要求的输入格式。

---

### 附录 B：`train_stage1_align.py` 训练主循环（示意）

```python
# 省略 import
from accelerate import Accelerator
from transformers import AutoProcessor, AutoTokenizer

acc = Accelerator(mixed_precision='bf16')

# 1) Processor / Tokenizer
proc = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
tok  = AutoTokenizer.from_pretrained('google/siglip-base-patch16-224')

# 2) Dataset / Dataloader
ds = PairCaptionDataset('data/rico_screen2words', 'data/rico_screen2words/captions.jsonl', proc, tok)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=False)

# 3) Model
model = SigLipDualEncoder(load_in_4bit=True, lora_r=16, lora_alpha=16)
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=0.05)

model, optim, loader = acc.prepare(model, optim, loader)

for epoch in range(5):
    model.train()
    for batch in loader:
        img, ids, mask = batch['pixel_values'], batch['input_ids'], batch['attention_mask']
        with acc.autocast():
            img_feat, txt_feat, s = model(img, ids, mask)
            loss = contrastive_loss(img_feat, txt_feat, s)
        acc.backward(loss)
        optim.step(); optim.zero_grad()
    # TODO: 评估与保存
```

---

### 附录 C：风险与合规

* 本项目仅用于学习/研究；自建数据需遵守对应 App 的使用条款与隐私政策；
* 不上传包含个人隐私的截图；必要时对头像、订单号等敏感字段做遮挡。

---

**版本**：v1.0
**维护人**：你自己（填写姓名/学号）
**最后更新**：填写日期
