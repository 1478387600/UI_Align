### Stage‑1 训练报告（RICO + Screen2Words 预适配）

概述
- 目标：使用 SigLIP ViT‑B/16 进行图文对齐（对称 InfoNCE），完成 Stage‑1 预适配，为 Stage‑2（自建数据对齐+分类）打基础。
- 模型：`google/siglip-base-patch16-224`，LoRA 仅注入 vision 分支，避免 text forward 冲突；支持 bf16/fp16，4bit 在服务器上自动回退。
- 硬件：A40 48G。

数据
- 本次训练使用 `22417` 对图文配对。
- 训练来源：`data/rico_screen2words/captions.jsonl` 或 `data/captions.jsonl`（两者等价为统一格式）。

关键配置（最终一轮）
- LoRA：`r=16, alpha=16, dropout=0.1`，`freeze_ratio=1.0`（LoRA‑only），仅作用视觉编码器。
- 训练：`precision=bf16`，`epochs=8`（最后一轮示例），`scheduler=cosine`，`warmup_ratio=0.03`。
- 批量：配置为较大批量与梯度累积（实际 steps/epoch≈58，drop_last=True）。
- 预处理：`image_size=224`。

修复/增强
- 彻底修复了 LoRA 与 SigLIP forward 的关键字冲突（安装 vision 前向过滤器，仅透传 `pixel_values`）。
- 增强评估：每个 epoch 结束自动抽样评估（默认 2000 对），记录 Recall@K、Precision@K、mAP@K、NDCG@K、MRR、Mean/Median Rank、相似度统计等，全部同步到 WandB。
- 数值稳定性：避免温度爆炸（对 `exp(logit_scale)` 做上限裁剪），损失回到合理量级。

训练与评估结果（采样评估，K∈{1,5,10}）
- 训练损失：从 ~1.79 ↘ ~1.21（8 个 epoch），收敛稳定。
- 检索总体：
  - Avg Recall@1 ≈ 49.55%，@5 ≈ 72.38%，@10 ≈ 80.15%。
  - Avg Precision@1 ≈ 0.4955，mAP@5 ≈ 0.582，NDCG@10 ≈ 0.643。
  - Avg MRR ≈ 0.600；Avg Mean Rank ≈ (i2t:14.13, t2i:17.08)；Median Rank ≈ (i2t:1, t2i:2)。
- 正负样本分布：
  - pos_sim_mean ≈ 0.122，neg_sim_mean ≈ 0.0006（分布可分）。

结论
- 对齐质量达到较好水平（Recall@1≈50%，MRR≈0.60，Median Rank≈1/2），已满足作为 Stage‑2 起点的要求。
- 若期望更高对齐性能，可继续微调：
  - 方案 A（稳健）：继续 LoRA‑only，延长到 10–15 个 epoch，或将 `image_size` 提至 336/384。
  - 方案 B（更强）：`freeze_ratio=0.8–0.9` 轻微解冻后部层，学习率保持 1e‑5–2e‑5，观察稳定性。

Stage‑1 复现命令（示例）
```bash
python src/train_stage1_align.py \
  --config configs/stage1.yaml \
  --use_wandb
```

（可按显存增大批量/分辨率）
```bash
python src/train_stage1_align.py --config configs/stage1.yaml \
  --per_device_batch_size 96 --gradient_accumulation_steps 4 \
  --image_size 336 --num_workers 12 --use_wandb
```

是否进入 Stage‑2？
- 建议：可以开始 Stage‑2（对齐 + 分类）微调。
  - 以当前 Stage‑1 权重为初始化（`resume_from: outputs/stage1`）。
  - LoRA 维持 vision‑only；分类头学习率相对更高（如 `head_lr=1e-3`）。
  - 若分类数据量较少，推荐 LoRA‑only 或轻微解冻；分类头可使用 `label_smoothing` 或 `focal loss`（可选）。

Stage‑2 启动命令（示例）
```bash
python src/train_stage2_align_cls.py \
  --config configs/stage2.yaml \
  --use_wandb
```

Stage‑2 监控指标（已在脚本中集成）
- 分类：loss、Top‑1/Top‑3 Acc、混淆矩阵、classification_report。
- 对齐：Recall@K、mAP@K、NDCG@K、MRR、Mean/Median Rank、相似度统计（可酌情保留）。

备注
- 如需满载显存提升吞吐：优先增大 `per_device_batch_size`，OOM 再提高 `gradient_accumulation_steps`；同时提高 `image_size` 有助于 UI 细节建模。
- 4‑bit 量化需匹配的 CUDA/bitsandbytes；当前已自动回退，不影响 GPU 训练。
