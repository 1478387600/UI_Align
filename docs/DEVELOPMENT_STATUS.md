# 开发现状（Development Status）

更新时间：请填写当前日期

## 1. 概述

项目已按 MVP 指南实现“移动 App 截图—文本对齐与页面分类”的完整训练与评测流水线，支持 Windows 与 Linux 双平台脚本执行，并提供示例数据生成与快速上手文档。

## 2. 当前已实现

- 数据集与切分（`src/datasets.py`）
  - `PairCaptionDataset`：图像-文本对齐数据集（Stage‑1/Stage‑2 对齐）
  - `CaptionClsDataset`：图像-文本+分类联合数据集（Stage‑2）
  - `ImageClsDataset`：纯分类数据集（备用）
  - `create_data_splits`：按比例划分 train/val/test
- 模型与训练
  - `SigLipDualEncoder`（`src/model_siglip_lora.py`）：基于 HF SigLIP 的双塔模型
  - QLoRA 4‑bit（bitsandbytes）+ LoRA（peft）注入；支持冻结前部层
  - 可选分类头 `ImageClassifierHead`；可训练参数统计
  - 训练脚本：
    - Stage‑1 预适配（`src/train_stage1_align.py`）：仅对齐（InfoNCE）
    - Stage‑2 微调（`src/train_stage2_align_cls.py`）：对齐+分类联合损失
- 损失与指标（`src/losses.py`）
  - `contrastive_loss`、`classification_loss`、`combined_loss`
  - 备用：`FocalLoss`、`TripletLoss`
  - 指标：`compute_accuracy`、`compute_retrieval_metrics`
- 评估与推理
  - 检索评估（`src/eval_retrieval.py`）：Recall@K、示例检索结果保存
  - 分类评估（`src/eval_cls.py`）：Top‑K、混淆矩阵、错误分析、报告
  - 推理演示（`src/infer_demo.py`）：单图分类 Top‑K + 文本相似度 + 可视化
- 配置与脚本
  - 配置：`configs/accelerate_config.yaml`、`configs/stage1.yaml`、`configs/stage2.yaml`
  - Linux：`scripts/linux/stage1.sh`、`stage2.sh`、`eval.sh`、`demo.sh`
  - Windows：`scripts/windows/stage1.bat`、`stage2.bat`、`eval.bat`、`demo.bat`
  - Makefile：`make stage1` / `make stage2` / `make eval` / `make demo img=...`
- 文档
  - `README.md`（入口说明）
  - `docs/QUICKSTART.md`（快速上手）
  - `docs/PROJECT_SUMMARY.md`（实现总结）
  - `docs/MVP_Guide_V1.md`（原始开发指导文档）

## 3. 数据与示例

- 示例数据生成器：`scripts/utils/generate_sample_data.py`
  - 生成 RICO 风格 10 张 + 自建三 App（mcdonalds/luckin/ctrip）各 5 页×3 张=45 张
  - 自动生成 `captions.jsonl`、`labels.jsonl`、`label_map.json`
  - 目录结构与 README/QUICKSTART 保持一致

## 4. 训练与评估现状

- Stage‑1：可在示例数据上完整跑通（流程验证）
- Stage‑2：联合训练可产出分类头与检索对齐能力
- 评估：支持检索 Recall@K、分类 Top‑K、混淆矩阵、错误分析与报告导出

注：示例数据仅用于流程校验，性能指标不代表真实生产效果。

## 5. 平台与工程支持

- 脚本双平台：Windows（.bat）与 Linux（bash）
- 显存友好：QLoRA 4‑bit + 冻结策略 + 混合精度，单卡 ~48GB 可训练
- 依赖集中在 `requirements.txt`；可通过 `scripts/utils/setup.py` 进行环境检查与安装

## 6. 待办与改进（Backlog）

- 训练：
  - Stage‑1 增加早停与“最优权重”保存
  - LoRA 目标模块自适配；更细粒度可训练层配置
- 模型：
  - 兼容不同 SigLIP 权重仓库的输出键名差异（`image_embeds`/`text_embeds`）
  - 分类头使用未归一化图像特征的可选路径与对比实验
- 数据：
  - 引入 OCR/布局轻量特征（Layout‑Lite）
  - 更完备的数据去重与清洗脚本
- 评估：
  - 分布漂移/跨 App 零样本实验脚本
  - 更多可视化与 Top‑N 错误样例导出
- 工程：
  - Dockerfile 与启动脚本
  - CI（lint/test/build）与训练日志可视化（W&B/TensorBoard）

## 7. Linux 服务器使用指引

```bash
# 安装依赖
python -m pip install -r requirements.txt

# 可选：环境检查与示例数据
python scripts/utils/setup.py
python scripts/utils/generate_sample_data.py

# 训练
bash scripts/linux/stage1.sh
bash scripts/linux/stage2.sh

# 评估与推理
bash scripts/linux/eval.sh
bash scripts/linux/demo.sh path/to/image.jpg
```

---
如需补充或修订，请直接编辑本文档并提交。