# 移动App截图-文本对齐与页面分类 (UI_Align)

基于SigLIP的移动应用界面截图文本对齐与页面分类系统，支持图像-文本检索和页面类型分类。

## 项目概述

本项目实现了一个多模态AI系统，能够：

1. **文本对齐**: 使截图图像与一句话摘要在同一嵌入空间对齐，支持图像-文本双向检索
2. **页面分类**: 根据截图图像预测页面类别（如首页、点单页、购物车等）

### 技术特点

- 🔥 基于**SigLIP ViT-B/16**双塔架构
- 🚀 支持**QLoRA 4-bit量化**，单卡48GB可训练
- 📊 **两阶段训练**：预适配 + 微调
- 🎯 **多任务学习**：对齐损失 + 分类损失
- 📱 专注移动App界面理解

## 环境要求

### 硬件要求
- GPU: 单卡 ≥48GB 显存（推荐RTX A6000/V100/A100）
- CPU: 任意
- 存储: ≥100GB（含数据与模型cache）

### 软件要求
- Python 3.10/3.11
- PyTorch ≥ 2.1
- CUDA ≥ 11.8

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone <repository_url>
cd UI_Align

# 安装依赖（建议）
python scripts/utils/setup.py

# 配置accelerate
accelerate config --config_file configs/accelerate_config.yaml
```

### 2. 数据准备

#### 公开数据（Stage-1预适配）
下载RICO和Screen2Words数据集：
```bash
# 将RICO图像放到 data/rico_screen2words/images/
# 将Screen2Words描述保存为 data/rico_screen2words/captions.jsonl
```

#### 自建数据（Stage-2微调）
收集目标App截图（麦当劳、瑞幸、航旅纵横等）：
```bash
# 图像放到对应目录
data/custom_app/images/mcdonalds/*.jpg
data/custom_app/images/luckin/*.jpg  
data/custom_app/images/ctrip/*.jpg

# 生成图像描述（使用Qwen-VL）
python src/qwen_caption_generator.py \
    --images_dir data/custom_app/images \
    --output_file data/custom_app/captions.jsonl

# 手动标注页面类别到 data/custom_app/labels.jsonl
# 配置类别映射 data/custom_app/label_map.json
```

### 3. 训练

#### Linux/macOS
```bash
# Stage-1
bash scripts/linux/stage1.sh

# Stage-2
bash scripts/linux/stage2.sh

# 或使用 Makefile（可选）
make stage1
make stage2
```

#### Windows
```bat
scripts\windows\stage1.bat
scripts\windows\stage2.bat
```

### 4. 评估

#### Linux/macOS
```bash
bash scripts/linux/eval.sh
```

#### Windows
```bat
scripts\windows\eval.bat
```

### 5. 推理演示

#### Linux/macOS
```bash
bash scripts/linux/demo.sh data/custom_app/images/mcdonalds/mcdonalds_home_001.jpg
```

#### Windows
```bat
scripts\windows\demo.bat data\custom_app\images\mcdonalds\mcdonalds_home_001.jpg
```

## 项目结构

```
UI_Align/
├── data/                          # 数据目录
│   ├── rico_screen2words/         # RICO+Screen2Words数据
│   └── custom_app/                # 自建数据
├── src/                           # 源代码
│   ├── datasets.py                # 数据集类
│   ├── model_siglip_lora.py       # SigLIP模型+QLoRA
│   ├── losses.py                  # 损失函数
│   ├── train_stage1_align.py      # Stage-1训练
│   ├── train_stage2_align_cls.py  # Stage-2训练
│   ├── eval_retrieval.py          # 检索评估
│   ├── eval_cls.py                # 分类评估
│   ├── infer_demo.py              # 推理演示
│   └── qwen_caption_generator.py  # 描述生成工具
├── configs/                       # 配置文件
│   ├── accelerate_config.yaml     # Accelerate配置
│   ├── stage1.yaml                # Stage-1训练配置
│   └── stage2.yaml                # Stage-2训练配置
├── outputs/                       # 训练输出
├── requirements.txt               # 依赖包
├── scripts/                       # 脚本（linux/windows 分目录，utils 工具）
└── README.md                      # 说明文档
```

## 详细使用说明

### 数据格式

#### captions.jsonl
```json
{"image":"mcd_001.jpg","caption":"麦当劳点单页，展示套餐和加购按钮"}
```

#### labels.jsonl  
```json
{"image":"mcd_001.jpg","label":"mcd_order"}
```

#### label_map.json
```json
{
  "mcd_home": 0,
  "mcd_order": 1,
  "luckin_home": 2,
  "ctrip_flight_search": 3
}
```

### 训练配置

#### Stage-1 关键参数
- `learning_rate`: 2e-5
- `batch_size`: 16 (per device) × 16 (gradient accumulation) = 256 (global)
- `epochs`: 5
- `lora_r`: 16, `lora_alpha`: 16
- `freeze_ratio`: 0.6

#### Stage-2 关键参数  
- `learning_rate`: 1e-5 (base), 1e-3 (classification head)
- `lambda_align`: 0.5 (对齐损失权重)
- `early_stopping_patience`: 3

### 评估指标

#### 检索指标
- Recall@1/5/10 (Image-to-Text)
- Recall@1/5/10 (Text-to-Image) 
- Average Recall@1/5/10

#### 分类指标
- Top-1/3/5 Accuracy
- Per-class Precision/Recall/F1
- Confusion Matrix

## 实验结果

### 预期性能
- **检索**: Avg Recall@1 > 60%, Recall@5 > 85%
- **分类**: Top-1 Accuracy > 85%, Top-3 > 95%

### 消融实验
- Stage-1 vs Stage-2
- 不同λ权重的影响
- LoRA vs 全参数微调

## 常见问题

### Q: 显存不足怎么办？
A: 
- 减少`per_device_batch_size`
- 增加`gradient_accumulation_steps`  
- 启用`gradient_checkpointing`
- 调整`freeze_ratio`

### Q: 如何添加新的App类别？
A:
1. 收集新App的截图数据
2. 更新`label_map.json`
3. 重新训练Stage-2

### Q: 如何优化检索性能？
A:
- 增加对齐损失权重`lambda_align`
- 使用更多的负样本
- 调整温度参数`logit_scale`

## 扩展方向

- **Layout理解**: 集成OCR和控件检测
- **零样本泛化**: 跨App类别的泛化能力
- **多语言支持**: 支持英文等其他语言描述
- **实时推理**: 模型压缩和加速优化

## 引用

如果此项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{ui_align_2024,
  title={Mobile App Screenshot Text Alignment and Page Classification},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/UI_Align}}
}
```

## 📖 文档导航

- [快速开始](docs/QUICKSTART.md) - 快速设置和运行指南  
- [项目总结](docs/PROJECT_SUMMARY.md) - 实现架构和性能预期
- [开发现状](docs/DEVELOPMENT_STATUS.md) - 详细功能实现与进度
- [当前状况](docs/CURRENT_STATUS.md) - 最新开发状态报告

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系：
- Email: your.email@example.com
- GitHub: [@your-username](https://github.com/your-username)