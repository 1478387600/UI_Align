# UI_Align 快速开始指南

## 🚀 一键开始

### 1. 环境检查和初始化
```bash
# 安装依赖并检查环境
python scripts/utils/setup.py

# 生成示例数据（用于快速测试）
python scripts/utils/generate_sample_data.py
```

### 2. 配置Accelerate
```bash
# Windows
accelerate config --config_file configs/accelerate_config.yaml

# 建议配置：
# - 单机单卡
# - 混合精度：bf16（如果支持）或fp16
# - 不使用DeepSpeed
```

### 2.5 RICO‑Screen2Words 准备步骤（真实数据）

> 若仅验证流程，可跳过本节使用“示例数据”。如进行真实训练，请先完成以下准备。

1) 准备目录
```bash
mkdir -p data/rico_screen2words/parquet
mkdir -p data/rico_screen2words/images
# 将 parquet 分片放入 data/rico_screen2words/parquet/
# 将 RICO 截图放入 data/rico_screen2words/images/（保持原有层级或扁平皆可）
```

2) 生成 captions.jsonl
```bash
python scripts/utils/prepare_rico_screen2words.py \
  --parquet_dir data/rico_screen2words/parquet \
  --images_dir  data/rico_screen2words/images \
  --output_file data/rico_screen2words/captions.jsonl \
  --image_suffix .jpg   # 可选：未匹配到真实文件时回退的统一后缀
```

说明：
- 脚本会递归匹配 images/ 下与 screen_id 同名的图片（任意后缀与子目录），未命中时使用 --image_suffix 兜底。
- 脚本会输出统计信息：总行数、写入数、字段缺失数、图片缺失数，便于排查数据缺失。
- 若 parquet 字段名不同，可先打印列名后调整映射：
  ```python
  from datasets import load_dataset
  ds = load_dataset('parquet', data_files={'train':'data/rico_screen2words/parquet/train-*.parquet'})
  print(ds['train'].column_names)
  ```

完成后，Stage‑1 训练将直接读取 data/rico_screen2words/captions.jsonl。

### 3. 开始训练

#### Linux/macOS
```bash
bash scripts/linux/stage1.sh
bash scripts/linux/stage2.sh

# 或使用 Makefile
make stage1
make stage2
```

#### Windows
```bat
scripts\windows\stage1.bat
scripts\windows\stage2.bat
```

### 4. 模型评估
```bash
# Linux/macOS
bash scripts/linux/eval.sh

# Windows
scripts\windows\eval.bat
```

### 5. 推理演示
```bash
# Linux/macOS
bash scripts/linux/demo.sh data/custom_app/images/mcdonalds/mcdonalds_home_001.jpg

# Windows
scripts\windows\demo.bat data\custom_app\images\mcdonalds\mcdonalds_home_001.jpg
```

## 📊 预期结果

### 训练时间（使用示例数据）
- **Stage-1**: ~10分钟（RTX 4090）
- **Stage-2**: ~15分钟（RTX 4090）

### 性能指标
- **检索**: Recall@1 > 80%, Recall@5 > 95%
- **分类**: Top-1 Accuracy > 90%

## 🔧 常见问题

### Q1: 显存不足
```bash
# 修改配置文件中的batch_size
# configs/stage1.yaml 或 configs/stage2.yaml
per_device_batch_size: 8  # 从16减少到8
gradient_accumulation_steps: 32  # 相应增加以保持全局batch size
```

### Q2: 训练速度慢
```bash
# 检查GPU利用率
nvidia-smi

# 确保使用了混合精度训练
precision: "bf16"  # 或 "fp16"
```

### Q3: 模型不收敛
```bash
# 调整学习率
learning_rate: 1e-5  # 减小学习率
warmup_ratio: 0.1    # 增加warmup比例
```

## 📁 输出文件说明

### 训练输出
```
outputs/
├── stage1/
│   ├── checkpoint-epoch-X/     # 训练检查点
│   └── final/                  # 最终模型
└── stage2/
    ├── best/                   # 最佳模型（用于推理）
    ├── checkpoint-epoch-X/     # 训练检查点
    └── final/                  # 最终模型
```

### 评估输出
```
eval_results/
├── retrieval/
│   ├── retrieval_metrics.json      # 检索指标
│   ├── category_metrics.json       # 各类别检索性能
│   ├── i2t_example_*.json          # 图像到文本检索示例
│   └── t2i_example_*.json          # 文本到图像检索示例
└── classification/
    ├── metrics.json                # 分类指标
    ├── predictions.json            # 预测结果
    ├── confusion_matrix.png        # 混淆矩阵
    └── classification_report.txt   # 分类报告
```

### 推理输出
```
demo_results/
├── inference_results.json     # 推理结果JSON
└── inference_results.png      # 可视化结果图
```

## 🎯 下一步

### 使用真实数据
1. **收集数据**: 下载RICO和Screen2Words数据集
2. **标注数据**: 收集目标App截图并标注
3. **生成描述**: 使用Qwen-VL生成图像描述
4. **重新训练**: 使用真实数据重新训练模型

### 模型优化
1. **超参数调优**: 调整学习率、batch size等
2. **数据增强**: 添加更多的数据增强策略
3. **模型结构**: 尝试不同的LoRA配置
4. **损失函数**: 实验不同的损失权重

### 部署应用
1. **模型转换**: 转换为ONNX或TensorRT格式
2. **API服务**: 部署为REST API服务
3. **前端界面**: 创建Web界面进行交互
4. **移动应用**: 集成到移动应用中

## 📞 获取帮助

遇到问题？
1. 查看 [README.md](../README.md) 获取详细文档
2. 提交Issue报告问题
3. 联系项目维护者