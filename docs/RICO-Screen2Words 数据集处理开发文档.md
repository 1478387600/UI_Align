📘 RICO-Screen2Words 数据集处理开发文档
目标

在 UI_Align 项目中，新增一套脚本流程，用于处理 RootsAutomation 提供的 RICO-Screen2Words parquet 数据集，并生成项目训练所需的标准文件：

data/rico_screen2words/images/
（存放 RICO 截图，需单独下载）

data/rico_screen2words/captions.jsonl
（由 parquet 转换而来，图文对齐样本）

这样，Stage-1 训练脚本即可直接使用。

📂 数据目录要求（与 README 保持一致）
UI_Align/
  data/
    rico_screen2words/
      parquet/                  # 存放下载的 parquet 分片
        train-00000-of-00008.parquet
        ...
        val-00001-of-00002.parquet
        test-00001-of-00002.parquet
      images/                   # 存放 RICO 截图（单独下载）
        0/xxxx.png
        0/yyyy.png
        ...
      captions.jsonl            # 由脚本生成

🛠️ 开发任务
1. 新建脚本

在 scripts/utils/ 下新增脚本：

scripts/utils/prepare_rico_screen2words.py

2. 依赖

确保项目依赖中包含：

pip install datasets pandas

3. 核心功能实现
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_rico_screen2words.py

功能:
- 读取 parquet 格式的 RICO-Screen2Words 数据集
- 转换为 captions.jsonl ({"image": xxx, "caption": yyy})
- 保持与 README 说明一致的目录结构
"""

import os
import json
from datasets import load_dataset

def prepare_rico_screen2words(parquet_dir, output_file):
    # 加载 parquet 分片
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(parquet_dir, "train-*.parquet"),
            "val":   os.path.join(parquet_dir, "val-*.parquet"),
            "test":  os.path.join(parquet_dir, "test-*.parquet"),
        },
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for split in ["train", "val", "test"]:
            for row in dataset[split]:
                # 注意字段名可能不同, 建议先 print(dataset["train"].column_names)
                screen_id = row.get("screen_id") or row.get("id")
                caption   = row.get("caption") or row.get("summary") or row.get("description")

                if screen_id is None or caption is None:
                    continue  # 跳过缺失数据

                # 文件后缀要和 RICO 截图保持一致 (通常是 .jpg 或 .png)
                image_name = f"{screen_id}.jpg"

                f.write(json.dumps({
                    "image": image_name,
                    "caption": caption.strip()
                }, ensure_ascii=False) + "\n")

    print(f"[OK] captions.jsonl 已生成: {output_file}")

if __name__ == "__main__":
    base_dir = "data/rico_screen2words"
    parquet_dir = os.path.join(base_dir, "parquet")
    output_file = os.path.join(base_dir, "captions.jsonl")

    prepare_rico_screen2words(parquet_dir, output_file)

🚀 使用方法

准备数据

下载 RICO 截图（放到 data/rico_screen2words/images/）

下载 RootsAutomation parquet 分片（放到 data/rico_screen2words/parquet/）

运行脚本

python scripts/utils/prepare_rico_screen2words.py


生成结果

脚本会在 data/rico_screen2words/ 下生成 captions.jsonl

格式如下：

{"image":"12345.jpg","caption":"Settings page with account options"}
{"image":"67890.jpg","caption":"Shopping cart page with product list"}

📑 与现有项目的衔接

captions.jsonl 会被 Stage-1 训练脚本 (src/train_stage1_align.py) 直接读取，无需改动。

项目 README中已有的数据目录说明保持不变，只是新增了 parquet → captions.jsonl 的转换步骤。

开发只需要保证 parquet 字段映射正确 (screen_id ↔ image，caption ↔ caption)。

⚠️ 注意事项

字段名检查
parquet 文件可能字段名不同，请用：

print(dataset["train"].column_names)


确认真实字段。

图片后缀
RICO 原始截图可能是 .png 或 .jpg，脚本要统一；否则训练时找不到图片。

覆盖率
部分 screen_id 可能在 parquet 有，但截图缺失，需要开发在日志里统计跳过条数。