#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_rico_screen2words.py

功能:
- 读取 parquet 格式的 RICO-Screen2Words 数据集
- 生成 captions.jsonl ( {"image": xxx, "caption": yyy} )
- 目录结构与项目 README/文档保持一致

使用:
  python scripts/utils/prepare_rico_screen2words.py \
    --parquet_dir data/rico_screen2words/parquet \
    --images_dir  data/rico_screen2words/images \
    --output_file data/rico_screen2words/captions.jsonl

说明:
- 若未指定 --images_dir，将不做图片存在性校验，直接使用 {screen_id}.jpg 作为文件名
- 若提供了 --images_dir，则会优先通过递归匹配查找真实文件（含子目录与后缀）
  - 查找顺序: 任意后缀匹配 (rglob f"{screen_id}.*")
  - 若未找到，按 --image_suffix 或默认 .jpg 写入（并统计 missing_images）
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def find_image_relative_path(images_dir: Path, screen_id: str, preferred_suffix: Optional[str]) -> Optional[str]:
    """在 images_dir 下递归查找与 screen_id 匹配的文件，返回相对路径（posix）。

    优先精确匹配任意后缀，其次回退到 preferred_suffix（如 .jpg/.png）。
    返回 None 表示未找到。
    """
    # 1) 尝试任意后缀匹配
    matches = list(images_dir.rglob(f"{screen_id}.*"))
    if matches:
        # 若存在多个匹配，取最短路径（更可能是扁平结构），否则取第一个
        matches.sort(key=lambda p: (len(p.as_posix()), p.suffix))
        return matches[0].relative_to(images_dir).as_posix()

    # 2) 若指定了偏好后缀，构造该路径并检测是否存在
    if preferred_suffix:
        candidate = images_dir / f"{screen_id}{preferred_suffix}"
        if candidate.exists():
            return candidate.relative_to(images_dir).as_posix()

    # 3) 常见后缀兜底
    for ext in (".jpg", ".png", ".jpeg"):
        candidate = images_dir / f"{screen_id}{ext}"
        if candidate.exists():
            return candidate.relative_to(images_dir).as_posix()

    return None


def prepare_rico_screen2words(parquet_dir: Path, output_file: Path, images_dir: Optional[Path], image_suffix: Optional[str]):
    # 规范化后缀（带点）
    preferred_suffix = None
    if image_suffix:
        preferred_suffix = image_suffix if image_suffix.startswith(".") else f".{image_suffix}"

    # 加载 parquet 分片
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(parquet_dir / "train-*.parquet"),
            "val":   str(parquet_dir / "val-*.parquet"),
            "test":  str(parquet_dir / "test-*.parquet"),
        },
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    written_rows = 0
    skipped_missing_fields = 0
    missing_images = 0

    # 收集列名用于提示
    try:
        column_names = dataset["train"].column_names
    except Exception:
        column_names = []

    print(f"[Info] Loaded parquet splits from: {parquet_dir}")
    if column_names:
        print(f"[Info] Train columns: {column_names}")

    with output_file.open("w", encoding="utf-8") as f:
        for split in ("train", "val", "test"):
            if split not in dataset:
                continue
            for row in dataset[split]:
                total_rows += 1
                screen_id = row.get("screen_id") or row.get("id") or row.get("image_id")
                caption = row.get("caption") or row.get("summary") or row.get("description")

                if screen_id is None or caption is None:
                    skipped_missing_fields += 1
                    continue

                screen_id = str(screen_id)
                caption = str(caption).strip()

                # 解析图片相对路径
                if images_dir and images_dir.exists():
                    rel_path = find_image_relative_path(images_dir, screen_id, preferred_suffix)
                    if rel_path is None:
                        # 未找到实际文件，退回到统一后缀命名并计数
                        miss_name = f"{screen_id}{preferred_suffix or '.jpg'}"
                        rel_path = miss_name
                        missing_images += 1
                else:
                    # 未提供 images_dir，则直接使用统一命名
                    rel_path = f"{screen_id}{preferred_suffix or '.jpg'}"

                f.write(json.dumps({
                    "image": rel_path,
                    "caption": caption
                }, ensure_ascii=False) + "\n")
                written_rows += 1

    print(f"[OK] captions.jsonl 已生成: {output_file}")
    print(f"[Stats] total_rows={total_rows}, written={written_rows}, missing_fields={skipped_missing_fields}, missing_images={missing_images}")
    if images_dir and missing_images > 0:
        print("[Warn] 有图片未在 images_dir 中找到，已使用统一后缀命名。请确认后缀与目录结构是否一致。")


def main():
    parser = argparse.ArgumentParser(description="Prepare RICO-Screen2Words captions.jsonl from parquet")
    parser.add_argument("--parquet_dir", type=str, default="data/rico_screen2words/parquet", help="parquet 分片目录")
    parser.add_argument("--images_dir", type=str, default="data/rico_screen2words/images", help="RICO 截图目录（用于匹配真实文件，含子目录）")
    parser.add_argument("--output_file", type=str, default="data/rico_screen2words/captions.jsonl", help="输出 captions.jsonl 路径")
    parser.add_argument("--image_suffix", type=str, default=None, help="统一图片后缀（如 .jpg 或 jpg）。若提供 images_dir 会优先匹配真实文件")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None
    output_file = Path(args.output_file)

    if not parquet_dir.exists():
        raise FileNotFoundError(f"parquet_dir 不存在: {parquet_dir}")

    prepare_rico_screen2words(
        parquet_dir=parquet_dir,
        output_file=output_file,
        images_dir=images_dir,
        image_suffix=args.image_suffix,
    )


if __name__ == "__main__":
    main()

