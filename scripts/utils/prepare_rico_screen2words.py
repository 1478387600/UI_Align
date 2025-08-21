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


def detect_column_name(columns, candidates):
    """在列名中根据候选集合和启发式规则选择最合适的列名。

    candidates: 优先顺序的候选名称列表（小写比较）。
    返回：匹配到的列名或 None。
    """
    lower_map = {c.lower(): c for c in columns}

    # 1) 直接精确匹配
    for name in candidates:
        if name in lower_map:
            return lower_map[name]

    # 2) 含有关键字的匹配（启发式）
    def pick_by_keywords(keywords):
        for col in columns:
            lc = col.lower()
            if all(k in lc for k in keywords):
                return col
        return None

    # id类
    id_kw = pick_by_keywords(["screen"]) or pick_by_keywords(["image","id"]) or pick_by_keywords(["screenshot"]) or pick_by_keywords(["id"])
    # caption类
    cap_kw = pick_by_keywords(["caption"]) or pick_by_keywords(["summary"]) or pick_by_keywords(["description"]) or pick_by_keywords(["text"]) or pick_by_keywords(["label"]) or pick_by_keywords(["sentence"]) or pick_by_keywords(["title"]) or pick_by_keywords(["comment"]) or pick_by_keywords(["content"]) or pick_by_keywords(["captions"]) 

    return id_kw if candidates is None else cap_kw


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


def prepare_rico_screen2words(
    parquet_dir: Path,
    output_file: Path,
    images_dir: Optional[Path],
    image_suffix: Optional[str],
    id_key: Optional[str] = None,
    caption_key: Optional[str] = None,
):
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

    # 自动探测列名（若未指定）
    detected_id_key = id_key
    detected_caption_key = caption_key
    if column_names:
        lower_cols = [c.lower() for c in column_names]
        if not detected_id_key:
            for name in ["screen_id","screenshot_id","image_id","id","screenId","screenshotId"]:
                if name.lower() in lower_cols:
                    detected_id_key = column_names[lower_cols.index(name.lower())]
                    break
        if not detected_caption_key:
            for name in ["caption","summary","description","text","label","sentence","title","captions","content"]:
                if name.lower() in lower_cols:
                    detected_caption_key = column_names[lower_cols.index(name.lower())]
                    break

    print(f"[Info] Using id_key={detected_id_key!r}, caption_key={detected_caption_key!r}")

    with output_file.open("w", encoding="utf-8") as f:
        for split in ("train", "val", "test"):
            if split not in dataset:
                continue
            for idx, row in enumerate(dataset[split]):
                total_rows += 1
                # 优先使用指定/探测到的列名
                screen_id = None
                caption = None
                if detected_id_key and detected_id_key in row:
                    screen_id = row.get(detected_id_key)
                else:
                    screen_id = row.get("screen_id") or row.get("id") or row.get("image_id") or row.get("screenshot_id")

                if detected_caption_key and detected_caption_key in row:
                    caption = row.get(detected_caption_key)
                else:
                    caption = row.get("caption") or row.get("summary") or row.get("description") or row.get("text") or row.get("label") or row.get("captions")

                if screen_id is None or caption is None:
                    skipped_missing_fields += 1
                    # 打印前几个缺失样例，辅助排查
                    if skipped_missing_fields <= 3:
                        print(f"[Warn] Missing fields at split={split}, idx={idx}. Sample keys={list(row.keys())[:10]}")
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
    parser.add_argument("--id_key", type=str, default=None, help="parquet中作为screen_id的列名（未指定则自动探测）")
    parser.add_argument("--caption_key", type=str, default=None, help="parquet中作为caption的列名（未指定则自动探测）")
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
        id_key=args.id_key,
        caption_key=args.caption_key,
    )


if __name__ == "__main__":
    main()

