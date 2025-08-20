"""
检索评估脚本：评估图像-文本检索性能
"""
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 导入自定义模块
from datasets import PairCaptionDataset, CaptionClsDataset
from model_siglip_lora import load_model_checkpoint, create_model_and_processor
from losses import compute_retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="预训练模型名称")
    
    # 数据参数
    parser.add_argument("--test_images", type=str, required=True,
                        help="测试图像目录")
    parser.add_argument("--test_captions", type=str, required=True,
                        help="测试captions文件")
    parser.add_argument("--test_labels", type=str, default=None,
                        help="测试labels文件（可选）")
    parser.add_argument("--label_map", type=str, default=None,
                        help="标签映射文件（可选）")
    parser.add_argument("--max_text_length", type=int, default=64,
                        help="文本最大长度")
    
    # 评估参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="输出目录")
    
    # 可视化参数
    parser.add_argument("--num_examples", type=int, default=10,
                        help="展示的检索示例数量")
    parser.add_argument("--save_examples", action="store_true",
                        help="是否保存检索示例图")
    
    return parser.parse_args()


def load_model_and_processor(model_path, model_name):
    """加载模型和处理器"""
    
    # 创建处理器
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型配置
    config_path = Path(model_path).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        num_classes = config.get('num_classes', None)
    else:
        num_classes = None
    
    # 加载模型
    model = load_model_checkpoint(
        model_path,
        model_name=model_name,
        num_classes=num_classes
    )
    
    return model, processor, tokenizer


def create_test_dataloader(args, processor, tokenizer):
    """创建测试数据加载器"""
    
    if args.test_labels and args.label_map:
        # 使用分类数据集
        dataset = CaptionClsDataset(
            root=args.test_images,
            jsonl_cap=args.test_captions,
            jsonl_lbl=args.test_labels,
            label_map_path=args.label_map,
            image_processor=processor,
            tokenizer=tokenizer,
            max_len=args.max_text_length
        )
    else:
        # 使用普通配对数据集
        dataset = PairCaptionDataset(
            root=args.test_images,
            jsonl=args.test_captions,
            image_processor=processor,
            tokenizer=tokenizer,
            max_len=args.max_text_length
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader, dataset


def extract_features(model, dataloader, device):
    """提取图像和文本特征"""
    
    model.eval()
    model.to(device)
    
    all_image_embeds = []
    all_text_embeds = []
    all_image_names = []
    all_captions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 收集特征
            all_image_embeds.append(outputs['image_embeds'].cpu())
            all_text_embeds.append(outputs['text_embeds'].cpu())
            all_image_names.extend(batch['image_name'])
            all_captions.extend(batch['caption'])
            
            if 'label' in batch:
                all_labels.extend(batch['label'].cpu().numpy())
    
    # 拼接特征
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    print(f"Extracted features for {len(image_embeds)} samples")
    
    return {
        'image_embeds': image_embeds,
        'text_embeds': text_embeds,
        'image_names': all_image_names,
        'captions': all_captions,
        'labels': all_labels if all_labels else None
    }


def evaluate_retrieval(image_embeds, text_embeds, k_values=(1, 5, 10)):
    """评估检索性能"""
    
    print("Computing retrieval metrics...")
    metrics = compute_retrieval_metrics(image_embeds, text_embeds, k_values)
    
    print("\nRetrieval Results:")
    print("=" * 50)
    
    for k in k_values:
        print(f"Recall@{k}:")
        print(f"  Image-to-Text: {metrics[f'i2t_recall@{k}']:.2f}%")
        print(f"  Text-to-Image: {metrics[f't2i_recall@{k}']:.2f}%")
        print(f"  Average: {metrics[f'avg_recall@{k}']:.2f}%")
        print()
    
    return metrics


def find_top_retrievals(query_embed, gallery_embeds, k=5):
    """找到top-k检索结果"""
    
    # 计算相似度
    similarity = query_embed @ gallery_embeds.t()
    
    # 获取top-k索引
    _, top_indices = similarity.topk(k, dim=-1)
    top_similarities = similarity.gather(-1, top_indices)
    
    return top_indices.cpu().numpy(), top_similarities.cpu().numpy()


def visualize_retrieval_examples(features, output_dir, num_examples=10, top_k=5):
    """可视化检索示例"""
    
    image_embeds = features['image_embeds']
    text_embeds = features['text_embeds']
    image_names = features['image_names']
    captions = features['captions']
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 随机选择一些查询
    indices = np.random.choice(len(image_embeds), min(num_examples, len(image_embeds)), replace=False)
    
    for i, idx in enumerate(indices):
        # Image-to-Text检索
        query_image_embed = image_embeds[idx:idx+1]
        top_indices, top_similarities = find_top_retrievals(query_image_embed, text_embeds, k=top_k)
        
        # 保存结果
        result = {
            'query_image': image_names[idx],
            'query_caption': captions[idx],
            'retrieved_texts': []
        }
        
        for j, (text_idx, sim) in enumerate(zip(top_indices[0], top_similarities[0])):
            result['retrieved_texts'].append({
                'rank': j + 1,
                'image': image_names[text_idx],
                'caption': captions[text_idx],
                'similarity': float(sim)
            })
        
        # 保存到文件
        with open(output_dir / f"i2t_example_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(indices)} image-to-text retrieval examples to {output_dir}")
    
    # Text-to-Image检索
    for i, idx in enumerate(indices):
        query_text_embed = text_embeds[idx:idx+1]
        top_indices, top_similarities = find_top_retrievals(query_text_embed, image_embeds, k=top_k)
        
        # 保存结果
        result = {
            'query_text': captions[idx],
            'query_image': image_names[idx],
            'retrieved_images': []
        }
        
        for j, (img_idx, sim) in enumerate(zip(top_indices[0], top_similarities[0])):
            result['retrieved_images'].append({
                'rank': j + 1,
                'image': image_names[img_idx],
                'caption': captions[img_idx],
                'similarity': float(sim)
            })
        
        # 保存到文件
        with open(output_dir / f"t2i_example_{i+1}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(indices)} text-to-image retrieval examples to {output_dir}")


def analyze_by_category(features, metrics):
    """按类别分析检索性能"""
    
    if features['labels'] is None:
        return
    
    labels = np.array(features['labels'])
    unique_labels = np.unique(labels)
    
    print("\nPer-category Analysis:")
    print("=" * 50)
    
    category_results = {}
    
    for label in unique_labels:
        # 获取该类别的样本索引
        mask = labels == label
        indices = np.where(mask)[0]
        
        if len(indices) < 2:  # 需要至少2个样本
            continue
        
        # 提取该类别的特征
        cat_image_embeds = features['image_embeds'][indices]
        cat_text_embeds = features['text_embeds'][indices]
        
        # 计算该类别的检索指标
        cat_metrics = compute_retrieval_metrics(cat_image_embeds, cat_text_embeds)
        category_results[int(label)] = cat_metrics
        
        print(f"Category {label} ({len(indices)} samples):")
        print(f"  Avg Recall@1: {cat_metrics['avg_recall@1']:.2f}%")
        print(f"  Avg Recall@5: {cat_metrics['avg_recall@5']:.2f}%")
    
    return category_results


def save_results(metrics, category_results, output_dir):
    """保存评估结果"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存总体结果
    with open(output_dir / "retrieval_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存类别结果
    if category_results:
        with open(output_dir / "category_metrics.json", 'w') as f:
            json.dump(category_results, f, indent=2)
    
    # 生成报告
    with open(output_dir / "evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write("检索评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("总体检索性能:\n")
        for k in [1, 5, 10]:
            if f'avg_recall@{k}' in metrics:
                f.write(f"  Recall@{k}: {metrics[f'avg_recall@{k}']:.2f}%\n")
        f.write("\n")
        
        if category_results:
            f.write("各类别检索性能:\n")
            for label, cat_metrics in category_results.items():
                f.write(f"  类别 {label}:\n")
                f.write(f"    Recall@1: {cat_metrics['avg_recall@1']:.2f}%\n")
                f.write(f"    Recall@5: {cat_metrics['avg_recall@5']:.2f}%\n")
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def main():
    args = parse_args()
    
    # 加载模型和处理器
    print("Loading model and processor...")
    model, processor, tokenizer = load_model_and_processor(args.model_path, args.model_name)
    
    # 创建测试数据加载器
    print("Creating test dataloader...")
    test_loader, test_dataset = create_test_dataloader(args, processor, tokenizer)
    
    # 提取特征
    print("Extracting features...")
    features = extract_features(model, test_loader, args.device)
    
    # 评估检索性能
    metrics = evaluate_retrieval(features['image_embeds'], features['text_embeds'])
    
    # 按类别分析
    category_results = analyze_by_category(features, metrics)
    
    # 可视化检索示例
    if args.save_examples:
        print("Generating retrieval examples...")
        visualize_retrieval_examples(features, args.output_dir, args.num_examples)
    
    # 保存结果
    save_results(metrics, category_results, args.output_dir)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()