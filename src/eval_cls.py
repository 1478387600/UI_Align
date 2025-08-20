"""
分类评估脚本：评估页面分类性能
"""
import os
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
from tqdm import tqdm

# 导入自定义模块
from datasets import CaptionClsDataset, ImageClsDataset
from model_siglip_lora import load_model_checkpoint
from losses import compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Classification Evaluation")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="预训练模型名称")
    
    # 数据参数
    parser.add_argument("--test_images", type=str, required=True,
                        help="测试图像目录")
    parser.add_argument("--test_labels", type=str, required=True,
                        help="测试labels文件")
    parser.add_argument("--label_map", type=str, required=True,
                        help="标签映射文件")
    parser.add_argument("--test_captions", type=str, default=None,
                        help="测试captions文件（可选）")
    parser.add_argument("--max_text_length", type=int, default=64,
                        help="文本最大长度")
    
    # 评估参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="输出目录")
    
    return parser.parse_args()


def load_model_and_processor(model_path, model_name):
    """加载模型和处理器"""
    
    # 创建处理器
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name) if model_name else None
    
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
    
    if args.test_captions:
        # 使用多任务数据集
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
        # 使用纯分类数据集
        dataset = ImageClsDataset(
            root=args.test_images,
            jsonl_lbl=args.test_labels,
            label_map_path=args.label_map,
            image_processor=processor
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


def evaluate_classification(model, dataloader, device, dataset):
    """评估分类性能"""
    
    model.eval()
    model.to(device)
    
    all_logits = []
    all_labels = []
    all_image_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label']
            
            # 前向传播
            if hasattr(dataset, 'tokenizer') and 'input_ids' in batch:
                # 多任务模型
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(pixel_values, input_ids, attention_mask)
            else:
                # 纯分类模型
                outputs = model(pixel_values)
            
            # 收集结果
            all_logits.append(outputs['cls_logits'].cpu())
            all_labels.append(labels)
            all_image_names.extend(batch['image_name'])
    
    # 拼接结果
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return logits, labels, all_image_names


def compute_metrics(logits, labels, label_names):
    """计算分类指标"""
    
    # 预测结果
    predictions = torch.argmax(logits, dim=1)
    
    # 基本准确率
    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    
    # Top-k准确率
    top_k_metrics = compute_accuracy(logits, labels, topk=(1, 3, 5))
    
    # 分类报告
    report = classification_report(
        labels.numpy(), 
        predictions.numpy(),
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(labels.numpy(), predictions.numpy())
    
    print("Classification Results:")
    print("=" * 50)
    print(f"Total samples: {len(labels)}")
    print(f"Top-1 Accuracy: {top_k_metrics['top1']:.2f}%")
    if len(label_names) >= 3:
        print(f"Top-3 Accuracy: {top_k_metrics['top3']:.2f}%")
    if len(label_names) >= 5:
        print(f"Top-5 Accuracy: {top_k_metrics['top5']:.2f}%")
    print()
    
    # 按类别显示结果
    print("Per-class Results:")
    print("-" * 30)
    for i, class_name in enumerate(label_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            support = report[str(i)]['support']
            print(f"{class_name:20s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, N={support}")
    
    return {
        'accuracy': accuracy,
        'top_k_metrics': top_k_metrics,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions.numpy(),
        'probabilities': torch.softmax(logits, dim=1).numpy()
    }


def plot_confusion_matrix(cm, class_names, output_dir, normalize=False):
    """绘制混淆矩阵"""
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        filename = 'confusion_matrix_normalized.png'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        filename = 'confusion_matrix.png'
        fmt = 'd'
    
    plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def analyze_errors(logits, labels, image_names, class_names, output_dir, top_n=10):
    """分析预测错误"""
    
    predictions = torch.argmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    
    # 找到错误预测的样本
    wrong_indices = (predictions != labels).nonzero().squeeze()
    if wrong_indices.dim() == 0:
        wrong_indices = wrong_indices.unsqueeze(0)
    
    if len(wrong_indices) == 0:
        print("No prediction errors found!")
        return
    
    # 计算预测置信度
    max_probs = torch.max(probabilities, dim=1)[0]
    
    # 分析最容易混淆的类别对
    error_pairs = {}
    for idx in wrong_indices:
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        pair = (true_label, pred_label)
        
        if pair not in error_pairs:
            error_pairs[pair] = []
        
        error_pairs[pair].append({
            'image': image_names[idx],
            'confidence': max_probs[idx].item(),
            'true_prob': probabilities[idx, true_label].item(),
            'pred_prob': probabilities[idx, pred_label].item()
        })
    
    # 按错误数量排序
    sorted_pairs = sorted(error_pairs.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nError Analysis ({len(wrong_indices)} total errors):")
    print("=" * 50)
    print("Most confused class pairs:")
    
    error_analysis = {}
    for i, ((true_idx, pred_idx), errors) in enumerate(sorted_pairs[:top_n]):
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        count = len(errors)
        avg_conf = np.mean([e['confidence'] for e in errors])
        
        print(f"{i+1:2d}. {true_name} → {pred_name}: {count} errors (avg conf: {avg_conf:.3f})")
        
        error_analysis[f"{true_name}_to_{pred_name}"] = {
            'count': count,
            'avg_confidence': avg_conf,
            'examples': errors[:5]  # 保存前5个例子
        }
    
    # 保存错误分析
    with open(Path(output_dir) / "error_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    return error_analysis


def save_predictions(predictions, probabilities, image_names, labels, class_names, output_dir):
    """保存预测结果"""
    
    results = []
    for i in range(len(predictions)):
        result = {
            'image': image_names[i],
            'true_label': class_names[labels[i]],
            'predicted_label': class_names[predictions[i]],
            'correct': predictions[i] == labels[i],
            'confidence': float(probabilities[i, predictions[i]]),
            'true_probability': float(probabilities[i, labels[i]]),
            'all_probabilities': {
                class_names[j]: float(probabilities[i, j]) 
                for j in range(len(class_names))
            }
        }
        results.append(result)
    
    # 保存到文件
    with open(Path(output_dir) / "predictions.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Predictions saved to {Path(output_dir) / 'predictions.json'}")


def generate_report(metrics, class_names, output_dir):
    """生成评估报告"""
    
    report_path = Path(output_dir) / "classification_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("分类评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 总体性能
        f.write("总体性能:\n")
        f.write(f"  Top-1 准确率: {metrics['top_k_metrics']['top1']:.2f}%\n")
        if 'top3' in metrics['top_k_metrics']:
            f.write(f"  Top-3 准确率: {metrics['top_k_metrics']['top3']:.2f}%\n")
        if 'top5' in metrics['top_k_metrics']:
            f.write(f"  Top-5 准确率: {metrics['top_k_metrics']['top5']:.2f}%\n")
        f.write(f"  宏平均 F1: {metrics['classification_report']['macro avg']['f1-score']:.3f}\n")
        f.write(f"  加权平均 F1: {metrics['classification_report']['weighted avg']['f1-score']:.3f}\n\n")
        
        # 各类别性能
        f.write("各类别性能:\n")
        f.write(f"{'类别':20s} {'精确率':>8s} {'召回率':>8s} {'F1分数':>8s} {'样本数':>8s}\n")
        f.write("-" * 60 + "\n")
        
        for i, class_name in enumerate(class_names):
            if str(i) in metrics['classification_report']:
                report = metrics['classification_report'][str(i)]
                f.write(f"{class_name:20s} {report['precision']:8.3f} {report['recall']:8.3f} "
                       f"{report['f1-score']:8.3f} {report['support']:8d}\n")
    
    print(f"Report saved to {report_path}")


def main():
    args = parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载标签映射
    with open(args.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    class_names = [None] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name
    
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # 加载模型和处理器
    print("Loading model and processor...")
    model, processor, tokenizer = load_model_and_processor(args.model_path, args.model_name)
    
    # 创建测试数据加载器
    print("Creating test dataloader...")
    test_loader, test_dataset = create_test_dataloader(args, processor, tokenizer)
    
    # 评估分类性能
    print("Evaluating classification performance...")
    logits, labels, image_names = evaluate_classification(model, test_loader, args.device, test_dataset)
    
    # 计算指标
    metrics = compute_metrics(logits, labels, class_names)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.output_dir)
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, args.output_dir, normalize=True)
    
    # 错误分析
    error_analysis = analyze_errors(logits, labels, image_names, class_names, args.output_dir)
    
    # 保存预测结果
    save_predictions(metrics['predictions'], metrics['probabilities'], 
                    image_names, labels.numpy(), class_names, args.output_dir)
    
    # 生成报告
    generate_report(metrics, class_names, args.output_dir)
    
    # 保存指标
    with open(Path(args.output_dir) / "metrics.json", 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        serializable_metrics = {
            'accuracy': metrics['accuracy'],
            'top_k_metrics': metrics['top_k_metrics'],
            'classification_report': metrics['classification_report']
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()