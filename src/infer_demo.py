"""
推理演示脚本：对单张图片进行分类预测和文本检索
"""
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from model_siglip_lora import load_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Demo")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="预训练模型名称")
    parser.add_argument("--label_map", type=str, required=True,
                        help="标签映射文件路径")
    
    # 输入参数
    parser.add_argument("--image", type=str, required=True,
                        help="输入图像路径")
    parser.add_argument("--text_queries", type=str, nargs='+', default=None,
                        help="文本查询列表（用于检索演示）")
    parser.add_argument("--text_file", type=str, default=None,
                        help="包含文本查询的文件路径")
    
    # 推理参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    parser.add_argument("--max_text_length", type=int, default=64,
                        help="文本最大长度")
    parser.add_argument("--top_k", type=int, default=5,
                        help="显示top-k预测结果")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="输出目录")
    parser.add_argument("--save_visualization", action="store_true",
                        help="是否保存可视化结果")
    
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


def load_image(image_path):
    """加载和预处理图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict_classification(model, image, processor, label_map, device, top_k=5):
    """预测图像分类"""
    
    if model.classifier is None:
        print("Model does not have a classification head.")
        return None
    
    # 预处理图像
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs['cls_logits']
        probabilities = F.softmax(logits, dim=1)
    
    # 获取top-k预测
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # 创建类别名称映射
    id2label = {v: k for k, v in label_map.items()}
    
    # 格式化结果
    predictions = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        predictions.append({
            'rank': i + 1,
            'class': id2label[idx],
            'probability': float(prob),
            'confidence': float(prob) * 100
        })
    
    return predictions


def compute_text_similarity(model, image, text_queries, processor, tokenizer, device):
    """计算图像与文本的相似度"""
    
    # 预处理图像
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    
    # 预处理文本
    text_inputs = tokenizer(
        text_queries,
        max_length=64,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        # 编码图像
        image_outputs = model(pixel_values)
        image_embeds = image_outputs['image_embeds']
        
        # 编码文本
        text_outputs = model(
            pixel_values=None,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        text_embeds = text_outputs['text_embeds']
        
        # 计算相似度
        similarities = (image_embeds @ text_embeds.t()).cpu().numpy()[0]
    
    # 排序结果
    sorted_indices = np.argsort(similarities)[::-1]
    
    results = []
    for i, idx in enumerate(sorted_indices):
        results.append({
            'rank': i + 1,
            'text': text_queries[idx],
            'similarity': float(similarities[idx]),
            'score': float(similarities[idx]) * 100
        })
    
    return results


def visualize_results(image, classification_results, similarity_results, output_dir, save_fig=False):
    """可视化结果"""
    
    fig = plt.figure(figsize=(15, 8))
    
    # 显示原图
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Input Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 显示分类结果
    if classification_results:
        plt.subplot(1, 3, 2)
        classes = [r['class'] for r in classification_results[:5]]
        probs = [r['probability'] for r in classification_results[:5]]
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(classes)))
        
        bars = plt.barh(range(len(classes)), probs, color=colors)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel('Probability')
        plt.title('Classification Results', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', ha='left', va='center', fontsize=10)
    
    # 显示文本相似度结果
    if similarity_results:
        plt.subplot(1, 3, 3)
        texts = [r['text'][:30] + '...' if len(r['text']) > 30 else r['text'] 
                for r in similarity_results[:5]]
        scores = [r['similarity'] for r in similarity_results[:5]]
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(texts)))
        
        bars = plt.barh(range(len(texts)), scores, color=colors)
        plt.yticks(range(len(texts)), texts)
        plt.xlabel('Similarity Score')
        plt.title('Text Similarity Results', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = Path(output_dir) / "inference_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_results(classification_results, similarity_results):
    """打印结果到控制台"""
    
    print("=" * 60)
    print("推理结果")
    print("=" * 60)
    
    # 分类结果
    if classification_results:
        print("\n分类预测:")
        print("-" * 40)
        for result in classification_results:
            print(f"{result['rank']:2d}. {result['class']:25s} {result['confidence']:6.2f}%")
    
    # 文本相似度结果
    if similarity_results:
        print("\n文本相似度:")
        print("-" * 40)
        for result in similarity_results:
            print(f"{result['rank']:2d}. {result['text']:40s} {result['score']:6.2f}")
    
    print("=" * 60)


def save_results(image_path, classification_results, similarity_results, output_dir):
    """保存结果到文件"""
    
    results = {
        'input_image': str(image_path),
        'timestamp': str(torch.cuda.Event().record() if torch.cuda.is_available() else "N/A"),
        'classification': classification_results,
        'text_similarity': similarity_results
    }
    
    output_path = Path(output_dir) / "inference_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    args = parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载标签映射
    with open(args.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    print(f"Loaded {len(label_map)} classes")
    
    # 加载模型和处理器
    print("Loading model and processor...")
    model, processor, tokenizer = load_model_and_processor(args.model_path, args.model_name)
    model.to(args.device)
    
    # 加载图像
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    if image is None:
        return
    
    print(f"Image loaded: {image.size}")
    
    # 准备文本查询
    text_queries = []
    if args.text_queries:
        text_queries.extend(args.text_queries)
    
    if args.text_file and Path(args.text_file).exists():
        with open(args.text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    text_queries.append(line)
    
    # 如果没有提供文本查询，使用默认的
    if not text_queries:
        text_queries = [
            "应用主页界面",
            "点单购买页面", 
            "购物车页面",
            "支付结算页面",
            "用户个人中心",
            "商品列表页面",
            "搜索筛选页面",
            "订单详情页面",
            "设置配置页面",
            "登录注册页面"
        ]
    
    print(f"Using {len(text_queries)} text queries")
    
    # 分类预测
    print("Performing classification...")
    classification_results = predict_classification(
        model, image, processor, label_map, args.device, args.top_k
    )
    
    # 文本相似度计算
    print("Computing text similarities...")
    similarity_results = compute_text_similarity(
        model, image, text_queries, processor, tokenizer, args.device
    )
    
    # 打印结果
    print_results(classification_results, similarity_results)
    
    # 可视化结果
    if args.save_visualization:
        print("Generating visualization...")
        visualize_results(
            image, classification_results, similarity_results, 
            args.output_dir, save_fig=True
        )
    else:
        visualize_results(
            image, classification_results, similarity_results, 
            args.output_dir, save_fig=False
        )
    
    # 保存结果
    save_results(args.image, classification_results, similarity_results, args.output_dir)
    
    print("Inference completed!")


if __name__ == "__main__":
    main()