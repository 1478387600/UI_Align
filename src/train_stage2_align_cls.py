"""
Stage-2训练脚本：自建数据微调（对齐+分类）
在Stage-1的基础上，使用自建数据进行多任务训练：对齐+分类
"""
import os
import json
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from datasets import CaptionClsDataset, create_data_splits
from model_siglip_lora import create_model_and_processor, load_model_checkpoint, get_model_info
from losses import combined_loss, compute_accuracy, compute_retrieval_metrics


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 Alignment + Classification Training")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="预训练模型名称")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Stage-1检查点路径")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="是否使用4bit量化")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--freeze_ratio", type=float, default=0.6,
                        help="冻结层的比例")
    
    # 数据参数
    parser.add_argument("--train_images", type=str, default="data/custom_app/images",
                        help="训练图像目录")
    parser.add_argument("--train_captions", type=str, default="data/custom_app/captions.jsonl",
                        help="训练captions文件")
    parser.add_argument("--train_labels", type=str, default="data/custom_app/labels.jsonl",
                        help="训练labels文件")
    parser.add_argument("--label_map", type=str, default="data/custom_app/label_map.json",
                        help="标签映射文件")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--max_text_length", type=int, default=64,
                        help="文本最大长度")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="测试集比例")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/stage2",
                        help="输出目录")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"],
                        help="混合精度")
    parser.add_argument("--per_device_batch_size", type=int, default=16,
                        help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="基础学习率")
    parser.add_argument("--head_learning_rate", type=float, default=1e-3,
                        help="分类头学习率")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减")
    parser.add_argument("--epochs", type=int, default=8,
                        help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例")
    parser.add_argument("--lambda_align", type=float, default=0.5,
                        help="对齐损失权重")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 日志参数
    parser.add_argument("--logging_steps", type=int, default=20,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="评估步数")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="ui-align-stage2",
                        help="wandb项目名")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径")
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(args, processor, tokenizer, accelerator):
    """创建数据加载器"""
    
    # 读取标签映射获取类别数
    with open(args.label_map, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    
    # 创建完整数据集
    full_dataset = CaptionClsDataset(
        root=args.train_images,
        jsonl_cap=args.train_captions,
        jsonl_lbl=args.train_labels,
        label_map_path=args.label_map,
        image_processor=processor,
        tokenizer=tokenizer,
        max_len=args.max_text_length
    )
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = create_data_splits(
        full_dataset, 
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        seed=args.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    logger.info(f"  Number of classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


def evaluate_model(model, eval_loader, accelerator, compute_retrieval=True):
    """评估模型"""
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    all_cls_logits = []
    all_labels = []
    
    total_loss = 0
    total_align_loss = 0
    total_cls_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 计算损失
            loss, align_loss, cls_loss, _, _ = combined_loss(
                outputs['image_embeds'],
                outputs['text_embeds'],
                outputs['cls_logits'],
                labels,
                outputs['logit_scale'],
                lambda_align=0.5  # 使用固定权重进行评估
            )
            
            total_loss += loss.item()
            total_align_loss += align_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
            
            # 收集结果用于指标计算
            if compute_retrieval:
                image_embeds = accelerator.gather(outputs['image_embeds'])
                text_embeds = accelerator.gather(outputs['text_embeds'])
                all_image_embeds.append(image_embeds.cpu())
                all_text_embeds.append(text_embeds.cpu())
            
            cls_logits = accelerator.gather(outputs['cls_logits'])
            labels = accelerator.gather(labels)
            all_cls_logits.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
    
    # 计算指标
    metrics = {
        'eval_loss': total_loss / num_batches,
        'eval_align_loss': total_align_loss / num_batches,
        'eval_cls_loss': total_cls_loss / num_batches
    }
    
    if accelerator.is_local_main_process:
        # 分类指标
        all_cls_logits = torch.cat(all_cls_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        cls_metrics = compute_accuracy(all_cls_logits, all_labels, topk=(1, 3))
        metrics.update({f'eval_cls_{k}': v for k, v in cls_metrics.items()})
        
        # 检索指标
        if compute_retrieval:
            all_image_embeds = torch.cat(all_image_embeds, dim=0)
            all_text_embeds = torch.cat(all_text_embeds, dim=0)
            
            retrieval_metrics = compute_retrieval_metrics(all_image_embeds, all_text_embeds)
            metrics.update({f'eval_{k}': v for k, v in retrieval_metrics.items()})
    
    return metrics


def train_one_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, args):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    total_align_loss = 0
    total_cls_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                       disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(progress_bar):
        with accelerator.autocast():
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            # 前向传播
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 计算联合损失
            loss, align_loss, cls_loss, _, _ = combined_loss(
                outputs['image_embeds'],
                outputs['text_embeds'],
                outputs['cls_logits'],
                labels,
                outputs['logit_scale'],
                lambda_align=args.lambda_align
            )
        
        # 反向传播
        accelerator.backward(loss)
        
        # 梯度累积
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 更新统计
        total_loss += loss.item()
        total_align_loss += align_loss.item()
        total_cls_loss += cls_loss.item()
        
        # 更新进度条
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'align': f"{align_loss.item():.4f}",
                'cls': f"{cls_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 日志记录
        if (step + 1) % args.logging_steps == 0:
            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/align_loss': align_loss.item(),
                    'train/cls_loss': cls_loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/epoch': epoch + (step + 1) / num_batches,
                    'train/step': epoch * num_batches + step + 1
                })
    
    return {
        'train_loss': total_loss / num_batches,
        'train_align_loss': total_align_loss / num_batches,
        'train_cls_loss': total_cls_loss / num_batches
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, accelerator, is_best=False):
    """保存检查点"""
    if accelerator.is_local_main_process:
        if is_best:
            save_dir = Path(output_dir) / "best"
        else:
            save_dir = Path(output_dir) / f"checkpoint-epoch-{epoch+1}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型状态
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
        }, save_dir / "pytorch_model.bin")
        
        # 保存配置
        config = {
            'model_name': unwrapped_model.model_name,
            'embed_dim': unwrapped_model.embed_dim,
            'num_classes': unwrapped_model.num_classes,
            'epoch': epoch,
            'step': step,
            'is_best': is_best
        }
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"{'Best model' if is_best else 'Checkpoint'} saved to {save_dir}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # 加载配置文件
    if args.config:
        config = load_config(args.config)
        # 更新args
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    setattr(args, key, value)
    
    # 初始化accelerator
    accelerator = Accelerator(
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    if accelerator.is_local_main_process and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage2-{args.model_name.split('/')[-1]}"
        )
    
    # 创建模型和处理器
    logger.info("Creating model and processor...")
    
    # 首先创建数据加载器以获取类别数
    _, processor, tokenizer = create_model_and_processor(model_name=args.model_name)
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        args, processor, tokenizer, accelerator
    )
    
    # 创建带分类头的模型
    if args.resume_from:
        logger.info(f"Loading model from {args.resume_from}")
        model = load_model_checkpoint(
            args.resume_from + "/final/pytorch_model.bin",
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            freeze_ratio=args.freeze_ratio,
            num_classes=num_classes
        )
    else:
        model, _, _ = create_model_and_processor(
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            freeze_ratio=args.freeze_ratio,
            num_classes=num_classes
        )
    
    # 打印模型信息
    if accelerator.is_local_main_process:
        get_model_info(model)
    
    # 创建优化器（不同学习率）
    base_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                head_params.append(param)
            else:
                base_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': head_params, 'lr': args.head_learning_rate}
    ], weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Accelerate准备
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    
    # 训练循环
    logger.info("Starting training...")
    best_metric = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, 
                                      accelerator, epoch, args)
        
        # 验证
        val_metrics = evaluate_model(model, val_loader, accelerator)
        
        # 合并指标
        epoch_metrics = {**train_metrics, **val_metrics}
        
        # 打印结果
        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['eval_loss']:.4f}")
            logger.info(f"  Val Top-1 Acc: {val_metrics['eval_cls_top1']:.2f}%")
            logger.info(f"  Val Avg Recall@1: {val_metrics.get('eval_avg_recall@1', 0):.2f}%")
        
        # 早停检查
        current_metric = val_metrics['eval_cls_top1']  # 使用Top-1准确率作为主要指标
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          epoch * len(train_loader), args.output_dir, accelerator, is_best=True)
        else:
            patience_counter += 1
        
        # 常规保存
        if (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          epoch * len(train_loader), args.output_dir, accelerator)
        
        # 记录到wandb
        if accelerator.is_local_main_process and args.use_wandb:
            wandb.log({
                **{f'epoch/{k}': v for k, v in epoch_metrics.items()},
                'epoch/epoch': epoch + 1,
                'epoch/best_metric': best_metric
            })
        
        # 早停
        if patience_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # 最终测试
    logger.info("Running final evaluation on test set...")
    test_metrics = evaluate_model(model, test_loader, accelerator)
    
    if accelerator.is_local_main_process:
        logger.info("Final Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        if args.use_wandb:
            wandb.log({f'final_test/{k}': v for k, v in test_metrics.items()})
    
    # 完成训练
    if accelerator.is_local_main_process and args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()