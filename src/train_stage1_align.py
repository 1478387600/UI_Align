"""
Stage-1训练脚本：RICO+Screen2Words预适配训练
仅进行图像-文本对齐训练，使用对比学习损失
"""
import os
import sys
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

# 导入自定义模块
from datasets import PairCaptionDataset
from model_siglip_lora import create_model_and_processor, get_model_info
from losses import contrastive_loss, compute_retrieval_metrics


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 Alignment Training")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-224",
                        help="预训练模型名称")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="是否使用4bit量化")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--freeze_ratio", type=float, default=0.6,
                        help="冻结层的比例")
    
    # 数据参数
    parser.add_argument("--train_images", type=str, default="data/rico_screen2words/images",
                        help="训练图像目录")
    parser.add_argument("--train_captions", type=str, default="data/rico_screen2words/captions.jsonl",
                        help="训练captions文件")
    parser.add_argument("--image_size", type=int, default=224,
                        help="图像尺寸")
    parser.add_argument("--max_text_length", type=int, default=64,
                        help="文本最大长度")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/stage1",
                        help="输出目录")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"],
                        help="混合精度")
    parser.add_argument("--per_device_batch_size", type=int, default=16,
                        help="每设备批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker 数")
    
    # 日志参数
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="ui-align-stage1",
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


def create_dataloader(args, processor, tokenizer, accelerator):
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = PairCaptionDataset(
        root=args.train_images,
        jsonl=args.train_captions,
        image_processor=processor,
        tokenizer=tokenizer,
        max_len=args.max_text_length
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    effective_bs = args.per_device_batch_size * max(1, args.gradient_accumulation_steps)
    logger.info(f"Created dataloader with {len(train_dataset)} samples")
    logger.info(f"per_device_batch_size={args.per_device_batch_size}, grad_acc_steps={args.gradient_accumulation_steps}, effective_batch~={effective_bs}, steps/epoch={len(train_loader)}")
    
    return train_loader


def evaluate_model(model, eval_loader, accelerator):
    """评估模型"""
    model.eval()
    
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 收集嵌入
            image_embeds = accelerator.gather(outputs['image_embeds'])
            text_embeds = accelerator.gather(outputs['text_embeds'])
            
            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
    
    # 计算检索指标
    if accelerator.is_local_main_process:
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        
        metrics = compute_retrieval_metrics(all_image_embeds, all_text_embeds)
        return metrics
    
    return {}


def train_one_epoch(model, train_loader, optimizer, scheduler, accelerator, epoch, args):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", 
                       disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(progress_bar):
        with accelerator.autocast():
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            # 前向传播
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 计算损失
            loss, logits_per_image, logits_per_text = contrastive_loss(
                outputs['image_embeds'], 
                outputs['text_embeds'], 
                outputs['logit_scale']
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
        
        # 更新进度条
        if accelerator.is_local_main_process:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(step+1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 日志记录
        if (step + 1) % args.logging_steps == 0:
            if accelerator.is_local_main_process and args.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/epoch': epoch + (step + 1) / num_batches,
                    'train/step': epoch * num_batches + step + 1
                })
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, accelerator):
    """保存检查点"""
    if accelerator.is_local_main_process:
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
            'step': step
        }
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Checkpoint saved to {save_dir}")


def _maybe_eval(model, args, processor, tokenizer, accelerator):
    if not hasattr(args, 'eval_enabled') or not args.eval_enabled:
        return {}
    # 构造一个较小的评估 DataLoader（从训练同源数据采样）
    dataset = PairCaptionDataset(
        root=args.train_images,
        jsonl=args.train_captions,
        image_processor=processor,
        tokenizer=tokenizer,
        max_len=args.max_text_length,
    )
    # 采样 eval_sample_size
    N = len(dataset)
    k = min(getattr(args, 'eval_sample_size', 2000), N)
    indices = torch.randperm(N)[:k].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    eval_loader = DataLoader(
        subset,
        batch_size=getattr(args, 'eval_batch_size', args.per_device_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    eval_loader = accelerator.prepare(eval_loader)
    metrics = evaluate_model(model, eval_loader, accelerator)
    return metrics


def main():
    args = parse_args()
    
    # 加载配置文件（CLI 优先于配置文件）
    if args.config:
        config = load_config(args.config)
        # 收集 CLI 显式提供的键（--key 或 --key=value）
        cli_keys = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                k = tok[2:].split("=")[0]
                cli_keys.add(k)
        # 更新 args：仅对未在 CLI 指定的键应用配置
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if key in cli_keys:
                        continue
                    setattr(args, key, value)

    # 强制类型纠正：防止 YAML/环境变量将数值读取为字符串
    def _to_float(x, name):
        try:
            return float(x)
        except Exception:
            logger.warning(f"Cast to float failed: {name}={x}")
            return x

    def _to_int(x, name):
        try:
            return int(x)
        except Exception:
            logger.warning(f"Cast to int failed: {name}={x}")
            return x

    def _to_bool(x, name):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(x)

    # 数值/布尔字段列表
    for k in ["learning_rate", "weight_decay", "warmup_ratio", "freeze_ratio"]:
        if hasattr(args, k):
            setattr(args, k, _to_float(getattr(args, k), k))
    for k in [
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "seed",
        "image_size",
        "max_text_length",
        "lora_r",
        "lora_alpha",
    ]:
        if hasattr(args, k):
            setattr(args, k, _to_int(getattr(args, k), k))
    for k in ["load_in_4bit", "use_wandb"]:
        if hasattr(args, k):
            setattr(args, k, _to_bool(getattr(args, k), k))
    
    # matmul 精度优化（加速 bfloat16/fp16）
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

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
            name=f"stage1-{args.model_name.split('/')[-1]}"
        )
    
    # 创建模型和处理器
    logger.info("Creating model and processor...")
    model, processor, tokenizer = create_model_and_processor(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_ratio=args.freeze_ratio
    )
    
    # 打印模型信息
    if accelerator.is_local_main_process:
        get_model_info(model)
    
    # 创建数据加载器
    logger.info("Creating dataloader...")
    train_loader = create_dataloader(args, processor, tokenizer, accelerator)
    
    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Accelerate准备
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    
    # 训练循环
    logger.info("Starting training...")
    best_recall = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, 
                                 accelerator, epoch, args)
        
        logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        # 可选评估（Recall@K）
        metrics = _maybe_eval(model, args, processor, tokenizer, accelerator)
        if metrics and accelerator.is_local_main_process:
            logger.info(f"Eval metrics (sampled): {metrics}")
            if args.use_wandb:
                wandb.log({f'eval/{k}': v for k, v in metrics.items()})
            # 维护 best_recall@5
            best_recall = max(best_recall, metrics.get('avg_recall@5', 0.0))
        
        # 保存检查点
        if (epoch + 1) % 1 == 0:  # 每个epoch保存一次
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          epoch * len(train_loader), args.output_dir, accelerator)
        
        # 记录到wandb
        if accelerator.is_local_main_process and args.use_wandb:
            wandb.log({
                'epoch/train_loss': avg_loss,
                'epoch/epoch': epoch + 1
            })
    
    # 保存最终模型
    if accelerator.is_local_main_process:
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), final_dir / "pytorch_model.bin")
        
        config = {
            'model_name': unwrapped_model.model_name,
            'embed_dim': unwrapped_model.embed_dim,
            'num_classes': unwrapped_model.num_classes,
            'final_epoch': args.epochs
        }
        
        with open(final_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Final model saved to {final_dir}")
    
    # 完成训练
    if accelerator.is_local_main_process and args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()