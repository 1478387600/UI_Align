"""
SigLIP双塔模型实现，支持QLoRA量化和LoRA微调
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
# 兼容服务器上 bitsandbytes 不可用/无 GPU 的场景
try:
    from bitsandbytes import BitsAndBytesConfig  # type: ignore
except Exception:  # noqa: BLE001
    BitsAndBytesConfig = None  # 动态回退：禁用4bit量化
import math


class SigLipDualEncoder(nn.Module):
    """SigLIP双塔编码器，支持图像-文本对齐和分类任务"""
    
    def __init__(self, 
                 model_name='google/siglip-base-patch16-224',
                 load_in_4bit=True,
                 lora_r=16,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 freeze_ratio=0.6,
                 num_classes=None,
                 device_map='auto'):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 设备与精度选择（bf16 优先，其次 fp16；CPU 用 fp32）
        def _choose_dtype() -> torch.dtype:
            if torch.cuda.is_available():
                try:
                    major_cc = torch.cuda.get_device_capability(0)[0]
                    return torch.bfloat16 if major_cc >= 8 else torch.float16  # Ampere(8.x) 及以上优先 bf16
                except Exception:  # noqa: BLE001
                    return torch.float16
            return torch.float32

        torch_dtype = _choose_dtype()

        # 量化配置（在 GPU 且 bnb 可用时才启用）
        bnb_available = (BitsAndBytesConfig is not None) and torch.cuda.is_available()
        if load_in_4bit and bnb_available:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        else:
            if load_in_4bit and not bnb_available:
                print("[Warn] bitsandbytes 不可用或未检测到GPU，已回退为非量化加载（float16/bfloat16/float32）")
            quantization_config = None
        
        # 加载预训练模型
        resolved_device_map = device_map if torch.cuda.is_available() else None
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            device_map=resolved_device_map,
            trust_remote_code=True,
        )
        
        # 获取模型配置
        self.config = self.model.config
        
        # 获取嵌入维度
        if hasattr(self.config, 'projection_dim'):
            self.embed_dim = self.config.projection_dim
        elif hasattr(self.config, 'hidden_size'):
            self.embed_dim = self.config.hidden_size
        else:
            self.embed_dim = 768  # 默认值
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
        # 冻结部分层
        self._freeze_layers(freeze_ratio)
        
        # 应用LoRA
        self._apply_lora(lora_r, lora_alpha, lora_dropout)
        
        # 分类头（如果需要）
        if num_classes is not None:
            self.classifier = ImageClassifierHead(self.embed_dim, num_classes)
        else:
            self.classifier = None
        
        print(f"Model initialized with embed_dim={self.embed_dim}, num_classes={num_classes}")
    
    def _freeze_layers(self, freeze_ratio):
        """冻结指定比例的层"""
        if freeze_ratio <= 0:
            return
        
        # 获取所有transformer层
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
            vision_layers = self.model.vision_model.encoder.layers
            text_layers = getattr(self.model.text_model.encoder, 'layers', [])
        else:
            # 尝试其他可能的结构
            vision_layers = []
            text_layers = []
        
        # 冻结vision layers
        if vision_layers:
            freeze_count = int(len(vision_layers) * freeze_ratio)
            for i in range(freeze_count):
                for param in vision_layers[i].parameters():
                    param.requires_grad = False
            print(f"Frozen {freeze_count}/{len(vision_layers)} vision layers")
        
        # 冻结text layers  
        if text_layers:
            freeze_count = int(len(text_layers) * freeze_ratio)
            for i in range(freeze_count):
                for param in text_layers[i].parameters():
                    param.requires_grad = False
            print(f"Frozen {freeze_count}/{len(text_layers)} text layers")
    
    def _apply_lora(self, lora_r, lora_alpha, lora_dropout):
        """应用LoRA配置，仅对 vision 分支与视觉投影层注入，避免 text 分支触发 inputs_embeds 路径。"""
        if lora_r is None or lora_r <= 0:
            print("[Info] LoRA disabled (r<=0)")
            return

        # 仅对 vision_model 子模块注入 LoRA，避免影响 text 塔 forward（导致 inputs_embeds 路径）
        vm = getattr(self.model, "vision_model", None)
        if vm is None:
            print("[Warn] vision_model not found on base model; skip LoRA")
            return

        vision_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
        inner_targets = []
        for name, _ in vm.named_modules():
            if any(name.endswith(sfx) for sfx in vision_suffixes):
                inner_targets.append(name)

        if not inner_targets:
            print("[Warn] No vision modules found for LoRA injection in vision_model; skipping LoRA")
            return

        # 创建LoRA配置（目标为精确模块名列表）
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=inner_targets,
            bias="none",
        )

        try:
            vm_lora = get_peft_model(vm, lora_config)
            self.model.vision_model = vm_lora
            print(f"Applied LoRA with r={lora_r}, alpha={lora_alpha} on {len(inner_targets)} vision modules (vision_model)")
        except Exception as e:
            print(f"Warning: Failed to apply LoRA on vision_model: {e}")
            print("Continuing without LoRA...")
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None, return_features=False):
        """
        前向传播（调用顶层 SigLIP 模型，避免将 text 关键字误传入 vision 子模块）
        """
        model_inputs = {"pixel_values": pixel_values}
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        outputs = self.model(**model_inputs, return_dict=True)

        # 未归一化特征用于分类
        image_features = outputs.image_embeds
        text_features = getattr(outputs, "text_embeds", None)

        # 归一化后的特征用于对齐损失
        image_embeds = F.normalize(image_features, dim=-1)
        text_embeds = F.normalize(text_features, dim=-1) if text_features is not None else None

        cls_logits = None
        if self.classifier is not None:
            cls_logits = self.classifier(image_features)

        result = {
            "image_embeds": image_embeds,
            "logit_scale": self.logit_scale.exp(),
        }
        if text_embeds is not None:
            result["text_embeds"] = text_embeds
        if cls_logits is not None:
            result["cls_logits"] = cls_logits
        if return_features:
            result["raw_image_features"] = image_features
            if text_features is not None:
                result["raw_text_features"] = text_features

        return result
    
    def encode_image(self, pixel_values):
        """编码图像"""
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
        return image_embeds
    
    def encode_text(self, input_ids, attention_mask):
        """编码文本"""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
        return text_embeds
    
    def get_trainable_params(self):
        """获取可训练参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params * 100
        }


class ImageClassifierHead(nn.Module):
    """图像分类头"""
    
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
        # 初始化
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


def create_model_and_processor(model_name='google/siglip-base-patch16-224', **model_kwargs):
    """创建模型和处理器的便捷函数"""
    
    # 创建处理器
    # 优先尝试 fast 版；如因缺少 torchvision 或其他后端导致 ImportError，则回退为非 fast 版
    try:
        processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    except (TypeError, ImportError, Exception):
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建模型
    model = SigLipDualEncoder(model_name=model_name, **model_kwargs)
    
    return model, processor, tokenizer


def load_model_checkpoint(checkpoint_path, model_name='google/siglip-base-patch16-224', **model_kwargs):
    """从检查点加载模型"""
    
    # 创建模型
    model = SigLipDualEncoder(model_name=model_name, **model_kwargs)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 加载状态字典
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Model loaded from {checkpoint_path}")
    
    return model


def get_model_info(model):
    """获取模型信息"""
    info = model.get_trainable_params()
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Trainable ratio: {info['trainable_ratio']:.2f}%")
    
    if model.num_classes:
        print(f"Number of classes: {model.num_classes}")
    
    return info