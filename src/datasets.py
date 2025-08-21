"""
数据集类实现：用于图像-文本对齐和页面分类任务
"""
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairCaptionDataset(Dataset):
    """图像-文本配对数据集，用于对比学习训练"""
    
    def __init__(self, root, jsonl, image_processor, tokenizer, max_len=64):
        """
        Args:
            root: 数据根目录
            jsonl: 包含image和caption的jsonl文件路径
            image_processor: 图像预处理器
            tokenizer: 文本分词器
            max_len: 文本最大长度
        """
        self.root = root
        self.items = []
        
        # 读取jsonl文件
        with open(jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        print(f"Loaded {len(self.items)} image-caption pairs from {jsonl}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        
        # 加载图像
        img_path = os.path.join(self.root, item['image'])
        if not os.path.exists(img_path):
            # 尝试在images子目录中查找
            img_path = os.path.join(self.root, 'images', item['image'])
            
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个默认的空白图像
            img = Image.new('RGB', (224, 224), color='white')
        
        # 图像预处理
        pixel_values = self.image_processor(images=img, return_tensors='pt')['pixel_values'][0]
        
        # 文本预处理（兼容无 attention_mask 的分词器）
        text_inputs = self.tokenizer(
            item['caption'], 
            max_length=self.max_len, 
            truncation=True,
            padding='max_length', 
            return_tensors='pt'
        )

        input_ids = text_inputs['input_ids'][0]
        # 有些 SigLIP/CLIP 系列分词器不会返回 attention_mask，这里做兜底
        if 'attention_mask' in text_inputs:
            attention_mask = text_inputs['attention_mask'][0]
        else:
            attention_mask = torch.ones_like(input_ids)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_name': item['image'],
            'caption': item['caption']
        }


class CaptionClsDataset(PairCaptionDataset):
    """图像-文本配对+分类数据集，用于多任务训练"""
    
    def __init__(self, root, jsonl_cap, jsonl_lbl, label_map_path, *args, **kwargs):
        """
        Args:
            root: 数据根目录
            jsonl_cap: 包含image和caption的jsonl文件路径
            jsonl_lbl: 包含image和label的jsonl文件路径
            label_map_path: 标签映射文件路径
            *args, **kwargs: 传递给父类的其他参数
        """
        super().__init__(root, jsonl_cap, *args, **kwargs)
        
        # 读取标签文件
        self.labels = {}
        with open(jsonl_lbl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    label_item = json.loads(line)
                    self.labels[label_item['image']] = label_item['label']
        
        # 读取标签映射
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)
        
        print(f"Loaded {len(self.labels)} labels with {self.num_classes} classes")

    def __getitem__(self, i):
        item = super().__getitem__(i)
        img_name = self.items[i]['image']
        
        # 添加标签信息
        if img_name in self.labels:
            label_name = self.labels[img_name]
            label_id = self.label2id[label_name]
        else:
            print(f"Warning: No label found for image {img_name}")
            label_id = 0  # 默认标签
        
        item['label'] = label_id
        item['label_name'] = self.id2label[label_id]
        
        return item


class ImageClsDataset(Dataset):
    """纯分类数据集，仅用于分类任务"""
    
    def __init__(self, root, jsonl_lbl, label_map_path, image_processor):
        """
        Args:
            root: 数据根目录
            jsonl_lbl: 包含image和label的jsonl文件路径
            label_map_path: 标签映射文件路径
            image_processor: 图像预处理器
        """
        self.root = root
        self.items = []
        
        # 读取标签文件
        with open(jsonl_lbl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))
        
        # 读取标签映射
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.label2id)
        self.image_processor = image_processor
        
        print(f"Loaded {len(self.items)} images for classification with {self.num_classes} classes")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        
        # 加载图像
        img_path = os.path.join(self.root, item['image'])
        if not os.path.exists(img_path):
            # 尝试在images子目录中查找
            img_path = os.path.join(self.root, 'images', item['image'])
            
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 创建一个默认的空白图像
            img = Image.new('RGB', (224, 224), color='white')
        
        # 图像预处理
        pixel_values = self.image_processor(images=img, return_tensors='pt')['pixel_values'][0]
        
        # 标签处理
        label_name = item['label']
        label_id = self.label2id[label_name]
        
        return {
            'pixel_values': pixel_values,
            'label': label_id,
            'label_name': label_name,
            'image_name': item['image']
        }


def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    """将数据集划分为训练、验证和测试集"""
    import random
    
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset