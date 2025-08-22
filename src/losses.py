"""
损失函数实现：对比学习损失和分类损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(img_embeds, text_embeds, logit_scale, temperature=None):
    """
    InfoNCE对比学习损失（对称版本）
    
    Args:
        img_embeds: 图像嵌入 [batch_size, embed_dim]
        text_embeds: 文本嵌入 [batch_size, embed_dim] 
        logit_scale: 可学习的温度参数
        temperature: 固定温度参数（如果提供则忽略logit_scale）
    
    Returns:
        loss: 对比学习损失
        logits_per_image: 图像到文本的logits
        logits_per_text: 文本到图像的logits
    """
    # 确保嵌入已经归一化
    img_embeds = F.normalize(img_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # 计算相似度矩阵
    if temperature is not None:
        scale = 1.0 / temperature
    else:
        # 使用可学习的对数温度：exp(logit_scale)，并对 exp 后的尺度做上限裁剪（<=100）
        scale = torch.clamp(torch.exp(logit_scale), max=100.0)
    
    logits_per_image = scale * img_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()
    
    # 创建标签（对角线为正样本）
    batch_size = img_embeds.size(0)
    labels = torch.arange(batch_size, device=img_embeds.device)
    
    # 计算双向交叉熵损失
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    
    # 对称损失
    loss = (loss_img + loss_text) / 2
    
    return loss, logits_per_image, logits_per_text


def classification_loss(logits, labels, label_smoothing=0.0):
    """
    分类交叉熵损失
    
    Args:
        logits: 分类logits [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        label_smoothing: 标签平滑参数
    
    Returns:
        loss: 分类损失
    """
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def combined_loss(img_embeds, text_embeds, cls_logits, labels, logit_scale, 
                  lambda_align=0.5, temperature=None, label_smoothing=0.0):
    """
    联合损失：对比学习损失 + 分类损失
    
    Args:
        img_embeds: 图像嵌入
        text_embeds: 文本嵌入
        cls_logits: 分类logits
        labels: 分类标签
        logit_scale: 温度参数
        lambda_align: 对比学习损失权重
        temperature: 固定温度参数
        label_smoothing: 标签平滑参数
    
    Returns:
        total_loss: 总损失
        align_loss: 对比学习损失
        cls_loss: 分类损失
        logits_per_image: 图像到文本的logits
        logits_per_text: 文本到图像的logits
    """
    # 对比学习损失
    align_loss, logits_per_image, logits_per_text = contrastive_loss(
        img_embeds, text_embeds, logit_scale, temperature
    )
    
    # 分类损失
    cls_loss = classification_loss(cls_logits, labels, label_smoothing)
    
    # 联合损失
    total_loss = lambda_align * align_loss + (1 - lambda_align) * cls_loss
    
    return total_loss, align_loss, cls_loss, logits_per_image, logits_per_text


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡问题
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TripletLoss(nn.Module):
    """
    三元组损失，用于增强对比学习
    """
    
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: 锚点嵌入 [batch_size, embed_dim]
            positive: 正样本嵌入 [batch_size, embed_dim]  
            negative: 负样本嵌入 [batch_size, embed_dim]
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def compute_accuracy(logits, labels, topk=(1,)):
    """
    计算Top-K准确率
    
    Args:
        logits: 预测logits [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        topk: 要计算的top-k值
    
    Returns:
        dict: 包含各个top-k准确率的字典
    """
    maxk = max(topk)
    batch_size = labels.size(0)
    
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size).item()
        res[f'top{k}'] = acc
    
    return res


def compute_retrieval_metrics(img_embeds, text_embeds, k_values=(1, 5, 10)):
    """
    计算检索指标：Recall@K
    
    Args:
        img_embeds: 图像嵌入 [N, embed_dim]
        text_embeds: 文本嵌入 [N, embed_dim]
        k_values: 要计算的k值
    
    Returns:
        dict: 包含各种检索指标的字典
    """
    # 确保嵌入已归一化
    img_embeds = F.normalize(img_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # 计算相似度矩阵
    similarity = img_embeds @ text_embeds.t()
    
    N = similarity.size(0)
    ground_truth = torch.arange(N, device=similarity.device)
    
    metrics = {}
    
    # Image-to-Text检索
    _, i2t_ranks = similarity.sort(dim=1, descending=True)
    i2t_gt_ranks = (i2t_ranks == ground_truth.unsqueeze(1)).nonzero()[:, 1]
    
    for k in k_values:
        recall = (i2t_gt_ranks < k).float().mean().item() * 100
        metrics[f'i2t_recall@{k}'] = recall
    
    # Text-to-Image检索
    _, t2i_ranks = similarity.t().sort(dim=1, descending=True)
    t2i_gt_ranks = (t2i_ranks == ground_truth.unsqueeze(1)).nonzero()[:, 1]
    
    for k in k_values:
        recall = (t2i_gt_ranks < k).float().mean().item() * 100
        metrics[f't2i_recall@{k}'] = recall
    
    # 平均检索性能
    for k in k_values:
        avg_recall = (metrics[f'i2t_recall@{k}'] + metrics[f't2i_recall@{k}']) / 2
        metrics[f'avg_recall@{k}'] = avg_recall
    
    return metrics


def _ranks_from_similarity(similarity: torch.Tensor):
    """给定相似度矩阵 S[N,N]，返回：
    - i2t_ranks: 每个图像查询对应其正样本文本的排名(1-based)
    - t2i_ranks: 每个文本查询对应其正样本图像的排名(1-based)
    """
    N = similarity.size(0)
    ground_truth = torch.arange(N, device=similarity.device)

    # Image-to-Text ranks
    _, i2t_sorted_idx = similarity.sort(dim=1, descending=True)
    i2t_positions = (i2t_sorted_idx == ground_truth.unsqueeze(1)).nonzero()[:, 1]
    i2t_ranks = i2t_positions + 1  # 1-based

    # Text-to-Image ranks
    _, t2i_sorted_idx = similarity.t().sort(dim=1, descending=True)
    t2i_positions = (t2i_sorted_idx == ground_truth.unsqueeze(1)).nonzero()[:, 1]
    t2i_ranks = t2i_positions + 1  # 1-based

    return i2t_ranks, t2i_ranks


def compute_retrieval_metrics_full(img_embeds, text_embeds, k_values=(1, 5, 10)):
    """
    完整检索指标：Recall@K、Precision@K、mAP@K、NDCG@K、MRR、Mean/Median Rank、
    以及正样本/负样本相似度统计（基于已归一化的嵌入）。
    """
    img_embeds = F.normalize(img_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    similarity = img_embeds @ text_embeds.t()  # [N,N]
    N = similarity.size(0)
    device = similarity.device

    # Ranks
    i2t_ranks, t2i_ranks = _ranks_from_similarity(similarity)

    metrics = {}

    # MRR
    metrics['i2t_mrr'] = (1.0 / i2t_ranks.float()).mean().item()
    metrics['t2i_mrr'] = (1.0 / t2i_ranks.float()).mean().item()
    metrics['avg_mrr'] = (metrics['i2t_mrr'] + metrics['t2i_mrr']) / 2

    # Mean/Median rank
    metrics['i2t_mean_rank'] = i2t_ranks.float().mean().item()
    metrics['t2i_mean_rank'] = t2i_ranks.float().mean().item()
    metrics['i2t_median_rank'] = i2t_ranks.median().item()
    metrics['t2i_median_rank'] = t2i_ranks.median().item()

    # Recall@K, Precision@K, mAP@K, NDCG@K（单一正样本情形的简化）
    for k in k_values:
        # Recall@K（单一正样本即命中率）
        i2t_recall_k = (i2t_ranks <= k).float().mean().item() * 100
        t2i_recall_k = (t2i_ranks <= k).float().mean().item() * 100
        metrics[f'i2t_recall@{k}'] = i2t_recall_k
        metrics[f't2i_recall@{k}'] = t2i_recall_k
        metrics[f'avg_recall@{k}'] = (i2t_recall_k + t2i_recall_k) / 2

        # Precision@K（单一相关项：命中则 1/K，否则 0）
        i2t_prec_k = (i2t_ranks <= k).float().mean().item() * (1.0 / k)
        t2i_prec_k = (t2i_ranks <= k).float().mean().item() * (1.0 / k)
        metrics[f'i2t_precision@{k}'] = i2t_prec_k
        metrics[f't2i_precision@{k}'] = t2i_prec_k
        metrics[f'avg_precision@{k}'] = (i2t_prec_k + t2i_prec_k) / 2

        # AP@K（单一相关项：命中则 1/rank 且 rank<=K，否则 0）
        i2t_ap_k = torch.where(i2t_ranks <= k, 1.0 / i2t_ranks.float(), torch.zeros_like(i2t_ranks, dtype=torch.float)).mean().item()
        t2i_ap_k = torch.where(t2i_ranks <= k, 1.0 / t2i_ranks.float(), torch.zeros_like(t2i_ranks, dtype=torch.float)).mean().item()
        metrics[f'i2t_map@{k}'] = i2t_ap_k
        metrics[f't2i_map@{k}'] = t2i_ap_k
        metrics[f'avg_map@{k}'] = (i2t_ap_k + t2i_ap_k) / 2

        # NDCG@K（单一相关项：命中则 1/log2(1+rank)，否则 0；IDCG=1）
        def _ndcg(ranks):
            ranks_f = ranks.float()
            hits = (ranks_f <= k).float()
            gains = hits / torch.log2(ranks_f + 1)
            return gains.mean().item()

        i2t_ndcg_k = _ndcg(i2t_ranks)
        t2i_ndcg_k = _ndcg(t2i_ranks)
        metrics[f'i2t_ndcg@{k}'] = i2t_ndcg_k
        metrics[f't2i_ndcg@{k}'] = t2i_ndcg_k
        metrics[f'avg_ndcg@{k}'] = (i2t_ndcg_k + t2i_ndcg_k) / 2

    # 相似度统计（对角线为正样本；非对角线视为负样本的近似）
    pos_sim = similarity.diag()
    neg_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    neg_sim = similarity[neg_mask]
    metrics['pos_sim_mean'] = pos_sim.mean().item()
    metrics['pos_sim_std'] = pos_sim.std(unbiased=False).item()
    metrics['neg_sim_mean'] = neg_sim.mean().item()
    metrics['neg_sim_std'] = neg_sim.std(unbiased=False).item()

    return metrics