"""
扰动检测器模块
用于检测图像patch和文本token的扰动程度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDisturbDetector(nn.Module):
    """
    图像patch扰动检测器
    检测每个patch token相对于清洁样本的扰动程度
    """
    def __init__(self, dim=768, use_learnable=True):
        """
        Args:
            dim: patch token的维度（ViT-L/14: 1024, ViT-B/16: 768, ViT-B/32: 512）
            use_learnable: 是否使用可学习的检测器（推理阶段）
        """
        super().__init__()
        self.dim = dim
        self.use_learnable = use_learnable
        
        if use_learnable:
            # 可学习的扰动检测网络（用于推理阶段）
            self.detector = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出[0,1]的扰动分数
            )
        
        # 存储清洁样本的统计信息（用于推理阶段）
        self.register_buffer('clean_patch_mean', torch.zeros(dim))
        self.register_buffer('clean_patch_std', torch.ones(dim))
        self.register_buffer('is_initialized', torch.tensor(False))
    
    def update_clean_statistics(self, clean_patches):
        """
        更新清洁样本的统计信息
        Args:
            clean_patches: (B, N, dim) - 清洁图像的patch特征
        """
        with torch.no_grad():
            # 计算所有patch的均值和标准差
            patches_flat = clean_patches.view(-1, self.dim)
            self.clean_patch_mean = patches_flat.mean(dim=0)
            self.clean_patch_std = patches_flat.std(dim=0) + 1e-6
            self.is_initialized = torch.tensor(True)
    
    def forward(self, patch_tokens, patch_tokens_clean=None, mode='train'):
        """
        计算patch的扰动分数
        
        Args:
            patch_tokens: (B, N, dim) - 当前patch特征（可能被扰动）
            patch_tokens_clean: (B, N, dim) - 清洁patch特征（训练时提供）
            mode: 'train' 或 'eval'
        
        Returns:
            disturb_score: (B, N) - 扰动分数，范围[0,1]，越高表示扰动越强
        """
        B, N, D = patch_tokens.shape
        
        if mode == 'train' and patch_tokens_clean is not None:
            # 训练阶段：使用可学习检测器，输出需要有梯度用于训练
            if self.use_learnable:
                # 可学习检测器的预测（有梯度）
                disturb_score = self.detector(patch_tokens).squeeze(-1)
            else:
                # 如果不使用可学习检测器，使用余弦相似度
                disturb_score = 1 - F.cosine_similarity(
                    patch_tokens, patch_tokens_clean, dim=-1
                )
                disturb_score = torch.clamp(disturb_score, 0, 1)
                
        else:
            # 推理阶段：使用可学习检测器或统计方法
            if self.use_learnable:
                disturb_score = self.detector(patch_tokens).squeeze(-1)
            else:
                # 基于统计的检测
                if not self.is_initialized:
                    # 如果未初始化，使用当前batch估计
                    self.update_clean_statistics(patch_tokens)
                
                # 计算标准化后的L2距离
                normalized_patches = (patch_tokens - self.clean_patch_mean) / self.clean_patch_std
                disturb_score = normalized_patches.norm(dim=-1) / (D ** 0.5)
                disturb_score = torch.sigmoid(disturb_score - 1.0)  # 中心化
        
        return disturb_score


class TokenDisturbDetector(nn.Module):
    """
    文本token语义偏离检测器
    通过KL散度检测token的语义分布偏移
    """
    def __init__(self, vocab_size=49408, dim=768):
        """
        Args:
            vocab_size: CLIP文本编码器的词汇表大小（默认49408）
            dim: token特征维度（768 for ViT-L/14）
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # 可学习的语义分布投影层
        self.proj = nn.Linear(dim, vocab_size)
        
        # 存储清洁文本token的分布统计
        # 这将在训练时从数据集统计得到
        self.register_buffer('clean_token_dist', 
                            torch.ones(vocab_size) / vocab_size)
        self.register_buffer('is_dist_initialized', torch.tensor(False))
    
    def update_clean_distribution(self, clean_text_tokens):
        """
        更新清洁文本token的分布统计
        Args:
            clean_text_tokens: (B, L, dim) - 清洁文本的token特征
        """
        with torch.no_grad():
            # 计算清洁文本的平均token分布
            clean_logits = self.proj(clean_text_tokens)
            clean_dist = F.softmax(clean_logits, dim=-1)
            
            # 在所有token上平均
            self.clean_token_dist = clean_dist.mean(dim=[0, 1])
            self.is_dist_initialized = torch.tensor(True)
    
    def forward(self, text_tokens, text_tokens_clean=None, mode='train'):
        """
        计算文本token的语义偏离分数
        
        Args:
            text_tokens: (B, L, dim) - 当前文本token特征（可能被扰动）
            text_tokens_clean: (B, L, dim) - 清洁文本token特征（训练时）
            mode: 'train' 或 'eval'
        
        Returns:
            sem_score: (B, L) - 语义偏离分数，范围[0, ∞)，越高表示偏离越大
        """
        B, L, D = text_tokens.shape
        
        # 计算当前文本的token分布
        adv_logits = self.proj(text_tokens)
        adv_dist = F.softmax(adv_logits, dim=-1)  # (B, L, vocab_size)
        
        if mode == 'train' and text_tokens_clean is not None:
            # 训练阶段：计算与清洁样本的KL散度
            clean_logits = self.proj(text_tokens_clean)
            clean_dist = F.softmax(clean_logits, dim=-1)
            
            # KL(adv || clean)
            sem_score = F.kl_div(
                adv_dist.log(),
                clean_dist,
                reduction='none'
            ).sum(dim=-1)  # (B, L)
            
            # 更新统计分布（移动平均）
            if self.is_dist_initialized:
                momentum = 0.99
                with torch.no_grad():
                    batch_clean_dist = clean_dist.mean(dim=[0, 1])
                    self.clean_token_dist = (
                        momentum * self.clean_token_dist + 
                        (1 - momentum) * batch_clean_dist
                    )
            else:
                self.update_clean_distribution(text_tokens_clean)
        
        else:
            # 推理阶段：与统计分布比较
            if not self.is_dist_initialized:
                # 如果未初始化，使用均匀分布
                clean_dist_expanded = self.clean_token_dist.unsqueeze(0).unsqueeze(0)
            else:
                clean_dist_expanded = self.clean_token_dist.unsqueeze(0).unsqueeze(0)
            
            # KL(adv || clean_stat)
            sem_score = F.kl_div(
                adv_dist.log(),
                clean_dist_expanded.expand_as(adv_dist),
                reduction='none'
            ).sum(dim=-1)  # (B, L)
        
        # 归一化到[0,1]范围（使用sigmoid）
        sem_score = torch.sigmoid(sem_score - 1.0)
        
        return sem_score


class DisturbDetectorTrainer(nn.Module):
    """
    扰动检测器的联合训练包装器
    同时训练图像和文本的扰动检测
    """
    def __init__(self, img_dim=768, text_dim=768, vocab_size=49408):
        super().__init__()
        self.img_detector = PatchDisturbDetector(dim=img_dim, use_learnable=True)
        self.text_detector = TokenDisturbDetector(vocab_size=vocab_size, dim=text_dim)
    
    def forward(self, img_tokens, img_tokens_clean, 
                text_tokens, text_tokens_clean):
        """
        联合训练两个检测器
        
        Returns:
            dict: {
                'img_disturb': (B, N_img) - 图像扰动分数
                'text_disturb': (B, N_text) - 文本扰动分数
                'detector_loss': 检测器的辅助损失（可选）
            }
        """
        # 检测扰动
        img_disturb = self.img_detector(img_tokens, img_tokens_clean, mode='train')
        text_disturb = self.text_detector(text_tokens, text_tokens_clean, mode='train')
        
        # 可选：添加检测器的辅助损失
        # 这里可以添加对检测器本身的监督（如对比学习）
        detector_loss = torch.tensor(0.0, device=img_tokens.device)
        
        return {
            'img_disturb': img_disturb,
            'text_disturb': text_disturb,
            'detector_loss': detector_loss
        }


# 测试代码
if __name__ == '__main__':
    print("测试扰动检测器...")
    
    # 测试图像patch检测器
    print("\n1. 测试 PatchDisturbDetector")
    patch_detector = PatchDisturbDetector(dim=768)
    
    # 模拟数据
    B, N, D = 4, 197, 768
    patches = torch.randn(B, N, D)
    patches_clean = torch.randn(B, N, D)
    
    # 训练模式
    disturb_scores = patch_detector(patches, patches_clean, mode='train')
    print(f"   扰动分数 shape: {disturb_scores.shape}")
    print(f"   扰动分数范围: [{disturb_scores.min():.3f}, {disturb_scores.max():.3f}]")
    
    # 推理模式
    disturb_scores_eval = patch_detector(patches, mode='eval')
    print(f"   推理模式扰动分数: [{disturb_scores_eval.min():.3f}, {disturb_scores_eval.max():.3f}]")
    
    # 测试文本token检测器
    print("\n2. 测试 TokenDisturbDetector")
    token_detector = TokenDisturbDetector(vocab_size=49408, dim=768)
    
    B, L, D = 4, 77, 768
    text_tokens = torch.randn(B, L, D)
    text_tokens_clean = torch.randn(B, L, D)
    
    sem_scores = token_detector(text_tokens, text_tokens_clean, mode='train')
    print(f"   语义偏离分数 shape: {sem_scores.shape}")
    print(f"   语义偏离范围: [{sem_scores.min():.3f}, {sem_scores.max():.3f}]")
    
    print("\n✅ 扰动检测器测试通过！")
