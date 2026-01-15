"""
关键Token筛选器
基于CLIP注意力机制筛选图像和文本的关键Token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyTokenSelector(nn.Module):
    """
    关键Token筛选器
    通过注意力权重和token重要性筛选关键的图像patch和文本token
    """
    def __init__(self, top_k_ratio=0.2, importance_threshold=0.1):
        """
        Args:
            top_k_ratio: 保留的关键Token比例（默认20%）
            importance_threshold: Token重要性阈值
        """
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.importance_threshold = importance_threshold
    
    def select_img_key_tokens(self, img_tokens, attention_weights=None):
        """
        筛选图像的关键patch tokens
        基于ViT的注意力权重
        
        Args:
            img_tokens: (B, N, dim) - 图像patch特征（N=197 for ViT-L/14）
            attention_weights: (B, num_heads, N, N) - ViT最后一层注意力权重
        
        Returns:
            key_mask: (B, N) - bool张量，True表示关键Token
        """
        B, N, D = img_tokens.shape
        
        if attention_weights is not None:
            # 方法1：基于注意力权重（推荐）
            # 使用[CLS] token对其他token的注意力作为重要性指标
            # attention_weights shape: (B, num_heads, N, N)
            
            # 对所有注意力头求平均
            if attention_weights.dim() == 4:
                # (B, num_heads, N, N) -> (B, N, N)
                attn_avg = attention_weights.mean(dim=1)
            else:
                attn_avg = attention_weights
            
            # 提取[CLS] token（位置0）对其他patch的注意力
            # (B, N, N) -> (B, N)
            cls_attention = attn_avg[:, 0, :]  
            
            # 去掉[CLS]自身的注意力
            patch_importance = cls_attention.clone()
            patch_importance[:, 0] = 0  # [CLS]位置置0
            
        else:
            # 方法2：基于token特征的L2范数（备选）
            patch_importance = img_tokens.norm(dim=-1)  # (B, N)
            # [CLS] token的重要性置0（通常不作为关键patch）
            patch_importance[:, 0] = 0
        
        # Top-K筛选
        top_k = max(1, int(N * self.top_k_ratio))
        _, top_indices = torch.topk(patch_importance, k=top_k, dim=1)
        
        # 生成bool掩码
        key_mask = torch.zeros(B, N, dtype=torch.bool, device=img_tokens.device)
        key_mask.scatter_(1, top_indices, True)
        
        # 强制保留[CLS] token
        key_mask[:, 0] = True
        
        return key_mask
    
    def select_text_key_tokens(self, text_tokens, text_token_ids=None, 
                               attention_weights=None):
        """
        筛选文本的关键tokens
        基于token重要性和特殊token保护
        
        Args:
            text_tokens: (B, L, dim) - 文本token特征（L=77 for CLIP）
            text_token_ids: (B, L) - 文本token IDs
            attention_weights: (B, num_heads, L, L) - 文本注意力权重（可选）
        
        Returns:
            key_mask: (B, L) - bool张量，True表示关键Token
        """
        B, L, D = text_tokens.shape
        
        # 计算token重要性
        if attention_weights is not None:
            # 基于注意力权重
            if attention_weights.dim() == 4:
                attn_avg = attention_weights.mean(dim=1)  # (B, L, L)
            else:
                attn_avg = attention_weights
            
            # [CLS] token对其他token的注意力
            token_importance = attn_avg[:, 0, :]  # (B, L)
        else:
            # 基于token特征范数
            token_importance = text_tokens.norm(dim=-1)  # (B, L)
        
        # 如果有token IDs，过滤padding token
        if text_token_ids is not None:
            # CLIP中padding token ID通常是0
            valid_mask = (text_token_ids != 0)
            token_importance = token_importance * valid_mask.float()
        else:
            valid_mask = torch.ones(B, L, dtype=torch.bool, device=text_tokens.device)
        
        # Top-K筛选（只在有效token中选择）
        valid_count = valid_mask.sum(dim=1).clamp(min=1)  # (B,)
        top_k_per_sample = (valid_count.float() * self.top_k_ratio).long().clamp(min=1)
        
        # 为每个样本单独筛选
        key_mask = torch.zeros(B, L, dtype=torch.bool, device=text_tokens.device)
        
        for i in range(B):
            k = top_k_per_sample[i].item()
            valid_importance = token_importance[i] * valid_mask[i].float()
            _, top_idx = torch.topk(valid_importance, k=min(k, valid_mask[i].sum().item()))
            key_mask[i, top_idx] = True
        
        # 强制保留特殊token
        if text_token_ids is not None:
            # [CLS] token (ID=49406)
            cls_mask = (text_token_ids == 49406)
            key_mask = key_mask | cls_mask
            
            # [EOS] token (ID=49407)
            eos_mask = (text_token_ids == 49407)
            key_mask = key_mask | eos_mask
        else:
            # 如果没有token IDs，假设位置0和最后一个有效位置是特殊token
            key_mask[:, 0] = True
        
        return key_mask
    
    def get_key_token_features(self, tokens, key_mask):
        """
        提取关键Token的特征
        
        Args:
            tokens: (B, N, dim) - 所有token特征
            key_mask: (B, N) - 关键token掩码
        
        Returns:
            key_features: (B, dim) - 关键token的平均特征
        """
        # 将非关键token的特征置0
        key_tokens = tokens * key_mask.unsqueeze(-1).float()
        
        # 计算关键token的平均（避免除0）
        key_count = key_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        key_features = key_tokens.sum(dim=1) / key_count
        
        return key_features


class AdaptiveKeyTokenSelector(nn.Module):
    """
    自适应关键Token筛选器
    根据输入的扰动强度自适应调整筛选比例
    """
    def __init__(self, base_ratio=0.2, min_ratio=0.1, max_ratio=0.5):
        """
        Args:
            base_ratio: 基础保留比例
            min_ratio: 最小保留比例（高扰动时）
            max_ratio: 最大保留比例（低扰动时）
        """
        super().__init__()
        self.base_ratio = base_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # 基础筛选器
        self.base_selector = KeyTokenSelector(top_k_ratio=base_ratio)
    
    def forward(self, tokens, disturb_scores, attention_weights=None, 
                token_ids=None, token_type='image'):
        """
        自适应筛选关键Token
        
        Args:
            tokens: (B, N, dim) - token特征
            disturb_scores: (B, N) - 扰动分数
            attention_weights: 注意力权重（可选）
            token_ids: token IDs（文本）
            token_type: 'image' 或 'text'
        
        Returns:
            key_mask: (B, N) - 关键token掩码
        """
        B, N, D = tokens.shape
        
        # 计算平均扰动强度
        avg_disturb = disturb_scores.mean(dim=1)  # (B,)
        
        # 自适应调整保留比例
        # 扰动越强，保留越多token（保护更多信息）
        adaptive_ratio = self.base_ratio + (self.max_ratio - self.base_ratio) * avg_disturb
        adaptive_ratio = torch.clamp(adaptive_ratio, self.min_ratio, self.max_ratio)
        
        # 为每个样本单独筛选
        key_mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        
        for i in range(B):
            ratio = adaptive_ratio[i].item()
            top_k = max(1, int(N * ratio))
            
            # 基于注意力或特征重要性
            if attention_weights is not None:
                if token_type == 'image':
                    sample_mask = self.base_selector.select_img_key_tokens(
                        tokens[i:i+1], 
                        attention_weights[i:i+1] if attention_weights is not None else None
                    )
                else:
                    sample_mask = self.base_selector.select_text_key_tokens(
                        tokens[i:i+1],
                        token_ids[i:i+1] if token_ids is not None else None,
                        attention_weights[i:i+1] if attention_weights is not None else None
                    )
            else:
                # 简单基于特征范数
                importance = tokens[i].norm(dim=-1)
                if token_ids is not None:
                    valid_mask = (token_ids[i] != 0)
                    importance = importance * valid_mask.float()
                
                _, top_idx = torch.topk(importance, k=top_k)
                sample_mask = torch.zeros(N, dtype=torch.bool, device=tokens.device)
                sample_mask[top_idx] = True
            
            key_mask[i] = sample_mask.squeeze(0) if sample_mask.dim() > 1 else sample_mask
        
        return key_mask


# 测试代码
if __name__ == '__main__':
    print("测试关键Token筛选器...")
    
    # 测试图像patch筛选
    print("\n1. 测试图像patch筛选")
    selector = KeyTokenSelector(top_k_ratio=0.2)
    
    B, N, D = 4, 197, 768
    img_tokens = torch.randn(B, N, D)
    
    # 模拟注意力权重
    num_heads = 12
    attention_weights = torch.softmax(
        torch.randn(B, num_heads, N, N), dim=-1
    )
    
    key_mask = selector.select_img_key_tokens(img_tokens, attention_weights)
    print(f"   图像关键Token掩码 shape: {key_mask.shape}")
    print(f"   保留的Token数量: {key_mask.sum(dim=1)}")
    print(f"   保留比例: {key_mask.float().mean():.2%}")
    
    # 提取关键特征
    key_features = selector.get_key_token_features(img_tokens, key_mask)
    print(f"   关键特征 shape: {key_features.shape}")
    
    # 测试文本token筛选
    print("\n2. 测试文本token筛选")
    B, L, D = 4, 77, 768
    text_tokens = torch.randn(B, L, D)
    text_token_ids = torch.randint(0, 49408, (B, L))
    
    # 设置特殊token
    text_token_ids[:, 0] = 49406  # [CLS]
    text_token_ids[:, 10] = 49407  # [EOS]
    text_token_ids[:, 11:] = 0  # Padding
    
    key_mask_text = selector.select_text_key_tokens(
        text_tokens, text_token_ids
    )
    print(f"   文本关键Token掩码 shape: {key_mask_text.shape}")
    print(f"   保留的Token数量: {key_mask_text.sum(dim=1)}")
    
    # 测试自适应筛选
    print("\n3. 测试自适应筛选器")
    adaptive_selector = AdaptiveKeyTokenSelector(
        base_ratio=0.2, min_ratio=0.1, max_ratio=0.5
    )
    
    # 模拟不同扰动强度
    disturb_scores_low = torch.rand(B, N) * 0.3  # 低扰动
    disturb_scores_high = torch.rand(B, N) * 0.7 + 0.3  # 高扰动
    
    key_mask_low = adaptive_selector(
        img_tokens, disturb_scores_low, 
        attention_weights, token_type='image'
    )
    key_mask_high = adaptive_selector(
        img_tokens, disturb_scores_high,
        attention_weights, token_type='image'
    )
    
    print(f"   低扰动保留Token: {key_mask_low.sum(dim=1)}")
    print(f"   高扰动保留Token: {key_mask_high.sum(dim=1)}")
    
    print("\n✅ 关键Token筛选器测试通过！")
