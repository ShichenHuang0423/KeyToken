"""
KeyToken融合损失函数
结合对比学习损失、鲁棒性损失、扰动检测损失和重建损失

核心思想：
1. 对比学习（InfoNCE）：强化图文对齐能力
2. L2鲁棒性：对抗样本与干净样本embedding对齐
3. MAE重建：修复被扰动的token
4. 扰动检测：监督检测器学习识别被扰动的token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyTokenLoss(nn.Module):
    """
    KeyToken融合损失函数（无监督版本 + 对比学习）
    
    L_total = λ_contrastive * L_contrastive + λ_robust * L_robust + λ_mae * L_mae + λ_detect * L_detect
    
    其中：
    - L_contrastive: 对比学习损失（InfoNCE），强化图文对齐
    - L_robust: 特征鲁棒性损失（L2），保持embedding稳定
    - L_mae: MAE重建损失，修复被扰动的token
    - L_detect: 扰动检测损失（可选），监督检测器学习
    """
    
    def __init__(self, 
                 contrastive_weight=1.0,
                 robust_weight=0.5,
                 mae_weight=1.0,
                 detect_weight=0.1,
                 temperature=0.07,
                 logit_scale=100.0):
        """
        Args:
            contrastive_weight: 对比学习损失权重
            robust_weight: 鲁棒性L2损失权重
            mae_weight: MAE重建损失权重
            detect_weight: 扰动检测损失权重
            temperature: 对比学习温度参数
            logit_scale: CLIP logit缩放因子（用于兼容旧接口）
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.robust_weight = robust_weight
        self.mae_weight = mae_weight
        self.detect_weight = detect_weight
        self.temperature = temperature
        self.logit_scale = logit_scale
    
    def contrastive_loss(self, image_embeddings, text_embeddings, targets=None):
        """
        对比学习损失（InfoNCE）
        
        当有targets时：使用类别文本embedding进行对比学习
        当无targets时：使用batch内的图文配对进行对比学习
        
        Args:
            image_embeddings: (B, D) 图像embedding
            text_embeddings: (D, num_classes) 或 (B, D) 文本embedding（假设已归一化）
            targets: (B,) 真实标签（可选）
        
        Returns:
            loss: 对比学习损失
        """
        # 确保image embedding已归一化
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        if targets is not None and text_embeddings.dim() == 2 and text_embeddings.size(0) != image_embeddings.size(0):
            # 使用类别文本embedding：text_embeddings是(D, num_classes)
            # 注意：CLIP的text_embeddings已经是归一化的，不需要再次归一化
            # 直接使用，避免破坏已有的归一化
            
            # 计算所有图文对的相似度
            logits = image_embeddings @ text_embeddings / self.temperature  # (B, num_classes)
            
            # 使用targets作为正样本索引
            loss = F.cross_entropy(logits, targets)
        else:
            # Batch内对比学习：假设batch内第i个图像和第i个文本是配对的
            text_embeddings = F.normalize(text_embeddings, dim=-1)
            
            # 计算相似度矩阵
            logits = image_embeddings @ text_embeddings.T / self.temperature  # (B, B)
            
            # 对角线是正样本
            batch_size = image_embeddings.size(0)
            labels = torch.arange(batch_size, device=image_embeddings.device)
            
            # 双向对比损失
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def robust_loss(self, embedding_adv, embedding_orig, key_mask=None):
        """
        关键Token级别的鲁棒性损失
        
        原理：只对关键Token的embedding施加L2约束，允许非关键Token自由变化
        这样既保持了关键语义的鲁棒性，又允许MAE等模块修复被扰动的token
        
        Args:
            embedding_adv: (B, D) 对抗样本embedding
            embedding_orig: (B, D) 原始干净样本embedding
            key_mask: (B, N) 关键token掩码（可选），True表示关键token
        
        Returns:
            loss: 加权L2损失
        """
        if key_mask is not None:
            # 计算每个样本的关键token比例作为权重
            # 关键token越多，L2约束越强；关键token越少，允许更多自由变化
            key_ratio = key_mask.float().mean(dim=1, keepdim=True)  # (B, 1)
            
            # 基础L2损失
            loss = F.mse_loss(embedding_adv, embedding_orig, reduction='none')  # (B, D)
            
            # 按关键token比例加权：关键token多→更强约束，关键token少→更弱约束
            # 这里用sqrt来缓和权重变化，避免权重过小
            weight = torch.sqrt(key_ratio + 0.1)  # (B, 1), 范围约[0.32, 1.05]
            loss = (loss * weight).sum(dim=1).mean()
        else:
            # 回退到原始L2
            loss = F.mse_loss(embedding_adv, embedding_orig, reduction='none')
            loss = loss.sum(dim=1).mean()
        return loss
    
    def detection_loss(self, pred_disturb, target_disturb):
        """
        扰动检测损失
        监督检测器学习识别被扰动的token
        
        Args:
            pred_disturb: (B, N) 预测的扰动分数
            target_disturb: (B, N) 真实的扰动程度（基于特征差异计算）
        
        Returns:
            loss: MSE损失（更稳定，允许连续值）
        """
        # 将target归一化到[0,1]范围，避免BCE的数值问题
        # 使用sigmoid归一化：大于1的值映射到接近1
        target_normalized = torch.sigmoid(target_disturb.detach() * 2 - 1)  # 0.5处映射到0.5
        
        # 使用MSE损失代替BCE，更稳定且允许连续值
        loss = F.mse_loss(pred_disturb, target_normalized)
        return loss
    
    def forward(self, 
                embedding_adv,
                embedding_orig,
                text_embeddings=None,
                targets=None,
                mae_loss=None,
                pred_disturb=None,
                target_disturb=None,
                key_mask=None):
        """
        计算总损失
        
        Args:
            embedding_adv: (B, D) 对抗样本的最终embedding
            embedding_orig: (B, D) 原始干净样本的embedding（来自原始模型）
            text_embeddings: (D, num_classes) 文本类别embedding（用于对比学习）
            targets: (B,) 真实标签（用于对比学习的正样本选择）
            mae_loss: 预计算的MAE重建损失（可选）
            pred_disturb: (B, N) 预测的扰动分数（可选）
            target_disturb: (B, N) 真实的扰动程度（可选）
            key_mask: (B, N) 关键token掩码（可选）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. 对比学习损失（强化图文对齐能力）
        if self.contrastive_weight > 0 and text_embeddings is not None:
            loss_contrastive = self.contrastive_loss(embedding_adv, text_embeddings, targets)
            total_loss = total_loss + self.contrastive_weight * loss_contrastive
            loss_dict['loss_contrastive'] = loss_contrastive.item()
        
        # 2. 鲁棒性损失（关键Token级别，保持关键语义稳定）
        if self.robust_weight > 0:
            loss_robust = self.robust_loss(embedding_adv, embedding_orig, key_mask)
            total_loss = total_loss + self.robust_weight * loss_robust
            loss_dict['loss_robust'] = loss_robust.item()
        
        # 3. MAE重建损失
        if self.mae_weight > 0 and mae_loss is not None:
            if isinstance(mae_loss, torch.Tensor):
                if mae_loss.dim() > 0:
                    mae_loss = mae_loss.mean()
                total_loss = total_loss + self.mae_weight * mae_loss
                loss_dict['loss_mae'] = mae_loss.item()
        
        # 4. 扰动检测损失（可选）
        if self.detect_weight > 0 and pred_disturb is not None and target_disturb is not None:
            loss_detect = self.detection_loss(pred_disturb, target_disturb)
            total_loss = total_loss + self.detect_weight * loss_detect
            loss_dict['loss_detect'] = loss_detect.item()
        
        loss_dict['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict


def compute_keytoken_loss(embedding_adv, 
                          embedding_orig,
                          targets=None,
                          text_embeddings=None,
                          mae_loss=None,
                          pred_disturb=None,
                          target_disturb=None,
                          key_mask=None,
                          contrastive_weight=1.0,
                          robust_weight=0.5,
                          mae_weight=1.0,
                          detect_weight=0.1,
                          temperature=0.07,
                          logit_scale=100.0):
    """
    便捷函数：计算KeyToken融合损失（对比学习版本）
    
    Args:
        embedding_adv: 对抗样本embedding
        embedding_orig: 原始embedding
        targets: 真实标签（用于对比学习正样本选择）
        text_embeddings: 文本类别embedding（用于对比学习）
        mae_loss: MAE重建损失
        pred_disturb: 预测的扰动分数
        target_disturb: 真实的扰动程度
        key_mask: 关键token掩码（用于关键Token级别Robust Loss）
        contrastive_weight: 对比学习损失权重
        robust_weight: 鲁棒性损失权重
        mae_weight: MAE损失权重
        detect_weight: 检测损失权重
        temperature: 对比学习温度参数
        logit_scale: logit缩放因子（兼容旧接口）
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失
    """
    loss_fn = KeyTokenLoss(
        contrastive_weight=contrastive_weight,
        robust_weight=robust_weight,
        mae_weight=mae_weight,
        detect_weight=detect_weight,
        temperature=temperature,
        logit_scale=logit_scale
    )
    
    return loss_fn(
        embedding_adv=embedding_adv,
        embedding_orig=embedding_orig,
        text_embeddings=text_embeddings,
        targets=targets,
        mae_loss=mae_loss,
        pred_disturb=pred_disturb,
        target_disturb=target_disturb,
        key_mask=key_mask
    )


# 测试代码
if __name__ == '__main__':
    print("测试KeyToken融合损失（对比学习版本）...")
    
    # 模拟数据
    B, D, num_classes = 4, 768, 1000
    N = 197  # ViT patch数量
    
    embedding_adv = torch.randn(B, D)
    embedding_adv = F.normalize(embedding_adv, dim=-1)
    embedding_orig = torch.randn(B, D)
    embedding_orig = F.normalize(embedding_orig, dim=-1)
    # text_embeddings: (D, num_classes) 格式
    text_embeddings = torch.randn(D, num_classes)
    text_embeddings = F.normalize(text_embeddings, dim=0)
    targets = torch.randint(0, num_classes, (B,))
    mae_loss = torch.tensor(0.5)
    pred_disturb = torch.sigmoid(torch.randn(B, N))
    target_disturb = torch.sigmoid(torch.randn(B, N))
    
    # 测试损失计算
    loss_fn = KeyTokenLoss(
        contrastive_weight=1.0,
        robust_weight=0.5,
        mae_weight=1.0,
        detect_weight=0.1,
        temperature=0.07
    )
    
    total_loss, loss_dict = loss_fn(
        embedding_adv=embedding_adv,
        embedding_orig=embedding_orig,
        text_embeddings=text_embeddings,
        targets=targets,
        mae_loss=mae_loss,
        pred_disturb=pred_disturb,
        target_disturb=target_disturb
    )
    
    print(f"\n损失计算结果:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n✅ KeyToken融合损失（对比学习版本）测试通过!")
