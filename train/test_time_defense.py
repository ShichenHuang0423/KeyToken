"""
Test-time Adversarial Defense Strategies (Training-free)
测试时对抗防御策略（无需训练）

包含两种方法：
1. Interpretability-Guided Defense: 基于神经元重要性排名
2. ZeroPur: 对抗净化（引导漂移 + 自适应投影）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class InterpretabilityGuidedDefense:
    """
    Interpretability-Guided Test-Time Adversarial Defense
    
    通过识别对输出类别重要的神经元来提高鲁棒性。
    无需训练，即插即用，计算开销小。
    
    Reference: Interpretability-Guided Test-Time Adversarial Defense
    """
    
    def __init__(self, model, layer_name: str = 'final_layer', top_k_ratio: float = 0.3):
        """
        Args:
            model: 目标模型
            layer_name: 要分析的层名称
            top_k_ratio: 保留top-k重要神经元的比例
        """
        self.model = model
        self.layer_name = layer_name
        self.top_k_ratio = top_k_ratio
        self.activations = {}
        self.gradients = {}
        
        # 注册hook
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向hook以捕获激活和梯度"""
        def forward_hook(module, input, output):
            self.activations['value'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0].detach()
        
        # 为最后一层注册hook（可根据实际模型调整）
        # 这里假设是全连接层或归一化后的特征
        # 实际使用时需要根据模型结构修改
        pass
    
    def compute_neuron_importance(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        计算神经元重要性
        
        Args:
            x: 输入图像 (B, C, H, W)
            target_class: 目标类别
        
        Returns:
            importance_scores: 神经元重要性分数 (B, D)
        """
        self.model.eval()
        
        # 前向传播
        x.requires_grad = True
        output = self.model(x)
        
        if output.dim() > 2:
            output = output.view(output.size(0), -1)
        
        # 计算对目标类别的梯度
        target_score = output[:, target_class].sum()
        target_score.backward(retain_graph=True)
        
        # 使用梯度 × 激活作为重要性度量（类似Grad-CAM）
        if x.grad is not None:
            # 简化版本：使用输入梯度的绝对值作为重要性
            importance = x.grad.abs().mean(dim=[2, 3])  # (B, C)
        else:
            # 备选方案：使用输出本身
            importance = output.abs()
        
        return importance
    
    def purify(self, x: torch.Tensor, predicted_class: int) -> torch.Tensor:
        """
        通过抑制不重要的神经元来净化输入
        
        Args:
            x: 输入图像 (B, C, H, W)
            predicted_class: 预测的类别
        
        Returns:
            purified_x: 净化后的图像
        """
        with torch.enable_grad():
            importance = self.compute_neuron_importance(x, predicted_class)
        
        # 选择top-k重要的通道
        k = max(1, int(importance.size(1) * self.top_k_ratio))
        _, top_indices = importance.topk(k, dim=1)
        
        # 创建mask（只保留重要通道）
        mask = torch.zeros_like(x)
        for b in range(x.size(0)):
            mask[b, top_indices[b]] = 1.0
        
        # 应用mask
        purified_x = x * mask
        
        return purified_x


class ZeroPurDefense:
    """
    ZeroPur: Succinct Training-Free Adversarial Purification
    
    假设对抗样本是自然图像流形的离群值，通过两步净化：
    1. Guided Shift: 通过模糊引导获得移位嵌入
    2. Adaptive Projection: 自适应投影到流形
    
    Reference: ZeroPur (ICLR 2024)
    """
    
    def __init__(self, 
                 sigma: float = 0.5,
                 alpha: float = 0.3,
                 num_steps: int = 5,
                 blur_kernel_size: int = 5):
        """
        Args:
            sigma: 高斯模糊的标准差
            alpha: 投影步长
            num_steps: 投影迭代次数
            blur_kernel_size: 模糊核大小
        """
        self.sigma = sigma
        self.alpha = alpha
        self.num_steps = num_steps
        self.blur_kernel_size = blur_kernel_size
    
    def gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """应用高斯模糊"""
        from torchvision.transforms import GaussianBlur
        
        blur = GaussianBlur(
            kernel_size=self.blur_kernel_size,
            sigma=self.sigma
        )
        
        # 对每个样本应用模糊
        blurred = torch.stack([blur(img) for img in x])
        
        return blurred
    
    def guided_shift(self, x_adv: torch.Tensor, model) -> torch.Tensor:
        """
        Guided Shift: 通过模糊版本引导获得移位嵌入
        
        Args:
            x_adv: 对抗样本 (B, C, H, W)
            model: 受害分类器
        
        Returns:
            shifted_embedding: 移位后的嵌入
        """
        # 获取对抗样本的嵌入
        with torch.no_grad():
            if hasattr(model, 'encode_image'):
                # CLIP-like模型
                emb_adv = model.encode_image(x_adv)
            else:
                emb_adv = model(x_adv)
            
            if emb_adv.dim() > 2:
                emb_adv = emb_adv.view(emb_adv.size(0), -1)
        
        # 获取模糊版本的嵌入
        x_blur = self.gaussian_blur(x_adv)
        
        with torch.no_grad():
            if hasattr(model, 'encode_image'):
                emb_blur = model.encode_image(x_blur)
            else:
                emb_blur = model(x_blur)
            
            if emb_blur.dim() > 2:
                emb_blur = emb_blur.view(emb_blur.size(0), -1)
        
        # 计算引导方向：从对抗嵌入指向模糊嵌入
        direction = F.normalize(emb_blur - emb_adv, dim=-1)
        
        # 移位嵌入
        shifted_emb = emb_adv + self.alpha * direction
        
        return shifted_emb, direction
    
    def adaptive_projection(self, 
                           x_adv: torch.Tensor, 
                           shifted_emb: torch.Tensor,
                           direction: torch.Tensor,
                           model) -> torch.Tensor:
        """
        Adaptive Projection: 自适应地将对抗图像投影到流形
        
        Args:
            x_adv: 对抗样本
            shifted_emb: 移位嵌入
            direction: 方向向量
            model: 受害分类器
        
        Returns:
            purified_x: 净化后的图像
        """
        x_purified = x_adv.clone()
        
        for step in range(self.num_steps):
            x_purified.requires_grad = True
            
            # 获取当前嵌入
            if hasattr(model, 'encode_image'):
                emb_current = model.encode_image(x_purified)
            else:
                emb_current = model(x_purified)
            
            if emb_current.dim() > 2:
                emb_current = emb_current.view(emb_current.size(0), -1)
            
            # 计算与移位嵌入的距离
            distance = ((emb_current - shifted_emb) ** 2).sum()
            
            # 反向传播得到梯度
            distance.backward()
            
            # 使用梯度更新图像，减少与移位嵌入的距离
            with torch.no_grad():
                grad = x_purified.grad
                
                # 直接使用梯度（已经包含了方向信息）
                # 梯度下降：减小distance，所以减去梯度
                x_purified = x_purified - self.alpha * grad.sign()
                
                # 保持在有效范围内
                x_purified = torch.clamp(x_purified, 0, 1)
                
                # 清零梯度
                x_purified.grad = None
            
            x_purified = x_purified.detach()
        
        return x_purified
    
    def purify(self, x_adv: torch.Tensor, model) -> torch.Tensor:
        """
        完整的ZeroPur净化流程
        
        Args:
            x_adv: 对抗样本 (B, C, H, W)
            model: 受害分类器
        
        Returns:
            purified_x: 净化后的图像
        """
        # 步骤1: Guided Shift
        shifted_emb, direction = self.guided_shift(x_adv, model)
        
        # 步骤2: Adaptive Projection
        purified_x = self.adaptive_projection(x_adv, shifted_emb, direction, model)
        
        return purified_x


class CombinedDefense:
    """
    组合防御：结合Interpretability-Guided和ZeroPur
    """
    
    def __init__(self, 
                 model,
                 use_interpretability: bool = True,
                 use_zeropur: bool = True,
                 **kwargs):
        """
        Args:
            model: 目标模型
            use_interpretability: 是否使用可解释性引导防御
            use_zeropur: 是否使用ZeroPur净化
            **kwargs: 各防御策略的参数
        """
        self.model = model
        self.use_interpretability = use_interpretability
        self.use_zeropur = use_zeropur
        
        if use_interpretability:
            self.interp_defense = InterpretabilityGuidedDefense(
                model,
                top_k_ratio=kwargs.get('top_k_ratio', 0.3)
            )
        
        if use_zeropur:
            self.zeropur_defense = ZeroPurDefense(
                sigma=kwargs.get('sigma', 0.5),
                alpha=kwargs.get('alpha', 0.3),
                num_steps=kwargs.get('num_steps', 5)
            )
    
    def purify(self, x: torch.Tensor, predicted_class: Optional[int] = None) -> torch.Tensor:
        """
        应用组合防御
        
        Args:
            x: 输入图像
            predicted_class: 预测类别（用于可解释性引导）
        
        Returns:
            purified_x: 净化后的图像
        """
        purified_x = x
        
        # 先应用ZeroPur（全局净化）
        if self.use_zeropur:
            purified_x = self.zeropur_defense.purify(purified_x, self.model)
        
        # 再应用可解释性引导（细粒度净化）
        if self.use_interpretability and predicted_class is not None:
            purified_x = self.interp_defense.purify(purified_x, predicted_class)
        
        return purified_x


def apply_test_time_defense(
    model,
    x: torch.Tensor,
    defense_type: str = 'zeropur',
    predicted_class: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    便捷函数：应用测试时防御
    
    Args:
        model: 目标模型
        x: 输入图像
        defense_type: 'zeropur', 'interpretability', 'combined'
        predicted_class: 预测类别
        **kwargs: 防御策略的参数
    
    Returns:
        purified_x: 净化后的图像
    """
    if defense_type == 'zeropur':
        defense = ZeroPurDefense(**kwargs)
        return defense.purify(x, model)
    
    elif defense_type == 'interpretability':
        defense = InterpretabilityGuidedDefense(model, **kwargs)
        return defense.purify(x, predicted_class)
    
    elif defense_type == 'combined':
        defense = CombinedDefense(model, **kwargs)
        return defense.purify(x, predicted_class)
    
    else:
        raise ValueError(f"Unknown defense type: {defense_type}")
