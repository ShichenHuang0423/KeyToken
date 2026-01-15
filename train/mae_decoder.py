"""
MAE (Masked Autoencoder) 解码器
用于重建被掩码的图像patch和文本token
"""

import torch
import torch.nn as nn
import math


class MAEDecoder(nn.Module):
    """
    MAE风格的Transformer解码器
    用于重建被掩码的Token特征
    """
    def __init__(self, dim=768, num_heads=8, num_layers=3, mlp_ratio=4.0, dropout=0.0):
        """
        Args:
            dim: token特征维度（768 for ViT-L/14）
            num_heads: 注意力头数
            num_layers: Transformer层数
            mlp_ratio: MLP隐藏层扩展比例
            dropout: dropout比例
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 可学习的掩码token嵌入
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(dim)
        
        # 输出投影（重建token特征）
        self.decoder_pred = nn.Linear(dim, dim, bias=True)
        
        # 位置编码（可学习）
        self.decoder_pos_embed = None  # 将在forward中动态初始化
    
    def _init_pos_embed(self, seq_len, device):
        """初始化位置编码"""
        if self.decoder_pos_embed is None or self.decoder_pos_embed.shape[1] != seq_len:
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, seq_len, self.dim, device=device)
            )
            nn.init.normal_(self.decoder_pos_embed, std=0.02)
    
    def forward(self, tokens, mask):
        """
        重建被掩码的token
        
        Args:
            tokens: (B, N, dim) - 输入token序列
            mask: (B, N) - bool掩码，True表示被掩码的位置
        
        Returns:
            reconstructed: (B, N, dim) - 重建的token特征
        """
        B, N, D = tokens.shape
        device = tokens.device
        
        # 初始化位置编码
        self._init_pos_embed(N, device)
        
        # 1. 将被掩码的token替换为mask_token
        tokens_with_mask = tokens.clone()
        mask_expanded = mask.unsqueeze(-1).expand(B, N, D)
        mask_token_expanded = self.mask_token.expand(B, N, D)
        tokens_with_mask = torch.where(mask_expanded, mask_token_expanded, tokens_with_mask)
        
        # 2. 添加位置编码
        x = tokens_with_mask + self.decoder_pos_embed
        
        # 3. 通过Transformer解码器层
        for layer in self.decoder_layers:
            x = layer(x)
        
        # 4. 归一化
        x = self.norm(x)
        
        # 5. 预测重建的token
        reconstructed = self.decoder_pred(x)
        
        return reconstructed
    
    def compute_reconstruction_loss(self, reconstructed, target, mask):
        """
        计算重建损失（仅在被掩码的位置）
        
        Args:
            reconstructed: (B, N, dim) - 重建的特征
            target: (B, N, dim) - 目标特征（清洁样本）
            mask: (B, N) - 掩码位置
        
        Returns:
            loss: 标量，MSE重建损失
        """
        # 只在被掩码的位置计算损失
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
        
        # MSE损失
        loss = (reconstructed - target) ** 2
        loss = (loss * mask_expanded.float()).sum() / (mask_expanded.sum() + 1e-8)
        
        return loss


class TransformerDecoderBlock(nn.Module):
    """
    Transformer解码器块
    包含自注意力和前馈网络
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, dim)
        Returns:
            x: (B, N, dim)
        """
        # 自注意力 + 残差
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x)
        )[0]
        
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        
        return x


class DualMAEDecoder(nn.Module):
    """
    双分支MAE解码器
    分别处理图像和文本的token重建
    """
    def __init__(self, img_dim=768, text_dim=768, num_heads=8, num_layers=3):
        """
        Args:
            img_dim: 图像token维度
            text_dim: 文本token维度
            num_heads: 注意力头数
            num_layers: 解码器层数
        """
        super().__init__()
        
        # 图像解码器
        self.img_decoder = MAEDecoder(
            dim=img_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 文本解码器
        self.text_decoder = MAEDecoder(
            dim=text_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    def forward(self, img_tokens, img_mask, text_tokens, text_mask):
        """
        同时重建图像和文本token
        
        Args:
            img_tokens: (B, N_img, dim_img)
            img_mask: (B, N_img)
            text_tokens: (B, N_text, dim_text)
            text_mask: (B, N_text)
        
        Returns:
            dict: {
                'img_recon': 重建的图像token
                'text_recon': 重建的文本token
            }
        """
        img_recon = self.img_decoder(img_tokens, img_mask)
        text_recon = self.text_decoder(text_tokens, text_mask)
        
        return {
            'img_recon': img_recon,
            'text_recon': text_recon
        }
    
    def compute_loss(self, img_recon, img_target, img_mask,
                     text_recon, text_target, text_mask,
                     text_weight=0.8):
        """
        计算总重建损失
        
        Args:
            img_recon, img_target, img_mask: 图像相关
            text_recon, text_target, text_mask: 文本相关
            text_weight: 文本损失权重（默认0.8，因为文本影响较小）
        
        Returns:
            dict: {
                'total_loss': 总损失
                'img_loss': 图像重建损失
                'text_loss': 文本重建损失
            }
        """
        img_loss = self.img_decoder.compute_reconstruction_loss(
            img_recon, img_target, img_mask
        )
        
        text_loss = self.text_decoder.compute_reconstruction_loss(
            text_recon, text_target, text_mask
        )
        
        total_loss = img_loss + text_weight * text_loss
        
        return {
            'total_loss': total_loss,
            'img_loss': img_loss,
            'text_loss': text_loss
        }


# 测试代码
if __name__ == '__main__':
    print("测试MAE解码器...")
    
    # 测试单个解码器
    print("\n1. 测试单分支MAE解码器")
    decoder = MAEDecoder(dim=768, num_heads=8, num_layers=3)
    
    # 模拟数据
    B, N, D = 4, 197, 768
    tokens = torch.randn(B, N, D)
    target = torch.randn(B, N, D)
    
    # 随机掩码50%的token
    mask = torch.rand(B, N) > 0.5
    print(f"   掩码比例: {mask.float().mean():.2%}")
    
    # 重建
    reconstructed = decoder(tokens, mask)
    print(f"   重建特征 shape: {reconstructed.shape}")
    
    # 计算损失
    loss = decoder.compute_reconstruction_loss(reconstructed, target, mask)
    print(f"   重建损失: {loss.item():.4f}")
    
    # 测试双分支解码器
    print("\n2. 测试双分支MAE解码器")
    dual_decoder = DualMAEDecoder(img_dim=768, text_dim=768)
    
    # 图像数据
    img_tokens = torch.randn(B, 197, 768)
    img_target = torch.randn(B, 197, 768)
    img_mask = torch.rand(B, 197) > 0.5
    
    # 文本数据
    text_tokens = torch.randn(B, 77, 768)
    text_target = torch.randn(B, 77, 768)
    text_mask = torch.rand(B, 77) > 0.6
    
    # 重建
    recon_results = dual_decoder(img_tokens, img_mask, text_tokens, text_mask)
    print(f"   图像重建 shape: {recon_results['img_recon'].shape}")
    print(f"   文本重建 shape: {recon_results['text_recon'].shape}")
    
    # 计算损失
    loss_dict = dual_decoder.compute_loss(
        recon_results['img_recon'], img_target, img_mask,
        recon_results['text_recon'], text_target, text_mask
    )
    print(f"   总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"   图像损失: {loss_dict['img_loss'].item():.4f}")
    print(f"   文本损失: {loss_dict['text_loss'].item():.4f}")
    
    # 测试梯度反传
    print("\n3. 测试梯度反传")
    optimizer = torch.optim.Adam(dual_decoder.parameters(), lr=1e-4)
    
    loss = loss_dict['total_loss']
    loss.backward()
    optimizer.step()
    print("   ✅ 梯度反传成功")
    
    print("\n✅ MAE解码器测试通过！")
