"""
增强版CLIP评估工具
支持加载完整的EnhancedClipVisionModel（包含所有训练的增强模块）
用于推理时使用PatchDisturbDetector、KeyTokenSelector等模块
"""

import sys
import os
import torch
import open_clip
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.adversarial_training_clip_enhanced import EnhancedClipVisionModel


def load_enhanced_clip_model(clip_model_name, checkpoint_path, args=None):
    """
    加载完整的增强CLIP模型（包含所有增强模块）
    
    Args:
        clip_model_name: CLIP模型名称（如'ViT-L-14'）
        checkpoint_path: checkpoint路径
        args: 训练时的参数（从checkpoint自动加载，或手动指定）
    
    Returns:
        enhanced_model: 完整的EnhancedClipVisionModel
        preprocessor_no_norm: 图像预处理（不含normalize）
        normalizer: normalize变换
    """
    print(f"📦 加载增强CLIP模型...")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取训练参数
    if args is None:
        if 'args' not in checkpoint:
            raise ValueError("Checkpoint中没有args，请手动提供args参数")
        # 正确创建args对象：使用argparse.Namespace或SimpleNamespace
        from argparse import Namespace
        args = Namespace(**checkpoint['args'])
    
    # 使用checkpoint中的clip_model_name（优先级高于函数参数）
    if hasattr(args, 'clip_model_name'):
        actual_model_name = args.clip_model_name
        print(f"📦 从checkpoint读取模型架构: {actual_model_name}")
    else:
        actual_model_name = clip_model_name
        print(f"⚠️  使用默认模型架构: {actual_model_name}")
    
    # 创建基础CLIP模型
    print(f"🔧 创建模型，架构名称: {actual_model_name}")
    base_model, _, image_processor = open_clip.create_model_and_transforms(
        actual_model_name, pretrained='openai', device='cpu'
    )
    
    # 验证创建的模型维度
    if hasattr(base_model, 'visual') and hasattr(base_model.visual, 'transformer'):
        if hasattr(base_model.visual.transformer, 'width'):
            print(f"🔍 创建的模型维度: {base_model.visual.transformer.width}")
        layer_count = len(base_model.visual.transformer.resblocks)
        print(f"🔍 Transformer层数: {layer_count}")
    
    # 图像预处理
    preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
    normalizer = image_processor.transforms[-1]
    
    # 创建EnhancedClipVisionModel (需要传入visual encoder和normalize)
    # 注意：EnhancedClipVisionModel期望的是visual encoder，不是完整CLIP模型
    enhanced_model = EnhancedClipVisionModel(base_model.visual, args, normalizer)
    
    # 加载权重
    if 'enhanced_model_state_dict' in checkpoint:
        enhanced_model.load_state_dict(checkpoint['enhanced_model_state_dict'])
        print(f"✅ 加载完整增强模型（包含所有增强模块）")
    else:
        enhanced_model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"⚠️  仅加载基础CLIP权重（增强模块未训练）")
    
    enhanced_model.eval()
    
    print(f"✅ 模型加载完成")
    print(f"   - PatchDisturbDetector: {'✅' if args.use_key_token_protection else '❌'}")
    print(f"   - KeyTokenSelector: {'✅' if args.use_key_token_protection else '❌'}")
    print(f"   - DualMAEDecoder: {'✅' if args.use_mae_recon else '❌'}")
    
    return enhanced_model, preprocessor_no_norm, normalizer


def load_clip_model_for_inference(checkpoint_path, clip_model_name='ViT-L-14', 
                                   use_enhanced_modules=True):
    """
    推理时加载CLIP模型
    
    Args:
        checkpoint_path: checkpoint路径
        clip_model_name: CLIP模型名称
        use_enhanced_modules: 是否使用增强模块（True=使用增强推理，False=仅基础CLIP）
    
    Returns:
        model: CLIP模型（可能是EnhancedClipVisionModel或基础CLIP）
        preprocessor_no_norm: 图像预处理
        normalizer: normalize变换
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if use_enhanced_modules and 'enhanced_model_state_dict' in checkpoint:
        # 使用增强模块
        return load_enhanced_clip_model(clip_model_name, checkpoint_path)
    else:
        # 仅使用基础CLIP
        print(f"📦 加载基础CLIP模型（不使用增强模块）")
        model, _, image_processor = open_clip.create_model_and_transforms(
            clip_model_name, pretrained='openai', device='cpu'
        )
        
        if 'model_state_dict' in checkpoint:
            model.visual.load_state_dict(checkpoint['model_state_dict'])
        elif 'enhanced_model_state_dict' in checkpoint:
            # 从增强模型中提取基础CLIP权重
            enhanced_state = checkpoint['enhanced_model_state_dict']
            clip_state = {k.replace('model.', ''): v 
                         for k, v in enhanced_state.items() 
                         if k.startswith('model.')}
            model.visual.load_state_dict(clip_state)
        
        model.eval()
        preprocessor_no_norm = transforms.Compose(image_processor.transforms[:-1])
        normalizer = image_processor.transforms[-1]
        
        print(f"✅ 基础CLIP模型加载完成")
        return model, preprocessor_no_norm, normalizer


@torch.no_grad()
def enhanced_inference(enhanced_model, images, mode='eval'):
    """
    使用增强模块进行推理
    
    Args:
        enhanced_model: EnhancedClipVisionModel
        images: 输入图像 tensor (B, C, H, W)
        mode: 'eval' = 推理模式（使用增强模块）
    
    Returns:
        embeddings: 图像embedding (B, dim)
        extra_info: 额外信息（扰动分数、关键token等）
    """
    # 使用EnhancedClipVisionModel的forward
    embeddings, extra_info = enhanced_model(images, mode=mode)
    
    return embeddings, extra_info


# 向后兼容：导出标准接口
def load_clip_model(clip_model_name, pretrained, beta=0.):
    """
    标准接口：加载CLIP模型（自动检测是否为增强模型）
    兼容原有的eval_utils.py接口
    """
    if isinstance(pretrained, str) and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        
        # 检测是否为增强模型
        if 'enhanced_model_state_dict' in checkpoint:
            print("🔍 检测到增强模型checkpoint")
            return load_enhanced_clip_model(clip_model_name, pretrained)
    
    # 否则使用标准加载方式
    from CLIP_eval.eval_utils import load_clip_model as load_clip_model_original
    return load_clip_model_original(clip_model_name, pretrained, beta)
