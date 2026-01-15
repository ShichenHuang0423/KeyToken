"""
LVLM工具 - 加载LLaVA和OpenFlamingo并替换CLIP encoder
"""

import torch
import open_clip
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from llava.model.builder import load_pretrained_model as load_llava_model
from llava.mm_utils import get_model_name_from_path
from open_flamingo import create_model_and_transforms


def load_robust_clip_encoder(checkpoint_path: str, clip_model_name: str = 'ViT-L-14', device='cuda'):
    """
    加载鲁棒CLIP vision encoder
    
    Args:
        checkpoint_path: CLIP checkpoint路径 (FARE/KeyToken)
        clip_model_name: CLIP架构名称
        device: 设备
        
    Returns:
        vision_model, preprocess, normalizer, is_enhanced
    """
    from CLIP_eval.eval_utils import load_clip_model as load_baseline_clip_model
    from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
    
    if 'fare' in checkpoint_path.lower() or 'tecoa' in checkpoint_path.lower():
        # FARE/TeCoA基线模型
        model, preprocessor_no_norm, normalizer = load_baseline_clip_model(
            clip_model_name, checkpoint_path
        )
        is_enhanced = False
    else:
        # KeyToken增强模型
        enhanced_model, preprocessor_no_norm, normalizer = load_enhanced_clip_model(
            clip_model_name, checkpoint_path
        )
        model = enhanced_model
        is_enhanced = True
    
    model = model.to(device)
    model.eval()
    
    return model, preprocessor_no_norm, normalizer, is_enhanced


def replace_llava_clip_encoder(
    llava_model_path: str,
    clip_checkpoint_path: str,
    clip_model_name: str = 'ViT-L-14',
    device: str = 'cuda',
    load_8bit: bool = False,
    load_4bit: bool = False
):
    """
    加载LLaVA并替换其CLIP vision encoder
    
    Args:
        llava_model_path: LLaVA模型路径
        clip_checkpoint_path: 鲁棒CLIP checkpoint
        clip_model_name: CLIP架构
        device: 设备
        load_8bit: 是否8bit量化
        load_4bit: 是否4bit量化
        
    Returns:
        tokenizer, model, image_processor, context_len, is_enhanced
    """
    print(f"🔄 加载LLaVA模型: {llava_model_path}")
    
    # 加载LLaVA
    model_name = get_model_name_from_path(llava_model_path)
    tokenizer, model, image_processor, context_len = load_llava_model(
        model_path=llava_model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device
    )
    
    print(f"🔄 替换CLIP vision encoder: {clip_checkpoint_path}")
    
    # 加载鲁棒CLIP encoder
    robust_vision_model, preprocess, normalizer, is_enhanced = load_robust_clip_encoder(
        clip_checkpoint_path, clip_model_name, device
    )
    
    # 替换LLaVA的vision tower
    if hasattr(model.get_vision_tower(), 'vision_tower'):
        # LLaVA-1.5结构
        original_vision_tower = model.get_vision_tower().vision_tower
        
        # 保留原始配置
        config = original_vision_tower.config if hasattr(original_vision_tower, 'config') else None
        
        # 替换vision encoder
        if is_enhanced:
            # KeyToken增强模型
            model.get_vision_tower().vision_tower = robust_vision_model.model.visual
        else:
            # FARE/TeCoA基线
            model.get_vision_tower().vision_tower = robust_vision_model.visual
        
        if config is not None:
            model.get_vision_tower().vision_tower.config = config
        
        print(f"✅ 已替换LLaVA vision encoder (is_enhanced={is_enhanced})")
    else:
        raise ValueError("无法找到LLaVA的vision tower")
    
    # 更新image_processor使用鲁棒CLIP的预处理
    # 注意：LLaVA使用特定的图像预处理，我们只替换normalize部分
    image_processor.image_mean = normalizer.mean.tolist()
    image_processor.image_std = normalizer.std.tolist()
    
    return tokenizer, model, image_processor, context_len, is_enhanced, normalizer


def replace_flamingo_clip_encoder(
    flamingo_checkpoint_path: str,
    clip_checkpoint_path: str,
    clip_model_name: str = 'ViT-L-14',
    lang_encoder_path: str = "mosaicml/mpt-7b",
    tokenizer_path: str = "mosaicml/mpt-7b",
    cross_attn_every_n_layers: int = 4,
    device: str = 'cuda'
):
    """
    加载OpenFlamingo并替换其CLIP vision encoder
    
    Args:
        flamingo_checkpoint_path: Flamingo checkpoint路径
        clip_checkpoint_path: 鲁棒CLIP checkpoint
        clip_model_name: CLIP架构
        lang_encoder_path: 语言模型路径
        tokenizer_path: Tokenizer路径
        cross_attn_every_n_layers: Cross attention频率
        device: 设备
        
    Returns:
        model, image_processor, tokenizer, is_enhanced, normalizer
    """
    print(f"🔄 加载OpenFlamingo模型: {flamingo_checkpoint_path}")
    
    # 加载鲁棒CLIP encoder
    robust_vision_model, preprocess, normalizer, is_enhanced = load_robust_clip_encoder(
        clip_checkpoint_path, clip_model_name, device
    )
    
    # 创建OpenFlamingo模型
    # 注意：我们不使用create_model_and_transforms，而是手动创建并替换vision encoder
    from open_flamingo.src.flamingo import Flamingo
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 加载语言模型
    print(f"🔄 加载语言模型: {lang_encoder_path}")
    lang_model = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    
    # 创建Flamingo模型
    model = Flamingo(
        vision_encoder=robust_vision_model.visual if not is_enhanced else robust_vision_model.model.visual,
        lang_encoder=lang_model,
        eoc_token_id=tokenizer.encode("<|endofchunk|>")[-1],
        media_token_id=tokenizer.encode("<image>")[-1],
        vis_dim=robust_vision_model.visual.output_dim if not is_enhanced else robust_vision_model.model.visual.output_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )
    
    # 加载Flamingo checkpoint
    print(f"🔄 加载Flamingo checkpoint...")
    checkpoint = torch.load(flamingo_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ 已加载OpenFlamingo (is_enhanced={is_enhanced})")
    
    return model, preprocess, tokenizer, is_enhanced, normalizer


def get_lvlm_model(
    lvlm_type: str,
    lvlm_path: str,
    clip_checkpoint: str,
    clip_model_name: str = 'ViT-L-14',
    device: str = 'cuda'
):
    """
    统一接口：加载LVLM并替换CLIP encoder
    
    Args:
        lvlm_type: 'llava' or 'flamingo'
        lvlm_path: LVLM模型路径
        clip_checkpoint: 鲁棒CLIP checkpoint
        clip_model_name: CLIP架构
        device: 设备
        
    Returns:
        根据lvlm_type返回相应的模型组件
    """
    if lvlm_type.lower() == 'llava':
        return replace_llava_clip_encoder(
            llava_model_path=lvlm_path,
            clip_checkpoint_path=clip_checkpoint,
            clip_model_name=clip_model_name,
            device=device
        )
    elif lvlm_type.lower() == 'flamingo':
        flamingo_checkpoint = f"{lvlm_path}/checkpoint.pt"
        return replace_flamingo_clip_encoder(
            flamingo_checkpoint_path=flamingo_checkpoint,
            clip_checkpoint_path=clip_checkpoint,
            clip_model_name=clip_model_name,
            device=device
        )
    else:
        raise ValueError(f"不支持的LVLM类型: {lvlm_type}")
