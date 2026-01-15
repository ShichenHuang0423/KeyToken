#!/usr/bin/env python3
"""测试增强模型加载"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
import torch

checkpoint_path = 'output/stage1_freeze_all/checkpoints/epoch_1.pt'

print("=" * 80)
print("测试加载增强模型")
print("=" * 80)

try:
    model, preprocessor, normalizer = load_enhanced_clip_model(
        'ViT-L-14',  # 这个参数应该会被checkpoint中的覆盖
        checkpoint_path
    )
    
    print("\n✅ 模型加载成功!")
    print(f"模型类型: {type(model)}")
    print(f"模型维度: {model.dim}")
    
    # 检查模型结构
    if hasattr(model, 'model'):
        print(f"基础CLIP层数: {len(model.model.transformer.resblocks)}")
    
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()
