#!/usr/bin/env python3
"""测试完整的评估流程（少量样本）"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("开始完整评估测试...")

import argparse
import torch
import torch.nn.functional as F
import open_clip
from CLIP_eval.eval_utils import load_clip_model

print("1. 创建参数...")
args = argparse.Namespace(
    clip_model_name='ViT-L-14',
    pretrained='models/fare_eps_4.pt',
    mode='baseline',
    imagenet_root='/home/ubuntu/data/KeyToken/datasets/imagenet',
    batch_size=64,
    max_samples=10,
    attack=True,
    eps=4.0,
    iterations=10,
    output_dir='output/test'
)

print("2. 设置device...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"   device: {device}")
print(f"   GPU数量: {torch.cuda.device_count()}")

print("3. 加载模型...")
model, preprocessor_no_norm, normalizer = load_clip_model(args.clip_model_name, args.pretrained)
print("   ✓ 模型加载成功")

print("4. 获取normalize参数...")
mean = normalizer.mean
std = normalizer.std
print(f"   mean: {mean[:3]}...")
print(f"   std: {std[:3]}...")

print("5. 包装模型...")
class ClipVisionModel(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, vision, output_normalize=False):
        vision = (vision - self.mean) / self.std
        embedding = self.model(vision)
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding

print("6. 尝试 .to(device)...")
try:
    wrapped_model = ClipVisionModel(model.visual, mean, std)
    print("   ✓ ClipVisionModel创建成功")
    
    print("   正在移动到GPU...")
    wrapped_model_gpu = wrapped_model.to(device)
    print("   ✅ .to(device) 成功!")
    
    print("7. 测试DataParallel...")
    if torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
        wrapped_parallel = torch.nn.DataParallel(wrapped_model_gpu, device_ids=gpu_ids)
        print(f"   ✅ DataParallel成功! GPU IDs: {gpu_ids}")
    
    print("\n✅ 所有测试通过！evaluate_robust.py没有问题")
    
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()
