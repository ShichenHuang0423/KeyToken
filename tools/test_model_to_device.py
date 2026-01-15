#!/usr/bin/env python3
"""测试模型.to(device)操作"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import open_clip

print("1. CUDA基础测试...")
print(f"   CUDA可用: {torch.cuda.is_available()}")
print(f"   GPU数量: {torch.cuda.device_count()}")
print(f"   当前设备: {torch.cuda.current_device()}")

print("\n2. 创建device对象...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"   device: {device}")

print("\n3. 加载CLIP模型到CPU...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device='cpu')
print(f"   ✓ 模型已加载到CPU")

print("\n4. 尝试将visual模型移动到GPU...")
try:
    visual_model = model.visual
    print(f"   visual模型类型: {type(visual_model)}")
    print(f"   开始 .to(device)...")
    visual_model_gpu = visual_model.to(device)
    print(f"   ✅ 成功移动到GPU!")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n5. 测试DataParallel...")
try:
    if torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
        print(f"   GPU IDs: {gpu_ids}")
        visual_parallel = torch.nn.DataParallel(visual_model, device_ids=gpu_ids)
        print(f"   ✅ DataParallel创建成功!")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
