#!/usr/bin/env python3
"""测试训练代码中数据的实际范围和eps参数"""

import sys
sys.path.insert(0, '/home/ubuntu/data/KeyToken')

import torch
from torchvision import transforms
from train.datasets import ImageNetDataset
import open_clip

# 加载CLIP模型获取preprocessor
clip_model, _, image_processor = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai'
)

# 查看完整的image_processor
print("完整image_processor:")
print(image_processor)
print()

# 分离normalize前的transforms
preprocessor_without_normalize = transforms.Compose(image_processor.transforms[:-1])
normalize = image_processor.transforms[-1]

print("preprocessor_without_normalize:")
print(preprocessor_without_normalize)
print()

print("normalize transform:")
print(normalize)
print()

# 加载一个样本测试
dataset = ImageNetDataset(
    root='/home/ubuntu/data/KeyToken/datasets/imagenet/val',
    transform=preprocessor_without_normalize,
)

data, target = dataset[0]
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")
print(f"Data min: {data.min():.6f}")
print(f"Data max: {data.max():.6f}")
print(f"Data range: [{data.min():.6f}, {data.max():.6f}]")
print()

# 测试eps=4的情况
eps = 4
print(f"如果eps={eps}:")
print(f"  - 扰动范围: [-{eps}, {eps}]")
print(f"  - 数据范围: [0, 1]")
print(f"  - clamp后实际最大扰动: 1.0 (整个[0,1]范围)")
print(f"  - 这意味着eps={eps}时，扰动几乎不受限制（远大于数据范围）")
print()

# 测试eps=4/255的情况
eps_normalized = 4/255
print(f"如果eps={eps_normalized:.6f} (4/255):")
print(f"  - 扰动范围: [-{eps_normalized:.6f}, {eps_normalized:.6f}]")
print(f"  - 占数据范围的比例: {eps_normalized*100:.2f}%")
print(f"  - 这是ImageNet标准的对抗扰动大小")
