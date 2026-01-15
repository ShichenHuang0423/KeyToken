#!/usr/bin/env python3
"""测试数据加载和预处理"""

import sys
sys.path.insert(0, '/home/ubuntu/data/KeyToken')

import torch
from torchvision import transforms
from train.datasets import ImageNetDataset

# 测试预处理
preprocessor = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 加载数据集
dataset = ImageNetDataset(
    root='/home/ubuntu/data/KeyToken/datasets/imagenet/val',
    transform=preprocessor,
)

# 测试第一个样本
data, target = dataset[0]
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Data min/max: {data.min():.4f}/{data.max():.4f}")
print(f"Target: {target}")

# 测试batch
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
batch_data, batch_targets = next(iter(dataloader))
print(f"\nBatch shape: {batch_data.shape}")
print(f"Batch min/max: {batch_data.min():.4f}/{batch_data.max():.4f}")
