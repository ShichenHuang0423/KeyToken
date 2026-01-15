#!/usr/bin/env python3
"""精简版评估脚本 - 用于调试"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from train.datasets import ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model  # 测试这个导入
from train.utils import AverageMeter
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


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


def main():
    print("开始评估...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    print(f"GPU数量: {num_gpus}, device: {device}")
    
    # 加载模型
    print("加载模型...")
    model, preprocessor_no_norm, normalizer = load_clip_model('ViT-L-14', 'models/fare_eps_4.pt')
    mean = normalizer.mean
    std = normalizer.std
    print(f"模型加载成功")
    
    # 包装模型
    print("包装模型...")
    wrapped_model = ClipVisionModel(model.visual, mean, std).to(device)
    print(".to(device)成功!")
    
    # DataParallel
    if num_gpus > 1:
        wrapped_model = torch.nn.DataParallel(wrapped_model, device_ids=gpu_ids)
        print(f"DataParallel成功! gpu_ids={gpu_ids}")
    
    print("✅ 评估准备完成!")


if __name__ == '__main__':
    main()
