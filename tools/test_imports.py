#!/usr/bin/env python3
"""测试各个导入模块"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("1. 测试基础导入...")
import argparse
print("✓ argparse")

print("2. 测试torch...")
import torch
print(f"✓ torch - CUDA可用: {torch.cuda.is_available()}, GPU数: {torch.cuda.device_count()}")

print("3. 测试torch.nn.functional...")
import torch.nn.functional as F
print("✓ torch.nn.functional")

print("4. 测试open_clip...")
import open_clip
print("✓ open_clip")

print("5. 测试DataLoader...")
from torch.utils.data import DataLoader
print("✓ DataLoader")

print("6. 测试transforms...")
from torchvision import transforms
print("✓ transforms")

print("7. 测试tqdm...")
from tqdm import tqdm
print("✓ tqdm")

print("8. 测试train.datasets...")
from train.datasets import ImageNetDataset
print("✓ train.datasets.ImageNetDataset")

print("9. 测试CLIP_eval.eval_utils...")
from CLIP_eval.eval_utils import load_clip_model
print("✓ CLIP_eval.eval_utils.load_clip_model")

print("10. 测试CLIP_eval.eval_utils_enhanced...")
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
print("✓ CLIP_eval.eval_utils_enhanced.load_enhanced_clip_model")

print("11. 测试train.utils...")
from train.utils import AverageMeter
print("✓ train.utils.AverageMeter")

print("12. 测试open_flamingo.eval...")
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
print("✓ open_flamingo.eval.classification_utils.IMAGENET_1K_CLASS_ID_TO_LABEL")

print("\n✅ 所有导入测试通过！")
