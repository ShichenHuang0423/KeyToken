#!/usr/bin/env python3
"""
评估基准模型脚本
用于评估OpenAI CLIP和FARE模型在ImageNet验证集上的CleanAcc和RobustAcc
"""

import sys
sys.path.insert(0, '/home/ubuntu/data/KeyToken')

import os
import argparse
import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from train.datasets import ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model
from train.pgd_train import pgd
from train.utils import AverageMeter
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class ClipVisionModel(torch.nn.Module):
    """CLIP Vision模型包装器，在forward内部应用normalize"""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        # 存储normalize参数而不是Transform对象
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, vision, output_normalize=False):
        # vision是[0,1]范围的原始图像，手动normalize
        vision = (vision - self.mean) / self.std
        embedding = self.model(vision)
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


def evaluate_model(args):
    """评估模型性能"""
    print("=" * 80)
    print(f"📊 评估配置:")
    print(f"   模型: {args.pretrained}")
    print(f"   攻击: {args.attack} (norm={args.norm}, eps={args.eps})")
    print(f"   数据集: ImageNet验证集")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("\n🔄 加载模型...")
    if args.pretrained == 'openai':
        clip_model, _, image_processor = open_clip.create_model_and_transforms(
            args.clip_model_name, pretrained='openai'
        )
        print("   ✓ 加载OpenAI CLIP预训练模型")
    else:
        # 加载FARE或其他模型
        clip_model, _, image_processor = load_clip_model(args.clip_model_name, args.pretrained)
        print(f"   ✓ 加载模型: {args.pretrained}")
    
    # 直接构建预处理pipeline（不包含normalize，保持[0,1]范围）
    preprocessor_without_normalize = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # CLIP标准normalize参数
    normalize_mean = [0.48145466, 0.4578275, 0.40821073]
    normalize_std = [0.26862954, 0.26130258, 0.27577711]
    del image_processor
    
    # 保存完整CLIP模型用于文本编码
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # 包装vision模型，传入normalize参数
    vision_model = ClipVisionModel(
        model=clip_model.visual,
        mean=normalize_mean,
        std=normalize_std
    )
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"   ✓ 使用 {torch.cuda.device_count()} 张GPU进行DataParallel")
        vision_model = torch.nn.DataParallel(vision_model)
    
    vision_model = vision_model.to(device)
    vision_model.eval()
    
    # 加载数据集（不包含normalize，保持[0,1]范围供PGD使用）
    print("\n🔄 加载ImageNet验证集...")
    dataset = ImageNetDataset(
        root=os.path.join(args.imagenet_root, 'val'),
        transform=preprocessor_without_normalize,
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"   ✓ 加载 {len(dataset)} 张图片")
    
    # 获取ImageNet类别文本嵌入
    print("\n🔄 计算类别文本嵌入...")
    template = 'This is a photo of a {}'
    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    text_tokens = open_clip.tokenize(texts)
    
    with torch.no_grad():
        # 分批处理避免OOM
        embedding_text_labels_norm = []
        for el in (text_tokens[:500], text_tokens[500:]):
            # 使用完整CLIP模型的文本编码器
            emb = clip_model.encode_text(el.to(device), normalize=True)
            embedding_text_labels_norm.append(emb.cpu())
        embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(device)
    print("   ✓ 文本嵌入计算完成")
    
    # 评估指标
    clean_acc_meter = AverageMeter('CleanAcc')
    robust_acc_meter = AverageMeter('RobustAcc')
    
    # 评估循环
    print(f"\n🔄 开始评估 (batch_size={args.batch_size})...")
    
    # 限制评估样本数（如果指定）
    total_batches = len(dataloader)
    if args.max_samples > 0:
        total_batches = min(total_batches, args.max_samples // args.batch_size)
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="评估进度", total=total_batches)):
            if batch_idx >= total_batches:
                break
            
            data = data.to(device)
            targets = targets.to(device)
            n_samples = data.shape[0]
            
            # 调试：打印数据形状
            if batch_idx == 0:
                print(f"\n   [DEBUG] data shape: {data.shape}")
                print(f"   [DEBUG] data min/max: {data.min():.4f}/{data.max():.4f}")
        
            # 1. 评估Clean Accuracy
            embedding_clean = vision_model(data, output_normalize=True)
            logits_clean = embedding_clean @ embedding_text_labels_norm
            pred_clean = logits_clean.argmax(dim=1)
            clean_acc = (pred_clean == targets).float().mean().item()
            clean_acc_meter.update(clean_acc, n_samples)
            
            # 2. 评估Robust Accuracy（如果指定攻击）
            if args.attack != 'none':
                # 设置模型为eval模式
                vision_model.eval()
                
                # 定义攻击损失函数：接收embedding作为输入（不是图像）
                # loss_fn(out, targets) 其中out是forward的返回值（embedding）
                def attack_loss_fn(emb_adv, targets):
                    # 负的余弦相似度 = 最大化距离
                    return -F.cosine_similarity(emb_adv, embedding_clean.detach(), dim=1).mean()
                
                # 生成对抗样本
                data_adv = pgd(
                    forward=lambda x, output_normalize: vision_model(x, output_normalize),
                    loss_fn=attack_loss_fn,  # loss_fn接收embedding
                    data_clean=data,
                    targets=None,
                    norm=args.norm,
                    eps=args.eps,
                    iterations=args.iterations_adv,
                    stepsize=args.stepsize_adv,
                    output_normalize=True,
                    perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                    mode='max',
                    verbose=False
                )
                
                # 评估对抗样本
                embedding_adv = vision_model(data_adv, output_normalize=True)
                logits_adv = embedding_adv @ embedding_text_labels_norm
                pred_adv = logits_adv.argmax(dim=1)
                robust_acc = (pred_adv == targets).float().mean().item()
                robust_acc_meter.update(robust_acc, n_samples)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("📊 评估结果:")
    print("=" * 80)
    print(f"模型: {args.pretrained}")
    print(f"CleanAcc:  {clean_acc_meter.avg:.4f}")
    if args.attack != 'none':
        print(f"RobustAcc: {robust_acc_meter.avg:.4f} (攻击: {args.attack}, norm={args.norm}, eps={args.eps})")
    print("=" * 80)
    
    # 保存结果到文件
    output_file = os.path.join(args.output_dir, f"{os.path.basename(args.pretrained).replace('.pt', '')}_results.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"Model: {args.pretrained}\n")
        f.write(f"CleanAcc: {clean_acc_meter.avg:.4f}\n")
        if args.attack != 'none':
            f.write(f"RobustAcc: {robust_acc_meter.avg:.4f}\n")
            f.write(f"Attack: {args.attack}, norm={args.norm}, eps={args.eps}\n")
    print(f"\n✅ 结果已保存到: {output_file}")
    
    return {
        'clean_acc': clean_acc_meter.avg,
        'robust_acc': robust_acc_meter.avg if args.attack != 'none' else None
    }


def main():
    parser = argparse.ArgumentParser(description='评估CLIP基准模型')
    
    # 模型参数
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='预训练模型路径或名称 (openai, models/fare_eps_4.pt, etc.)')
    
    # 数据参数
    parser.add_argument('--imagenet_root', type=str, 
                       default='/home/ubuntu/data/KeyToken/datasets/imagenet',
                       help='ImageNet数据集根目录')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='最大评估样本数 (-1表示全部)')
    
    # 攻击参数
    parser.add_argument('--attack', type=str, default='pgd',
                       choices=['pgd', 'none'],
                       help='攻击类型')
    parser.add_argument('--norm', type=str, default='linf',
                       choices=['linf', 'l2'],
                       help='扰动范数')
    parser.add_argument('--eps', type=float, default=4.0,
                       help='扰动幅度')
    parser.add_argument('--iterations_adv', type=int, default=10,
                       help='攻击迭代次数')
    parser.add_argument('--stepsize_adv', type=float, default=1.0,
                       help='攻击步长')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output/baseline_eval',
                       help='输出目录')
    parser.add_argument('--gpu', type=str, default='0',
                       help='使用的GPU编号，多卡用逗号分隔 (例如: 0,5,6,7)')
    
    args = parser.parse_args()
    
    # 转换eps和stepsize（与训练代码一致）
    args.eps /= 255
    args.stepsize_adv /= 255
    
    # 设置使用的GPU
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_list = args.gpu.split(',')
    print(f"🎮 使用GPU: {args.gpu} (共{len(gpu_list)}张卡)")
    
    # 评估模型
    results = evaluate_model(args)


if __name__ == '__main__':
    main()
