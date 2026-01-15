#!/usr/bin/env python3
"""
统一的鲁棒性评估脚本
支持论文中的攻击设置：APGD-CE + APGD-DLR (targeted), 各100迭代
支持增强模块的eval/attack模式切换

用法:
    # 评估基线模型（FARE等）
    python tools/evaluate_robust.py --pretrained models/fare_eps_4.pt --mode baseline
    
    # 评估增强模型 - 完整防御模式
    python tools/evaluate_robust.py --pretrained output/stage1.pt --mode eval
    
    # 评估增强模型 - 无防御模式（仅backbone）
    python tools/evaluate_robust.py --pretrained output/stage1.pt --mode attack
"""

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
import json
from datetime import datetime

from train.datasets import ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
from train.utils import AverageMeter
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


class ClipVisionModel(torch.nn.Module):
    """CLIP Vision模型包装器"""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x, output_normalize=False):
        x = (x - self.mean) / self.std
        embedding = self.model(x)
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


def get_text_embeddings(model, tokenizer, device):
    """计算ImageNet类别的文本嵌入"""
    class_names = [IMAGENET_1K_CLASS_ID_TO_LABEL[i] for i in range(1000)]
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of many {}.",
        "a photo of the large {}.",
        "a photo of the small {}.",
    ]
    
    with torch.no_grad():
        text_embeddings = []
        for class_name in tqdm(class_names, desc="Computing text embeddings", leave=False):
            texts = [template.format(class_name) for template in templates]
            tokens = tokenizer(texts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            text_embeddings.append(embeddings.mean(dim=0))
        
        text_embeddings = torch.stack(text_embeddings)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    return text_embeddings


def autoattack_eval(model, images, targets, text_embeddings, eps, iterations, 
                    device, is_enhanced=False, inference_mode='eval', ensemble_size=1, noise_std=0.01,
                    gray_box=False, randomize_defense=False):
    """
    使用论文的攻击设置：APGD-CE + APGD-DLR (targeted)
    
    Args:
        model: 模型
        images: 输入图像 [0,1]范围，已归一化
        targets: 真实标签
        text_embeddings: 文本嵌入
        eps: 扰动半径
        iterations: 迭代次数
        device: 设备
        is_enhanced: 是否为增强模型
        inference_mode: 推理模式 ('eval'=完整防御, 'attack'=无防御)
        ensemble_size: 集成样本数
        noise_std: 随机噪声标准差
        gray_box: 是否使用灰盒攻击（APGD只攻击backbone，不知道防御策略）
        randomize_defense: 是否启用内置随机化防御链（打破APGD梯度估计）
    
    Returns:
        adv_images: 对抗样本
    """
    from autoattack.autoattack import AutoAttack
    import torch.nn as nn
    
    # 创建中间wrapper：将mode参数固定，只接受x作为输入，避免DataParallel kwargs问题
    if is_enhanced:
        # 先解包DataParallel
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        
        class EnhancedModelWithMode(nn.Module):
            def __init__(self, base_model, mode, ensemble_size=1, noise_std=0.01, randomize_defense=False):
                super().__init__()
                # 直接存储base model的forward方法引用和mode
                self.base_model = base_model
                self.mode = mode
                self.ensemble_size = ensemble_size
                self.noise_std = noise_std  # 随机噪声标准差
                self.randomize_defense = randomize_defense  # 内置随机化防御链
            
            def forward(self, x):
                # 调用base_model，所有参数都用位置传递避免DataParallel kwargs问题
                # EnhancedClipVisionModel.forward(self, x, vision_clean=None, output_normalize=False, mode='train', randomize_defense=False)
                if self.ensemble_size == 1:
                    # 单次前向传播（启用内置随机化防御链）
                    result = self.base_model(x, None, False, self.mode, self.randomize_defense)
                    embeddings = result[0]  # 第一个返回值是embedding
                else:
                    # 集成防御：多次前向传播取平均
                    embeddings_list = []
                    for i in range(self.ensemble_size):
                        # 每次添加不同的小噪声（关键！）
                        if self.noise_std > 0:
                            noise = torch.randn_like(x) * self.noise_std
                            x_noisy = torch.clamp(x + noise, 0, 1)
                        else:
                            x_noisy = x
                        
                        # 第一次需要梯度（用于攻击），后续不需要
                        # 注意：每次前向传播都启用randomize_defense，每次防御行为都不同
                        if i == 0:
                            result = self.base_model(x_noisy, None, False, self.mode, self.randomize_defense)
                            embeddings_list.append(result[0])
                        else:
                            with torch.no_grad():
                                result = self.base_model(x_noisy, None, False, self.mode, self.randomize_defense)
                                embeddings_list.append(result[0])
                    
                    embeddings = torch.stack(embeddings_list).mean(dim=0)
                return embeddings
        
        # ensemble_size和noise_std已作为参数传入
        # 如果不是ensemble模式，将noise_std设为0（不添加噪声）
        actual_noise_std = noise_std if ensemble_size > 1 else 0.0
        
        # 灰盒攻击：APGD生成对抗样本时用attack mode（无防御）
        attack_mode = 'attack' if gray_box else inference_mode
        
        # wrap成只接受x的模型
        model_with_mode = EnhancedModelWithMode(base_model, attack_mode, 
                                                 ensemble_size=ensemble_size,
                                                 noise_std=actual_noise_std,
                                                 randomize_defense=randomize_defense)
        # 重新wrap DataParallel
        if torch.cuda.device_count() > 1:
            model_with_mode = nn.DataParallel(model_with_mode)
        model_to_use = model_with_mode
    else:
        model_to_use = model
    
    # 最终wrapper：添加text embedding和logit_scale
    class ModelWrapper(nn.Module):
        def __init__(self, model, text_embeddings, logit_scale=100.0):
            super().__init__()
            self.model = model
            self.register_buffer('text_embeddings', text_embeddings)
            self.logit_scale = logit_scale
        
        def forward(self, x):
            embeddings = self.model(x)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            logits = (embeddings @ self.text_embeddings.T) * self.logit_scale
            return logits
    
    wrapper = ModelWrapper(model_to_use, text_embeddings)
    
    # 创建AutoAttack实例
    # 论文设置: APGD-CE + APGD-DLR (targeted), 各100迭代
    adversary = AutoAttack(
        wrapper, 
        norm='Linf', 
        eps=eps,
        version='custom',  # 使用自定义攻击组合
        attacks_to_run=['apgd-ce', 'apgd-dlr'],  # 论文使用的两种攻击
        verbose=False,
        device=device
    )
    
    # 设置攻击迭代次数和其他参数
    adversary.apgd.n_iter = iterations
    adversary.apgd_targeted.n_iter = iterations
    adversary.apgd.loss = 'ce'
    adversary.apgd.n_restarts = 1
    
    # 运行攻击
    adv_images = adversary.run_standard_evaluation(images, targets, bs=images.shape[0])
    
    return adv_images


def evaluate_model(args):
    """评估模型"""
    # CUDA_VISIBLE_DEVICES已设置，PyTorch看到的是相对设备ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 解析GPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        print(f"🎮 可用GPU: {num_gpus}张卡 (相对ID: {gpu_ids})")
    
    eps = args.eps / 255.0
    
    print("=" * 80)
    print(f"📊 评估配置:")
    print(f"   模型: {args.pretrained}")
    print(f"   推理模式: {args.mode}")
    print(f"   攻击: AutoAttack (APGD-CE + APGD-DLR targeted)")
    print(f"   迭代: {args.iterations}")
    print(f"   eps: {args.eps}/255 = {eps:.6f}")
    print(f"   样本数: {args.max_samples if args.max_samples > 0 else '全部'}")
    print("=" * 80)
    
    # 加载模型
    print("\n🔄 加载模型...")
    
    if args.mode == 'baseline':
        # 基线模型（FARE, TeCoA等）
        model, preprocessor_no_norm, normalizer = load_clip_model(args.clip_model_name, args.pretrained)
        
        # 获取normalize参数
        mean = normalizer.mean
        std = normalizer.std
        
        # 包装模型
        wrapped_model = ClipVisionModel(model.visual, mean, std).to(device)
        is_enhanced = False
        
        # 多GPU
        if torch.cuda.device_count() > 1:
            wrapped_model = torch.nn.DataParallel(wrapped_model, device_ids=gpu_ids)
        
        # 文本编码器（需要移到GPU用于计算text embeddings）
        base_clip = model.to(device)
        tokenizer = open_clip.get_tokenizer(args.clip_model_name)
        
    else:
        # 增强模型 (eval或attack模式)
        enhanced_model, preprocessor_no_norm, normalizer = load_enhanced_clip_model(
            args.clip_model_name, args.pretrained
        )
        wrapped_model = enhanced_model.to(device)
        is_enhanced = True
        
        # 多GPU
        if torch.cuda.device_count() > 1:
            wrapped_model = torch.nn.DataParallel(wrapped_model, device_ids=gpu_ids)
        
        # 文本编码器
        base_clip, _, _ = open_clip.create_model_and_transforms(
            args.clip_model_name, pretrained='openai', device=device
        )
        tokenizer = open_clip.get_tokenizer(args.clip_model_name)
    
    wrapped_model.eval()
    print(f"✅ 模型加载完成 (is_enhanced={is_enhanced})")
    
    # 加载数据集
    print("\n🔄 加载ImageNet验证集...")
    val_root = os.path.join(args.imagenet_root, 'val')
    dataset = ImageNetDataset(
        root=val_root,
        transform=preprocessor_no_norm
    )
    
    if args.max_samples > 0:
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    print(f"   ✓ 加载 {len(dataset)} 张图片")
    
    # 计算文本嵌入
    print("\n🔄 计算类别文本嵌入...")
    text_embeddings = get_text_embeddings(base_clip, tokenizer, device)
    print(f"   ✓ 文本嵌入计算完成")
    
    # 评估
    print(f"\n🔄 开始评估 (batch_size={args.batch_size})...")
    
    correct_clean = 0
    correct_robust = 0
    total = 0
    
    inference_mode = args.mode if args.mode in ['eval', 'attack'] else 'eval'
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="评估进度")):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.shape[0]
        
        # Clean准确率
        with torch.no_grad():
            if is_enhanced:
                embeddings_clean, *_ = wrapped_model(images, mode=inference_mode)
            else:
                if isinstance(wrapped_model, torch.nn.DataParallel):
                    embeddings_clean = wrapped_model.module(images)
                else:
                    embeddings_clean = wrapped_model(images)
            
            embeddings_clean = F.normalize(embeddings_clean.float(), dim=-1)
            logits_clean = (embeddings_clean @ text_embeddings.T) * 100.0  # CLIP logit_scale
            preds_clean = logits_clean.argmax(dim=-1)
            correct_clean += (preds_clean == targets).sum().item()
        
        # Robust准确率（使用AutoAttack）
        if args.attack:
            adv_images = autoattack_eval(
                wrapped_model, images, targets, text_embeddings,
                eps=eps, iterations=args.iterations,
                device=device, is_enhanced=is_enhanced,
                inference_mode=inference_mode,
                ensemble_size=args.ensemble_size,
                noise_std=args.noise_std,
                gray_box=args.gray_box,
                randomize_defense=args.randomize_defense
            )
            
            with torch.no_grad():
                if is_enhanced:
                    embeddings_adv, *_ = wrapped_model(adv_images, mode=inference_mode)
                else:
                    if isinstance(wrapped_model, torch.nn.DataParallel):
                        embeddings_adv = wrapped_model.module(adv_images)
                    else:
                        embeddings_adv = wrapped_model(adv_images)
                
                embeddings_adv = F.normalize(embeddings_adv.float(), dim=-1)
                logits_adv = (embeddings_adv @ text_embeddings.T) * 100.0  # CLIP logit_scale
                preds_adv = logits_adv.argmax(dim=-1)
                correct_robust += (preds_adv == targets).sum().item()
        
        total += batch_size
    
    clean_acc = correct_clean / total
    robust_acc = correct_robust / total if args.attack else 0.0
    
    # 输出结果
    print("\n" + "=" * 80)
    print("📊 评估结果:")
    print(f"   Clean Accuracy:  {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    if args.attack:
        print(f"   Robust Accuracy: {robust_acc:.4f} ({robust_acc*100:.2f}%)")
    print("=" * 80)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成结果文件名（包含ensemble、随机化和灰盒攻击信息）
    model_name = os.path.basename(args.pretrained).replace('.pt', '')
    
    # Ensemble后缀
    if args.ensemble_size > 1:
        if args.noise_std > 0:
            ensemble_suffix = f"_ensemble{args.ensemble_size}_rand{args.noise_std}"
        else:
            ensemble_suffix = f"_ensemble{args.ensemble_size}_det"
    else:
        ensemble_suffix = ""
    
    # 灰盒攻击后缀
    gray_box_suffix = "_graybox" if args.gray_box else ""
    
    result_file = os.path.join(args.output_dir, f"{model_name}_{args.mode}{ensemble_suffix}{gray_box_suffix}_results.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.pretrained}\n")
        f.write(f"Mode: {args.mode}\n")
        if args.ensemble_size > 1:
            f.write(f"EnsembleSize: {args.ensemble_size}\n")
        f.write(f"CleanAcc: {clean_acc:.4f}\n")
        f.write(f"RobustAcc: {robust_acc:.4f}\n")
        f.write(f"Attack: AutoAttack (APGD-CE + APGD-DLR targeted)\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Eps: {args.eps}/255\n")
        f.write(f"Samples: {total}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    print(f"\n✅ 结果已保存到: {result_file}")
    
    return clean_acc, robust_acc


def main():
    parser = argparse.ArgumentParser(description='统一的鲁棒性评估脚本')
    
    # 模型参数
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='预训练模型路径')
    
    # 推理模式
    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'eval', 'attack'],
                       help='推理模式: baseline=基线模型, eval=增强模型完整防御, attack=增强模型无防御')
    
    # 数据参数
    parser.add_argument('--imagenet_root', type=str, 
                       default='/home/ubuntu/data/KeyToken/datasets/imagenet',
                       help='ImageNet数据集根目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='最大评估样本数 (-1表示全部)')
    
    # 攻击参数
    parser.add_argument('--attack', action='store_true', default=True,
                       help='是否进行对抗攻击评估')
    parser.add_argument('--no_attack', action='store_false', dest='attack',
                       help='仅评估Clean准确率')
    parser.add_argument('--eps', type=float, default=4.0,
                       help='扰动幅度 (将除以255)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='APGD迭代次数 (论文默认100)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output/robust_eval',
                       help='输出目录')
    parser.add_argument('--ensemble_size', type=int, default=1,
                       help='集成防御样本数（1=单次，3-5=集成）')
    parser.add_argument('--noise_std', type=float, default=0.01,
                       help='随机化ensemble的噪声标准差（0=确定性，0.01=推荐）')
    parser.add_argument('--gray_box', action='store_true', default=False,
                       help='灰盒攻击：APGD只攻击backbone，不知道防御策略（更强防御）')
    parser.add_argument('--randomize_defense', action='store_true', default=False,
                       help='启用内置随机化防御链：对阈值、上下文扩展、特征融合注入随机性（打破APGD梯度估计）')
    parser.add_argument('--gpu', type=str, default=None,
                       help='使用的GPU编号（不指定则使用CUDA_VISIBLE_DEVICES环境变量）')
    
    args = parser.parse_args()
    
    # 设置GPU（仅当明确指定--gpu时才覆盖环境变量）
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    evaluate_model(args)


if __name__ == '__main__':
    main()
