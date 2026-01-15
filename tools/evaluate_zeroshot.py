#!/usr/bin/env python3
"""
零样本分类评估脚本 - 基于FARE论文设置
支持ImageNet和13个零样本数据集
攻击: APGD-CE + APGD-DLR (100 iterations each)
Eps: 2/255 and 4/255
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.adversarial_training_clip_enhanced import EnhancedClipVisionModel
from train.test_time_defense import ZeroPurDefense, InterpretabilityGuidedDefense, CombinedDefense
from autoattack import AutoAttack
import open_clip
from CLIP_eval.eval_utils import load_clip_model as load_baseline_clip_model
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL


# 13个零样本数据集的配置
ZEROSHOT_DATASETS = {
    'imagenet': {'path': 'datasets/imagenet/val', 'templates': 'imagenet'},
    'cifar10': {'path': 'datasets/webdatasets/cifar10/test', 'templates': 'simple'},
    'cifar100': {'path': 'datasets/webdatasets/cifar100/test', 'templates': 'simple'},
    'flowers102': {'path': 'datasets/webdatasets/flowers/test', 'templates': 'flowers'},
    'imagenet_r': {'path': 'datasets/webdatasets/imagenet_r/test', 'templates': 'simple'},
    'imagenet_sketch': {'path': 'datasets/webdatasets/imagenet_sketch/test', 'templates': 'simple'},
    'pets': {'path': 'datasets/webdatasets/pets/test', 'templates': 'pets'},
    'cars': {'path': 'datasets/webdatasets/cars/test', 'templates': 'simple'},
    'dtd': {'path': 'datasets/webdatasets/dtd/test', 'templates': 'dtd'},
    'caltech101': {'path': 'datasets/webdatasets/caltech101/test', 'templates': 'simple'},
    'aircraft': {'path': 'datasets/webdatasets/fgvc_aircraft/test', 'templates': 'aircraft'},
    'eurosat': {'path': 'datasets/webdatasets/eurosat/test', 'templates': 'simple'},
    'pcam': {'path': 'datasets/webdatasets/pcam/test', 'templates': 'simple'},
    'stl10': {'path': 'datasets/webdatasets/stl10/test', 'templates': 'simple'},
}


# CLIP prompt templates
PROMPT_TEMPLATES = {
    'imagenet': [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a photo of many {}.',
        'a photo of the large {}.',
        'a photo of the small {}.',
    ],
    'simple': ['a photo of a {}.'],
    'flowers': ['a photo of a {}, a type of flower.'],
    'food': ['a photo of {}, a type of food.'],
    'pets': ['a photo of a {}, a type of pet.'],
    'aircraft': ['a photo of a {}, a type of aircraft.'],
    'dtd': ['{} texture.'],
}


class VisionEncoderWrapper(torch.nn.Module):
    """将encode_image包装到forward中，使DataParallel能正确分布计算"""
    def __init__(self, clip_model, is_enhanced=False):
        super().__init__()
        self.clip_model = clip_model
        self.is_enhanced = is_enhanced
    
    def forward(self, x, mode='eval', gray_box=False):
        if self.is_enhanced:
            if gray_box:
                # 灰盒：只攻击backbone
                image_emb, _, _, _, _, _ = self.clip_model(x, mode='attack')
            else:
                # 白盒：攻击完整防御
                image_emb, _, _, _, _, _ = self.clip_model(x, mode=mode)
        else:
            image_emb = self.clip_model.encode_image(x)
        return image_emb
    
    def encode_text(self, text):
        """Delegate to clip_model"""
        return self.clip_model.encode_text(text)


def load_clip_model(checkpoint_path, device, clip_model_name='ViT-L-14'):
    """加载CLIP模型 - 使用CLIP_eval/eval_utils.py中的标准加载方式"""
    print(f"🔄 加载模型: {checkpoint_path}")
    print(f"📦 CLIP架构: {clip_model_name}")
    
    checkpoint_lower = checkpoint_path.lower()
    is_enhanced = not ('fare' in checkpoint_lower or 'tecoa' in checkpoint_lower)
    
    if is_enhanced:
        print("📦 加载KeyToken增强模型...")
        enhanced_model, preprocessor_no_norm, normalizer = load_enhanced_clip_model(
            clip_model_name, checkpoint_path
        )
        vision_base = enhanced_model
        
        # 独立加载文本模型 (openai权重)
        text_clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained='openai', device='cpu'
        )
        text_model = text_clip_model.to(device)
        
    else:
        # FARE/TeCoA模型 - 使用eval_utils中的标准加载函数
        print("📦 加载FARE/TeCoA模型...")
        base_model, preprocessor_no_norm, normalizer = load_baseline_clip_model(
            clip_model_name, checkpoint_path
        )
        vision_base = base_model
        text_model = base_model.to(device)  # baseline模型包含完整CLIP
    
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    preprocess = preprocessor_no_norm
    
    # Vision encoder wrapper
    model = vision_base.to(device)
    
    # 创建Vision Encoder Wrapper
    vision_wrapper = VisionEncoderWrapper(model, is_enhanced)
    
    # 多GPU支持 - 包装wrapper而不是原始模型
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"💻 使用 {num_gpus} 张GPU (DataParallel)")
        vision_wrapper = torch.nn.DataParallel(vision_wrapper)
    
    vision_wrapper.eval()
    print(f"✅ 模型加载完成 (is_enhanced={is_enhanced})")
    
    # 返回vision_wrapper作为主模型，同时保留text_model用于文本编码
    return vision_wrapper, text_model, tokenizer, preprocess, normalizer, is_enhanced


def load_dataset(dataset_name, preprocess, max_samples=-1):
    """加载数据集"""
    config = ZEROSHOT_DATASETS[dataset_name]
    dataset_path = config['path']
    
    print(f"🔄 加载数据集: {dataset_name}")
    print(f"   路径: {dataset_path}")
    
    from torchvision import datasets, transforms
    from torchvision.datasets import ImageFolder
    
    # ImageNet使用标准ImageFolder格式
    if dataset_name == 'imagenet':
        dataset = ImageFolder(dataset_path, transform=preprocess)
    # webdataset格式数据集
    elif 'webdatasets' in dataset_path:
        import webdataset as wds
        from PIL import Image
        import io
        import glob
        
        # webdataset加载逻辑 - 使用glob展开tar文件列表
        tar_pattern = os.path.join(dataset_path, "*.tar")
        tar_files = sorted(glob.glob(tar_pattern))
        
        if not tar_files:
            raise FileNotFoundError(f"No .tar files found in {dataset_path}")
        
        dataset_wds = wds.WebDataset(tar_files).decode("pil").to_tuple("jpg;png;webp", "cls")
        
        # 读取类别名称
        dataset_root = os.path.dirname(dataset_path)  # 去掉 /test 后缀
        classnames_file = os.path.join(dataset_root, "classnames.txt")
        if os.path.exists(classnames_file):
            with open(classnames_file, 'r') as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            classes = None
        
        # 转换为列表以支持len和索引访问
        dataset_list = []
        for img, label in dataset_wds:
            if preprocess is not None:
                img = preprocess(img)
            dataset_list.append((img, label))
            if max_samples > 0 and len(dataset_list) >= max_samples:
                break
        
        # 创建简单的包装类
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data, classes=None):
                self.data = data
                self.classes = classes
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SimpleDataset(dataset_list, classes)
    else:
        # 其他数据集使用ImageFolder格式
        dataset = ImageFolder(dataset_path, transform=preprocess)
    
    if max_samples > 0 and 'webdatasets' not in dataset_path:
        indices = torch.randperm(len(dataset))[:max_samples]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"   ✓ 加载 {len(dataset)} 张图片")
    
    return dataset


def get_text_embeddings(text_model, tokenizer, classnames, templates, device):
    """计算类别文本嵌入"""
    print(f"🔄 计算文本嵌入 ({len(classnames)} 类)...")
    
    text_embeddings = []
    
    with torch.no_grad():
        for classname in tqdm(classnames, desc="Text embeddings"):
            # 对每个类别应用所有模板
            texts = [template.format(classname) for template in templates]
            texts_tokenized = tokenizer(texts).to(device)
            
            # 编码文本（处理可能的DataParallel包装）
            text_encoder = text_model.module if isinstance(text_model, torch.nn.DataParallel) else text_model
            class_embeddings = text_encoder.encode_text(texts_tokenized)
            
            # 归一化并平均
            class_embeddings = F.normalize(class_embeddings, dim=-1)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=-1)
            
            text_embeddings.append(class_embedding)
    
    text_embeddings = torch.stack(text_embeddings)
    print(f"   ✓ 文本嵌入: {text_embeddings.shape}")
    
    return text_embeddings


def evaluate_clean(model, dataloader, text_embeddings, device, normalizer, is_enhanced=False, mode='eval'):
    """评估干净样本准确率"""
    print("🔄 评估干净样本...")
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Clean eval"):
            images = images.to(device)
            labels = labels.to(device)
            
            # baseline模型需要额外normalize；增强模型内部已处理
            if not is_enhanced:
                images = normalizer(images)
            
            # 编码图像 - VisionEncoderWrapper已包装了DataParallel
            image_embeddings = model(images)
            
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            
            # 分类
            logits = 100.0 * image_embeddings @ text_embeddings.T
            predictions = logits.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"   Clean Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def evaluate_robust(model, dataloader, text_embeddings, device, eps, iterations,
                   normalizer, is_enhanced=False, mode='eval', gray_box=False, defense_type=None, noise_std=0.0):
    """评估对抗鲁棒性"""
    print(f"🔄 评估对抗鲁棒性 (eps={eps}, iter={iterations})...")
    if defense_type:
        print(f"   🛡️  测试时防御: {defense_type}")
    if noise_std > 0:
        print(f"   🎲 输入噪声: std={noise_std} (Randomized Smoothing)")
    
    # 创建分类包装器
    class CLIPClassifier(torch.nn.Module):
        def __init__(self, vision_model, text_embeddings, normalizer, is_enhanced, mode, gray_box, noise_std=0.0):
            super().__init__()
            self.vision_model = vision_model
            # 注册为buffer使DataParallel能正确分配到对应设备
            self.register_buffer('text_embeddings', text_embeddings)
            self.normalizer = normalizer
            self.is_enhanced = is_enhanced
            self.mode = mode
            self.gray_box = gray_box
            self.noise_std = noise_std
        
        def forward(self, x):
            # 应用输入噪声（Randomized Smoothing）
            if self.noise_std > 0:
                noise = torch.randn_like(x) * self.noise_std
                x = x + noise
                x = torch.clamp(x, 0, 1)
            
            # baseline模型需要额外normalize；增强模型内部已处理
            if not self.is_enhanced:
                x = self.normalizer(x)
            
            # VisionEncoderWrapper已处理DataParallel和is_enhanced逻辑
            image_emb = self.vision_model(x)
            
            image_emb = F.normalize(image_emb, dim=-1)
            logits = 100.0 * image_emb @ self.text_embeddings.T
            return logits
    
    # 解包vision_model的DataParallel（如果存在），然后对整个classifier进行DataParallel
    # 避免双重包装导致性能下降
    base_vision_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    classifier = CLIPClassifier(base_vision_model, text_embeddings, normalizer, is_enhanced, mode, gray_box, noise_std)
    
    # 多GPU包装classifier - AutoAttack需要对整个classifier进行DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        classifier = torch.nn.DataParallel(classifier)
    
    classifier.eval()
    
    # 初始化测试时防御
    test_defense = None
    if defense_type and is_enhanced:
        if defense_type == 'zeropur':
            test_defense = ZeroPurDefense(sigma=0.5, alpha=0.3, num_steps=5)
        elif defense_type == 'interpretability':
            test_defense = InterpretabilityGuidedDefense(model, top_k_ratio=0.3)
        elif defense_type == 'combined':
            # 组合防御：同时使用ZeroPur和Interpretability-Guided
            test_defense = CombinedDefense(
                model,
                use_interpretability=True,
                use_zeropur=True,
                sigma=0.5,
                alpha=0.3,
                num_steps=5,
                top_k_ratio=0.3
            )
    
    # AutoAttack配置 - 严格按照FARE论文设置
    adversary = AutoAttack(
        classifier,
        norm='Linf',
        eps=eps,
        version='custom',  # 使用custom版本，与evaluate_robust.py一致
        attacks_to_run=['apgd-ce', 'apgd-dlr'],  # FARE论文使用的两种攻击
        verbose=False,
        device=device
    )
    
    # 设置攻击迭代次数
    adversary.apgd.n_iter = iterations
    adversary.apgd_targeted.n_iter = iterations
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Robust eval"):
        images = images.to(device)
        labels = labels.to(device)
        
        # AutoAttack
        with torch.enable_grad():
            adv_images = adversary.run_standard_evaluation(images, labels, bs=images.size(0))
        
        # 应用测试时防御（如果启用）
        if test_defense is not None:
            with torch.no_grad():
                logits_pred = classifier(adv_images)
                pred_classes = logits_pred.argmax(dim=-1)
            
            if defense_type == 'zeropur':
                with torch.enable_grad():
                    adv_images = test_defense.purify(adv_images, model)
            elif defense_type == 'interpretability':
                purified_images = []
                for i in range(adv_images.size(0)):
                    purified = test_defense.purify(
                        adv_images[i:i+1], 
                        pred_classes[i].item()
                    )
                    purified_images.append(purified)
                adv_images = torch.cat(purified_images, dim=0)
            elif defense_type == 'combined':
                purified_images = []
                for i in range(adv_images.size(0)):
                    with torch.enable_grad():
                        purified = test_defense.purify(
                            adv_images[i:i+1], 
                            pred_classes[i].item()
                        )
                    purified_images.append(purified)
                adv_images = torch.cat(purified_images, dim=0)
        
        # 评估
        with torch.no_grad():
            logits = classifier(adv_images)
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"   Robust Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='零样本分类鲁棒性评估')
    
    # 模型配置
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       choices=list(ZEROSHOT_DATASETS.keys()),
                       help='数据集名称')
    
    # 攻击配置
    parser.add_argument('--eps', type=float, default=4.0,
                       help='扰动强度 (x/255)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='APGD迭代次数')
    parser.add_argument('--gray_box', action='store_true',
                       help='灰盒攻击（仅攻击backbone）')
    
    # 评估配置
    parser.add_argument('--mode', type=str, default='eval',
                       choices=['eval', 'baseline'],
                       help='推理模式')
    parser.add_argument('--batch_size', type=int, default=64)  # 4GPU可用64
    parser.add_argument('--robust_batch_size', type=int, default=64,
                       help='Robust eval batch size (same as batch_size with multi-GPU)')
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='最大样本数（-1表示全部）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/zeroshot_eval')
    
    # 测试时防御策略
    parser.add_argument('--defense', type=str, default=None,
                       choices=[None, 'zeropur', 'interpretability', 'combined'],
                       help='测试时防御策略（无需训练）。combined=ZeroPur+Interpretability')
    parser.add_argument('--noise_std', type=float, default=0.0,
                       help='输入随机噪声标准差（Randomized Smoothing），0=确定性')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    # 加载模型 - vision_model是DataParallel包装的VisionEncoderWrapper, text_model是原始CLIP模型
    vision_model, text_model, tokenizer, preprocess, normalizer, is_enhanced = load_clip_model(
        args.checkpoint, device, args.clip_model_name
    )
    
    # 加载数据集
    dataset = load_dataset(args.dataset, preprocess, args.max_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # 获取类别名称和模板
    if args.dataset == 'imagenet':
        # ImageNet需要使用人类可读的类别名称，而不是WordNet ID
        classnames = list(IMAGENET_1K_CLASS_ID_TO_LABEL.values())
    else:
        classnames = dataset.dataset.classes if hasattr(dataset, 'dataset') else dataset.classes
    template_key = ZEROSHOT_DATASETS[args.dataset]['templates']
    templates = PROMPT_TEMPLATES[template_key]
    
    # 计算文本嵌入 - 使用原始text_model
    text_embeddings = get_text_embeddings(
        text_model, tokenizer, classnames, templates, device
    )
    
    # 评估
    print("=" * 80)
    print(f"📊 零样本分类评估")
    print(f"   数据集: {args.dataset}")
    print(f"   模型: {args.checkpoint}")
    print(f"   Eps: {args.eps}/255")
    print("=" * 80)
    
    # 干净样本 - 使用vision_model
    clean_acc = evaluate_clean(vision_model, dataloader, text_embeddings, device, normalizer, is_enhanced, args.mode)
    
    # 对抗样本 - 使用更小的batch_size避免OOM
    eps_normalized = args.eps / 255.0
    robust_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.robust_batch_size, shuffle=False, num_workers=4
    )
    robust_acc = evaluate_robust(
        vision_model, robust_dataloader, text_embeddings, device,
        eps_normalized, args.iterations, normalizer, is_enhanced, args.mode, args.gray_box,
        defense_type=args.defense, noise_std=args.noise_std
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.checkpoint).replace('.pt', '')
    eps_str = f"eps{int(args.eps)}"
    gray_str = "_graybox" if args.gray_box else ""
    defense_str = f"_{args.defense}" if args.defense else ""
    noise_str = f"_noise{args.noise_std}" if args.noise_std > 0 else ""
    
    result_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{eps_str}{gray_str}{defense_str}{noise_str}_results.json"
    )
    
    results = {
        'model': args.checkpoint,
        'dataset': args.dataset,
        'mode': args.mode,
        'eps': args.eps,
        'iterations': args.iterations,
        'gray_box': args.gray_box,
        'defense': args.defense,
        'noise_std': args.noise_std,
        'clean_acc': clean_acc,
        'robust_acc': robust_acc,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"✅ 结果已保存: {result_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
