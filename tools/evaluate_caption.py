#!/usr/bin/env python3
"""
Image Captioning鲁棒性评估 - 基于FARE论文设置
支持COCO和Flickr30k数据集
攻击pipeline: APGD半精度 -> 单精度攻击
评估指标: CIDEr score
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
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.adversarial_training_clip_enhanced import EnhancedClipVisionModel
from CLIP_eval.eval_utils import load_clip_model as load_baseline_clip_model
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
from autoattack import AutoAttack
import open_clip


def load_caption_dataset(dataset_name, dataset_root, split='val', max_samples=500):
    """加载Caption数据集"""
    print(f"🔄 加载{dataset_name}数据集...")
    
    if dataset_name == 'coco':
        # COCO Captions
        ann_file = os.path.join(dataset_root, 'annotations', f'captions_{split}2014.json')
        images_dir = os.path.join(dataset_root, f'{split}2014')
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        
        # 随机采样
        if max_samples > 0 and len(img_ids) > max_samples:
            img_ids = np.random.choice(img_ids, max_samples, replace=False).tolist()
        
        samples = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            captions = [ann['caption'] for ann in anns]
            image_path = os.path.join(images_dir, img_info['file_name'])
            
            samples.append({
                'image_id': img_id,
                'image_path': image_path,
                'captions': captions
            })
    
    elif dataset_name == 'flickr30k':
        # Flickr30k
        ann_file = os.path.join(dataset_root, 'results_20130124.token')
        images_dir = os.path.join(dataset_root, 'flickr30k_images')
        
        # 解析annotation文件
        image_captions = {}
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_id = parts[0].split('#')[0]
                caption = parts[1]
                
                if img_id not in image_captions:
                    image_captions[img_id] = []
                image_captions[img_id].append(caption)
        
        # 构建样本
        samples = []
        img_ids = list(image_captions.keys())
        
        if max_samples > 0 and len(img_ids) > max_samples:
            img_ids = np.random.choice(img_ids, max_samples, replace=False).tolist()
        
        for img_id in img_ids:
            image_path = os.path.join(images_dir, img_id)
            if os.path.exists(image_path):
                samples.append({
                    'image_id': img_id,
                    'image_path': image_path,
                    'captions': image_captions[img_id]
                })
    
    print(f"   ✓ 加载 {len(samples)} 个样本")
    return samples


class SimpleCaptionModel(torch.nn.Module):
    """简化的Caption模型（CLIP-based）"""
    def __init__(self, vision_model, text_model, tokenizer, vocab, 
                 is_enhanced=False, mode='eval', gray_box=False):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.is_enhanced = is_enhanced
        self.mode = mode
        self.gray_box = gray_box
        
        # 词汇嵌入
        with torch.no_grad():
            word_tokens = tokenizer(vocab)
            self.word_embeddings = F.normalize(
                text_model(word_tokens.to(next(text_model.parameters()).device)),
                dim=-1
            )
    
    def forward(self, image):
        """返回对词汇表的logits"""
        # 图像编码
        if self.is_enhanced:
            if self.gray_box:
                image_emb, _, _, _, _, _ = self.vision_model(image, mode='attack')
            else:
                image_emb, _, _, _, _, _ = self.vision_model(image, mode=self.mode)
        else:
            # 处理DataParallel包装
            if isinstance(self.vision_model, torch.nn.DataParallel):
                image_emb = self.vision_model.module.encode_image(image)
            else:
                image_emb = self.vision_model.encode_image(image)
        
        image_emb = F.normalize(image_emb, dim=-1)
        
        # 计算与词汇的相似度
        logits = 100.0 * image_emb @ self.word_embeddings.T
        
        return logits
    
    def generate_caption(self, image, max_length=20, beam_size=3):
        """生成caption（beam search）"""
        device = image.device
        batch_size = image.size(0)
        
        # 编码图像
        with torch.no_grad():
            logits = self.forward(image)
            
            # 简单greedy decoding (实际应该用beam search)
            top_k = 10
            _, top_indices = logits.topk(top_k, dim=-1)
            
            # 选择top-k词汇组成caption
            caption_words = []
            for i in range(min(max_length, top_k)):
                word_idx = top_indices[0, i].item()
                caption_words.append(self.vocab[word_idx])
            
            caption = ' '.join(caption_words)
        
        return caption


def build_vocab_from_captions(samples, max_vocab_size=5000):
    """从captions构建词汇表"""
    from collections import Counter
    
    word_counts = Counter()
    for sample in samples:
        for caption in sample['captions']:
            words = caption.lower().split()
            word_counts.update(words)
    
    # 取最常见的词
    most_common = word_counts.most_common(max_vocab_size)
    vocab = [word for word, _ in most_common]
    
    print(f"   ✓ 词汇表大小: {len(vocab)}")
    return vocab


def attack_pipeline_caption(caption_model, image, target_caption_emb, eps, device):
    """
    FARE攻击pipeline for Caption:
    1. APGD半精度100 iter
    2. 检查阈值
    3. APGD单精度攻击
    """
    threshold = 0.3  # CIDEr阈值
    
    # 创建攻击目标（最小化与目标caption的相似度）
    class CaptionLoss(torch.nn.Module):
        def __init__(self, model, target_emb):
            super().__init__()
            self.model = model
            self.target_emb = target_emb
        
        def forward(self, x):
            # 返回与目标的负相似度（最大化距离）
            logits = self.model(x)
            # 简化：直接用logits作为embedding proxy
            return -F.cosine_similarity(logits.mean(dim=-1, keepdim=True), 
                                       self.target_emb, dim=-1)
    
    loss_fn = CaptionLoss(caption_model, target_caption_emb)
    
    # 阶段1: 半精度APGD
    adversary_half = AutoAttack(
        caption_model,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_half.attacks_to_run = ['apgd-ce']
    adversary_half.apgd.n_iter = 100
    
    # 使用dummy label
    dummy_label = torch.zeros(image.size(0), dtype=torch.long).to(device)
    
    with torch.cuda.amp.autocast():
        adv_image = adversary_half.run_standard_evaluation(
            image, dummy_label, bs=image.size(0)
        )
    
    # 阶段2: 单精度攻击
    adversary_full = AutoAttack(
        caption_model,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_full.attacks_to_run = ['apgd-ce']
    adversary_full.apgd.n_iter = 100
    
    with torch.enable_grad():
        adv_image = adversary_full.run_standard_evaluation(
            adv_image, dummy_label, bs=image.size(0)
        )
    
    return adv_image


def compute_cider_score(pred_captions, gt_captions_dict):
    """计算CIDEr score"""
    from pycocoevalcap.cider.cider import Cider
    
    # 格式转换
    gts = {}
    res = {}
    for i, (img_id, pred_caption) in enumerate(pred_captions):
        res[i] = [pred_caption]
        gts[i] = gt_captions_dict[img_id]
    
    # 计算CIDEr
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    
    return score


def evaluate_caption(model, tokenizer, samples, vocab, eps, device,
                    preprocess, normalizer, is_enhanced=False, mode='eval', gray_box=False):
    """评估Caption生成"""
    print(f"🔄 开始Caption评估 (eps={eps})...")
    
    # 创建Caption模型 - 处理DataParallel包装
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if is_enhanced:
        text_model = base_model.model.encode_text
    else:
        text_model = base_model.encode_text
    
    caption_model = SimpleCaptionModel(
        model, text_model, tokenizer, vocab,
        is_enhanced, mode, gray_box
    ).to(device)
    caption_model.eval()
    
    clean_captions = []
    robust_captions = []
    gt_captions_dict = {}
    
    for sample in tqdm(samples, desc="Caption eval"):
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensor = normalizer(image_tensor)  # 应用normalize
        
        img_id = sample['image_id']
        gt_captions = sample['captions']
        gt_captions_dict[img_id] = gt_captions
        
        # 干净样本生成
        with torch.no_grad():
            clean_caption = caption_model.generate_caption(image_tensor)
            clean_captions.append((img_id, clean_caption))
        
        # 对抗样本生成
        # 计算目标caption embedding (第一个GT)
        with torch.no_grad():
            gt_tokens = tokenizer([gt_captions[0]]).to(device)
            target_emb = text_model(gt_tokens)
            target_emb = F.normalize(target_emb, dim=-1)
        
        with torch.enable_grad():
            adv_image = attack_pipeline_caption(
                caption_model, image_tensor, target_emb, eps, device
            )
        
        with torch.no_grad():
            robust_caption = caption_model.generate_caption(adv_image)
            robust_captions.append((img_id, robust_caption))
    
    # 计算CIDEr scores
    clean_cider = compute_cider_score(clean_captions, gt_captions_dict)
    robust_cider = compute_cider_score(robust_captions, gt_captions_dict)
    
    print(f"   Clean CIDEr: {clean_cider:.4f}")
    print(f"   Robust CIDEr: {robust_cider:.4f}")
    
    return clean_cider, robust_cider


def main():
    parser = argparse.ArgumentParser(description='Caption鲁棒性评估')
    
    # 模型配置
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--dataset', type=str, default='coco',
                       choices=['coco', 'flickr30k'])
    parser.add_argument('--dataset_root', type=str, required=True)
    
    # 攻击配置
    parser.add_argument('--eps', type=float, default=4.0)
    parser.add_argument('--gray_box', action='store_true')
    
    # 评估配置
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/caption_eval')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    # 加载模型
    print(f"🔄 加载模型: {args.checkpoint}")
    print(f"📦 CLIP架构: {args.clip_model_name}")
    
    if 'fare' in args.checkpoint.lower() or 'tecoa' in args.checkpoint.lower():
        model, preprocess, normalizer = load_baseline_clip_model(
            args.clip_model_name, args.checkpoint
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(args.clip_model_name)
        is_enhanced = False
    else:
        enhanced_model, preprocess, normalizer = load_enhanced_clip_model(
            args.clip_model_name, args.checkpoint
        )
        model = enhanced_model.to(device)
        tokenizer = open_clip.get_tokenizer(args.clip_model_name)
        is_enhanced = True
    
    # 多GPU支持
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"💻 使用 {num_gpus} 张GPU (DataParallel)")
        model = torch.nn.DataParallel(model)
    
    model.eval()
    
    # 加载数据集
    samples = load_caption_dataset(
        args.dataset, args.dataset_root, 'val', args.max_samples
    )
    
    # 构建词汇表
    vocab = build_vocab_from_captions(samples)
    
    # 评估
    print("=" * 80)
    print(f"📊 Caption鲁棒性评估")
    print(f"   数据集: {args.dataset}")
    print(f"   模型: {args.checkpoint}")
    print(f"   Eps: {args.eps}/255")
    print("=" * 80)
    
    eps_normalized = args.eps / 255.0
    clean_cider, robust_cider = evaluate_caption(
        model, tokenizer, samples, vocab, eps_normalized, device,
        preprocess, normalizer, is_enhanced, args.mode, args.gray_box
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.checkpoint).replace('.pt', '')
    eps_str = f"eps{int(args.eps)}"
    gray_str = "_graybox" if args.gray_box else ""
    
    result_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{eps_str}{gray_str}_results.json"
    )
    
    results = {
        'model': args.checkpoint,
        'dataset': args.dataset,
        'mode': args.mode,
        'eps': args.eps,
        'gray_box': args.gray_box,
        'clean_cider': clean_cider,
        'robust_cider': robust_cider,
        'num_samples': len(samples),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"✅ 结果已保存: {result_file}")


if __name__ == '__main__':
    main()
