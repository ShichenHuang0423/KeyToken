#!/usr/bin/env python3
"""
VQA鲁棒性评估脚本 - 基于FARE论文设置
支持VQAv2和TextVQA数据集
攻击pipeline: APGD半精度100iter -> 单精度攻击 -> Targeted攻击
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.adversarial_training_clip_enhanced import EnhancedClipVisionModel
from CLIP_eval.eval_utils import load_clip_model as load_baseline_clip_model
from CLIP_eval.eval_utils_enhanced import load_enhanced_clip_model
from autoattack import AutoAttack
import open_clip


def load_vqa_dataset(dataset_name, dataset_root, split='val', max_samples=500):
    """加载VQA数据集"""
    print(f"🔄 加载{dataset_name}数据集...")
    
    if dataset_name == 'vqav2':
        # VQAv2格式
        questions_file = os.path.join(dataset_root, f'v2_OpenEnded_mscoco_{split}2014_questions.json')
        annotations_file = os.path.join(dataset_root, f'v2_mscoco_{split}2014_annotations.json')
        images_dir = os.path.join(dataset_root, f'{split}2014')
        
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
        
        # 构建样本
        samples = []
        for q, a in zip(questions_data['questions'], annotations_data['annotations']):
            image_id = q['image_id']
            image_path = os.path.join(images_dir, f"COCO_{split}2014_{image_id:012d}.jpg")
            
            samples.append({
                'image_path': image_path,
                'question': q['question'],
                'answer': a['multiple_choice_answer'],
                'all_answers': [ans['answer'] for ans in a['answers']]
            })
    
    elif dataset_name == 'textvqa':
        # TextVQA格式
        annotations_file = os.path.join(dataset_root, f'TextVQA_{split}.json')
        images_dir = os.path.join(dataset_root, 'train_images')
        
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data['data']:
            samples.append({
                'image_path': os.path.join(images_dir, item['image_id'] + '.jpg'),
                'question': item['question'],
                'answer': item['answers'][0],
                'all_answers': item['answers']
            })
    
    # 随机采样
    if max_samples > 0 and len(samples) > max_samples:
        indices = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in indices]
    
    print(f"   ✓ 加载 {len(samples)} 个样本")
    return samples


def vqa_accuracy(pred_answer, gt_answers):
    """计算VQA准确率（考虑多个ground truth）"""
    pred_answer = pred_answer.lower().strip()
    
    # VQA评估规则：至少3个标注者给出相同答案才算正确
    answer_counts = {}
    for ans in gt_answers:
        ans = ans.lower().strip()
        answer_counts[ans] = answer_counts.get(ans, 0) + 1
    
    for ans, count in answer_counts.items():
        if pred_answer == ans and count >= 3:
            return 1.0
        elif pred_answer == ans:
            return count / 3.0
    
    return 0.0


class VQAModel(torch.nn.Module):
    """VQA模型包装器（CLIP + 简单答案预测）"""
    def __init__(self, vision_model, text_model, tokenizer, answer_vocab, 
                 is_enhanced=False, mode='eval', gray_box=False):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.answer_vocab = answer_vocab
        self.is_enhanced = is_enhanced
        self.mode = mode
        self.gray_box = gray_box
        
        # 答案嵌入
        with torch.no_grad():
            answer_texts = tokenizer(answer_vocab)
            self.answer_embeddings = F.normalize(
                text_model(answer_texts.to(next(text_model.parameters()).device)),
                dim=-1
            )
    
    def forward(self, image, question_text):
        """前向传播"""
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
        
        # 问题编码
        question_tokens = self.tokenizer([question_text]).to(image.device)
        question_emb = self.text_model(question_tokens)
        question_emb = F.normalize(question_emb, dim=-1)
        
        # 多模态融合（简单拼接）
        multimodal_emb = (image_emb + question_emb) / 2.0
        
        # 答案预测
        logits = 100.0 * multimodal_emb @ self.answer_embeddings.T
        
        return logits


def attack_pipeline_vqa(vqa_model, image, question, answer_idx, eps, device):
    """
    FARE攻击pipeline for VQA:
    1. APGD半精度100 iter
    2. 检查阈值，如果未达到则继续
    3. APGD单精度攻击
    4. Targeted攻击（单精度）
    """
    threshold = 0.5  # 分数阈值
    
    # 阶段1: 半精度APGD
    adversary_half = AutoAttack(
        vqa_model,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_half.attacks_to_run = ['apgd-ce']
    adversary_half.apgd.n_iter = 100
    
    with torch.cuda.amp.autocast():
        adv_image = adversary_half.run_standard_evaluation(
            image.unsqueeze(0),
            torch.tensor([answer_idx]).to(device),
            bs=1
        )
    
    # 检查是否需要继续攻击
    with torch.no_grad():
        logits = vqa_model(adv_image, question)
        score = F.softmax(logits, dim=-1)[0, answer_idx].item()
    
    if score < threshold:
        return adv_image  # 攻击成功，提前返回
    
    # 阶段2: 单精度APGD
    adversary_full = AutoAttack(
        vqa_model,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_full.attacks_to_run = ['apgd-ce']
    adversary_full.apgd.n_iter = 100
    
    with torch.enable_grad():
        adv_image = adversary_full.run_standard_evaluation(
            adv_image,
            torch.tensor([answer_idx]).to(device),
            bs=1
        )
    
    # 检查是否需要targeted攻击
    with torch.no_grad():
        logits = vqa_model(adv_image, question)
        score = F.softmax(logits, dim=-1)[0, answer_idx].item()
    
    if score < threshold:
        return adv_image
    
    # 阶段3: Targeted攻击
    adversary_targeted = AutoAttack(
        vqa_model,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_targeted.attacks_to_run = ['apgd-dlr']
    adversary_targeted.apgd_targeted.n_iter = 100
    
    with torch.enable_grad():
        adv_image = adversary_targeted.run_standard_evaluation(
            adv_image,
            torch.tensor([answer_idx]).to(device),
            bs=1
        )
    
    return adv_image


def evaluate_vqa(model, tokenizer, samples, answer_vocab, eps, device,
                preprocess, normalizer, is_enhanced=False, mode='eval', gray_box=False):
    """评估VQA"""
    print(f"🔄 开始VQA评估 (eps={eps})...")
    
    # 创建VQA模型 - 处理DataParallel包装
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if is_enhanced:
        text_model = base_model.model.encode_text
    else:
        text_model = base_model.encode_text
    
    vqa_model = VQAModel(
        model, text_model, tokenizer, answer_vocab,
        is_enhanced, mode, gray_box
    ).to(device)
    vqa_model.eval()
    
    # 获取答案索引映射
    answer_to_idx = {ans: i for i, ans in enumerate(answer_vocab)}
    
    clean_correct = 0
    robust_correct = 0
    total = 0
    
    for sample in tqdm(samples, desc="VQA eval"):
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensor = normalizer(image_tensor)  # 应用normalize
        
        question = sample['question']
        gt_answer = sample['answer']
        gt_answers = sample['all_answers']
        
        # 找到ground truth在词表中的索引
        if gt_answer not in answer_to_idx:
            continue  # 跳过不在词表中的答案
        
        answer_idx = answer_to_idx[gt_answer]
        
        # 干净样本评估
        with torch.no_grad():
            logits = vqa_model(image_tensor, question)
            pred_idx = logits.argmax(dim=-1).item()
            pred_answer = answer_vocab[pred_idx]
            
            acc = vqa_accuracy(pred_answer, gt_answers)
            clean_correct += acc
        
        # 对抗样本评估
        with torch.enable_grad():
            adv_image = attack_pipeline_vqa(
                vqa_model, image_tensor, question, answer_idx, eps, device
            )
        
        with torch.no_grad():
            logits = vqa_model(adv_image, question)
            pred_idx = logits.argmax(dim=-1).item()
            pred_answer = answer_vocab[pred_idx]
            
            acc = vqa_accuracy(pred_answer, gt_answers)
            robust_correct += acc
        
        total += 1
    
    clean_acc = clean_correct / total if total > 0 else 0
    robust_acc = robust_correct / total if total > 0 else 0
    
    print(f"   Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    print(f"   Robust Accuracy: {robust_acc:.4f} ({robust_acc*100:.2f}%)")
    
    return clean_acc, robust_acc


def build_answer_vocabulary(samples, top_k=3000):
    """构建答案词表（取最常见的答案）"""
    from collections import Counter
    
    answer_counts = Counter()
    for sample in samples:
        for ans in sample['all_answers']:
            answer_counts[ans.lower().strip()] += 1
    
    # 取top_k最常见答案
    most_common = answer_counts.most_common(top_k)
    answer_vocab = [ans for ans, _ in most_common]
    
    print(f"   ✓ 答案词表大小: {len(answer_vocab)}")
    return answer_vocab


def main():
    parser = argparse.ArgumentParser(description='VQA鲁棒性评估')
    
    # 模型配置
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--dataset', type=str, default='vqav2',
                       choices=['vqav2', 'textvqa'])
    parser.add_argument('--dataset_root', type=str, required=True)
    
    # 攻击配置
    parser.add_argument('--eps', type=float, default=4.0)
    parser.add_argument('--gray_box', action='store_true')
    
    # 评估配置
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/vqa_eval')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    # 加载模型
    print(f"🔄 加载模型: {args.checkpoint}")
    print(f"📦 CLIP架构: {args.clip_model_name}")
    
    if 'fare' in args.checkpoint.lower() or 'tecoa' in args.checkpoint.lower():
        # 使用统一的加载函数
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
    samples = load_vqa_dataset(
        args.dataset, args.dataset_root, 'val', args.max_samples
    )
    
    # 构建答案词表
    answer_vocab = build_answer_vocabulary(samples)
    
    # 评估
    print("=" * 80)
    print(f"📊 VQA鲁棒性评估")
    print(f"   数据集: {args.dataset}")
    print(f"   模型: {args.checkpoint}")
    print(f"   Eps: {args.eps}/255")
    print("=" * 80)
    
    eps_normalized = args.eps / 255.0
    clean_acc, robust_acc = evaluate_vqa(
        model, tokenizer, samples, answer_vocab, eps_normalized, device,
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
        'clean_acc': clean_acc,
        'robust_acc': robust_acc,
        'num_samples': len(samples),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"✅ 结果已保存: {result_file}")


if __name__ == '__main__':
    main()
