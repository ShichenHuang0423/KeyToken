#!/usr/bin/env python3
"""
POPE幻觉评估脚本 - Polling-based Object Probing Evaluation
评估LVLM的对象幻觉问题
二分类任务：对象是否在图像中
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
import open_clip


def load_pope_dataset(dataset_root, split='random'):
    """
    加载POPE数据集
    split: random, popular, adversarial
    """
    print(f"🔄 加载POPE数据集 (split={split})...")
    
    # POPE annotation文件
    ann_file = os.path.join(dataset_root, f'coco_pope_{split}.json')
    images_dir = os.path.join(dataset_root, 'val2014')
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        image_id = item['image']
        question = item['text']  # "Is there a [object] in the image?"
        answer = item['label']  # "yes" or "no"
        
        # 提取对象名称
        object_name = question.replace("Is there a ", "").replace(" in the image?", "").strip()
        
        image_path = os.path.join(images_dir, image_id)
        
        samples.append({
            'image_path': image_path,
            'question': question,
            'object': object_name,
            'answer': answer
        })
    
    print(f"   ✓ 加载 {len(samples)} 个样本")
    return samples


class POPEModel(torch.nn.Module):
    """POPE二分类模型（基于CLIP）"""
    def __init__(self, vision_model, text_model, tokenizer, 
                 is_enhanced=False, mode='eval'):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.is_enhanced = is_enhanced
        self.mode = mode
        
        # Yes/No嵌入
        with torch.no_grad():
            yes_no_tokens = tokenizer(["yes", "no"])
            device = next(text_model.parameters()).device
            self.answer_embeddings = F.normalize(
                text_model(yes_no_tokens.to(device)),
                dim=-1
            )
    
    def forward(self, image, question):
        """返回yes/no的logits"""
        # 图像编码
        if self.is_enhanced:
            image_emb, _, _, _, _, _ = self.vision_model(image, mode=self.mode)
        else:
            # 处理DataParallel包装
            if isinstance(self.vision_model, torch.nn.DataParallel):
                image_emb = self.vision_model.module.encode_image(image)
            else:
                image_emb = self.vision_model.encode_image(image)
        
        image_emb = F.normalize(image_emb, dim=-1)
        
        # 问题编码
        question_tokens = self.tokenizer([question]).to(image.device)
        question_emb = self.text_model(question_tokens)
        question_emb = F.normalize(question_emb, dim=-1)
        
        # 多模态融合
        multimodal_emb = (image_emb + question_emb) / 2.0
        
        # Yes/No预测
        logits = 100.0 * multimodal_emb @ self.answer_embeddings.T
        
        return logits


def evaluate_pope(model, tokenizer, samples, device, preprocess, normalizer, is_enhanced=False, mode='eval'):
    """评估POPE"""
    print("🔄 开始POPE评估...")
    
    # 创建POPE模型 - 处理DataParallel包装
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if is_enhanced:
        text_model = base_model.model.encode_text
    else:
        text_model = base_model.encode_text
    
    pope_model = POPEModel(
        model, text_model, tokenizer,
        is_enhanced, mode
    ).to(device)
    pope_model.eval()
    
    correct = 0
    total = 0
    
    # 统计指标
    true_positive = 0  # 正确识别存在的对象
    false_positive = 0  # 错误识别不存在的对象（幻觉）
    true_negative = 0  # 正确识别不存在的对象
    false_negative = 0  # 错误识别存在的对象
    
    predictions = []
    
    for sample in tqdm(samples, desc="POPE eval"):
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        image_tensor = normalizer(image_tensor)  # 应用normalize
        
        question = sample['question']
        gt_answer = sample['answer']
        
        # 预测
        with torch.no_grad():
            logits = pope_model(image_tensor, question)
            pred_idx = logits.argmax(dim=-1).item()
            pred_answer = "yes" if pred_idx == 0 else "no"
            
            predictions.append({
                'question': question,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer
            })
            
            # 统计
            if pred_answer == gt_answer:
                correct += 1
                if gt_answer == "yes":
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if pred_answer == "yes" and gt_answer == "no":
                    false_positive += 1  # 幻觉
                elif pred_answer == "no" and gt_answer == "yes":
                    false_negative += 1
            
            total += 1
    
    # 计算指标
    accuracy = correct / total if total > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 幻觉率
    hallucination_rate = false_positive / total if total > 0 else 0
    
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1: {f1:.4f}")
    print(f"   Hallucination Rate: {hallucination_rate:.4f} ({hallucination_rate*100:.2f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hallucination_rate': hallucination_rate,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative
    }, predictions


def main():
    parser = argparse.ArgumentParser(description='POPE幻觉评估')
    
    # 模型配置
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP模型架构')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='POPE数据集根目录')
    parser.add_argument('--split', type=str, default='random',
                       choices=['random', 'popular', 'adversarial'])
    
    # 评估配置
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/pope_eval')
    
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
    samples = load_pope_dataset(args.dataset_root, args.split)
    
    # 评估
    print("=" * 80)
    print(f"📊 POPE幻觉评估")
    print(f"   Split: {args.split}")
    print(f"   模型: {args.checkpoint}")
    print("=" * 80)
    
    metrics, predictions = evaluate_pope(
        model, tokenizer, samples, device,
        preprocess, normalizer, is_enhanced, args.mode
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.checkpoint).replace('.pt', '')
    
    result_file = os.path.join(
        args.output_dir,
        f"{model_name}_pope_{args.split}_results.json"
    )
    
    results = {
        'model': args.checkpoint,
        'split': args.split,
        'mode': args.mode,
        'metrics': metrics,
        'num_samples': len(samples),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存详细预测
    pred_file = os.path.join(
        args.output_dir,
        f"{model_name}_pope_{args.split}_predictions.json"
    )
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("=" * 80)
    print(f"✅ 结果已保存: {result_file}")
    print(f"✅ 预测已保存: {pred_file}")


if __name__ == '__main__':
    main()
