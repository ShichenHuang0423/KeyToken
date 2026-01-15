#!/usr/bin/env python3
"""
完整LVLM VQA评估 - 使用LLaVA和OpenFlamingo
遵循FARE论文设置
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
from autoattack import AutoAttack

# LLaVA imports
try:
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    print("警告: LLaVA未安装，请安装llava包")

# 导入LVLM工具
from lvlm_utils import get_lvlm_model


def load_vqa_dataset(dataset_name: str, dataset_root: str, max_samples: int = -1):
    """加载VQA数据集"""
    if dataset_name == 'vqav2':
        questions_file = os.path.join(dataset_root, 'v2_OpenEnded_mscoco_val2014_questions.json')
        annotations_file = os.path.join(dataset_root, 'v2_mscoco_val2014_annotations.json')
        images_dir = os.path.join(dataset_root, 'val2014')
        
        with open(questions_file) as f:
            questions_data = json.load(f)
        with open(annotations_file) as f:
            annotations_data = json.load(f)
        
        # 创建答案映射
        ann_dict = {ann['question_id']: ann for ann in annotations_data['annotations']}
        
        samples = []
        for q in questions_data['questions']:
            qid = q['question_id']
            if qid not in ann_dict:
                continue
            
            img_id = q['image_id']
            img_path = os.path.join(images_dir, f"COCO_val2014_{img_id:012d}.jpg")
            
            if not os.path.exists(img_path):
                continue
            
            answer = ann_dict[qid]['multiple_choice_answer']
            
            samples.append({
                'image_path': img_path,
                'question': q['question'],
                'answer': answer,
                'question_id': qid
            })
    
    elif dataset_name == 'textvqa':
        annotations_file = os.path.join(dataset_root, 'TextVQA_0.5.1_val.json')
        images_dir = os.path.join(dataset_root, 'train_images')
        
        with open(annotations_file) as f:
            data = json.load(f)
        
        samples = []
        for item in data['data']:
            img_path = os.path.join(images_dir, item['image_id'] + '.jpg')
            if not os.path.exists(img_path):
                continue
            
            # 使用最常见的答案
            answers = item['answers']
            answer = max(set(answers), key=answers.count)
            
            samples.append({
                'image_path': img_path,
                'question': item['question'],
                'answer': answer,
                'question_id': item['question_id']
            })
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 随机采样
    if max_samples > 0 and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)
    
    print(f"✓ 加载 {len(samples)} 个VQA样本")
    return samples


def llava_generate_answer(model, tokenizer, image_processor, image, question, device='cuda'):
    """使用LLaVA生成答案"""
    disable_torch_init()
    
    # 准备conversation
    conv = conv_templates["llava_v1"].copy()
    
    # 添加图像token
    if model.config.mm_use_im_start_end:
        question = f"<im_start><image><im_end>\n{question}"
    else:
        question = f"<image>\n{question}"
    
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # 处理图像
    image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
        )
    
    # 解码
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return outputs


def flamingo_generate_answer(model, tokenizer, image_processor, image, question, device='cuda'):
    """使用OpenFlamingo生成答案"""
    # 准备prompt - OpenFlamingo zero-shot格式
    prompt = f"<image>Question: {question} Answer:"
    
    # 处理图像
    image_tensor = image_processor(image).unsqueeze(0).to(device)
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            vision_x=image_tensor,
            lang_x=input_ids,
            max_new_tokens=32,
            num_beams=3,
        )
    
    # 解码
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 提取答案（去掉prompt部分）
    if "Answer:" in outputs:
        answer = outputs.split("Answer:")[-1].strip()
    else:
        answer = outputs.strip()
    
    return answer


class LVLMVQAWrapper(torch.nn.Module):
    """LVLM VQA包装器 - 用于AutoAttack"""
    def __init__(self, lvlm_type, model, tokenizer, image_processor, normalizer, target_answer, device='cuda'):
        super().__init__()
        self.lvlm_type = lvlm_type
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.normalizer = normalizer
        self.target_answer = target_answer.lower()
        self.device = device
    
    def forward(self, images):
        """
        返回logits: [batch_size, 2]
        logits[:, 0] = 答案不匹配的分数
        logits[:, 1] = 答案匹配的分数
        """
        batch_size = images.size(0)
        logits = torch.zeros(batch_size, 2, device=images.device)
        
        for i in range(batch_size):
            # 反归一化
            img_tensor = images[i]
            img_tensor = img_tensor * self.normalizer.std.view(3, 1, 1).to(img_tensor.device) + \
                        self.normalizer.mean.view(3, 1, 1).to(img_tensor.device)
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转PIL
            img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            # 生成答案
            if self.lvlm_type == 'llava':
                generated = llava_generate_answer(
                    self.model, self.tokenizer, self.image_processor,
                    pil_image, self.question, self.device
                )
            else:
                generated = flamingo_generate_answer(
                    self.model, self.tokenizer, self.image_processor,
                    pil_image, self.question, self.device
                )
            
            # 简单匹配
            match_score = 1.0 if self.target_answer in generated.lower() else 0.0
            
            logits[i, 0] = 1.0 - match_score
            logits[i, 1] = match_score
        
        return logits
    
    def set_question(self, question):
        """设置当前问题"""
        self.question = question


def attack_vqa_sample(wrapper, image, question, answer, eps, device):
    """
    FARE三阶段攻击pipeline for VQA
    """
    wrapper.set_question(question)
    
    threshold = 0.5
    
    # 阶段1: 半精度APGD
    adversary_half = AutoAttack(
        wrapper,
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
            torch.tensor([1]).to(device),  # target: 匹配答案
            bs=1
        )
    
    # 检查
    with torch.no_grad():
        logits = wrapper(adv_image)
        score = F.softmax(logits, dim=-1)[0, 1].item()
    
    if score < threshold:
        return adv_image
    
    # 阶段2: 单精度APGD
    adversary_full = AutoAttack(
        wrapper,
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
            torch.tensor([1]).to(device),
            bs=1
        )
    
    # 检查
    with torch.no_grad():
        logits = wrapper(adv_image)
        score = F.softmax(logits, dim=-1)[0, 1].item()
    
    if score < threshold:
        return adv_image
    
    # 阶段3: Targeted攻击
    adversary_targeted = AutoAttack(
        wrapper,
        norm='Linf',
        eps=eps,
        version='custom',
        verbose=False
    )
    adversary_targeted.attacks_to_run = ['apgd-dlr']
    adversary_targeted.apgd.n_iter = 100
    
    with torch.enable_grad():
        adv_image = adversary_targeted.run_standard_evaluation(
            adv_image,
            torch.tensor([1]).to(device),
            bs=1
        )
    
    return adv_image


def evaluate_vqa_lvlm(args):
    """主评估函数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print(f"📊 LVLM VQA评估 (FARE设置)")
    print(f"   LVLM: {args.lvlm_type}")
    print(f"   数据集: {args.dataset}")
    print(f"   CLIP: {args.clip_checkpoint}")
    print(f"   Eps: {args.eps}/255")
    print("=" * 80)
    
    # 加载LVLM
    if args.lvlm_type == 'llava':
        tokenizer, model, image_processor, context_len, is_enhanced, normalizer = get_lvlm_model(
            lvlm_type='llava',
            lvlm_path=args.lvlm_path,
            clip_checkpoint=args.clip_checkpoint,
            clip_model_name=args.clip_model_name,
            device=device
        )
    else:
        model, image_processor, tokenizer, is_enhanced, normalizer = get_lvlm_model(
            lvlm_type='flamingo',
            lvlm_path=args.lvlm_path,
            clip_checkpoint=args.clip_checkpoint,
            clip_model_name=args.clip_model_name,
            device=device
        )
    
    # 加载数据集
    samples = load_vqa_dataset(args.dataset, args.dataset_root, args.max_samples)
    
    # 评估clean accuracy
    print("\n🔄 评估干净样本...")
    clean_correct = 0
    
    for sample in tqdm(samples, desc="Clean eval"):
        image = Image.open(sample['image_path']).convert('RGB')
        question = sample['question']
        gt_answer = sample['answer'].lower()
        
        if args.lvlm_type == 'llava':
            generated = llava_generate_answer(model, tokenizer, image_processor, image, question, device)
        else:
            generated = flamingo_generate_answer(model, tokenizer, image_processor, image, question, device)
        
        if gt_answer in generated.lower():
            clean_correct += 1
    
    clean_acc = clean_correct / len(samples)
    print(f"   Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    
    # 评估robust accuracy
    if args.eps > 0:
        print(f"\n🔄 评估对抗鲁棒性 (eps={args.eps/255:.10f})...")
        
        eps_normalized = args.eps / 255.0
        robust_correct = 0
        
        for sample in tqdm(samples, desc="Robust eval"):
            image = Image.open(sample['image_path']).convert('RGB')
            question = sample['question']
            gt_answer = sample['answer']
            
            # 预处理图像
            image_tensor = image_processor(image).to(device)
            
            # 归一化
            image_normalized = normalizer(image_tensor)
            
            # 创建wrapper
            wrapper = LVLMVQAWrapper(
                args.lvlm_type, model, tokenizer, image_processor,
                normalizer, gt_answer, device
            )
            
            # 攻击
            adv_image = attack_vqa_sample(
                wrapper, image_normalized, question, gt_answer,
                eps_normalized, device
            )
            
            # 反归一化
            adv_image = adv_image * normalizer.std.view(1, 3, 1, 1).to(device) + \
                       normalizer.mean.view(1, 3, 1, 1).to(device)
            adv_image = torch.clamp(adv_image, 0, 1)
            
            # 转PIL
            adv_img_np = (adv_image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            adv_pil = Image.fromarray(adv_img_np)
            
            # 生成答案
            if args.lvlm_type == 'llava':
                generated = llava_generate_answer(model, tokenizer, image_processor, adv_pil, question, device)
            else:
                generated = flamingo_generate_answer(model, tokenizer, image_processor, adv_pil, question, device)
            
            if gt_answer.lower() in generated.lower():
                robust_correct += 1
        
        robust_acc = robust_correct / len(samples)
        print(f"   Robust Accuracy: {robust_acc:.4f} ({robust_acc*100:.2f}%)")
    else:
        robust_acc = None
    
    # 保存结果
    results = {
        'lvlm_type': args.lvlm_type,
        'lvlm_path': args.lvlm_path,
        'clip_checkpoint': args.clip_checkpoint,
        'dataset': args.dataset,
        'eps': args.eps,
        'max_samples': args.max_samples,
        'clean_acc': clean_acc,
        'robust_acc': robust_acc,
        'is_enhanced': is_enhanced
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.lvlm_type}_{args.dataset}_eps{args.eps}_results.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='LVLM VQA评估 (FARE设置)')
    
    # LVLM配置
    parser.add_argument('--lvlm_type', type=str, required=True,
                       choices=['llava', 'flamingo'],
                       help='LVLM类型')
    parser.add_argument('--lvlm_path', type=str, required=True,
                       help='LVLM模型路径')
    
    # CLIP配置
    parser.add_argument('--clip_checkpoint', type=str, required=True,
                       help='鲁棒CLIP checkpoint路径')
    parser.add_argument('--clip_model_name', type=str, default='ViT-L-14',
                       help='CLIP架构')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['vqav2', 'textvqa'],
                       help='VQA数据集')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大样本数')
    
    # 攻击配置
    parser.add_argument('--eps', type=int, default=4,
                       help='扰动强度 (2 or 4 for 2/255 or 4/255)')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/lvlm_vqa')
    
    args = parser.parse_args()
    
    evaluate_vqa_lvlm(args)


if __name__ == '__main__':
    main()
