#!/usr/bin/env python3
"""
完整LVLM Caption评估 - 使用LLaVA和OpenFlamingo
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
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# LLaVA imports
try:
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
except ImportError:
    print("警告: LLaVA未安装")

# 导入LVLM工具
from lvlm_utils import get_lvlm_model


def load_caption_dataset(dataset_name: str, dataset_root: str, max_samples: int = -1):
    """加载Caption数据集"""
    if dataset_name == 'coco':
        ann_file = os.path.join(dataset_root, 'annotations/captions_val2014.json')
        images_dir = os.path.join(dataset_root, 'val2014')
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        
        samples = []
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
            
            # 获取参考captions
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            
            samples.append({
                'image_id': img_id,
                'image_path': img_path,
                'captions': captions
            })
    
    elif dataset_name == 'flickr30k':
        ann_file = os.path.join(dataset_root, 'results_20130124.token')
        images_dir = os.path.join(dataset_root, 'flickr30k_images')
        
        # 解析Flickr30k annotations
        image_captions = {}
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_name = parts[0].split('#')[0]
                caption = parts[1]
                
                if img_name not in image_captions:
                    image_captions[img_name] = []
                image_captions[img_name].append(caption)
        
        samples = []
        for img_name, captions in image_captions.items():
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                continue
            
            samples.append({
                'image_id': img_name,
                'image_path': img_path,
                'captions': captions
            })
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 随机采样
    if max_samples > 0 and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)
    
    print(f"✓ 加载 {len(samples)} 个Caption样本")
    return samples


def llava_generate_caption(model, tokenizer, image_processor, image, device='cuda'):
    """使用LLaVA生成caption"""
    disable_torch_init()
    
    # 准备conversation
    conv = conv_templates["llava_v1"].copy()
    
    # Caption prompt
    question = "Provide a detailed description of the image."
    
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
    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return caption


def flamingo_generate_caption(model, tokenizer, image_processor, image, device='cuda'):
    """使用OpenFlamingo生成caption"""
    # OpenFlamingo zero-shot caption prompt
    prompt = "<image>Output: A caption of this image:"
    
    # 处理图像
    image_tensor = image_processor(image).unsqueeze(0).to(device)
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # 生成
    with torch.inference_mode():
        output_ids = model.generate(
            vision_x=image_tensor,
            lang_x=input_ids,
            max_new_tokens=64,
            num_beams=3,
        )
    
    # 解码
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 提取caption（去掉prompt部分）
    if ":" in caption:
        caption = caption.split(":")[-1].strip()
    
    return caption


class LVLMCaptionWrapper(torch.nn.Module):
    """LVLM Caption包装器 - 用于AutoAttack"""
    def __init__(self, lvlm_type, model, tokenizer, image_processor, normalizer, 
                 reference_captions, device='cuda'):
        super().__init__()
        self.lvlm_type = lvlm_type
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.normalizer = normalizer
        self.reference_captions = reference_captions
        self.device = device
        
        # 使用CLIP计算参考caption embeddings（简化）
        # 在实际实现中应该使用CIDEr等指标
    
    def forward(self, images):
        """
        返回负的相似度分数（越高越好的攻击）
        """
        batch_size = images.size(0)
        scores = []
        
        for i in range(batch_size):
            # 反归一化
            img_tensor = images[i]
            img_tensor = img_tensor * self.normalizer.std.view(3, 1, 1).to(img_tensor.device) + \
                        self.normalizer.mean.view(3, 1, 1).to(img_tensor.device)
            img_tensor = torch.clamp(img_tensor, 0, 1)
            
            # 转PIL
            img_np = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            # 生成caption
            if self.lvlm_type == 'llava':
                generated = llava_generate_caption(
                    self.model, self.tokenizer, self.image_processor,
                    pil_image, self.device
                )
            else:
                generated = flamingo_generate_caption(
                    self.model, self.tokenizer, self.image_processor,
                    pil_image, self.device
                )
            
            # 计算与参考captions的相似度（简化：关键词overlap）
            gen_words = set(generated.lower().split())
            max_overlap = 0
            for ref_cap in self.reference_captions:
                ref_words = set(ref_cap.lower().split())
                overlap = len(gen_words & ref_words) / max(len(gen_words), 1)
                max_overlap = max(max_overlap, overlap)
            
            # 返回负分数（攻击目标是最小化overlap）
            scores.append(-max_overlap)
        
        return torch.tensor(scores, device=images.device).unsqueeze(1)


def attack_caption_sample(wrapper, image, eps, device):
    """
    FARE两阶段攻击pipeline for Caption
    """
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
            torch.tensor([0]).to(device),  # dummy label
            bs=1
        )
    
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
            torch.tensor([0]).to(device),
            bs=1
        )
    
    return adv_image


def compute_cider_score(generated_captions, reference_captions_dict):
    """计算CIDEr分数"""
    try:
        # 创建临时文件用于COCOEvalCap
        import tempfile
        
        # 准备results格式
        results = []
        for img_id, caption in generated_captions.items():
            results.append({
                'image_id': img_id,
                'caption': caption
            })
        
        # 准备annotations格式
        annotations = []
        images = []
        ann_id = 0
        for img_id, captions in reference_captions_dict.items():
            images.append({'id': img_id})
            for cap in captions:
                annotations.append({
                    'image_id': img_id,
                    'id': ann_id,
                    'caption': cap
                })
                ann_id += 1
        
        # 创建COCO格式
        coco_format = {
            'images': images,
            'annotations': annotations
        }
        
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_format, f)
            ann_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            res_file = f.name
        
        # 计算CIDEr
        coco = COCO(ann_file)
        coco_result = coco.loadRes(res_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()
        
        # 清理临时文件
        os.remove(ann_file)
        os.remove(res_file)
        
        return coco_eval.eval['CIDEr']
    
    except Exception as e:
        print(f"警告: CIDEr计算失败: {e}")
        return 0.0


def evaluate_caption_lvlm(args):
    """主评估函数"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print(f"📊 LVLM Caption评估 (FARE设置)")
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
    samples = load_caption_dataset(args.dataset, args.dataset_root, args.max_samples)
    
    # 评估clean captions
    print("\n🔄 评估干净样本...")
    clean_captions = {}
    reference_captions = {}
    
    for sample in tqdm(samples, desc="Clean eval"):
        image = Image.open(sample['image_path']).convert('RGB')
        img_id = sample['image_id']
        
        if args.lvlm_type == 'llava':
            caption = llava_generate_caption(model, tokenizer, image_processor, image, device)
        else:
            caption = flamingo_generate_caption(model, tokenizer, image_processor, image, device)
        
        clean_captions[img_id] = caption
        reference_captions[img_id] = sample['captions']
    
    clean_cider = compute_cider_score(clean_captions, reference_captions)
    print(f"   Clean CIDEr: {clean_cider:.4f}")
    
    # 评估robust captions
    if args.eps > 0:
        print(f"\n🔄 评估对抗鲁棒性 (eps={args.eps/255:.10f})...")
        
        eps_normalized = args.eps / 255.0
        robust_captions = {}
        
        for sample in tqdm(samples, desc="Robust eval"):
            image = Image.open(sample['image_path']).convert('RGB')
            img_id = sample['image_id']
            
            # 预处理图像
            image_tensor = image_processor(image).to(device)
            
            # 归一化
            image_normalized = normalizer(image_tensor)
            
            # 创建wrapper
            wrapper = LVLMCaptionWrapper(
                args.lvlm_type, model, tokenizer, image_processor,
                normalizer, sample['captions'], device
            )
            
            # 攻击
            adv_image = attack_caption_sample(wrapper, image_normalized, eps_normalized, device)
            
            # 反归一化
            adv_image = adv_image * normalizer.std.view(1, 3, 1, 1).to(device) + \
                       normalizer.mean.view(1, 3, 1, 1).to(device)
            adv_image = torch.clamp(adv_image, 0, 1)
            
            # 转PIL
            adv_img_np = (adv_image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            adv_pil = Image.fromarray(adv_img_np)
            
            # 生成caption
            if args.lvlm_type == 'llava':
                caption = llava_generate_caption(model, tokenizer, image_processor, adv_pil, device)
            else:
                caption = flamingo_generate_caption(model, tokenizer, image_processor, adv_pil, device)
            
            robust_captions[img_id] = caption
        
        robust_cider = compute_cider_score(robust_captions, reference_captions)
        print(f"   Robust CIDEr: {robust_cider:.4f}")
        
        cider_drop = (clean_cider - robust_cider) / clean_cider * 100 if clean_cider > 0 else 0
        print(f"   CIDEr Drop: {cider_drop:.2f}%")
    else:
        robust_cider = None
        cider_drop = None
    
    # 保存结果
    results = {
        'lvlm_type': args.lvlm_type,
        'lvlm_path': args.lvlm_path,
        'clip_checkpoint': args.clip_checkpoint,
        'dataset': args.dataset,
        'eps': args.eps,
        'max_samples': args.max_samples,
        'clean_cider': clean_cider,
        'robust_cider': robust_cider,
        'cider_drop': cider_drop,
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
    parser = argparse.ArgumentParser(description='LVLM Caption评估 (FARE设置)')
    
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
                       choices=['coco', 'flickr30k'],
                       help='Caption数据集')
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大样本数')
    
    # 攻击配置
    parser.add_argument('--eps', type=int, default=4,
                       help='扰动强度 (2 or 4 for 2/255 or 4/255)')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='output/lvlm_caption')
    
    args = parser.parse_args()
    
    evaluate_caption_lvlm(args)


if __name__ == '__main__':
    main()
