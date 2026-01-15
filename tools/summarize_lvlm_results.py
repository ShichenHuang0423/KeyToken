#!/usr/bin/env python3
"""
LVLM评估结果汇总脚本
"""

import os
import argparse
import json
import glob
from datetime import datetime
from collections import defaultdict


def load_lvlm_results(input_dir):
    """加载所有LVLM评估结果"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 遍历VQA和Caption子目录
    for task_dir in ['vqa', 'caption']:
        task_path = os.path.join(input_dir, task_dir)
        if not os.path.exists(task_path):
            continue
        
        # 查找所有结果文件
        result_files = glob.glob(os.path.join(task_path, '*_results.json'))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # 提取信息
            clip_name = os.path.basename(data['clip_checkpoint']).replace('.pt', '')
            lvlm_type = data['lvlm_type']
            dataset = data['dataset']
            eps = data['eps']
            
            key = f"{dataset}_eps{eps}"
            
            # 存储结果
            results[clip_name][lvlm_type][task_dir][key] = data
    
    return results


def generate_markdown_report(results):
    """生成Markdown格式报告"""
    md = []
    
    md.append("# LVLM鲁棒性评估报告 (FARE设置)\n")
    md.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("---\n\n")
    
    # VQA结果
    md.append("## 视觉问答 (VQA)\n\n")
    
    for lvlm_type in ['llava', 'flamingo']:
        md.append(f"### {lvlm_type.upper()}\n\n")
        md.append("| CLIP Model | Dataset | Eps | Clean Acc | Robust Acc | Acc Drop |\n")
        md.append("|------------|---------|-----|-----------|------------|----------|\n")
        
        for clip_name in sorted(results.keys()):
            if lvlm_type in results[clip_name]:
                if 'vqa' in results[clip_name][lvlm_type]:
                    for key, data in sorted(results[clip_name][lvlm_type]['vqa'].items()):
                        dataset = data['dataset']
                        eps = data['eps']
                        clean_acc = data.get('clean_acc', 0) * 100
                        robust_acc = data.get('robust_acc', 0) * 100
                        drop = clean_acc - robust_acc
                        
                        md.append(f"| {clip_name} | {dataset} | {eps}/255 | "
                                 f"{clean_acc:.2f}% | {robust_acc:.2f}% | {drop:.2f}% |\n")
        
        md.append("\n")
    
    # Caption结果
    md.append("## 图像描述 (Caption)\n\n")
    
    for lvlm_type in ['llava', 'flamingo']:
        md.append(f"### {lvlm_type.upper()}\n\n")
        md.append("| CLIP Model | Dataset | Eps | Clean CIDEr | Robust CIDEr | CIDEr Drop |\n")
        md.append("|------------|---------|-----|-------------|--------------|------------|\n")
        
        for clip_name in sorted(results.keys()):
            if lvlm_type in results[clip_name]:
                if 'caption' in results[clip_name][lvlm_type]:
                    for key, data in sorted(results[clip_name][lvlm_type]['caption'].items()):
                        dataset = data['dataset']
                        eps = data['eps']
                        clean_cider = data.get('clean_cider', 0)
                        robust_cider = data.get('robust_cider', 0)
                        drop_pct = (clean_cider - robust_cider) / clean_cider * 100 if clean_cider > 0 else 0
                        
                        md.append(f"| {clip_name} | {dataset} | {eps}/255 | "
                                 f"{clean_cider:.4f} | {robust_cider:.4f} | {drop_pct:.2f}% |\n")
        
        md.append("\n")
    
    # 对比总结
    md.append("## 对比总结\n\n")
    md.append("### 平均性能\n\n")
    
    # 计算平均性能
    avg_stats = defaultdict(lambda: defaultdict(lambda: {'clean': [], 'robust': []}))
    
    for clip_name in results.keys():
        for lvlm_type in results[clip_name].keys():
            # VQA平均
            if 'vqa' in results[clip_name][lvlm_type]:
                for data in results[clip_name][lvlm_type]['vqa'].values():
                    avg_stats[clip_name][f"{lvlm_type}_vqa"]['clean'].append(data.get('clean_acc', 0))
                    avg_stats[clip_name][f"{lvlm_type}_vqa"]['robust'].append(data.get('robust_acc', 0))
            
            # Caption平均
            if 'caption' in results[clip_name][lvlm_type]:
                for data in results[clip_name][lvlm_type]['caption'].values():
                    avg_stats[clip_name][f"{lvlm_type}_caption"]['clean'].append(data.get('clean_cider', 0))
                    avg_stats[clip_name][f"{lvlm_type}_caption"]['robust'].append(data.get('robust_cider', 0))
    
    md.append("| CLIP Model | Task | Clean | Robust | Drop |\n")
    md.append("|------------|------|-------|--------|------|\n")
    
    for clip_name in sorted(avg_stats.keys()):
        for task_key in sorted(avg_stats[clip_name].keys()):
            stats = avg_stats[clip_name][task_key]
            avg_clean = sum(stats['clean']) / len(stats['clean']) if stats['clean'] else 0
            avg_robust = sum(stats['robust']) / len(stats['robust']) if stats['robust'] else 0
            
            if 'vqa' in task_key:
                avg_clean *= 100
                avg_robust *= 100
                drop = avg_clean - avg_robust
                md.append(f"| {clip_name} | {task_key} | {avg_clean:.2f}% | {avg_robust:.2f}% | {drop:.2f}% |\n")
            else:
                drop_pct = (avg_clean - avg_robust) / avg_clean * 100 if avg_clean > 0 else 0
                md.append(f"| {clip_name} | {task_key} | {avg_clean:.4f} | {avg_robust:.4f} | {drop_pct:.2f}% |\n")
    
    md.append("\n")
    
    # 添加说明
    md.append("---\n\n")
    md.append("## 评估说明\n\n")
    md.append("- **VQA任务**: 使用FARE三阶段攻击pipeline (半精度APGD → 单精度APGD → Targeted)\n")
    md.append("- **Caption任务**: 使用FARE两阶段攻击pipeline (半精度APGD → 单精度APGD)\n")
    md.append("- **评估样本**: VQA和Caption各500个随机样本\n")
    md.append("- **LVLM模型**: LLaVA-1.5 7B, OpenFlamingo 9B\n")
    md.append("- **攻击类型**: 灰盒攻击（仅攻击CLIP vision encoder）\n")
    
    return ''.join(md)


def main():
    parser = argparse.ArgumentParser(description='汇总LVLM评估结果')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='评估结果目录')
    parser.add_argument('--output_file', type=str, default='lvlm_summary_report.json',
                       help='输出JSON文件')
    
    args = parser.parse_args()
    
    print("🔄 加载LVLM评估结果...")
    results = load_lvlm_results(args.input_dir)
    
    if not results:
        print("❌ 未找到评估结果")
        return
    
    print(f"   ✓ 找到 {len(results)} 个CLIP模型的结果")
    
    # 保存JSON
    output_data = {
        'results': dict(results),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"   ✓ 已保存: {args.output_file}")
    
    # 生成Markdown报告
    md_file = args.output_file.replace('.json', '.md')
    md_content = generate_markdown_report(results)
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"   ✓ 已保存: {md_file}")
    
    print("\n✅ 汇总完成!")


if __name__ == '__main__':
    main()
