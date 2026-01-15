#!/usr/bin/env python3
"""
结果汇总脚本 - 生成多任务评估的汇总报告
"""

import os
import argparse
import json
import glob
from datetime import datetime
from collections import defaultdict


def load_results(input_dir):
    """加载所有评估结果"""
    results = defaultdict(lambda: defaultdict(dict))
    
    # 遍历所有子目录
    for task_dir in ['zeroshot', 'vqa', 'caption', 'pope']:
        task_path = os.path.join(input_dir, task_dir)
        if not os.path.exists(task_path):
            continue
        
        # 查找所有结果文件
        result_files = glob.glob(os.path.join(task_path, '*_results.json'))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # 提取信息
            model_path = data.get('model', '')
            model_name = os.path.basename(model_path).replace('.pt', '')
            
            if 'dataset' in data:
                dataset = data['dataset']
            elif 'split' in data:
                dataset = f"pope_{data['split']}"
            else:
                continue
            
            # 存储结果
            results[model_name][task_dir][dataset] = data
    
    return results


def generate_markdown_table(results):
    """生成Markdown格式的结果表格"""
    md = []
    
    md.append("# 多任务鲁棒性评估报告\n")
    md.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append("---\n")
    
    # 零样本分类结果
    md.append("## 零样本分类 (Zero-Shot Classification)\n")
    md.append("| 模型 | 数据集 | Eps | Gray-box | Clean Acc | Robust Acc | 相对下降 |\n")
    md.append("|------|--------|-----|----------|-----------|------------|----------|\n")
    
    for model_name in sorted(results.keys()):
        if 'zeroshot' in results[model_name]:
            for dataset, data in sorted(results[model_name]['zeroshot'].items()):
                clean_acc = data.get('clean_acc', 0) * 100
                robust_acc = data.get('robust_acc', 0) * 100
                eps = data.get('eps', 0)
                gray_box = "✓" if data.get('gray_box', False) else ""
                drop = (clean_acc - robust_acc) / clean_acc * 100 if clean_acc > 0 else 0
                
                md.append(f"| {model_name} | {dataset} | {eps}/255 | {gray_box} | "
                         f"{clean_acc:.2f}% | {robust_acc:.2f}% | {drop:.1f}% |\n")
    
    # VQA结果
    md.append("\n## 视觉问答 (VQA)\n")
    md.append("| 模型 | 数据集 | Eps | Gray-box | Clean Acc | Robust Acc | 相对下降 |\n")
    md.append("|------|--------|-----|----------|-----------|------------|----------|\n")
    
    for model_name in sorted(results.keys()):
        if 'vqa' in results[model_name]:
            for dataset, data in sorted(results[model_name]['vqa'].items()):
                clean_acc = data.get('clean_acc', 0) * 100
                robust_acc = data.get('robust_acc', 0) * 100
                eps = data.get('eps', 0)
                gray_box = "✓" if data.get('gray_box', False) else ""
                drop = (clean_acc - robust_acc) / clean_acc * 100 if clean_acc > 0 else 0
                
                md.append(f"| {model_name} | {dataset} | {eps}/255 | {gray_box} | "
                         f"{clean_acc:.2f}% | {robust_acc:.2f}% | {drop:.1f}% |\n")
    
    # Caption结果
    md.append("\n## 图像描述 (Caption)\n")
    md.append("| 模型 | 数据集 | Eps | Gray-box | Clean CIDEr | Robust CIDEr | 相对下降 |\n")
    md.append("|------|--------|-----|----------|-------------|--------------|----------|\n")
    
    for model_name in sorted(results.keys()):
        if 'caption' in results[model_name]:
            for dataset, data in sorted(results[model_name]['caption'].items()):
                clean_cider = data.get('clean_cider', 0) * 100
                robust_cider = data.get('robust_cider', 0) * 100
                eps = data.get('eps', 0)
                gray_box = "✓" if data.get('gray_box', False) else ""
                drop = (clean_cider - robust_cider) / clean_cider * 100 if clean_cider > 0 else 0
                
                md.append(f"| {model_name} | {dataset} | {eps}/255 | {gray_box} | "
                         f"{clean_cider:.2f} | {robust_cider:.2f} | {drop:.1f}% |\n")
    
    # POPE结果
    md.append("\n## 幻觉评估 (POPE)\n")
    md.append("| 模型 | Split | Accuracy | Precision | Recall | F1 | Hallucination Rate |\n")
    md.append("|------|-------|----------|-----------|--------|----|-----------------|\n")
    
    for model_name in sorted(results.keys()):
        if 'pope' in results[model_name]:
            for dataset, data in sorted(results[model_name]['pope'].items()):
                metrics = data.get('metrics', {})
                accuracy = metrics.get('accuracy', 0) * 100
                precision = metrics.get('precision', 0) * 100
                recall = metrics.get('recall', 0) * 100
                f1 = metrics.get('f1', 0) * 100
                hall_rate = metrics.get('hallucination_rate', 0) * 100
                split = dataset.replace('pope_', '')
                
                md.append(f"| {model_name} | {split} | {accuracy:.2f}% | {precision:.2f}% | "
                         f"{recall:.2f}% | {f1:.2f}% | {hall_rate:.2f}% |\n")
    
    return ''.join(md)


def generate_comparison_summary(results):
    """生成对比汇总"""
    summary = {
        'models': list(results.keys()),
        'tasks': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 对比各任务的平均性能
    for task in ['zeroshot', 'vqa', 'caption', 'pope']:
        task_summary = {}
        
        for model_name in results.keys():
            if task not in results[model_name]:
                continue
            
            if task == 'pope':
                # POPE: 平均accuracy和幻觉率
                accuracies = []
                hall_rates = []
                for data in results[model_name][task].values():
                    metrics = data.get('metrics', {})
                    accuracies.append(metrics.get('accuracy', 0))
                    hall_rates.append(metrics.get('hallucination_rate', 0))
                
                task_summary[model_name] = {
                    'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                    'avg_hallucination_rate': sum(hall_rates) / len(hall_rates) if hall_rates else 0
                }
            else:
                # 其他任务: 平均clean和robust性能
                clean_scores = []
                robust_scores = []
                
                for data in results[model_name][task].values():
                    if task == 'caption':
                        clean_scores.append(data.get('clean_cider', 0))
                        robust_scores.append(data.get('robust_cider', 0))
                    else:
                        clean_scores.append(data.get('clean_acc', 0))
                        robust_scores.append(data.get('robust_acc', 0))
                
                task_summary[model_name] = {
                    'avg_clean': sum(clean_scores) / len(clean_scores) if clean_scores else 0,
                    'avg_robust': sum(robust_scores) / len(robust_scores) if robust_scores else 0
                }
        
        summary['tasks'][task] = task_summary
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='汇总多任务评估结果')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='评估结果目录')
    parser.add_argument('--output_file', type=str, default='summary_report.json',
                       help='输出JSON文件')
    
    args = parser.parse_args()
    
    print("🔄 加载评估结果...")
    results = load_results(args.input_dir)
    
    if not results:
        print("❌ 未找到评估结果")
        return
    
    print(f"   ✓ 找到 {len(results)} 个模型的结果")
    
    # 生成对比汇总
    print("🔄 生成对比汇总...")
    summary = generate_comparison_summary(results)
    
    # 保存JSON
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ 已保存: {args.output_file}")
    
    # 生成Markdown报告
    md_file = args.output_file.replace('.json', '.md')
    md_content = generate_markdown_table(results)
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"   ✓ 已保存: {md_file}")
    
    # 打印简要汇总
    print("\n" + "=" * 80)
    print("📊 评估汇总")
    print("=" * 80)
    
    for task, task_summary in summary['tasks'].items():
        print(f"\n【{task.upper()}】")
        for model_name, metrics in task_summary.items():
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ 汇总完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
