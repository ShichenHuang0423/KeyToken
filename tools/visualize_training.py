#!/usr/bin/env python3
"""
训练日志可视化工具
解析train.log文件，绘制Loss、MAE、CleanAcc、RobustAcc、FeatDiff折线图
支持多个训练日志文件对比，以及原始CLIP和FARE模型基准对比

基准数据来源说明：
1. OpenAI CLIP (Baseline):
   - 使用OpenAI预训练的ViT-L-14 CLIP模型
   - 在ImageNet验证集上测试得到CleanAcc和RobustAcc
   
2. FARE模型 (eps=2 和 eps=4):
   - FARE: Feature-Aware Robust CLIP模型
   - eps=2: 使用L2范数约束（eps=2）进行对抗训练的FARE模型
   - eps=4: 使用L-infinity范数约束（eps=4）进行对抗训练的FARE模型
   - 模型权重路径: models/fare_eps_2.pt, models/fare_eps_4.pt

如何获取基准数据：
  需要运行评估脚本来获取准确的基准性能：
  
  # 评估OpenAI CLIP
  python -m train.adversarial_training_clip \
      --clip_model_name ViT-L-14 --pretrained openai \
      --dataset imagenet --attack pgd --norm linf --eps 4.0 \
      --iterations_adv 10 --stepsize_adv 1.0 \
      --eval_only --batch_size 128
  
  # 评估FARE eps=4
  python -m train.adversarial_training_clip \
      --clip_model_name ViT-L-14 --pretrained models/fare_eps_4.pt \
      --dataset imagenet --attack pgd --norm linf --eps 4.0 \
      --iterations_adv 10 --stepsize_adv 1.0 \
      --eval_only --batch_size 128

注意：
- 基准模型显示为星形标记点（不是线），因为它们是预训练模型的固定性能
- 训练模型显示为折线图，展示训练过程中指标的变化趋势
- 所有关键数值都会自动标注在图表上
- 请通过命令行参数提供真实的评估数据，不要使用默认占位符
"""

import re
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def parse_log_file(log_path: str) -> Dict[str, List[float]]:
    """
    解析训练日志文件
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        包含各指标数据的字典，格式: {'step': [...], 'loss': [...], 'mae': [...], ...}
    """
    data = {
        'step': [],
        'loss': [],
        'contrastive': [],
        'l2': [],
        'mae': [],
        'detect': [],
        'clean_acc': [],
        'robust_acc': [],
        'feat_diff': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个步骤块
    # 匹配格式: [x.x%] Step N/M ... Loss: ... CleanAcc: ... RobustAcc: ... FeatDiff: ...
    pattern = r'\[(\d+\.\d+)%\]\s+Step\s+(\d+)/(\d+).*?Loss:\s+([\d.]+).*?Contrastive:\s+([\d.]+).*?L2:\s+([\d.]+).*?MAE:\s+([\d.]+).*?Detect:\s+([\d.]+).*?CleanAcc:\s+([\d.]+).*?RobustAcc:\s+([\d.]+).*?FeatDiff:\s+([\d.]+)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            step = int(match.group(2))
            loss = float(match.group(4))
            contrastive = float(match.group(5))
            l2 = float(match.group(6))
            mae = float(match.group(7))
            detect = float(match.group(8))
            clean_acc = float(match.group(9))
            robust_acc = float(match.group(10))
            feat_diff = float(match.group(11))
            
            data['step'].append(step)
            data['loss'].append(loss)
            data['contrastive'].append(contrastive)
            data['l2'].append(l2)
            data['mae'].append(mae)
            data['detect'].append(detect)
            data['clean_acc'].append(clean_acc)
            data['robust_acc'].append(robust_acc)
            data['feat_diff'].append(feat_diff)
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse step {match.group(2)}: {e}")
            continue
    
    return data


def plot_training_curves(
    training_logs: Dict[str, Dict[str, List[float]]],
    baseline_models: Dict[str, Dict[str, float]] = None,
    output_dir: str = None
):
    """
    绘制训练曲线
    
    Args:
        training_logs: 训练日志数据，格式: {模型名称: 数据字典}
        baseline_models: 基准模型数据，格式: {模型名称: {'clean_acc': x, 'robust_acc': y}}
        output_dir: 输出目录路径
    """
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体和图表样式
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 创建6个子图: Loss, CleanAcc, RobustAcc, FeatDiff, MAE, Loss Components
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics Comparison (KeyToken Enhanced)', fontsize=16, fontweight='bold')
    
    # 定义颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_logs) + (len(baseline_models) if baseline_models else 0)))
    
    # 绘制训练曲线
    for idx, (model_name, data) in enumerate(training_logs.items()):
        steps = data['step']
        color = colors[idx]
        
        # Loss
        if not all(np.isnan(data['loss'])):
            axes[0, 0].plot(steps, data['loss'], label=model_name, color=color, linewidth=2, marker='o', markersize=3)
            # 标注最后一个点的数值
            if steps:
                last_val = data['loss'][-1]
                if not np.isnan(last_val):
                    axes[0, 0].annotate(f"{last_val:.2f}", 
                                       xy=(steps[-1], last_val),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, color=color, fontweight='bold')
        
        # FeatDiff (移到[0,1]位置)
        if not all(np.isnan(data['feat_diff'])):
            axes[0, 1].plot(steps, data['feat_diff'], label=model_name, color=color, linewidth=2, marker='o', markersize=3)
            # 标注最后一个点的数值
            if steps:
                last_val = data['feat_diff'][-1]
                if not np.isnan(last_val):
                    axes[0, 1].annotate(f"{last_val:.4f}", 
                                       xy=(steps[-1], last_val),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, color=color, fontweight='bold')
        
        # CleanAcc
        axes[0, 2].plot(steps, data['clean_acc'], label=model_name, color=color, linewidth=2, marker='o', markersize=3)
        # 标注最后一个点的数值
        if steps:
            last_val = data['clean_acc'][-1]
            axes[0, 2].annotate(f"{last_val:.4f}", 
                               xy=(steps[-1], last_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=color, fontweight='bold')
        
        # RobustAcc
        axes[1, 0].plot(steps, data['robust_acc'], label=model_name, color=color, linewidth=2, marker='o', markersize=3)
        # 标注最后一个点的数值
        if steps:
            last_val = data['robust_acc'][-1]
            axes[1, 0].annotate(f"{last_val:.4f}", 
                               xy=(steps[-1], last_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=color, fontweight='bold')
        
        # MAE (移到[1,1]位置)
        if not all(np.isnan(data['mae'])):
            axes[1, 1].plot(steps, data['mae'], label=model_name, color=color, linewidth=2, marker='o', markersize=3)
            # 标注最后一个点的数值
            if steps:
                last_val = data['mae'][-1]
                if not np.isnan(last_val):
                    axes[1, 1].annotate(f"{last_val:.4f}", 
                                       xy=(steps[-1], last_val),
                                       xytext=(5, 5), textcoords='offset points',
                                       fontsize=8, color=color, fontweight='bold')
        
        # Loss Components (新增[1,2]位置：损失分解)
        if not all(np.isnan(data['contrastive'])):
            axes[1, 2].plot(steps, data['contrastive'], label=f'{model_name} - Contrastive', 
                           color=color, linewidth=2, marker='o', markersize=2, linestyle='-')
        if not all(np.isnan(data['l2'])):
            axes[1, 2].plot(steps, data['l2'], label=f'{model_name} - L2 Robust', 
                           color=color, linewidth=2, marker='s', markersize=2, linestyle='--')
        if not all(np.isnan(data['mae'])):
            axes[1, 2].plot(steps, data['mae'], label=f'{model_name} - MAE', 
                           color=color, linewidth=2, marker='^', markersize=2, linestyle='-.')
        if not all(np.isnan(data['detect'])):
            axes[1, 2].plot(steps, data['detect'], label=f'{model_name} - Detect', 
                           color=color, linewidth=2, marker='d', markersize=2, linestyle=':')
    
    # 添加基准模型（散点，显示在x轴起点位置）
    if baseline_models:
        for idx, (model_name, metrics) in enumerate(baseline_models.items()):
            color = colors[len(training_logs) + idx]
            
            # 获取所有训练日志的step范围
            all_steps = []
            for data in training_logs.values():
                all_steps.extend(data['step'])
            if all_steps:
                # 基准点显示在第一个step位置
                baseline_x = min(all_steps)
                
                # 标注位置错开，避免重叠
                offset_y = 15 + idx * 25  # 每个基准模型错开25个点
                
                # CleanAcc基准点
                if 'clean_acc' in metrics:
                    axes[0, 2].scatter(baseline_x, metrics['clean_acc'], color=color, 
                                      s=200, marker='*', edgecolors='black', linewidth=1.5,
                                      label=f'{model_name}', zorder=10)
                    # 添加数值标注
                    axes[0, 2].annotate(f"{metrics['clean_acc']:.4f}", 
                                       xy=(baseline_x, metrics['clean_acc']),
                                       xytext=(10, offset_y), textcoords='offset points',
                                       fontsize=9, fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                
                # RobustAcc基准点
                if 'robust_acc' in metrics:
                    axes[1, 0].scatter(baseline_x, metrics['robust_acc'], color=color, 
                                      s=200, marker='*', edgecolors='black', linewidth=1.5,
                                      label=f'{model_name}', zorder=10)
                    # 添加数值标注
                    axes[1, 0].annotate(f"{metrics['robust_acc']:.4f}", 
                                       xy=(baseline_x, metrics['robust_acc']),
                                       xytext=(10, offset_y), textcoords='offset points',
                                       fontsize=9, fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # 设置子图标题和标签
    axes[0, 0].set_title('Total Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Feature Difference (FeatDiff)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('FeatDiff')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Clean Accuracy', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend(loc='best')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.0])
    
    axes[1, 0].set_title('Robust Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.0])  # 设置为[0, 1.0]以显示鲁棒模型的高RobustAcc
    
    axes[1, 1].set_title('MAE Reconstruction Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 损失分解子图（右下角）
    axes[1, 2].set_title('Loss Components Breakdown', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Loss Value')
    axes[1, 2].legend(loc='best', fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='训练日志可视化工具',
        epilog='示例: python3 tools/visualize_training.py --log_dirs output/stage1_freeze_all '
               '--baseline_clip_clean_acc 0.726 --baseline_clip_robust_acc 0.0 '
               '--fare_eps4_clean_acc 0.692 --fare_eps4_robust_acc 0.005')
    parser.add_argument('--log_dirs', nargs='+', required=True,
                       help='训练日志目录列表，例如: output/stage1_freeze_all output/stage2_unfreeze_6')
    parser.add_argument('--baseline_clip_clean_acc', type=float, default=None,
                       help='原始CLIP的CleanAcc (必须提供真实评估数据)')
    parser.add_argument('--baseline_clip_robust_acc', type=float, default=None,
                       help='原始CLIP的RobustAcc (必须提供真实评估数据)')
    parser.add_argument('--fare_eps4_clean_acc', type=float, default=None,
                       help='FARE eps=4模型的CleanAcc (必须提供真实评估数据)')
    parser.add_argument('--fare_eps4_robust_acc', type=float, default=None,
                       help='FARE eps=4模型的RobustAcc (必须提供真实评估数据)')
    parser.add_argument('--fare_eps2_clean_acc', type=float, default=None,
                       help='FARE eps=2模型的CleanAcc (必须提供真实评估数据)')
    parser.add_argument('--fare_eps2_robust_acc', type=float, default=None,
                       help='FARE eps=2模型的RobustAcc (必须提供真实评估数据)')
    parser.add_argument('--output_dir', default='output/visualizations',
                       help='输出目录 (默认: output/visualizations)')
    
    args = parser.parse_args()
    
    # 解析所有训练日志
    training_logs = {}
    for log_dir in args.log_dirs:
        log_path = os.path.join(log_dir, 'train.log')
        if os.path.exists(log_path):
            model_name = os.path.basename(log_dir)
            print(f"📊 解析日志: {log_path}")
            data = parse_log_file(log_path)
            if data['step']:
                training_logs[model_name] = data
                print(f"   ✓ 成功提取 {len(data['step'])} 个数据点")
            else:
                print(f"   ⚠️  警告: 未找到有效数据")
        else:
            print(f"⚠️  警告: 日志文件不存在: {log_path}")
    
    if not training_logs:
        print("❌ 错误: 未找到任何有效的训练日志")
        return
    
    # 定义基准模型（只在提供了数据时才添加）
    baseline_models = {}
    
    # OpenAI CLIP
    if args.baseline_clip_clean_acc is not None or args.baseline_clip_robust_acc is not None:
        baseline_models['OpenAI CLIP (Baseline)'] = {}
        if args.baseline_clip_clean_acc is not None:
            baseline_models['OpenAI CLIP (Baseline)']['clean_acc'] = args.baseline_clip_clean_acc
        if args.baseline_clip_robust_acc is not None:
            baseline_models['OpenAI CLIP (Baseline)']['robust_acc'] = args.baseline_clip_robust_acc
    
    # FARE eps=4
    if args.fare_eps4_clean_acc is not None or args.fare_eps4_robust_acc is not None:
        baseline_models['FARE (eps=4)'] = {}
        if args.fare_eps4_clean_acc is not None:
            baseline_models['FARE (eps=4)']['clean_acc'] = args.fare_eps4_clean_acc
        if args.fare_eps4_robust_acc is not None:
            baseline_models['FARE (eps=4)']['robust_acc'] = args.fare_eps4_robust_acc
    
    # FARE eps=2
    if args.fare_eps2_clean_acc is not None or args.fare_eps2_robust_acc is not None:
        baseline_models['FARE (eps=2)'] = {}
        if args.fare_eps2_clean_acc is not None:
            baseline_models['FARE (eps=2)']['clean_acc'] = args.fare_eps2_clean_acc
        if args.fare_eps2_robust_acc is not None:
            baseline_models['FARE (eps=2)']['robust_acc'] = args.fare_eps2_robust_acc
    
    print(f"\n📈 绘制训练曲线...")
    print(f"   训练模型: {', '.join(training_logs.keys())}")
    print(f"   基准模型: {', '.join(baseline_models.keys())}")
    
    # 绘制图表
    plot_training_curves(training_logs, baseline_models, args.output_dir)


if __name__ == '__main__':
    main()
