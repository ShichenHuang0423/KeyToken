#!/bin/bash
# ============================================================================
# 后台运行鲁棒性评估
# ============================================================================
#
# 【使用方法】
#   1. 先编辑 bash/evaluate_robust.sh 中的配置:
#      - EVAL_TASKS: 评估任务列表（任务名称|权重路径|推理模式）
#      - MAX_SAMPLES: 评估样本数（-1为全量50000）
#      - GPU_ID: 使用的GPU
#
#   2. 然后运行此脚本:
#      bash bash/run_robust_eval.sh
#
# 【推理模式】
#   - baseline: 标准CLIP推理，不使用增强模块
#   - eval: 增强模型 + 完整防御（扰动检测+Token过滤+特征融合）
#   - attack: 增强模型 + 无防御（仅backbone）
#
# ============================================================================

OUTPUT_DIR="output/robust_eval"
mkdir -p $OUTPUT_DIR

# 后台运行（在nohup中显式设置CUDA和conda环境）
nohup bash -c "export CUDA_VISIBLE_DEVICES=4,5,6,7 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate keytoken && bash bash/evaluate_robust.sh" > $OUTPUT_DIR/nohup.out 2>&1 &

PID=$!
echo "=========================================="
echo "✅ 评估任务已在后台启动"
echo "=========================================="
echo "进程ID: $PID"
echo "日志: $OUTPUT_DIR/nohup.out"
echo ""
echo "📝 注意: 评估配置在 bash/evaluate_robust.sh 中设置"
echo ""
echo "监控命令:"
echo "  实时日志: tail -f $OUTPUT_DIR/nohup.out"
echo "  查看进程: ps -p $PID"
echo "  停止任务: kill $PID"
echo "=========================================="
