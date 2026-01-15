#!/bin/bash
# ============================================================================
# 后台运行多任务评估的nohup脚本
# ============================================================================

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit

# 创建输出目录
OUTPUT_DIR="output/multi_task_eval"
mkdir -p "$OUTPUT_DIR"

# 激活conda环境并运行评估
nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate keytoken && bash bash/evaluate_all_tasks.sh" > "$OUTPUT_DIR/nohup.out" 2>&1 &

# 获取进程ID
PID=$!

echo "=========================================="
echo "🚀 多任务评估已启动"
echo "   进程ID: $PID"
echo "   日志文件: $OUTPUT_DIR/nohup.out"
echo "=========================================="
echo ""
echo "📝 监控命令:"
echo "   查看日志: tail -f $OUTPUT_DIR/nohup.out"
echo "   查看进程: ps aux | grep evaluate_all_tasks"
echo "   停止评估: kill $PID"
echo ""
