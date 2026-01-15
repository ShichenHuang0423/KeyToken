#!/bin/bash
# 停止所有训练进程的脚本

echo "🔍 查找训练进程..."
echo ""

# 查找所有adversarial_training_clip_enhanced相关进程
PIDS=$(ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "✅ 没有发现运行中的训练进程"
    exit 0
fi

echo "发现以下训练进程："
echo "----------------------------------------"
ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep | awk '{printf "PID: %s | CPU: %s%% | MEM: %s%% | TIME: %s\n", $2, $3, $4, $10}'
echo "----------------------------------------"
echo ""

# 统计进程数量
COUNT=$(echo "$PIDS" | wc -w)
echo "总共 $COUNT 个进程"
echo ""

# 询问确认
read -p "确认停止所有训练进程? (y/n): " confirm

if [[ "$confirm" != "y" ]]; then
    echo "❌ 已取消"
    exit 0
fi

echo ""
echo "⏸️  正在停止训练进程..."

# 停止所有进程
for pid in $PIDS; do
    echo "  停止 PID: $pid"
    kill $pid
done

echo ""
echo "⏳ 等待进程退出..."
sleep 2

# 检查是否还有残留进程
REMAINING=$(ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep | wc -l)

if [ $REMAINING -gt 0 ]; then
    echo "⚠️  仍有 $REMAINING 个进程未退出，使用 kill -9 强制停止..."
    PIDS=$(ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep | awk '{print $2}')
    for pid in $PIDS; do
        echo "  强制停止 PID: $pid"
        kill -9 $pid
    done
    sleep 1
fi

# 最终检查
FINAL=$(ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep | wc -l)

if [ $FINAL -eq 0 ]; then
    echo ""
    echo "✅ 所有训练进程已成功停止！"
else
    echo ""
    echo "❌ 仍有进程未停止，请手动检查："
    ps aux | grep "adversarial_training_clip_enhanced" | grep -v grep
fi
