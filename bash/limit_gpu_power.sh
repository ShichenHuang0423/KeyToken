#!/bin/bash

# 限制GPU功耗，防止电源过载
# A6000: 300W → 250W (降低17%)
# 性能影响: 约5-10%，但系统稳定性大幅提升

echo "================================================"
echo "  限制GPU功耗，防止电源过载"
echo "================================================"

# 检查是否有root权限
if ! sudo -n true 2>/dev/null; then
    echo "❌ 需要sudo权限来设置GPU功耗限制"
    echo "请运行: sudo bash $0"
    exit 1
fi

# 获取GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "检测到 $GPU_COUNT 张GPU"

# A6000功耗限制 (300W → 250W)
# RTX 3090功耗限制 (350W → 280W)
POWER_LIMIT=250  # 瓦特

echo ""
echo "设置功耗限制: ${POWER_LIMIT}W/卡"
echo ""

for i in $(seq 0 $((GPU_COUNT-1))); do
    # 获取当前GPU型号
    GPU_NAME=$(nvidia-smi -i $i --query-gpu=name --format=csv,noheader)
    CURRENT_LIMIT=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits)
    
    # 根据GPU型号设置合适的功耗限制
    if [[ "$GPU_NAME" == *"A6000"* ]]; then
        POWER_LIMIT=250  # A6000: 300W → 250W
    elif [[ "$GPU_NAME" == *"3090"* ]]; then
        POWER_LIMIT=280  # RTX 3090: 350W → 280W
    elif [[ "$GPU_NAME" == *"4090"* ]]; then
        POWER_LIMIT=350  # RTX 4090: 450W → 350W
    else
        POWER_LIMIT=200  # 默认保守值
    fi
    
    echo "GPU $i ($GPU_NAME):"
    echo "  当前限制: ${CURRENT_LIMIT}W"
    echo "  设置为: ${POWER_LIMIT}W"
    
    sudo nvidia-smi -i $i -pl $POWER_LIMIT
    
    if [ $? -eq 0 ]; then
        NEW_LIMIT=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits)
        echo "  ✅ 设置成功: ${NEW_LIMIT}W"
    else
        echo "  ❌ 设置失败"
    fi
    echo ""
done

echo "================================================"
echo "  功耗限制设置完成"
echo "================================================"
echo ""
echo "预计总功耗降低:"
if [[ "$GPU_NAME" == *"A6000"* ]]; then
    echo "  原: 8 × 300W = 2400W"
    echo "  现: 8 × 250W = 2000W"
    echo "  节省: 400W (17%)"
elif [[ "$GPU_NAME" == *"3090"* ]]; then
    echo "  原: 8 × 350W = 2800W"
    echo "  现: 8 × 280W = 2240W"
    echo "  节省: 560W (20%)"
fi
echo ""
echo "注意: 功耗限制在系统重启后会重置"
echo "      如需持久化，请添加到开机脚本"
echo ""
