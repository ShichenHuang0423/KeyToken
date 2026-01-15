#!/bin/bash
# ============================================================================
# KeyToken融合Loss训练脚本
# 使用分类损失(CE) + 鲁棒性损失(L2) + MAE重建损失
# ============================================================================

set -e

echo "=============================================="
echo "🎯 KeyToken融合Loss训练"
echo "=============================================="
echo ""

# ============================================================================
# 配置参数
# ============================================================================

# 数据路径
IMAGENET_ROOT=~/data/KeyToken/datasets/imagenet

# 预训练模型（从FARE eps=4开始）
PRETRAINED_MODEL=models/fare_eps_4.pt

# 训练参数
STEPS=10000          # 总步数（先用较少步数验证）
WARMUP=1000          # warmup步数
BATCH_SIZE=64        # 单卡batch size
LR=1e-5              # 学习率

# KeyToken Loss权重
CLS_WEIGHT=1.0       # 分类损失权重（最重要）
ROBUST_WEIGHT=0.5    # 鲁棒性L2损失权重
MAE_WEIGHT=1.0       # MAE重建损失权重
DETECT_WEIGHT=0.1    # 扰动检测损失权重

# 对抗攻击参数
EPS=4                # 扰动强度 (4/255)
ATTACK_ITERS=10      # PGD迭代次数

# 显存优化
USE_AMP=True
GRADIENT_ACCUMULATION=2
MEMORY_EFFICIENT=True

# I/O优化
NUM_WORKERS=4
PREFETCH_FACTOR=2

# 冻结设置（可选：冻结CLIP backbone只训练新模块）
FREEZE_BACKBONE=False
FREEZE_LAYERS=0

# 实验名称
EXPERIMENT_NAME="keytoken_loss_phase1"

# ============================================================================
# 检查环境
# ============================================================================

echo "检查环境..."

# 检查数据集
if [ ! -d "$IMAGENET_ROOT" ]; then
    echo "❌ 错误: ImageNet数据集不存在: $IMAGENET_ROOT"
    exit 1
fi
echo "✓ ImageNet数据集: $IMAGENET_ROOT"

# 检查预训练模型
if [ ! -f "$PRETRAINED_MODEL" ] && [ "$PRETRAINED_MODEL" != "openai" ]; then
    echo "⚠️  警告: 预训练模型不存在: $PRETRAINED_MODEL"
    echo "   将使用OpenAI CLIP作为起点"
    PRETRAINED_MODEL="openai"
fi
echo "✓ 预训练模型: $PRETRAINED_MODEL"

# 检查GPU
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "✓ 可用GPU数量: $GPU_COUNT"

# 创建输出目录
OUTPUT_DIR="output/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "✓ 输出目录: $OUTPUT_DIR"

# ============================================================================
# 显示配置
# ============================================================================

echo ""
echo "=============================================="
echo "训练配置"
echo "=============================================="
echo "  实验名称: $EXPERIMENT_NAME"
echo "  预训练模型: $PRETRAINED_MODEL"
echo "  总步数: $STEPS"
echo "  Batch Size: $BATCH_SIZE × $GRADIENT_ACCUMULATION = $(($BATCH_SIZE * $GRADIENT_ACCUMULATION))"
echo "  学习率: $LR"
echo ""
echo "KeyToken Loss权重:"
echo "  分类损失(CE): $CLS_WEIGHT"
echo "  鲁棒性损失(L2): $ROBUST_WEIGHT"
echo "  MAE重建损失: $MAE_WEIGHT"
echo "  扰动检测损失: $DETECT_WEIGHT"
echo ""
echo "对抗攻击:"
echo "  扰动强度: $EPS/255"
echo "  攻击迭代: $ATTACK_ITERS"
echo "=============================================="
echo ""

# ============================================================================
# 开始训练
# ============================================================================

echo "开始训练 (后台运行)..."
echo "日志输出: $OUTPUT_DIR/train.log"
echo ""

# 使用nohup后台运行
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained $PRETRAINED_MODEL \
    --dataset imagenet \
    --imagenet_root $IMAGENET_ROOT \
    --steps $STEPS \
    --warmup $WARMUP \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --wd 1e-4 \
    --opt adamw \
    --attack pgd \
    --inner_loss l2 \
    --norm linf \
    --eps $EPS \
    --iterations_adv $ATTACK_ITERS \
    --stepsize_adv 1 \
    --use_keytoken_loss True \
    --cls_weight $CLS_WEIGHT \
    --robust_weight $ROBUST_WEIGHT \
    --mae_weight $MAE_WEIGHT \
    --detect_weight $DETECT_WEIGHT \
    --use_mae_recon True \
    --use_key_token_protection True \
    --key_token_ratio 0.2 \
    --mask_ratio 0.5 \
    --adaptive_masking False \
    --freeze_clip_backbone $FREEZE_BACKBONE \
    --freeze_encoder_layers $FREEZE_LAYERS \
    --use_amp $USE_AMP \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --memory_efficient_mode $MEMORY_EFFICIENT \
    --num_workers $NUM_WORKERS \
    --prefetch_factor $PREFETCH_FACTOR \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_dir $OUTPUT_DIR \
    --overwrite False \
    --wandb False \
    --log_freq 10 \
    --eval_freq 10 \
    --save_checkpoints True \
    --checkpoint_freq 1000 \
    > "$OUTPUT_DIR/train.log" 2>&1 &

TRAIN_PID=$!
echo "✓ 训练进程已启动 (PID: $TRAIN_PID)"
echo ""

# 等待几秒检查是否启动成功
sleep 5
if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ 训练正在运行..."
    echo ""
    echo "=============================================="
    echo "监控命令:"
    echo "  查看日志: tail -f $OUTPUT_DIR/train.log"
    echo "  查看GPU: watch -n 1 nvidia-smi"
    echo "  停止训练: kill $TRAIN_PID"
    echo "=============================================="
else
    echo "❌ 训练启动失败，查看日志: $OUTPUT_DIR/train.log"
    tail -20 "$OUTPUT_DIR/train.log"
    exit 1
fi
