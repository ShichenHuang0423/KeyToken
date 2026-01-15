#!/bin/bash

# 渐进式解冻训练脚本 - A6000优化版
# 硬件: 8x A6000 48GB
# 相比RTX 4090版本：GPU数量翻倍 + 显存翻倍 = 4倍算力提升
# ⚡ 优化策略：更大batch size + 减少梯度累积 = 更快训练速度

echo "================================================"
echo "  KeyToken 渐进式解冻训练 (A6000优化版)"
echo "  数据集: ImageNet (128万图像)"
echo "  GPU: 4x A6000 48GB"
echo "  ⚡ 优化: AMP混合精度 + 更大batch"
echo "================================================"
echo ""
echo "🆕 代码版本: v2.0 - Attack Mode (2024-12-21)"
echo "   ✅ 对抗样本生成使用attack模式（无防御）"
echo "   ✅ 训练时使用train模式（完整增强）"
echo "   ✅ 预期FeatDiff: 0.05-0.15（显著提升）"
echo "================================================"
echo ""
echo "✅ 增强模块保存功能已启用:"
echo "   - PatchDisturbDetector (扰动检测器)"
echo "   - KeyTokenSelector (关键Token选择器)"
echo "   - DualMAEDecoder (MAE重建器)"
echo "   所有模块权重将保存到checkpoint中"
echo "================================================"
echo ""
echo "请选择训练阶段："
echo "  0 - 完全解冻训练 (常规训练，不冻结，4 epochs)"
echo "  1 - 阶段1: 冻结CLIP backbone (只训练新增模块, 1 epoch)"
echo "  2 - 阶段2: 解冻后6层 (从阶段1恢复, 1.2 epochs)"
echo "  3 - 阶段3: 解冻后12层 (从阶段2恢复, 1 epoch)"
echo "  4 - 阶段4: 完全解冻微调 (从阶段3恢复, 1 epoch)"
echo ""
read -p "输入阶段编号 (0-4): " STAGE

# 验证输入
if ! [[ "$STAGE" =~ ^[0-4]$ ]]; then
    echo "❌ 无效输入！请输入0-4之间的数字"
    exit 1
fi

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keytoken

# GPU配置 - 使用4张A6000
CUDA_GPUS="0,1,2,3"

# 通用配置
IMAGENET_ROOT=~/data/KeyToken/datasets/imagenet
PRETRAINED_MODEL=models/fare_eps_4.pt  # 从FARE模型继续训练

# ⚡ 显存与I/O优化参数 (默认开启)
USE_AMP="True"              # 混合精度训练，节省~30%显存
MEMORY_EFFICIENT="True"     # 内存高效模式

# 🚨 I/O优化：A6000服务器使用SSD，可以用更多workers
NUM_WORKERS=6               # SSD可以用更多workers
PREFETCH_FACTOR=4           # SSD预读取可以更多

# 阶段配置
case $STAGE in
    0)
        STAGE_NAME="stage0_full_training"
        # ⚡ A6000 48GB配置（考虑MAE decoder显存需求）
        # 计算逻辑：
        # - 单卡batch_size=48 (为MAE decoder预留显存)
        # - 4卡总batch = 4 × 48 = 192
        # - 梯度累积=1
        # - 有效batch = 192
        # - MAE decoder约需15GB额外显存
        # 
        # Steps计算（optimizer更新步数）：
        # - 每个epoch的dataloader迭代 = 1,281,167 / 192 ≈ 6,673
        # - 每个epoch的optimizer步数 = 6,673 / 1 = 6,673
        # - 4 epochs = 6,673 × 4 = 26,692 步
        STEPS=26692
        WARMUP=5000  # ✅ 延长warmup防止初始不稳定
        BATCH_SIZE=192  # 4 GPU × 48/GPU (MAE显存优化)
        GRADIENT_ACCUMULATION=1
        LR="1e-5"  # ✅ 保持原始学习率（loss故意不除以维度，需小lr）
        FREEZE_BACKBONE="False"
        FREEZE_LAYERS=0
        RESUME=""
        SEED=42  # 🎲 Stage 0独立训练链的固定种子
        EPOCHS=4
        DESC="完全解冻训练 (4 epochs, A6000优化)"
        ;;
    1)
        STAGE_NAME="stage1_freeze_all"
        # 冻结时可以更大batch (只训练新模块，显存需求更小)
        # - 单卡batch_size=96 (冻结backbone，显存需求小)
        # - 4卡总batch = 4 × 96 = 384
        # - 梯度累积=1
        # - 每个epoch = 1,281,167 / 384 ≈ 3,337 步
        STEPS=3337
        WARMUP=300
        BATCH_SIZE=384
        GRADIENT_ACCUMULATION=1
        LR="5e-4"
        FREEZE_BACKBONE="True"
        FREEZE_LAYERS=0
        RESUME=""
        SEED=123  # 🎲 Stage 1-4渐进式训练链的起点种子（与Stage 0不同）
        EPOCHS=1
        DESC="冻结CLIP backbone (1 epoch)"
        ;;
    2)
        STAGE_NAME="stage2_unfreeze_6layers"
        # 1.2 epochs = 1.2 × 5,005 = 6,006 步
        STEPS=6006
        WARMUP=600
        BATCH_SIZE=256
        GRADIENT_ACCUMULATION=1
        LR="3e-4"
        FREEZE_BACKBONE="True"
        FREEZE_LAYERS=18
        RESUME="output/stage1_freeze_all/checkpoints/epoch_1.pt"
        SEED=""  # 🎲 从checkpoint恢复，继承Stage 1的种子
        EPOCHS="1.2"
        DESC="解冻后6层 (从Stage1续)"
        ;;
    3)
        STAGE_NAME="stage3_unfreeze_12layers"
        # 1 epoch = 5,005 步
        STEPS=5005
        WARMUP=500
        BATCH_SIZE=256
        GRADIENT_ACCUMULATION=1
        LR="1e-4"
        FREEZE_BACKBONE="True"
        FREEZE_LAYERS=12
        RESUME="output/stage2_unfreeze_6layers/checkpoints/epoch_2.pt"
        SEED=""  # 🎲 从checkpoint恢复，继承Stage 2的种子
        EPOCHS=1
        DESC="解冻后12层 (从Stage2续)"
        ;;
    4)
        STAGE_NAME="stage4_full_finetune"
        # 1 epoch = 5,005 步
        STEPS=5005
        WARMUP=500
        BATCH_SIZE=256
        GRADIENT_ACCUMULATION=1
        LR="5e-5"
        FREEZE_BACKBONE="False"
        FREEZE_LAYERS=0
        RESUME="output/stage3_unfreeze_12layers/checkpoints/epoch_3.pt"
        SEED=""  # 🎲 从checkpoint恢复，继承Stage 3的种子
        EPOCHS=1
        DESC="完全解冻微调 (从Stage3续)"
        ;;
esac

# 创建输出目录
mkdir -p output/${STAGE_NAME}/checkpoints

# 显示配置
echo ""
echo "🎯 训练配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "阶段: ${STAGE_NAME}"
echo "描述: ${DESC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU数量: 4x A6000 48GB"
echo "总Batch Size: ${BATCH_SIZE}"
echo "梯度累积: ${GRADIENT_ACCUMULATION}"
echo "有效Batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "训练步数: ${STEPS}"
echo "Warmup步数: ${WARMUP}"
echo "学习率: ${LR}"
echo "Epochs: ${EPOCHS}"
if [ ! -z "$RESUME" ]; then
    echo "恢复检查点: ${RESUME}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "确认开始训练? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "训练已取消"
    exit 0
fi

# 设置日志
LOG_FILE="output/${STAGE_NAME}/train.log"
NOHUP_LOG="output/${STAGE_NAME}/nohup.log"

echo "⚡ 训练开始: $(date)"
echo "日志文件: ${LOG_FILE}"
echo "nohup日志: ${NOHUP_LOG}"
echo ""

# 使用nohup后台运行 - 与4090脚本结构完全一致
# ⚡ 显存优化参数: use_amp, gradient_accumulation_steps, memory_efficient_mode
echo "开始训练 Stage $STAGE (后台运行)..."
echo "日志输出: ${NOHUP_LOG}"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_GPUS} nohup python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained ${PRETRAINED_MODEL} \
    --dataset imagenet \
    --imagenet_root ${IMAGENET_ROOT} \
    --steps ${STEPS} \
    --warmup ${WARMUP} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd 1e-4 \
    --opt adamw \
    --attack pgd \
    --inner_loss l2 \
    --norm linf \
    --eps 4 \
    --iterations_adv 10 \
    --stepsize_adv 1 \
    --use_keytoken_loss True \
    --contrastive_weight 1.0 \
    --contrastive_temperature 0.07 \
    --robust_weight 0.1 \
    --detect_weight 0.1 \
    --use_mae_recon True \
    --use_key_token_protection True \
    --mae_weight 1.0 \
    --text_recon_weight 0.5 \
    --key_token_ratio 0.2 \
    --mask_ratio 0.5 \
    --adaptive_masking False \
    --freeze_clip_backbone ${FREEZE_BACKBONE} \
    --freeze_encoder_layers ${FREEZE_LAYERS} \
    --use_amp ${USE_AMP} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --memory_efficient_mode ${MEMORY_EFFICIENT} \
    --num_workers ${NUM_WORKERS} \
    --prefetch_factor ${PREFETCH_FACTOR} \
    --experiment_name "${STAGE_NAME}" \
    --output_dir output/${STAGE_NAME} \
    --overwrite False \
    --wandb False \
    --log_freq 10 \
    --eval_freq 10 \
    --save_checkpoints True \
    --checkpoint_freq 2000 \
    --resume "${RESUME}" \
    --seed ${SEED} > ${NOHUP_LOG} 2>&1 &
TRAIN_PID=$!

echo ""
echo "====================================="
echo "✅ 训练已启动！"
echo "====================================="
echo "PID: ${TRAIN_PID}"
echo "日志文件: ${NOHUP_LOG}"
echo ""
echo "🔍 查看训练进度:"
echo "  tail -f ${NOHUP_LOG}"
echo ""
echo "🔍 检查进程状态:"
echo "  ps aux | grep ${TRAIN_PID}"
echo ""
echo "⏸️  停止训练:"
echo "  kill ${TRAIN_PID}"
echo ""
echo "💾 输出目录: output/${STAGE_NAME}"
echo "====================================="
