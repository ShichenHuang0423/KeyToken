#!/bin/bash

# 渐进式解冻训练脚本 - 交互式阶段选择 (显存优化版 v2)
# 支持阶段0-4：完全解冻/冻结backbone/渐进解冻
# ⚡ 显存优化：支持AMP混合精度训练、梯度累积

echo "================================================"
echo "  KeyToken 渐进式解冻训练 (显存优化版)"
echo "  数据集: ImageNet (128万图像)"
echo "  GPU: 4x RTX 4090 (GPU 1-4, 稳定配置)"
echo "  ⚡ 新增: AMP混合精度 + 梯度累积"
echo "================================================"
echo ""
echo "🆕 代码版本: v2.0 - Attack Mode (2024-12-21)"
echo "   ✅ 对抗样本生成使用attack模式（无防御）"
echo "   ✅ 训练时使用train模式（完整增强）"
echo "   ✅ 预期FeatDiff: 0.05-0.15（显著提升）"
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

# 计算：ImageNet约128万图像，4 GPU
# 4卡配置更稳定，避免满功耗崩溃风险

# ⚡ 显存与I/O优化参数 (默认开启)
USE_AMP="True"              # 混合精度训练，节省~30%显存
GRADIENT_ACCUMULATION=3     # 梯度累积步数，有效batch=batch_size*accumulation
MEMORY_EFFICIENT="True"     # 内存高效模式

# 🚨 I/O优化：针对HDD磁盘瓶颈
# - 磁盘使用率94%，需要极致优化I/O
# - 降低DataLoader workers避免随机I/O
NUM_WORKERS=2               # HDD严重瓶颈时降到2 (原4)
PREFETCH_FACTOR=2           # 降低预读取，减少I/O压力 (原4)

# 根据阶段设置参数
case $STAGE in
    0)
        STAGE_NAME="stage0_full_training"
        # 重新计算：4 GPU × 12/GPU = 48总batch（实际batch size）
        # 每个epoch ≈ 1,281,167 / 48 ≈ 26,690 steps
        # 4 epochs = 106,760 steps
        STEPS=106760
        WARMUP=5000
        # ⚡ RTX 4090 24GB稳定配置 (4卡)
        # - 单卡batch_size=12，安全裕度充足
        # - 梯度累积3倍 = 有效batch 144
        BATCH_SIZE=48
        GRADIENT_ACCUMULATION=3
        LR="1e-5"
        FREEZE_BACKBONE="False"
        FREEZE_LAYERS=0
        RESUME=""
        SEED=42  # 🎲 Stage 0独立训练链的固定种子
        EPOCHS=4
        DESC="完全解冻训练 (4 epochs, 4090优化)"
        ;;
    1)
        STAGE_NAME="stage1_freeze_all"
        # ⚡ 优化显存利用率（24GB显存应充分使用）
        # - 冻结时显存需求小，可以用更大batch
        # - 每卡batch=32，4卡总batch=128
        # - 每个epoch = 1,281,167 / 128 ≈ 10,009 dataloader迭代
        # - optimizer步数 = 10,009 / 2 ≈ 5,005 步
        # - 预期显存：~12-15GB/卡 (50-60%利用率)
        STEPS=5005
        WARMUP=500
        BATCH_SIZE=128  # 4卡 × 32/卡，充分利用显存
        GRADIENT_ACCUMULATION=2
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
        # ⚡ 优化显存利用率：解冻6层后显存需求增加
        # - 每卡batch=24，4卡总batch=96
        # - 每个epoch = 1,281,167 / 96 ≈ 13,345 dataloader迭代
        # - optimizer步数 = 13,345 / 2 ≈ 6,673 步
        # - 1.2 epochs = 6,673 × 1.2 ≈ 8,008 步
        # - 预期显存：~15-18GB/卡 (60-75%利用率)
        STEPS=8008
        WARMUP=500
        BATCH_SIZE=96  # 4卡 × 24/卡
        GRADIENT_ACCUMULATION=2
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
        # ⚡ 优化显存利用率：解冻12层显存需求更大
        # - 每卡batch=20，4卡总batch=80
        # - 每个epoch = 1,281,167 / 80 ≈ 16,015 dataloader迭代
        # - optimizer步数 = 16,015 / 2 ≈ 8,008 步
        # - 预期显存：~16-19GB/卡 (65-80%利用率)
        STEPS=8008
        WARMUP=500
        BATCH_SIZE=80  # 4卡 × 20/卡
        GRADIENT_ACCUMULATION=2
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
        # ⚡ 优化显存利用率：完全解冻显存需求最大
        # - 每卡batch=16，4卡总batch=64
        # - 每个epoch = 1,281,167 / 64 ≈ 20,019 dataloader迭代
        # - optimizer步数 = 20,019 / 2 ≈ 10,010 步
        # - 预期显存：~18-21GB/卡 (75-85%利用率)
        STEPS=10010
        WARMUP=500
        BATCH_SIZE=64  # 4卡 × 16/卡
        GRADIENT_ACCUMULATION=2
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

echo ""
echo "================================================"
echo "  训练配置"
echo "================================================"
echo "阶段: Stage $STAGE - $DESC"
echo "输出目录: output/${STAGE_NAME}"
echo "训练步数: $STEPS steps (~$EPOCHS epoch)"
echo "Batch Size: $BATCH_SIZE (每GPU: $((BATCH_SIZE/4)))"
echo "梯度累积: $GRADIENT_ACCUMULATION (有效batch: $((BATCH_SIZE*GRADIENT_ACCUMULATION)))"
echo "学习率: $LR"
echo "Warmup: $WARMUP steps"
echo "冻结策略: backbone=$FREEZE_BACKBONE, layers=$FREEZE_LAYERS"
echo ""
echo "⚡ 显存优化:"
echo "   混合精度(AMP): $USE_AMP"
echo "   内存高效模式: $MEMORY_EFFICIENT"
echo ""
echo "🚨 I/O优化 (针对HDD磁盘瓶颈):"
echo "   DataLoader workers: $NUM_WORKERS (降低随机I/O)"
echo "   Prefetch factor: $PREFETCH_FACTOR (减少I/O压力)"
echo "   ⚠️  磁盘使用率94%，请定期清理空间"
if [ ! -z "$RESUME" ]; then
    echo "恢复自: $RESUME"
fi
echo "================================================"
echo ""
read -p "按Enter开始训练，或Ctrl+C取消..."
echo ""

# 开始训练
echo "开始训练 Stage $STAGE (后台运行)..."
echo "日志输出: output/${STAGE_NAME}/train.log"
echo ""

# 使用nohup后台运行，CUDA_VISIBLE_DEVICES在命令行设置避免环境变量问题
# ⚡ 新增显存优化参数: use_amp, gradient_accumulation_steps, memory_efficient_mode

# 构建基础命令
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained models/fare_eps_4.pt \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps $STEPS \
    --warmup $WARMUP \
    --batch_size $BATCH_SIZE \
    --lr $LR \
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
    --text_recon_weight 0.8 \
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
    --experiment_name \"${STAGE_NAME}\" \
    --output_dir output/${STAGE_NAME} \
    --overwrite False \
    --wandb False \
    --log_freq 10 \
    --eval_freq 10 \
    --save_checkpoints True \
    --checkpoint_freq 2000 \
    --resume \"$RESUME\""

# 只在SEED非空时添加--seed参数
if [ ! -z "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# 执行命令
eval "$CMD > output/${STAGE_NAME}/train.log 2>&1 &"

# 保存训练进程PID
TRAIN_PID=$!
echo $TRAIN_PID > output/${STAGE_NAME}/train.pid

echo ""
echo "================================================"
echo "  Stage $STAGE 训练已启动（后台运行）"
echo "================================================"
echo " 进程ID: $TRAIN_PID"
echo ""
echo " 监控命令:"
echo "   实时日志: tail -f output/${STAGE_NAME}/train.log"
echo "   查看进程: ps -p $TRAIN_PID"
echo "   GPU状态: watch -n 2 nvidia-smi"
echo "   停止训练: kill $TRAIN_PID"
echo ""
echo " 查看训练进度:"
echo "   grep 'Step' output/${STAGE_NAME}/train.log | tail -10"
echo ""
echo " 特征差异分析:"
echo "   grep 'FeatDiff' output/${STAGE_NAME}/train.log | tail -20"
echo ""
if [ "$STAGE" == "1" ]; then
    echo " 下一步: 运行 Stage 2 (解冻后6层)"
    echo "   bash bash/train_enhanced_4epochs.sh  # 选择 2"
elif [ "$STAGE" == "2" ]; then
    echo " 下一步: 运行 Stage 3 (解冻后12层)"
    echo "   bash bash/train_enhanced_4epochs.sh  # 选择 3"
elif [ "$STAGE" == "3" ]; then
    echo " 下一步: 运行 Stage 4 (完全解冻微调，可选)"
    echo "   bash bash/train_enhanced_4epochs.sh  # 选择 4"
fi
echo "================================================"
echo ""
echo "提示: SSH断开后训练会继续运行"
echo "重新登录后可用以下命令监控:"
echo "  tail -f output/${STAGE_NAME}/train.log"
echo ""
