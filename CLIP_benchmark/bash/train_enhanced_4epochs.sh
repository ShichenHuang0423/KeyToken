#!/bin/bash

# 完整训练脚本（ImageNet, 4 epochs）with 进度条和日志
# 每个epoch保存一次模型权重

echo "================================================"
echo " KeyToken 增强版训练 - 4 Epochs (带进度条)"
echo " 数据集: ImageNet (128万图像)"
echo " GPU: 8x RTX A6000 (48GB each)"
echo " 每个epoch保存模型"
echo " 预计时间: ~12天 (4 epochs, APGD, 8卡)"
echo " 日志文件: output/enhanced_4epochs/train.log"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keytoken

# 使用全部 8 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 计算：ImageNet约128万图像
# 实际配置：8 GPU × batch_size 14/GPU = 总 batch 112
# 每个epoch ≈ 1,281,167 / 112 ≈ 11,439 steps
# 4 epochs = 11,439 × 4 = 45,756 steps
# 每个epoch都保存权重，可随时停止测试
# 每个epoch结束时会自动保存模型

# 创建输出目录
mkdir -p output/enhanced_4epochs/checkpoints

# 完整训练配置 - 4 epochs（实测稳定配置，避免OOM）
# -u: 无缓冲输出，tee: 同时输出到终端和日志文件
python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained models/fare_eps_4.pt \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps 45756 \
    --warmup 2000 \
    --batch_size 112 \
    --lr 1e-5 \
    --wd 1e-4 \
    --opt adamw \
    --attack pgd \
    --inner_loss l2 \
    --loss l2 \
    --clean_weight 0.1 \
    --loss_clean l2 \
    --norm linf \
    --eps 4 \
    --iterations_adv 10 \
    --stepsize_adv 1 \
    --use_mae_recon True \
    --use_key_token_protection True \
    --mae_weight 0.1 \
    --text_recon_weight 0.8 \
    --key_token_ratio 0.2 \
    --mask_ratio 0.5 \
    --adaptive_masking False \
    --experiment_name "enhanced_4epochs" \
    --output_dir output/enhanced_4epochs \
    --overwrite False \
    --wandb False \
    --log_freq 50 \
    --eval_freq 100 \
    --save_checkpoints True 2>&1 | tee output/enhanced_4epochs/train.log

echo ""
echo "================================================"
echo " 训练完成！"
echo " 每个epoch的模型保存在:"
echo " - output/enhanced_4epochs/checkpoints/epoch_1.pt     (Epoch 1, ~11439步)"
echo " - output/enhanced_4epochs/checkpoints/epoch_2.pt     (Epoch 2, ~22878步)"
echo " - output/enhanced_4epochs/checkpoints/epoch_3.pt     (Epoch 3, ~34317步)"
echo " - output/enhanced_4epochs/checkpoints/epoch_4.pt     (Epoch 4, ~45756步/Final)"
echo " + 每10%步数也会保存checkpoint (step_4575.pt, step_9151.pt, ...)"
echo " 日志文件: output/enhanced_4epochs/train.log"
echo "================================================"
