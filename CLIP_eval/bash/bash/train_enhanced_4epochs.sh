#!/bin/bash

# 完整训练脚本（ImageNet, 4 epochs）with 进度条和日志
# 每个epoch保存一次模型权重

echo "================================================"
echo " KeyToken 增强版训练 - 4 Epochs (带进度条)"
echo " 数据集: ImageNet (128万图像)"
echo " GPU: 7x RTX A6000 (48GB each)"
echo " 每个epoch保存模型"
echo " 预计时间: 10-12 小时"
echo " 日志文件: output/enhanced_4epochs/train.log"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keytoken

# 使用 7 张空闲 GPU（避开 GPU 4，它有其他任务）
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7

# 计算：ImageNet约128万图像
# 优化后：7 GPU × batch_size ~14/GPU = 总 batch 96（实测稳定配置）
# 每个epoch ≈ 1,281,167 / 96 ≈ 13,345 steps
# 4 epochs ≈ 53,380 steps

# 创建输出目录
mkdir -p output/enhanced_4epochs/checkpoints

# 完整训练配置 - 4 epochs（实测稳定配置，避免OOM）
# -u: 无缓冲输出，tee: 同时输出到终端和日志文件
python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained models/fare_eps_4.pt \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps 53380 \
    --warmup 2000 \
    --batch_size 96 \
    --lr 1e-5 \
    --wd 1e-4 \
    --opt adamw \
    --attack apgd \
    --inner_loss l2 \
    --loss l2 \
    --clean_weight 0.1 \
    --loss_clean l2 \
    --norm linf \
    --eps 4 \
    --iterations_adv 10 \
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
echo " - output/enhanced_4epochs/checkpoints/step_13345.pt  (Epoch 1)"
echo " - output/enhanced_4epochs/checkpoints/step_26690.pt (Epoch 2)"
echo " - output/enhanced_4epochs/checkpoints/step_40035.pt (Epoch 3)"
echo " - output/enhanced_4epochs/checkpoints/step_53380.pt (Epoch 4/Final)"
echo " 日志文件: output/enhanced_4epochs/train.log"
echo "================================================"
