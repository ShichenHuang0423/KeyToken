#!/bin/bash

# 完整训练脚本（ImageNet, 20000 steps）
# 使用所有8张RTX 3090

echo "================================================"
echo " KeyToken 增强版完整训练"
echo " 数据集: ImageNet"
echo " GPU: 8x RTX 3090"
echo " 预计时间: 12-16小时"
echo "================================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate keytoken

# 使用所有8张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 完整训练配置
python -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps 20000 \
    --warmup 1400 \
    --batch_size 256 \
    --lr 1e-5 \
    --wd 1e-4 \
    --opt adamw \
    --attack apgd \
    --inner_loss l2 \
    --loss l2 \
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
    --experiment_name "enhanced_clip_vitl14_eps4_mae" \
    --output_dir output/enhanced_clip_vitl14_eps4_mae \
    --overwrite False \
    --wandb True \
    --log_freq 10 \
    --eval_freq 50 \
    --save_checkpoints True

echo ""
echo "================================================"
echo " 训练完成！"
echo " 模型保存: output/enhanced_clip_vitl14_eps4_mae/checkpoints/final.pt"
echo "================================================"
