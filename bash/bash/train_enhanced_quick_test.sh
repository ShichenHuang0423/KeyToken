#!/bin/bash

# 快速测试脚本（CIFAR-10, 1000 steps）
# 用于验证增强功能是否正常工作

echo "================================================"
echo " KeyToken 增强版快速测试"
echo "================================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate keytoken

# 设置CUDA设备（避开显存占用的GPU 2,6,7）
export CUDA_VISIBLE_DEVICES=2,6,7

# 快速测试配置
python -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained openai \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps 500 \
    --warmup 100 \
    --batch_size 32\
    --lr 1e-5 \
    --attack apgd \
    --eps 4 \
    --iterations_adv 10 \
    --use_mae_recon True \
    --use_key_token_protection True \
    --mae_weight 0.1 \
    --key_token_ratio 0.2 \
    --mask_ratio 0.5 \
    --adaptive_masking False \
    --experiment_name "enhanced_quick_test" \
    --output_dir output/enhanced_quick_test \
    --overwrite True \
    --wandb False \
    --log_freq 10 \
    --save_checkpoints True

echo ""
echo "================================================"
echo " 快速测试完成！"
echo " 检查输出: output/enhanced_quick_test/"
echo "================================================"
