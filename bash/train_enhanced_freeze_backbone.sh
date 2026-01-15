#!/bin/bash

# 冻结CLIP backbone训练脚本 - 只训练新增模块
# 适用场景：快速迭代新增模块，减少显存和训练时间

echo "================================================"
echo " KeyToken 增强版训练 - 冻结CLIP Backbone"
echo " 策略: 只训练新增模块 (MAE decoder, disturb detector, key selector)"
echo " 数据集: ImageNet (128万图像)"
echo " GPU: 8x RTX A6000 (48GB each)"
echo " 预计显存节省: ~60% (梯度+优化器状态)"
echo " 预计训练加速: ~2-3x"
echo "================================================"

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keytoken

# 使用全部 8 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 冻结backbone后可以使用更大的batch size
# 8 GPU × batch_size 24/GPU = 总 batch 192 (vs 未冻结时112)
# 每个epoch ≈ 1,281,167 / 192 ≈ 6,673 steps
# 4 epochs = 6,673 × 4 = 26,692 steps

# 创建输出目录
mkdir -p output/enhanced_4epochs_frozen/checkpoints

# 冻结CLIP backbone训练配置
python -u -m train.adversarial_training_clip_enhanced \
    --clip_model_name ViT-L-14 \
    --pretrained models/fare_eps_4.pt \
    --dataset imagenet \
    --imagenet_root ~/data/KeyToken/datasets/imagenet \
    --steps 26692 \
    --warmup 1500 \
    --batch_size 192 \
    --lr 5e-4 \
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
    --freeze_clip_backbone True \
    --freeze_encoder_layers 0 \
    --experiment_name "enhanced_4epochs_frozen" \
    --output_dir output/enhanced_4epochs_frozen \
    --overwrite False \
    --wandb False \
    --log_freq 50 \
    --eval_freq 100 \
    --save_checkpoints True \
    --checkpoint_freq 500 \
    --resume "" 2>&1 | tee output/enhanced_4epochs_frozen/train.log

echo ""
echo "================================================"
echo " 训练完成！"
echo " 每个epoch的模型保存在:"
echo " - output/enhanced_4epochs_frozen/checkpoints/epoch_1.pt"
echo " - output/enhanced_4epochs_frozen/checkpoints/epoch_2.pt"
echo " - output/enhanced_4epochs_frozen/checkpoints/epoch_3.pt"
echo " - output/enhanced_4epochs_frozen/checkpoints/epoch_4.pt"
echo " 日志文件: output/enhanced_4epochs_frozen/train.log"
echo "================================================"
