#!/bin/bash

# 优化版评估脚本 - 跳过eps=0的攻击测试
# 只评估对抗鲁棒性

echo "================================================"
echo " 评估 KeyToken 增强模型（优化版）"
echo "================================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate keytoken

# 使用多个GPU加速评估
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

MODEL_PATH="output/enhanced_quick_test/checkpoints/final.pt"
IMAGENET_ROOT="~/data/KeyToken/datasets/imagenet"

echo ""
echo "1️⃣ 评估对抗鲁棒性 (PGD ε=2/255)..."
python -m CLIP_eval.clip_robustbench \
    --clip_model_name ViT-L-14 \
    --pretrained $MODEL_PATH \
    --dataset imagenet \
    --imagenet_root $IMAGENET_ROOT \
    --batch_size 64 \
    --n_samples_imagenet 1000 \
    --eps 2 \
    --wandb False

echo ""
echo "2️⃣ 评估对抗鲁棒性 (PGD ε=4/255)..."
python -m CLIP_eval.clip_robustbench \
    --clip_model_name ViT-L-14 \
    --pretrained $MODEL_PATH \
    --dataset imagenet \
    --imagenet_root $IMAGENET_ROOT \
    --batch_size 64 \
    --n_samples_imagenet 1000 \
    --eps 4 \
    --wandb False

echo ""
echo "================================================"
echo " 评估完成！"
echo " Clean准确率: 58.40% (已从之前测试获得)"
echo "================================================"
