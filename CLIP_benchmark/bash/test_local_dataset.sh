#!/bin/bash
# 快速测试本地数据集（CIFAR-10）
# 用于验证本地 WebDataset 是否正常工作

set -e
export PYTHONPATH="../":"${PYTHONPATH}"

SECONDS=0

# ===== 配置参数 =====
DATASET_ROOT=~/data/KeyToken/datasets/webdatasets
SAVE_DIR=~/data/KeyToken/zero_shot_results/test
mkdir -p "$SAVE_DIR"

# 测试参数
SAMPLES=100    # 只测试 100 个样本
BS=32
DATASET=cifar10
MODEL=ViT-L-14
PRETRAINED=openai

echo "======================================"
echo "测试本地 WebDataset"
echo "======================================"
echo "数据集: $DATASET"
echo "路径: $DATASET_ROOT/$DATASET"
echo "模型: $MODEL ($PRETRAINED)"
echo "样本数: $SAMPLES"
echo "======================================"

# 检查数据集是否存在
if [ ! -d "$DATASET_ROOT/$DATASET" ]; then
    echo "❌ 错误: 数据集不存在: $DATASET_ROOT/$DATASET"
    exit 1
fi

echo "✅ 数据集目录存在"
echo ""
echo "数据集结构:"
ls -lh "$DATASET_ROOT/$DATASET/"
echo ""
echo "开始评估..."
echo "======================================"

# 运行评估
python -m clip_benchmark.cli eval \
  --dataset_root "$DATASET_ROOT" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --pretrained "$PRETRAINED" \
  --output "${SAVE_DIR}/test_{model}_{pretrained}_{dataset}_{n_samples}.json" \
  --attack none \
  --batch_size $BS \
  --n_samples $SAMPLES \
  --verbose

echo ""
echo "======================================"
echo "✅ 测试完成！"
echo "======================================"
echo "运行时间: $SECONDS 秒"
echo "结果: ${SAVE_DIR}/test_${MODEL}_${PRETRAINED}_${DATASET}_${SAMPLES}.json"
echo ""
echo "如果测试成功，可以运行完整评估："
echo "  cd ~/data/KeyToken/CLIP_benchmark"
echo "  ./bash/eval_local_clean.sh"
