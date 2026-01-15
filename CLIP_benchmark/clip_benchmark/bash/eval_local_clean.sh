#!/bin/bash
# Zero-Shot 评估（干净样本）- 使用本地 WebDataset
# 路径：~/data/KeyToken/datasets/webdatasets/

set -e
export PYTHONPATH="../":"${PYTHONPATH}"

SECONDS=0

# ===== 配置参数 =====
# 本地数据集根目录
DATASET_ROOT=~/data/KeyToken/datasets/webdatasets

# 评估参数
SAMPLES=1000   # 每个数据集的样本数 (-1 = 全部)
BS=128         # Batch size

# 输出目录
SAVE_DIR=~/data/KeyToken/zero_shot_results/clean
mkdir -p "$SAVE_DIR"

# 数据集列表（本地）
DATASET_FILE=benchmark/datasets_local.txt

# 模型列表（本地）
MODEL_FILE=benchmark/models_local.txt

echo "======================================"
echo "CLIP Zero-Shot 评估（干净样本）"
echo "======================================"
echo "数据集根目录: $DATASET_ROOT"
echo "数据集配置: $DATASET_FILE"
echo "模型配置: $MODEL_FILE"
echo "样本数: $SAMPLES"
echo "Batch Size: $BS"
echo "输出目录: $SAVE_DIR"
echo "======================================"

# 运行评估
python -m clip_benchmark.cli eval \
  --dataset_root "$DATASET_ROOT/{dataset}" \
  --dataset "$DATASET_FILE" \
  --pretrained_model "$MODEL_FILE" \
  --output "${SAVE_DIR}/clean_{model}_{pretrained}_{dataset}_{n_samples}_bs{bs}.json" \
  --attack none \
  --batch_size $BS \
  --n_samples $SAMPLES \
  --verbose

hours=$((SECONDS / 3600))
minutes=$(( (SECONDS % 3600) / 60 ))
echo ""
echo "======================================"
echo "[Runtime] $hours h $minutes min"
echo "======================================"
echo "结果保存在: $SAVE_DIR"
