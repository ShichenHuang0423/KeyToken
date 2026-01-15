#!/bin/bash
# ============================================================================
# 多任务评估脚本 - 基于FARE论文设置
# 支持：零样本分类、VQA、Caption、POPE等任务
# 对比模型：FARE-4、KeyToken Epoch4
# ============================================================================

set -e

# ============================================================================
# 配置参数
# ============================================================================
IMAGENET_ROOT="/home/ubuntu/data/KeyToken/datasets/imagenet"
VQAV2_ROOT="/home/ubuntu/data/KeyToken/datasets/VQAv2"
TEXTVQA_ROOT="/home/ubuntu/data/KeyToken/datasets/textvqa"
COCO_ROOT="/home/ubuntu/data/KeyToken/datasets/coco"
FLICKR_ROOT="/home/ubuntu/data/KeyToken/datasets/flickr30k"
POPE_ROOT="/home/ubuntu/data/KeyToken/datasets/llava/eval/pope"
CLIP_EVAL_ROOT="/home/ubuntu/data/KeyToken/datasets/CLIP_eval"

OUTPUT_DIR="output/multi_task_eval"
mkdir -p "$OUTPUT_DIR"

# GPU配置
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 模型列表
# 格式: "模型路径|模型名称|推理模式|防御策略|noise_std"
# 防御策略: none, combined (仅对KeyToken有效)
# combined = ZeroPur + Interpretability-Guided (像素空间净化 + 特征空间净化)
# noise_std: 输入随机噪声标准差，0=确定性，0.01=推荐值
MODELS=(
    # "models/fare_eps_4.pt|FARE-4|baseline|none|0"
    # 纯模型评估（无噪声，目标RACC>33.8%）
    #"models/stage0_epoch4.pt|KeyToken-E4|eval|none|0"
    # 测试时防御策略（无噪声）
   # "models/stage0_epoch4.pt|KeyToken-E4-ZeroPur|eval|zeropur|0"
    "models/stage0_epoch4.pt|KeyToken-E4-Combined|eval|combined|0"
)

# ============================================================================
# 任务定义
# ============================================================================

# 格式: "任务类型|数据集|eps|max_samples"
# 任务类型: zeroshot, vqa, caption, pope
# eps: 2 或 4 (对应 2/255 或 4/255)
# max_samples: -1=全部, 其他数字=采样数量
# 注意：按照FARE论文，所有模型都使用灰盒攻击（只攻击vision backbone）

EVAL_TASKS=(
    # ============================================================================
    # 零样本分类 (ImageNet + 13个数据集) - FARE论文设置
    # 攻击: APGD-CE + APGD-DLR (targeted), 100 iterations
    # ============================================================================
    # "zeroshot|imagenet|2|-1"
    # "zeroshot|imagenet|4|-1"  # 暂时注释，测试其他数据集
    
    # 测试其他数据集是否可以正常加载
    "zeroshot|cifar10|4|-1"
    "zeroshot|cifar100|4|-1"
    "zeroshot|flowers102|4|-1"
    "zeroshot|pets|4|-1"
    "zeroshot|cars|4|-1"
    "zeroshot|dtd|4|-1"
    "zeroshot|caltech101|4|-1"
    "zeroshot|aircraft|4|-1"
    "zeroshot|eurosat|4|-1"
    "zeroshot|imagenet_r|4|-1"
    "zeroshot|imagenet_sketch|4|-1"
    "zeroshot|pcam|4|-1"
    "zeroshot|stl10|4|-1"

    # VQA、Caption和POPE任务保持注释
    "vqa|vqav2|2|500"
    "vqa|vqav2|4|500"
    "vqa|textvqa|2|500"
    "vqa|textvqa|4|500"
    "caption|coco|2|500"
    "caption|coco|4|500"
    "caption|flickr30k|2|500"
    "caption|flickr30k|4|500"
    "pope|random|0|-1"
    "pope|popular|0|-1"
    "pope|adversarial|0|-1"
)

# ============================================================================
# 评估函数
# ============================================================================

evaluate_task() {
    local task_type=$1
    local dataset=$2
    local eps=$3
    local max_samples=$4
    local model_path=$5
    local model_name=$6
    local mode=$7
    local defense=$8
    local noise_std=$9
    
    echo ""
    echo "=========================================="
    echo "📊 评估任务: ${task_type^^} - $dataset"
    echo "   模型: $model_name"
    echo "   Eps: $eps/255 (灰盒攻击)"
    echo "   Samples: $max_samples"
    if [ "$defense" != "none" ]; then
        echo "   🛡️  测试时防御: $defense"
    fi
    if (( $(echo "$noise_std > 0" | bc -l) )); then
        echo "   🎲 输入噪声: std=$noise_std (Randomized Smoothing)"
    fi
    echo "=========================================="
    
    # 检查模型文件
    if [ ! -f "$model_path" ]; then
        echo "❌ 模型不存在: $model_path"
        return 1
    fi
    
    case $task_type in
        "zeroshot")
            # 零样本分类 - FARE论文设置
            local args=(
                python tools/evaluate_zeroshot.py
                --checkpoint "$model_path"
                --clip_model_name "ViT-L-14"
                --dataset "$dataset"
                --eps "$eps"
                --mode "$mode"
                --batch_size 64
                --device cuda
                --output_dir "$OUTPUT_DIR/zeroshot"
                --gray_box
            )
            
            if [ $max_samples -gt 0 ]; then
                args+=(--max_samples "$max_samples")
            fi
            
            # 添加测试时防御策略（仅对KeyToken有效）
            if [ "$defense" != "none" ]; then
                args+=(--defense "$defense")
            fi
            
            # 添加输入噪声（Randomized Smoothing）
            if (( $(echo "$noise_std > 0" | bc -l) )); then
                args+=(--noise_std "$noise_std")
            fi
            
            "${args[@]}"
            ;;
            
        "vqa")
            # VQA评估 - FARE论文攻击pipeline
            local dataset_root=""
            if [ "$dataset" = "vqav2" ]; then
                dataset_root="$VQAV2_ROOT"
            elif [ "$dataset" = "textvqa" ]; then
                dataset_root="$TEXTVQA_ROOT"
            fi
            
            local args=(
                python tools/evaluate_vqa.py
                --checkpoint "$model_path"
                --clip_model_name "ViT-L-14"
                --dataset "$dataset"
                --dataset_root "$dataset_root"
                --eps "$eps"
                --mode "$mode"
                --max_samples "$max_samples"
                --device cuda
                --output_dir "$OUTPUT_DIR/vqa"
                --gray_box
            )
            
            "${args[@]}"
            ;;
            
        "caption")
            # Caption评估 - FARE论文攻击pipeline
            local dataset_root=""
            if [ "$dataset" = "coco" ]; then
                dataset_root="$COCO_ROOT"
            elif [ "$dataset" = "flickr30k" ]; then
                dataset_root="$FLICKR_ROOT"
            fi
            
            local args=(
                python tools/evaluate_caption.py
                --checkpoint "$model_path"
                --clip_model_name "ViT-L-14"
                --dataset "$dataset"
                --dataset_root "$dataset_root"
                --eps "$eps"
                --mode "$mode"
                --max_samples "$max_samples"
                --device cuda
                --output_dir "$OUTPUT_DIR/caption"
                --gray_box
            )
            
            "${args[@]}"
            ;;
            
        "pope")
            # POPE评估
            local args=(
                python tools/evaluate_pope.py
                --checkpoint "$model_path"
                --clip_model_name "ViT-L-14"
                --dataset_root "$POPE_ROOT"
                --split "$dataset"
                --mode "$mode"
                --device cuda
                --output_dir "$OUTPUT_DIR/pope"
            )
            
            "${args[@]}"
            ;;
            
        *)
            echo "❌ 未知任务类型: $task_type"
            return 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✅ 任务完成: ${task_type} - $dataset"
    else
        echo "❌ 任务失败: ${task_type} - $dataset"
    fi
}

# ============================================================================
# 主循环
# ============================================================================

echo "=========================================="
echo "🚀 开始多任务评估"
echo "=========================================="
echo "模型数量: ${#MODELS[@]}"
echo "任务数量: ${#EVAL_TASKS[@]}"
echo "总评估数: $((${#MODELS[@]} * ${#EVAL_TASKS[@]}))"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 记录开始时间
START_TIME=$(date +%s)

# 循环所有模型
for model_config in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name mode defense noise_std <<< "$model_config"
    
    echo ""
    echo "=========================================="
    echo "🔧 模型: $model_name"
    echo "   路径: $model_path"
    echo "   模式: $mode"
    if [ "$defense" != "none" ]; then
        echo "   防御: $defense"
    fi
    if (( $(echo "$noise_std > 0" | bc -l) )); then
        echo "   噪声: $noise_std"
    fi
    echo "=========================================="
    
    # 循环所有任务
    for task_config in "${EVAL_TASKS[@]}"; do
        IFS='|' read -r task_type dataset eps max_samples <<< "$task_config"
        
        # 对于带防御的模型，只在zeroshot任务上评估（其他任务不支持测试时防御）
        if [[ "$defense" != "none" && "$task_type" != "zeroshot" ]]; then
            echo "⏭️  跳过非零样本任务 (测试时防御仅用于zeroshot): ${task_type} - $dataset"
            continue
        fi
        
        # 执行评估
        evaluate_task "$task_type" "$dataset" "$eps" "$max_samples" \
                     "$model_path" "$model_name" "$mode" "$defense" "$noise_std"
        
        # 短暂休息
        sleep 2
    done
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ 全部评估完成!"
echo "   总耗时: ${HOURS}h ${MINUTES}m"
echo "   结果目录: $OUTPUT_DIR"
echo "=========================================="

# 生成汇总报告
python tools/summarize_results.py --input_dir "$OUTPUT_DIR" --output_file "$OUTPUT_DIR/summary_report.json"
