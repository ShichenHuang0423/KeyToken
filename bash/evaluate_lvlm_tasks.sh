#!/bin/bash
# ============================================================================
# 完整LVLM评估脚本 - 使用真实的LLaVA和OpenFlamingo
# 遵循FARE论文设置
# ============================================================================

set -e

# ============================================================================
# 配置参数
# ============================================================================

# 数据集路径
VQAV2_ROOT="/home/ubuntu/data/KeyToken/datasets/VQAv2"
TEXTVQA_ROOT="/home/ubuntu/data/KeyToken/datasets/textvqa"
COCO_ROOT="/home/ubuntu/data/KeyToken/datasets/coco"
FLICKR_ROOT="/home/ubuntu/data/KeyToken/datasets/flickr30k"

# LVLM模型路径
LLAVA_PATH="/home/ubuntu/data/KeyToken/models/llava-v1.5-7b"
FLAMINGO_PATH="/home/ubuntu/data/KeyToken/models/openflamingo/OpenFlamingo-9B-vitl-mpt7b"

# 输出目录
OUTPUT_DIR="output/lvlm_eval"
mkdir -p "$OUTPUT_DIR"

# GPU配置 - LVLM需要大显存，建议单GPU运行
export CUDA_VISIBLE_DEVICES=4

# ============================================================================
# 评估配置
# ============================================================================

# CLIP模型列表 - 格式: "路径|名称"
CLIP_MODELS=(
    "models/fare_eps_2.pt|FARE-2"
    "models/fare_eps_4.pt|FARE-4"
    "models/stage0_epoch4.pt|KeyToken-E4"
)

# 任务列表 - 格式: "任务类型|lvlm_type|数据集|eps|max_samples"
EVAL_TASKS=(
    # VQA任务
    "vqa|llava|vqav2|2|500"
    "vqa|llava|vqav2|4|500"
    "vqa|llava|textvqa|2|500"
    "vqa|llava|textvqa|4|500"
    
    "vqa|flamingo|vqav2|2|500"
    "vqa|flamingo|vqav2|4|500"
    "vqa|flamingo|textvqa|2|500"
    "vqa|flamingo|textvqa|4|500"
    
    # Caption任务
    "caption|llava|coco|2|500"
    "caption|llava|coco|4|500"
    "caption|llava|flickr30k|2|500"
    "caption|llava|flickr30k|4|500"
    
    "caption|flamingo|coco|2|500"
    "caption|flamingo|coco|4|500"
    "caption|flamingo|flickr30k|2|500"
    "caption|flamingo|flickr30k|4|500"
)

# ============================================================================
# 评估函数
# ============================================================================

evaluate_lvlm_task() {
    local task_type=$1
    local lvlm_type=$2
    local dataset=$3
    local eps=$4
    local max_samples=$5
    local clip_path=$6
    local clip_name=$7
    
    echo ""
    echo "=========================================="
    echo "📊 评估任务: ${task_type^^} - $lvlm_type - $dataset"
    echo "   CLIP: $clip_name"
    echo "   Eps: $eps/255"
    echo "   Samples: $max_samples"
    echo "=========================================="
    
    # 检查CLIP模型
    if [ ! -f "$clip_path" ]; then
        echo "❌ CLIP模型不存在: $clip_path"
        return 1
    fi
    
    # 设置LVLM路径
    local lvlm_path=""
    if [ "$lvlm_type" = "llava" ]; then
        lvlm_path="$LLAVA_PATH"
    elif [ "$lvlm_type" = "flamingo" ]; then
        lvlm_path="$FLAMINGO_PATH"
    else
        echo "❌ 不支持的LVLM类型: $lvlm_type"
        return 1
    fi
    
    # 检查LVLM模型
    if [ ! -d "$lvlm_path" ]; then
        echo "❌ LVLM模型不存在: $lvlm_path"
        return 1
    fi
    
    case $task_type in
        "vqa")
            # VQA评估
            local dataset_root=""
            if [ "$dataset" = "vqav2" ]; then
                dataset_root="$VQAV2_ROOT"
            elif [ "$dataset" = "textvqa" ]; then
                dataset_root="$TEXTVQA_ROOT"
            fi
            
            python tools/evaluate_vqa_lvlm.py \
                --lvlm_type "$lvlm_type" \
                --lvlm_path "$lvlm_path" \
                --clip_checkpoint "$clip_path" \
                --clip_model_name "ViT-L-14" \
                --dataset "$dataset" \
                --dataset_root "$dataset_root" \
                --eps "$eps" \
                --max_samples "$max_samples" \
                --device cuda \
                --output_dir "$OUTPUT_DIR/vqa"
            ;;
            
        "caption")
            # Caption评估
            local dataset_root=""
            if [ "$dataset" = "coco" ]; then
                dataset_root="$COCO_ROOT"
            elif [ "$dataset" = "flickr30k" ]; then
                dataset_root="$FLICKR_ROOT"
            fi
            
            python tools/evaluate_caption_lvlm.py \
                --lvlm_type "$lvlm_type" \
                --lvlm_path "$lvlm_path" \
                --clip_checkpoint "$clip_path" \
                --clip_model_name "ViT-L-14" \
                --dataset "$dataset" \
                --dataset_root "$dataset_root" \
                --eps "$eps" \
                --max_samples "$max_samples" \
                --device cuda \
                --output_dir "$OUTPUT_DIR/caption"
            ;;
            
        *)
            echo "❌ 未知任务类型: $task_type"
            return 1
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "✅ 任务完成: ${task_type} - $lvlm_type - $dataset"
    else
        echo "❌ 任务失败: ${task_type} - $lvlm_type - $dataset"
    fi
}

# ============================================================================
# 主循环
# ============================================================================

echo "=========================================="
echo "🚀 开始LVLM评估 (FARE设置)"
echo "=========================================="
echo "CLIP模型数量: ${#CLIP_MODELS[@]}"
echo "任务数量: ${#EVAL_TASKS[@]}"
echo "总评估数: $((${#CLIP_MODELS[@]} * ${#EVAL_TASKS[@]}))"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 记录开始时间
START_TIME=$(date +%s)

# 循环所有CLIP模型
for clip_config in "${CLIP_MODELS[@]}"; do
    IFS='|' read -r clip_path clip_name <<< "$clip_config"
    
    echo ""
    echo "=========================================="
    echo "🔧 CLIP模型: $clip_name"
    echo "   路径: $clip_path"
    echo "=========================================="
    
    # 循环所有任务
    for task_config in "${EVAL_TASKS[@]}"; do
        IFS='|' read -r task_type lvlm_type dataset eps max_samples <<< "$task_config"
        
        # 执行评估
        evaluate_lvlm_task "$task_type" "$lvlm_type" "$dataset" "$eps" "$max_samples" \
                          "$clip_path" "$clip_name"
        
        # 短暂休息（LVLM评估较慢）
        sleep 5
    done
done

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "✅ 全部LVLM评估完成!"
echo "   总耗时: ${HOURS}h ${MINUTES}m"
echo "   结果目录: $OUTPUT_DIR"
echo "=========================================="

# 生成汇总报告
echo "🔄 生成汇总报告..."
python tools/summarize_lvlm_results.py --input_dir "$OUTPUT_DIR" --output_file "$OUTPUT_DIR/lvlm_summary_report.json"

echo "✅ 完成!"
