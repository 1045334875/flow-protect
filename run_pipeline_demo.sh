#!/bin/bash

# Configuration
# 使用 FlowEdit 模块中的 YAML 文件作为测试数据
DATASET_YAML="modules/FlowEdit/edits.yaml"

# ============================================================
# 保护方法配置
# ============================================================
# 支持的保护方法:
#   - flowedit_protection: FlowEdit 保护 (频域、纹理、特征、速度场)
#       NOISE_CONFIG: freq, texture, feature, velocity, freq_texture, all
#   - diff_protect: Diff-Protect/Mist 保护
#       ATTACK_MODE: mist, advdm, sds, sdsT, texture_only
#   - pid: PID 保护
#   - atk_pdm: AtkPDM 保护
# ============================================================

# 定义要运行的实验配置数组
# 格式: "PROTECTION_METHOD:PROTECTION_MODEL:CONFIG_OR_MODE"
EXPERIMENTS=(
    # FlowEdit Protection 实验
    "flowedit_protection:sd3:freq"
    "flowedit_protection:sd3:texture"
    "flowedit_protection:sd3:freq_texture"
    # "flowedit_protection:sd3:all"
    
    # Diff-Protect 实验 (取消注释以运行)
    # "diff_protect:sd1.4:mist"
    # "diff_protect:sd1.4:advdm"
    # "diff_protect:sd1.4:sds"
    # "diff_protect:sd1.4:texture_only"
    
    # PID 实验 (取消注释以运行)
    # "pid:sd1.4:default"
    
    # AtkPDM 实验 (取消注释以运行)
    # "atk_pdm:sd1.4:default"
)

# 设置编辑方法 (支持 flow_edit, 设置为空则跳过编辑)
EDITING_METHOD="flow_edit"
EDIT_MODEL="sd3"  # 或 flux

echo "========================================================"
echo "Starting Batch Protection Pipeline"
echo "Dataset: $DATASET_YAML"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo "Editing: $EDITING_METHOD (Model: $EDIT_MODEL)"
echo "========================================================"

# 循环执行每个实验
for exp in "${EXPERIMENTS[@]}"; do
    # 解析实验配置
    IFS=':' read -r PROTECTION_METHOD PROTECTION_MODEL CONFIG <<< "$exp"
    
    echo ""
    echo "========================================================"
    echo "Running experiment: $PROTECTION_METHOD ($CONFIG)"
    echo "========================================================"
    
    # 根据保护方法设置参数
    PROTECTION_ARGS=""
    OUTPUT_SUFFIX=""
    
    if [ "$PROTECTION_METHOD" == "flowedit_protection" ]; then
        # FlowEdit Protection 参数
        OUTPUT_SUFFIX="${CONFIG}"
        case "$CONFIG" in
            "freq")
                PROTECTION_ARGS="--freq_enabled --no_texture_enabled --no_feature_enabled --no_velocity_enabled"
                ;;
            "texture")
                PROTECTION_ARGS="--texture_enabled --no_freq_enabled --no_feature_enabled --no_velocity_enabled"
                ;;
            "feature")
                PROTECTION_ARGS="--feature_enabled --no_freq_enabled --no_texture_enabled --no_velocity_enabled"
                ;;
            "velocity")
                PROTECTION_ARGS="--velocity_enabled --no_freq_enabled --no_texture_enabled --no_feature_enabled"
                ;;
            "freq_texture")
                PROTECTION_ARGS="--freq_enabled --texture_enabled --no_feature_enabled --no_velocity_enabled"
                ;;
            "all")
                PROTECTION_ARGS="--freq_enabled --texture_enabled --feature_enabled --velocity_enabled"
                ;;
            *)
                PROTECTION_ARGS="--freq_enabled --texture_enabled --no_feature_enabled --no_velocity_enabled"
                OUTPUT_SUFFIX="freq_texture"
                ;;
        esac
    elif [ "$PROTECTION_METHOD" == "diff_protect" ]; then
        # Diff-Protect 参数
        OUTPUT_SUFFIX="${CONFIG}"
        PROTECTION_ARGS="--attack_mode $CONFIG"
    else
        # PID / AtkPDM 等其他方法
        OUTPUT_SUFFIX="default"
    fi
    
    # 生成输出目录
    OUTPUT_DIR="results/${PROTECTION_METHOD}_${OUTPUT_SUFFIX}_${EDITING_METHOD}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Output Directory: $OUTPUT_DIR"
    echo "Protection Args: $PROTECTION_ARGS"
    
    # 运行 main.py
    python main.py \
        --dataset_yaml "$DATASET_YAML" \
        --output_dir "$OUTPUT_DIR" \
        --protection_method "$PROTECTION_METHOD" \
        --protection_model "$PROTECTION_MODEL" \
        --editing_method "$EDITING_METHOD" \
        --edit_model "$EDIT_MODEL" \
        $PROTECTION_ARGS
    
    echo "Experiment $PROTECTION_METHOD ($CONFIG) finished."
    echo "Results saved to $OUTPUT_DIR"
done

echo ""
echo "========================================================"
echo "All experiments completed!"
echo "========================================================"
