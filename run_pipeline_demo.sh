#!/bin/bash

# Configuration
# 使用 FlowEdit 模块中的 YAML 文件作为测试数据
DATASET_YAML="modules/FlowEdit/edits.yaml"
OUTPUT_DIR="results/flowedit_demo"

# 设置保护方法 (支持 pid, diff_protect, atk_pdm, flowedit_protection)
# FlowEdit Protection 使用频域、纹理、特征和速度场保护
PROTECTION_METHOD="flowedit_protection"
PROTECTION_MODEL="sd3"
# FlowEdit Protection 参数 (不使用 attack_mode，而是使用噪声类型配置)
NOISE_CONFIG="freq_texture"  # freq, texture, feature, velocity, freq_texture, all

# 设置编辑方法 (支持 flow_edit)
EDITING_METHOD="flow_edit"
EDIT_MODEL="sd3"  # 或 flux

# 自动生成输出目录名称 (加上 noise_config 以区分)
OUTPUT_DIR="results/${PROTECTION_METHOD}_${NOISE_CONFIG}_${EDITING_METHOD}"
mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting FlowEdit Protection Pipeline Demo"
echo "Dataset: $DATASET_YAML"
echo "Output Directory: $OUTPUT_DIR"
echo "Protection: $PROTECTION_METHOD (Config: $NOISE_CONFIG, Model: $PROTECTION_MODEL)"
echo "Editing: $EDITING_METHOD (Model: $EDIT_MODEL)"
echo "========================================================"

# 运行 main.py
# main.py 已被修改，会自动检测 protected/ 目录下是否存在已保护的图像
# 如果存在，将跳过保护步骤，直接使用现有图像进行编辑和评估

# 根据 NOISE_CONFIG 设置 FlowEdit Protection 参数
case "$NOISE_CONFIG" in
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
    # Default: freq + texture
    PROTECTION_ARGS="--freq_enabled --texture_enabled --no_feature_enabled --no_velocity_enabled"
    ;;
esac

python main.py \
  --dataset_yaml "$DATASET_YAML" \
  --output_dir "$OUTPUT_DIR" \
  --protection_method "$PROTECTION_METHOD" \
  --protection_model "$PROTECTION_MODEL" \
  --editing_method "$EDITING_METHOD" \
  --edit_model "$EDIT_MODEL" \
  $PROTECTION_ARGS

echo "========================================================"
echo "Pipeline execution finished."
echo "Results saved to $OUTPUT_DIR"
