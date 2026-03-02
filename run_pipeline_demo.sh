#!/bin/bash

# Configuration
# 使用 FlowEdit 模块中的 YAML 文件作为测试数据
DATASET_YAML="modules/FlowEdit/edits.yaml"
OUTPUT_DIR="results/flowedit_demo"

# 设置保护方法 (支持 pid, diff_protect, atk_pdm)
# Mist 是 diff_protect 的一种模式
PROTECTION_METHOD="diff_protect"
PROTECTION_MODEL="sd1.4"

# 设置编辑方法 (支持 flow_edit)
EDITING_METHOD="flow_edit"
EDIT_MODEL="sd3"  # 或 flux

# 自动生成输出目录名称
OUTPUT_DIR="results/${PROTECTION_METHOD}_${EDITING_METHOD}"mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo "Starting FlowEdit Protection Pipeline Demo"
echo "Dataset: $DATASET_YAML"
echo "Output Directory: $OUTPUT_DIR"
echo "Protection: $PROTECTION_METHOD (Model: $PROTECTION_MODEL)"
echo "Editing: $EDITING_METHOD (Model: $EDIT_MODEL)"
echo "========================================================"

# 运行 main.py
# main.py 已被修改，会自动检测 protected/ 目录下是否存在已保护的图像
# 如果存在，将跳过保护步骤，直接使用现有图像进行编辑和评估
python main.py \
  --dataset_yaml "$DATASET_YAML" \
  --output_dir "$OUTPUT_DIR" \
  --protection_method "$PROTECTION_METHOD" \
  --protection_model "$PROTECTION_MODEL" \
  --attack_mode "mist" \
  --editing_method "$EDITING_METHOD" \
  --edit_model "$EDIT_MODEL"

echo "========================================================"
echo "Pipeline execution finished."
echo "Results saved to $OUTPUT_DIR"
