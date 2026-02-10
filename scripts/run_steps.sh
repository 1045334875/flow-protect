#!/bin/bash

# Usage: ./scripts/run_steps.sh [config.json]
CONFIG_FILE=${1:-config.json}

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file not found: $CONFIG_FILE"
  exit 1
fi

# Helper to read json values
get_json_value() {
  local key=$1
  if command -v jq &> /dev/null; then
    jq -r ".$key" "$CONFIG_FILE"
  else
    python3 - <<PY
import json
print(json.load(open("$CONFIG_FILE")).get("$key", ""))
PY
  fi
}

INPUT_IMAGE=$(get_json_value input_image)
OUTPUT_DIR=$(get_json_value output_dir)
PROTECTION_METHOD=$(get_json_value protection_method)
PROTECTION_MODEL=$(get_json_value protection_model)
EDITING_METHOD=$(get_json_value editing_method)
EDIT_MODEL=$(get_json_value edit_model)
SOURCE_PROMPT=$(get_json_value source_prompt)
TARGET_PROMPT=$(get_json_value target_prompt)

if [ -z "$INPUT_IMAGE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PROTECTION_METHOD" ] || [ -z "$EDITING_METHOD" ]; then
  echo "Missing required fields in config.json"
  exit 1
fi

# Prepare output structure
mkdir -p "$OUTPUT_DIR/original" "$OUTPUT_DIR/protected" "$OUTPUT_DIR/edited"
FILENAME=$(basename "$INPUT_IMAGE")
BASENAME="${FILENAME%.*}"
cp "$INPUT_IMAGE" "$OUTPUT_DIR/original/$FILENAME"

PROTECTED_IMAGE="$OUTPUT_DIR/protected/$FILENAME"
EDITED_IMAGE="$OUTPUT_DIR/edited/${BASENAME}_edit_0.png"
METRICS_JSON="$OUTPUT_DIR/metrics.json"

# Conda setup
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi

# Step 1: Protection
case "$PROTECTION_METHOD" in
  "atk_pdm") ENV_NAME="atkpdm" ;;
  "pid") ENV_NAME="pid_env" ;;
  "diff_protect") ENV_NAME="mist" ;;
  *) echo "Unknown protection method: $PROTECTION_METHOD"; exit 1 ;;
esac

conda activate "$ENV_NAME" || exit 1
python scripts/run_protection.py \
  --input_image "$INPUT_IMAGE" \
  --output_dir "$OUTPUT_DIR" \
  --protection_method "$PROTECTION_METHOD" \
  --protection_model "$PROTECTION_MODEL" \
  --source_prompt "$SOURCE_PROMPT" \
  --output_image "$PROTECTED_IMAGE" || exit 1

# Step 2: Editing
case "$EDITING_METHOD" in
  "flow_edit") ENV_NAME="flowedit" ;;
  "dreambooth") ENV_NAME="flowedit" ;;
  *) echo "Unknown editing method: $EDITING_METHOD"; exit 1 ;;
esac

conda activate "$ENV_NAME" || exit 1
python scripts/run_editing.py \
  --input_image "$PROTECTED_IMAGE" \
  --output_dir "$OUTPUT_DIR" \
  --editing_method "$EDITING_METHOD" \
  --edit_model "$EDIT_MODEL" \
  --source_prompt "$SOURCE_PROMPT" \
  --target_prompt "$TARGET_PROMPT" \
  --output_image "$EDITED_IMAGE" || exit 1

# Step 3: Evaluation
conda activate "flowedit" || exit 1
python scripts/run_evaluate.py \
  --original_image "$OUTPUT_DIR/original/$FILENAME" \
  --protected_image "$PROTECTED_IMAGE" \
  --edited_image "$EDITED_IMAGE" \
  --target_prompt "$TARGET_PROMPT" \
  --output_json "$METRICS_JSON" || exit 1

echo "All steps complete. Metrics saved to $METRICS_JSON"
