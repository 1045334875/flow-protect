#!/bin/bash

# Configuration
CONFIG_FILE="config.json"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    # Fallback to python for parsing json if jq is not available
    PROTECTION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['protection_method'])")
else
    PROTECTION_METHOD=$(jq -r '.protection_method' $CONFIG_FILE)
fi

# Determine Environment based on Method
case "$PROTECTION_METHOD" in
  "atk_pdm")
    ENV_NAME="atkpdm"
    ;;
  "pid")
    ENV_NAME="pid_env"
    ;;
  "diff_protect")
    ENV_NAME="mist"
    ;;
  *)
    echo "Unknown protection method: $PROTECTION_METHOD. Using default environment 'base'."
    ENV_NAME="base"
    ;;
esac

# Activate Conda Environment
# Note: This assumes conda is initialized in your shell.
# If running from a script where conda isn't available, we might need to source it.
# Attempt to find conda hook
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Try to rely on shell integration
    eval "$(conda shell.bash hook)"
fi

conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment '$ENV_NAME'."
    echo "Please ensure the environment exists. You can create it using:"
    echo "conda create -n $ENV_NAME python=3.8 (or 3.10 depending on method)"
    exit 1
fi

echo "Activated environment: $ENV_NAME for method: $PROTECTION_METHOD"

# Run the pipeline
echo "Running pipeline with config: $CONFIG_FILE"
python main.py --config $CONFIG_FILE

