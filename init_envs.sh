#!/bin/bash

# Function to check if a conda env exists
env_exists() {
    conda env list | grep -q "^$1 "
}

# Function to create and setup environment
setup_env() {
    ENV_NAME=$1
    PYTHON_VERSION=$2
    REQ_FILE=$3
    CONDA_FILE=$4 # Optional: Path to environment.yml

    echo "=================================================="
    echo "Setting up environment: $ENV_NAME"
    echo "=================================================="

    if env_exists $ENV_NAME; then
        echo "Environment '$ENV_NAME' already exists. Skipping creation."
    else
        if [ -n "$CONDA_FILE" ] && [ -f "$CONDA_FILE" ]; then
            echo "Creating from $CONDA_FILE..."
            conda env create -f "$CONDA_FILE" -n $ENV_NAME
        else
            echo "Creating with Python $PYTHON_VERSION..."
            conda create -n $ENV_NAME python=$PYTHON_VERSION -y
        fi
    fi

    # Activate environment to install pip packages
    # Need to source conda.sh first
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        eval "$(conda shell.bash hook)"
    fi

    conda activate $ENV_NAME
    
    # Install dependencies
    if [ -n "$REQ_FILE" ] && [ -f "$REQ_FILE" ]; then
        echo "Installing pip requirements from $REQ_FILE..."
        # Upgrade pip first
        pip install --upgrade pip
        pip install -r "$REQ_FILE"
    fi
    
    echo "Environment $ENV_NAME setup complete."
    echo ""
}

# 1. Setup FlowEdit Environment
setup_env "flowedit" "3.10" "requirements_flowedit.txt" ""

# 2. Setup PID Environment
# PID uses Python 3.8 and has a requirements.txt
setup_env "pid_env" "3.8" "modules/Diffusion-PID-Protection/requirements.txt" ""

# 3. Setup AtkPDM Environment
# AtkPDM uses Python 3.10 and has a requirements.txt
setup_env "atkpdm" "3.10" "modules/AtkPDM/requirements.txt" ""

# 4. Setup Diff-Protect (Mist) Environment
# Mist uses an env.yml file
setup_env "mist" "3.8" "" "modules/Diff-Protect/env.yml"

echo "All environments initialized!"
