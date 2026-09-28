#!/bin/bash

# Usage: ./run_main.sh --config_path <path_to_config> --log_level <log_level> --remove_checkpoints <True/False>

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the workspace root (parent of scripts directory)
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to workspace root to ensure main.py is accessible
cd "$WORKSPACE_ROOT" || exit 1

# Activate virtual environment if it exists
if [ -d "$WORKSPACE_ROOT/venv" ]; then
    source "$WORKSPACE_ROOT/venv/bin/activate"
elif [ -d "$WORKSPACE_ROOT/.venv" ]; then
    source "$WORKSPACE_ROOT/.venv/bin/activate"
fi

# Default values for arguments
CONFIG_PATH="./configs/config_time_llm.gin"
LOG_LEVEL="INFO"
REMOVE_CHECKPOINTS="False"  # Default value for the flag
# Generate random port to avoid conflicts
master_port=$((RANDOM % 10000 + 20000))
num_process=1
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_path) CONFIG_PATH="$2"; shift ;;
        --log_level) LOG_LEVEL="$2"; shift ;;
        --remove_checkpoints) REMOVE_CHECKPOINTS="$2"; shift ;;
        --master_port) master_port="$2"; shift ;;  # Allow custom port
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute the Python script with accelerate launch and provided arguments
accelerate launch  \
 --num_processes $num_process \
 --main_process_port $master_port\
 --num_machines=1 --dynamo_backend=no --mixed_precision bf16 \
    main.py \
    --config_path "$CONFIG_PATH" \
    --log_level "$LOG_LEVEL" \
    --remove_checkpoints "$REMOVE_CHECKPOINTS"  # Pass the flag to Python script
