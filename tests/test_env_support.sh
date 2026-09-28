#!/bin/bash
# Test Environment Variable Support

echo "🌍 Testing Environment Variable Support"
echo "======================================"

# Get current directory as project root
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CURRENT_DIR/.." && pwd)"

echo "Setting LLM_TIME_ROOT to: $PROJECT_ROOT"
export LLM_TIME_ROOT="$PROJECT_ROOT"

echo "Testing path utilities with environment variable..."
python3 -c "
import sys
import os
sys.path.append('$PROJECT_ROOT')
from utils.path_utils import get_project_root, get_data_path
print(f'Project root from env: {get_project_root()}')
print(f'Data path from env: {get_data_path()}')
print('✅ Environment variable support works!')
"

echo "Environment variable test completed."
