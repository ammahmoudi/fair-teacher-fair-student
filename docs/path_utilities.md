# Path Utilities Guide

## Overview

The Fair Teacher–Fair Student project uses dynamic path resolution to make it portable across different users and systems. The `utils/path_utils.py` module provides centralized path management.

## Quick Start

```python
from utils.path_utils import get_project_root, get_data_path, get_models_path

# Get project root (automatically detected)
project_root = get_project_root()

# Get data paths
data_path = get_data_path()
ohiot1dm_data = get_data_path("ohiot1dm", "raw_standardized")

# Get models path
models_path = get_models_path()
```

## Key Functions

- `get_project_root()` - Auto-detects project root directory
- `get_data_path(*subdirs)` - Gets data directory path
- `get_models_path(*subdirs)` - Gets models directory path
- `get_configs_path(*subdirs)` - Gets configs directory path
- `get_scripts_path(*subdirs)` - Gets scripts directory path
- `get_arrow_data_path(dataset, scenario, patient_id)` - Arrow file paths
- `get_formatted_data_path(context_len, pred_len)` - Formatted data paths

## Environment Variable

Set `LLM_TIME_ROOT` to override automatic detection:

```bash
# Recommended: use new variable name
export FAIR_TEACHER_ROOT=/custom/path/to/fair-teacher-fair-student

# Or use legacy variable (backwards compatible)
export LLM_TIME_ROOT=/custom/path/to/fair-teacher-fair-student
```

## Testing

Run tests to verify path utilities:

```bash
python tests/test_dynamic_paths.py
bash tests/test_env_support.sh
```