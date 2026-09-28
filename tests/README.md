# Tests Directory

This directory contains test scripts for the Fair Teacher–Fair Student project.

## Available Tests

- `test_paths.py` - Simple test for path utilities functionality
- `test_dynamic_paths.py` - Comprehensive test suite for dynamic path implementation
- `test_env_support.sh` - Shell script to test environment variable support

## Running Tests

```bash
# Quick path utilities test
python tests/test_paths.py

# Comprehensive path test suite
python tests/test_dynamic_paths.py

# Environment variable support test
bash tests/test_env_support.sh
```

## Adding New Tests

When adding new tests, follow the naming convention `test_*.py` for Python tests and `test_*.sh` for shell tests.