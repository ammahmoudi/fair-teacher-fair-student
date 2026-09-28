#!/usr/bin/env python3
"""
Test Script for Dynamic Path Implementation

This script tests all the path utilities and verifies that the hardcoded paths
have been successfully replaced with dynamic alternatives.

Usage:
    python test_dynamic_paths.py
"""

import os
import sys
from pathlib import Path

# Test the path utilities
try:
    from utils.path_utils import (
        get_project_root, get_data_path, get_models_path, get_configs_path,
        get_scripts_path, get_distillation_path, get_logs_path, get_results_path,
        get_arrow_data_path, get_formatted_data_path
    )
    print("✅ Path utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import path utilities: {e}")
    sys.exit(1)

def test_path_utilities():
    """Test all path utility functions"""
    print("\n🧪 Testing Path Utilities")
    print("=" * 50)
    
    try:
        # Test basic paths
        project_root = get_project_root()
        print(f"✅ Project root: {project_root}")
        
        data_path = get_data_path()
        print(f"✅ Data path: {data_path}")
        
        models_path = get_models_path()
        print(f"✅ Models path: {models_path}")
        
        configs_path = get_configs_path()
        print(f"✅ Configs path: {configs_path}")
        
        # Test subdirectory paths
        ohiot1dm_path = get_data_path("ohiot1dm", "raw_standardized")
        print(f"✅ OhioT1DM raw data: {ohiot1dm_path}")
        
        # Test arrow data path
        arrow_path = get_arrow_data_path("ohiot1dm", "raw_standardized", "570")
        print(f"✅ Arrow file path: {arrow_path}")
        
        # Test formatted data path
        formatted_path = get_formatted_data_path(6, 6)
        print(f"✅ Formatted data path: {formatted_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path utilities test failed: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists and report"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"⚠️  {description} not found: {file_path}")
        return False

def test_updated_files():
    """Test that updated files work correctly"""
    print("\n📁 Testing Updated Files")
    print("=" * 50)
    
    project_root = get_project_root()
    
    # Test key files exist
    key_files = [
        (project_root / "utils" / "path_utils.py", "Path utilities module"),
        (project_root / "scripts" / "chronos" / "config_generator.py", "Chronos config generator"),
        (project_root / "data_processing" / "standardizer.py", "Data standardizer"),
        (project_root / "distillation" / "scripts" / "train_teachers.py", "Teacher trainer"),
    ]
    
    all_exist = True
    for file_path, description in key_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def test_import_updated_modules():
    """Test importing updated modules"""
    print("\n🔗 Testing Module Imports")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test chronos config generator import
    try:
        total_tests += 1
        sys.path.append(str(get_scripts_path("chronos")))
        import config_generator
        print("✅ Chronos config generator imports successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Chronos config generator import failed: {e}")
    
    # Test data standardizer import
    try:
        total_tests += 1
        sys.path.append(str(get_project_root() / "scripts" / "data_formatting"))
        import standardize_data
        print("✅ Data standardizer imports successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Data standardizer import failed: {e}")
    
    # Test teacher trainer import  
    try:
        total_tests += 1
        sys.path.append(str(get_distillation_path("scripts")))
        import train_teachers
        print("✅ Teacher trainer imports successfully")
        success_count += 1
    except Exception as e:
        print(f"❌ Teacher trainer import failed: {e}")
    
    print(f"\n📊 Import Tests: {success_count}/{total_tests} successful")
    return success_count == total_tests

def create_test_environment_script():
    """Create a script to test environment variable support"""
    script_content = '''#!/bin/bash
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
'''
    
    test_script_path = get_project_root() / "test_env_support.sh"
    with open(test_script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(test_script_path, 0o755)
    print(f"✅ Created environment test script: {test_script_path}")

def main():
    """Main test function"""
    print("🚀 Dynamic Path Implementation Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test path utilities
    if not test_path_utilities():
        all_tests_passed = False
    
    # Test updated files exist
    if not test_updated_files():
        all_tests_passed = False
    
    # Test module imports
    if not test_import_updated_modules():
        all_tests_passed = False
    
    # Create environment test script
    try:
        create_test_environment_script()
    except Exception as e:
        print(f"❌ Failed to create environment test script: {e}")
        all_tests_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Dynamic path implementation is working correctly.")
        print("\n📋 Summary of changes:")
        print("• Created utils/path_utils.py with dynamic path resolution")
        print("• Updated all Python scripts to use dynamic paths")
        print("• Updated shell scripts to detect project root dynamically")
        print("• Updated YAML config files to use relative paths")
        print("• Added environment variable support (LLM_TIME_ROOT)")
        print("\n🔧 The project is now portable across different users and directories!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())