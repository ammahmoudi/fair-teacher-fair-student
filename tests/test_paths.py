#!/usr/bin/env python3
"""
Simple test to verify path utilities work correctly.
Run this after any changes to path utilities.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_path_utilities():
    """Test basic path utilities functionality."""
    try:
        from utils.path_utils import get_project_root, get_data_path, get_models_path
        
        project_root = get_project_root()
        data_path = get_data_path()
        models_path = get_models_path()
        
        print(f"✅ Project root: {project_root}")
        print(f"✅ Data path: {data_path}")
        print(f"✅ Models path: {models_path}")
        print("✅ Path utilities working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Path utilities test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_path_utilities()
    sys.exit(0 if success else 1)