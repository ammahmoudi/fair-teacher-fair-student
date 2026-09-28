#!/usr/bin/env python3
"""
Convenience wrapper for the comprehensive efficiency runner.

This script provides easy access to the efficiency toolkit from the project root.

Usage:
    python efficiency_runner.py [args]
    
Examples:
    # Run all efficiency experiments
    python efficiency_runner.py
    
    # Analyze existing results only
    python efficiency_runner.py --analyze-only
    
    # Run experiments and analyze
    python efficiency_runner.py --analyze
    
    # Dry run (show what would be executed)
    python efficiency_runner.py --dry-run
"""

import sys
from pathlib import Path

# Add the efficiency toolkit to the Python path
toolkit_path = Path(__file__).parent / "efficiency_toolkit"
sys.path.insert(0, str(toolkit_path))

if __name__ == "__main__":
    from core.comprehensive_efficiency_runner import main
    sys.exit(main())