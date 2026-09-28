#!/usr/bin/env python3
"""
Quick Data Processing Script

This script provides quick commands for common data processing tasks.
It's a simpler interface to the complete data pipeline.

Usage:
    # Process ohiot1dm dataset completely
    python quick_process.py ohiot1dm

    # Process d1namo dataset completely  
    python quick_process.py d1namo

    # Process all datasets
    python quick_process.py all

    # Process specific scenarios only
    python quick_process.py ohiot1dm --scenarios raw,noisy

    # Dry run
    python quick_process.py ohiot1dm --dry-run
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_complete_pipeline(dataset, scenarios='all', dry_run=False):
    """Run the complete data pipeline."""
    script_dir = Path(__file__).parent
    pipeline_script = script_dir / "complete_data_pipeline.py"
    
    cmd = ['python', str(pipeline_script)]
    
    if dataset == 'all':
        cmd.append('--all')
    else:
        cmd.extend(['--dataset', dataset])
    
    if scenarios != 'all':
        cmd.extend(['--scenarios', scenarios])
    
    if dry_run:
        cmd.append('--dry-run')
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode

def main():
    parser = argparse.ArgumentParser(
        description="Quick Data Processing for Fair Teacher–Fair Student",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('dataset', choices=['ohiot1dm', 'd1namo', 'all'],
                       help='Dataset to process')
    parser.add_argument('--scenarios', default='all',
                       help='Comma-separated scenarios (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed')
    
    args = parser.parse_args()
    
    print(f"🚀 Quick processing for {args.dataset}")
    if args.scenarios != 'all':
        print(f"📋 Scenarios: {args.scenarios}")
    
    exit_code = run_complete_pipeline(args.dataset, args.scenarios, args.dry_run)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()