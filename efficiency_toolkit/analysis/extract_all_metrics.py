#!/usr/bin/env python3
"""
Comprehensive Metrics Extraction Tool

This script extracts metrics from all experiment results and creates consolidated CSV files
for both Chronos and Time-LLM experiments.

Usage:
    python extract_all_metrics.py
    python extract_all_metrics.py --chronos-only
    python extract_all_metrics.py --time-llm-only
    python extract_all_metrics.py --output-dir ./results
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def extract_chronos_metrics(experiments_dir="./experiments", output_dir="./"):
    """Extract metrics from all Chronos experiments."""
    try:
        from scripts.utilities.extract_metrics import extract_metrics_to_csv
        
        # Find Chronos experiment directories
        chronos_dirs = []
        if os.path.exists(experiments_dir):
            for item in os.listdir(experiments_dir):
                if item.startswith("chronos_"):
                    chronos_dirs.append(os.path.join(experiments_dir, item))
        
        if not chronos_dirs:
            print("❌ No Chronos experiment directories found")
            return False
        
        print(f"🔍 Found {len(chronos_dirs)} Chronos experiment directories")
        
        # Extract comprehensive metrics
        comprehensive_csv = os.path.join(output_dir, "chronos_comprehensive_results.csv")
        extract_metrics_to_csv(base_dir=experiments_dir, output_csv=comprehensive_csv)
        print(f"✅ Chronos comprehensive metrics saved to: {comprehensive_csv}")
        
        # Extract individual experiment metrics
        for exp_dir in chronos_dirs:
            exp_name = os.path.basename(exp_dir)
            individual_csv = os.path.join(output_dir, f"chronos_{exp_name}_results.csv")
            extract_metrics_to_csv(base_dir=exp_dir, output_csv=individual_csv)
            print(f"📊 Individual metrics saved to: {individual_csv}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting Chronos metrics: {e}")
        return False

def extract_time_llm_metrics(experiments_dir="./experiments", output_dir="./"):
    """Extract metrics from all Time-LLM experiments."""
    try:
        from scripts.utilities.extract_metrics import extract_metrics_to_csv
        
        # Find Time-LLM experiment directories
        time_llm_dirs = []
        if os.path.exists(experiments_dir):
            for item in os.listdir(experiments_dir):
                if item.startswith("time_llm_"):
                    time_llm_dirs.append(os.path.join(experiments_dir, item))
        
        if not time_llm_dirs:
            print("❌ No Time-LLM experiment directories found")
            return False
        
        print(f"🔍 Found {len(time_llm_dirs)} Time-LLM experiment directories")
        
        # Extract comprehensive metrics
        comprehensive_csv = os.path.join(output_dir, "time_llm_comprehensive_results.csv")
        extract_metrics_to_csv(base_dir=experiments_dir, output_csv=comprehensive_csv)
        print(f"✅ Time-LLM comprehensive metrics saved to: {comprehensive_csv}")
        
        # Extract individual experiment metrics
        for exp_dir in time_llm_dirs:
            exp_name = os.path.basename(exp_dir)
            individual_csv = os.path.join(output_dir, f"time_llm_{exp_name}_results.csv")
            extract_metrics_to_csv(base_dir=exp_dir, output_csv=individual_csv)
            print(f"📊 Individual metrics saved to: {individual_csv}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting Time-LLM metrics: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Metrics Extraction Tool")
    parser.add_argument("--experiments_dir", default="./experiments",
                       help="Base directory containing experiment results")
    parser.add_argument("--output_dir", default="./",
                       help="Directory to save CSV files")
    parser.add_argument("--chronos_only", action="store_true",
                       help="Extract only Chronos metrics")
    parser.add_argument("--time_llm_only", action="store_true",
                       help="Extract only Time-LLM metrics")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📊 Comprehensive Metrics Extraction Tool")
    print("=" * 50)
    print(f"📁 Experiments directory: {os.path.abspath(args.experiments_dir)}")
    print(f"💾 Output directory: {os.path.abspath(args.output_dir)}")
    
    success_count = 0
    total_count = 0
    
    # Extract Chronos metrics
    if not args.time_llm_only:
        print(f"\n🤖 Extracting Chronos Metrics...")
        total_count += 1
        if extract_chronos_metrics(args.experiments_dir, args.output_dir):
            success_count += 1
    
    # Extract Time-LLM metrics
    if not args.chronos_only:
        print(f"\n🕐 Extracting Time-LLM Metrics...")
        total_count += 1
        if extract_time_llm_metrics(args.experiments_dir, args.output_dir):
            success_count += 1
    
    print(f"\n✅ Metrics Extraction Complete!")
    print(f"📈 Success: {success_count}/{total_count} operations")
    
    # List generated CSV files
    csv_files = [f for f in os.listdir(args.output_dir) if f.endswith('.csv') and ('chronos' in f or 'time_llm' in f)]
    if csv_files:
        print(f"\n📄 Generated CSV files:")
        for csv_file in sorted(csv_files):
            file_path = os.path.join(args.output_dir, csv_file)
            file_size = os.path.getsize(file_path)
            print(f"   - {csv_file} ({file_size} bytes)")

if __name__ == "__main__":
    main()