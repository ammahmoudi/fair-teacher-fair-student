#!/usr/bin/env python3
"""
Quick test of the efficiency analysis functionality
"""

import os
import sys
import glob
import json
from pathlib import Path

def test_find_experiments():
    """Test finding completed experiments."""
    base_dir = Path(__file__).resolve().parents[1]
    
    experiments = {
        'time_llm_training': [],
        'time_llm_inference': [],
        'chronos_training': [],
        'chronos_inference': [],
        'distillation': [],
        'distillation_inference': []
    }
    
    # Find comprehensive performance reports
    base_pattern = "efficiency_experiments/experiments/**/comprehensive_performance_report*.json"
    report_files = glob.glob(str(base_dir / base_pattern), recursive=True)
    
    # Also find distillation results
    distillation_pattern = "efficiency_experiments/distillation_experiments/**/comprehensive_performance_report*.json"
    distillation_files = glob.glob(str(base_dir / distillation_pattern), recursive=True)
    report_files.extend(distillation_files)
    
    print(f"Found {len(report_files)} total report files")
    
    for report_file in report_files:
        try:
            # Parse experiment type from path using the fixed logic
            path_parts = Path(report_file).parts
            # Find experiment directory by looking for directories containing model type names
            experiment_dir = "unknown"
            for part in path_parts:
                if any(exp_type in part for exp_type in ['time_llm_inference', 'time_llm_training', 'chronos_inference', 'chronos_training', 'distillation_inference', 'distillation']):
                    experiment_dir = part
                    break
            
            print(f"Report: {Path(report_file).name}")
            print(f"  Experiment dir detected: {experiment_dir}")
            
            # Determine experiment category
            if 'time_llm_training' in experiment_dir:
                category = 'time_llm_training'
            elif 'time_llm_inference' in experiment_dir:
                category = 'time_llm_inference'
            elif 'chronos_training' in experiment_dir:
                category = 'chronos_training'
            elif 'chronos_inference' in experiment_dir:
                category = 'chronos_inference'
            elif 'distillation_inference' in experiment_dir:
                category = 'distillation_inference'
            elif 'distillation' in str(report_file):
                category = 'distillation'
            else:
                print(f"  -> No category matched for {experiment_dir}")
                continue
            
            print(f"  -> Categorized as: {category}")
            experiments[category].append(report_file)
            
        except Exception as e:
            print(f"  -> Error processing {report_file}: {e}")
        
        print()  # Empty line for readability
    
    # Print summary
    total = sum(len(exps) for exps in experiments.values())
    print(f"📊 Found {total} completed experiments:")
    for category, files in experiments.items():
        if files:
            print(f"  • {category.replace('_', ' ').title()}: {len(files)}")
    
    return experiments

if __name__ == "__main__":
    test_find_experiments()
