#!/usr/bin/env python3
"""
Rebuild Results CSV from Pipeline Directories
==============================================

Scans all pipeline run directories and extracts results from JSON files
to create a complete pipeline_results.csv file.

Usage:
    python scripts/analysis/rebuild_results_csv.py --results-dir distillation_pairs_comparison
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import sys

def extract_results_from_pipeline(pipeline_dir):
    """Extract results from a single pipeline directory."""
    pipeline_dir = Path(pipeline_dir)
    
    # Find patient subdirectory
    patient_dirs = list(pipeline_dir.glob("patient_*"))
    if not patient_dirs:
        return None
    
    patient_dir = patient_dirs[0]
    patient_id = patient_dir.name.replace("patient_", "")
    
    result = {
        'timestamp': datetime.fromtimestamp(pipeline_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_id': patient_dir.name,
        'pipeline_dir': str(pipeline_dir),
        'patient_ids': patient_id,
        'dataset_name': 'ohiot1dm',  # Default
        'seed': 42,  # Default
        'teacher_model': '',
        'student_model': '',
        'learning_rate': 0.001,
        'batch_size': 32,
        'teacher_epochs': 10,
        'student_epochs': 10,
        'distill_epochs': 10,
        'alpha': 0.5,
        'beta': 0.5,
        'teacher_rmse': None,
        'teacher_mae': None,
        'teacher_mape': None,
        'teacher_training_time': None,
        'teacher_status': '',
        'student_baseline_rmse': None,
        'student_baseline_mae': None,
        'student_baseline_mape': None,
        'student_training_time': None,
        'student_status': '',
        'distilled_rmse': None,
        'distilled_mae': None,
        'distilled_mape': None,
        'distillation_time': None,
        'distillation_status': '',
        'teacher_to_distilled_rmse_improvement': None,
        'teacher_to_distilled_rmse_improvement_pct': None,
        'student_to_distilled_rmse_improvement': None,
        'student_to_distilled_rmse_improvement_pct': None,
        'pipeline_status': '',
        'total_runtime': None,
        'notes': f'Patient {patient_id} complete 3-phase pipeline run'
    }
    
    # Extract teacher results
    teacher_summary = patient_dir / "phase_1_teacher" / "teacher_training_summary.json"
    if teacher_summary.exists():
        try:
            with open(teacher_summary, 'r') as f:
                teacher_data = json.load(f)
                result['teacher_model'] = teacher_data.get('model_name', '')
                result['teacher_rmse'] = teacher_data.get('performance_metrics', {}).get('rmse')
                result['teacher_mae'] = teacher_data.get('performance_metrics', {}).get('mae')
                result['teacher_mape'] = teacher_data.get('performance_metrics', {}).get('mape')
                result['teacher_training_time'] = teacher_data.get('training_time_seconds')
                result['teacher_status'] = 'completed'
                result['teacher_epochs'] = teacher_data.get('epochs', 10)
        except Exception as e:
            print(f"Warning: Could not read teacher summary from {teacher_summary}: {e}")
    
    # Extract student baseline results
    student_summary = patient_dir / "phase_2_student" / "student_baseline_summary.json"
    if student_summary.exists():
        try:
            with open(student_summary, 'r') as f:
                student_data = json.load(f)
                result['student_model'] = student_data.get('model_name', '')
                result['student_baseline_rmse'] = student_data.get('performance_metrics', {}).get('rmse')
                result['student_baseline_mae'] = student_data.get('performance_metrics', {}).get('mae')
                result['student_baseline_mape'] = student_data.get('performance_metrics', {}).get('mape')
                result['student_training_time'] = student_data.get('training_time_seconds')
                result['student_status'] = 'completed'
                result['student_epochs'] = student_data.get('epochs', 10)
        except Exception as e:
            print(f"Warning: Could not read student summary from {student_summary}: {e}")
    
    # Extract distillation results
    distill_summary = patient_dir / "phase_3_distillation" / "distillation_summary.json"
    if distill_summary.exists():
        try:
            with open(distill_summary, 'r') as f:
                distill_data = json.load(f)
                result['distilled_rmse'] = distill_data.get('performance_metrics', {}).get('rmse')
                result['distilled_mae'] = distill_data.get('performance_metrics', {}).get('mae')
                result['distilled_mape'] = distill_data.get('performance_metrics', {}).get('mape')
                result['distillation_time'] = distill_data.get('training_time_seconds')
                result['distillation_status'] = 'completed'
                result['distill_epochs'] = distill_data.get('epochs', 10)
                result['alpha'] = distill_data.get('alpha', 0.5)
                result['beta'] = distill_data.get('beta', 0.5)
        except Exception as e:
            print(f"Warning: Could not read distillation summary from {distill_summary}: {e}")
    
    # Calculate improvements
    if result['teacher_rmse'] is not None and result['distilled_rmse'] is not None:
        improvement = result['teacher_rmse'] - result['distilled_rmse']
        result['teacher_to_distilled_rmse_improvement'] = improvement
        result['teacher_to_distilled_rmse_improvement_pct'] = (improvement / result['teacher_rmse']) * 100
    
    if result['student_baseline_rmse'] is not None and result['distilled_rmse'] is not None:
        improvement = result['student_baseline_rmse'] - result['distilled_rmse']
        result['student_to_distilled_rmse_improvement'] = improvement
        result['student_to_distilled_rmse_improvement_pct'] = (improvement / result['student_baseline_rmse']) * 100
    
    # Calculate total runtime
    times = []
    if result['teacher_training_time']:
        times.append(result['teacher_training_time'])
    if result['student_training_time']:
        times.append(result['student_training_time'])
    if result['distillation_time']:
        times.append(result['distillation_time'])
    
    if times:
        result['total_runtime'] = sum(times)
    
    # Determine pipeline status
    if (result['teacher_status'] == 'completed' and 
        result['student_status'] == 'completed' and 
        result['distillation_status'] == 'completed'):
        result['pipeline_status'] = 'SUCCESS'
    else:
        result['pipeline_status'] = 'INCOMPLETE'
    
    # Only return if we have minimum required data
    if result['teacher_model'] and result['student_model']:
        return result
    
    return None


def rebuild_csv(results_dir, output_file=None):
    """Rebuild the complete CSV from all pipeline directories."""
    results_dir = Path(results_dir)
    pipeline_runs_dir = results_dir / "pipeline_runs"
    
    if not pipeline_runs_dir.exists():
        print(f"Error: {pipeline_runs_dir} does not exist!")
        return False
    
    # Find all pipeline directories
    pipeline_dirs = sorted(pipeline_runs_dir.glob("pipeline_*"))
    
    print(f"🔍 Found {len(pipeline_dirs)} pipeline directories")
    print(f"📊 Extracting results...")
    
    results = []
    skipped = 0
    
    for i, pipeline_dir in enumerate(pipeline_dirs, 1):
        result = extract_results_from_pipeline(pipeline_dir)
        if result:
            results.append(result)
            print(f"  ✓ {i}/{len(pipeline_dirs)}: {result['teacher_model'][:30]} → {result['student_model'][:30]}")
        else:
            skipped += 1
            print(f"  ⊘ {i}/{len(pipeline_dirs)}: {pipeline_dir.name} (incomplete data)")
    
    print(f"\n✅ Extracted {len(results)} complete experiments")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} incomplete experiments")
    
    if not results:
        print("❌ No valid results found!")
        return False
    
    # Write to CSV
    if output_file is None:
        output_file = results_dir / "pipeline_results.csv"
    
    fieldnames = [
        'timestamp', 'pipeline_id', 'pipeline_dir', 'patient_ids', 'dataset_name', 'seed',
        'teacher_model', 'student_model', 'learning_rate', 'batch_size',
        'teacher_epochs', 'student_epochs', 'distill_epochs', 'alpha', 'beta',
        'teacher_rmse', 'teacher_mae', 'teacher_mape', 'teacher_training_time', 'teacher_status',
        'student_baseline_rmse', 'student_baseline_mae', 'student_baseline_mape', 
        'student_training_time', 'student_status',
        'distilled_rmse', 'distilled_mae', 'distilled_mape', 'distillation_time', 'distillation_status',
        'teacher_to_distilled_rmse_improvement', 'teacher_to_distilled_rmse_improvement_pct',
        'student_to_distilled_rmse_improvement', 'student_to_distilled_rmse_improvement_pct',
        'pipeline_status', 'total_runtime', 'notes'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ CSV file written to: {output_file}")
    print(f"📊 Total records: {len(results)}")
    
    # Print summary statistics
    successful = sum(1 for r in results if r['pipeline_status'] == 'SUCCESS')
    print(f"🎯 Successful pipelines: {successful}/{len(results)}")
    
    teachers = set(r['teacher_model'] for r in results)
    students = set(r['student_model'] for r in results)
    print(f"👨‍🏫 Unique teachers: {len(teachers)}")
    print(f"👨‍🎓 Unique students: {len(students)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild pipeline_results.csv from individual pipeline run directories"
    )
    parser.add_argument(
        '--results-dir',
        default='distillation_pairs_comparison',
        help='Directory containing pipeline_runs subdirectory'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV file path (default: results-dir/pipeline_results.csv)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Backup existing CSV before overwriting'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"❌ Error: {results_dir} does not exist!")
        sys.exit(1)
    
    # Backup existing CSV if requested
    if args.backup:
        existing_csv = results_dir / "pipeline_results.csv"
        if existing_csv.exists():
            backup_path = results_dir / f"pipeline_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            existing_csv.rename(backup_path)
            print(f"📦 Backed up existing CSV to: {backup_path}")
    
    success = rebuild_csv(results_dir, args.output)
    
    if not success:
        sys.exit(1)
    
    print("\n🎉 CSV rebuild complete!")


if __name__ == "__main__":
    main()
