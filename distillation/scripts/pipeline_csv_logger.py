#!/usr/bin/env python3
"""
Pipeline CSV Logger for Knowledge Distillation Experiments
Logs all pipeline run data and metrics to a centralized CSV file.
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse


class PipelineCSVLogger:
    """Handles CSV logging for complete pipeline runs with all phases and metrics."""
    
    def __init__(self, csv_file_path: str = "distillation_experiments/pipeline_results.csv"):
        """
        Initialize the CSV logger.
        
        Args:
            csv_file_path: Path to the CSV file for logging results
        """
        self.csv_file_path = csv_file_path
        self.ensure_csv_directory()
        self.ensure_csv_headers()
    
    def ensure_csv_directory(self):
        """Ensure the directory for the CSV file exists."""
        os.makedirs(os.path.dirname(self.csv_file_path), exist_ok=True)
    
    def get_csv_headers(self) -> List[str]:
        """Get the standardized CSV headers for pipeline logging."""
        return [
            # Run metadata
            'timestamp',
            'pipeline_id',
            'pipeline_dir',
            
            # Run parameters
            'patient_ids',
            'dataset_name',
            'seed',
            'teacher_model',
            'student_model',
            
            # Training hyperparameters
            'learning_rate',
            'batch_size',
            'teacher_epochs',
            'student_epochs',
            'distill_epochs',
            
            # Distillation hyperparameters
            'alpha',
            'beta',
            
            # Phase 1: Teacher metrics
            'teacher_rmse',
            'teacher_mae', 
            'teacher_mape',
            'teacher_training_time',
            'teacher_status',
            
            # Phase 2: Student baseline metrics
            'student_baseline_rmse',
            'student_baseline_mae',
            'student_baseline_mape',
            'student_training_time',
            'student_status',
            
            # Phase 3: Distillation metrics
            'distilled_rmse',
            'distilled_mae',
            'distilled_mape',
            'distillation_time',
            'distillation_status',
            
            # Performance improvements
            'teacher_to_distilled_rmse_improvement',
            'teacher_to_distilled_rmse_improvement_pct',
            'student_to_distilled_rmse_improvement',
            'student_to_distilled_rmse_improvement_pct',
            
            # Overall pipeline
            'pipeline_status',
            'total_runtime',
            'notes'
        ]
    
    def ensure_csv_headers(self):
        """Ensure the CSV file exists with proper headers."""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.get_csv_headers())
            print(f"✓ Created new CSV log file: {self.csv_file_path}")
    
    def load_metrics_from_json(self, json_file_path: str) -> Dict[str, Any]:
        """Load metrics from a JSON summary file."""
        try:
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                return data
            else:
                print(f"⚠️  Metrics file not found: {json_file_path}")
                return {}
        except Exception as e:
            print(f"❌ Error loading metrics from {json_file_path}: {e}")
            return {}
    
    def extract_metrics_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from loaded JSON data."""
        metrics = {}
        
        # Try to extract performance metrics
        if 'performance_metrics' in data:
            perf = data['performance_metrics']
            metrics['rmse'] = perf.get('rmse', None)
            metrics['mae'] = perf.get('mae', None)
            metrics['mape'] = perf.get('mape', None)
        
        # Try to extract training time
        if 'training_time' in data:
            metrics['training_time'] = data['training_time']
        elif 'total_time' in data:
            metrics['training_time'] = data['total_time']
        
        # Extract status
        metrics['status'] = data.get('status', 'unknown')
        
        return metrics
    
    def calculate_improvements(self, teacher_rmse: float, student_rmse: float, distilled_rmse: float) -> Dict[str, float]:
        """Calculate performance improvements."""
        improvements = {}
        
        if teacher_rmse and distilled_rmse:
            improvements['teacher_to_distilled_rmse_improvement'] = teacher_rmse - distilled_rmse
            improvements['teacher_to_distilled_rmse_improvement_pct'] = ((teacher_rmse - distilled_rmse) / teacher_rmse) * 100
        
        if student_rmse and distilled_rmse:
            improvements['student_to_distilled_rmse_improvement'] = student_rmse - distilled_rmse
            improvements['student_to_distilled_rmse_improvement_pct'] = ((student_rmse - distilled_rmse) / student_rmse) * 100
        
        return improvements
    
    def log_pipeline_run(self, 
                        pipeline_dir: str,
                        run_params: Dict[str, Any],
                        teacher_metrics_file: Optional[str] = None,
                        student_metrics_file: Optional[str] = None,
                        distillation_metrics_file: Optional[str] = None,
                        total_runtime: Optional[float] = None,
                        notes: str = ""):
        """
        Log a complete pipeline run to CSV.
        
        Args:
            pipeline_dir: Directory containing the pipeline run
            run_params: Dictionary of run parameters (from pipeline script)
            teacher_metrics_file: Path to teacher training summary JSON
            student_metrics_file: Path to student training summary JSON  
            distillation_metrics_file: Path to distillation summary JSON
            total_runtime: Total pipeline runtime in seconds
            notes: Additional notes about the run
        """
        
        # Load metrics from each phase
        teacher_data = self.load_metrics_from_json(teacher_metrics_file) if teacher_metrics_file else {}
        student_data = self.load_metrics_from_json(student_metrics_file) if student_metrics_file else {}
        distillation_data = self.load_metrics_from_json(distillation_metrics_file) if distillation_metrics_file else {}
        
        # Extract metrics
        teacher_metrics = self.extract_metrics_from_data(teacher_data)
        student_metrics = self.extract_metrics_from_data(student_data)
        distillation_metrics = self.extract_metrics_from_data(distillation_data)
        
        # Calculate improvements
        teacher_rmse = teacher_metrics.get('rmse')
        student_rmse = student_metrics.get('rmse')
        distilled_rmse = distillation_metrics.get('rmse')
        
        improvements = self.calculate_improvements(teacher_rmse, student_rmse, distilled_rmse)
        
        # Determine overall pipeline status
        pipeline_status = "SUCCESS"
        if teacher_metrics.get('status') == 'failed' or student_metrics.get('status') == 'failed' or distillation_metrics.get('status') == 'failed':
            pipeline_status = "FAILED"
        elif not teacher_metrics or not student_metrics or not distillation_metrics:
            pipeline_status = "INCOMPLETE"
        
        # Create the row data
        row_data = {
            # Run metadata
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'pipeline_id': os.path.basename(pipeline_dir),
            'pipeline_dir': pipeline_dir,
            
            # Run parameters
            'patient_ids': run_params.get('patients', ''),
            'dataset_name': run_params.get('dataset_name', ''),
            'seed': run_params.get('seed', ''),
            'teacher_model': run_params.get('teacher', ''),
            'student_model': run_params.get('student', ''),
            
            # Training hyperparameters
            'learning_rate': run_params.get('lr', ''),
            'batch_size': run_params.get('batch_size', ''),
            'teacher_epochs': run_params.get('teacher_epochs', ''),
            'student_epochs': run_params.get('student_epochs', ''),
            'distill_epochs': run_params.get('distill_epochs', ''),
            
            # Distillation hyperparameters
            'alpha': run_params.get('alpha', ''),
            'beta': run_params.get('beta', ''),
            
            # Phase 1: Teacher metrics
            'teacher_rmse': teacher_metrics.get('rmse', ''),
            'teacher_mae': teacher_metrics.get('mae', ''),
            'teacher_mape': teacher_metrics.get('mape', ''),
            'teacher_training_time': teacher_metrics.get('training_time', ''),
            'teacher_status': teacher_metrics.get('status', ''),
            
            # Phase 2: Student baseline metrics
            'student_baseline_rmse': student_metrics.get('rmse', ''),
            'student_baseline_mae': student_metrics.get('mae', ''),
            'student_baseline_mape': student_metrics.get('mape', ''),
            'student_training_time': student_metrics.get('training_time', ''),
            'student_status': student_metrics.get('status', ''),
            
            # Phase 3: Distillation metrics
            'distilled_rmse': distillation_metrics.get('rmse', ''),
            'distilled_mae': distillation_metrics.get('mae', ''),
            'distilled_mape': distillation_metrics.get('mape', ''),
            'distillation_time': distillation_metrics.get('training_time', ''),
            'distillation_status': distillation_metrics.get('status', ''),
            
            # Performance improvements
            'teacher_to_distilled_rmse_improvement': improvements.get('teacher_to_distilled_rmse_improvement', ''),
            'teacher_to_distilled_rmse_improvement_pct': improvements.get('teacher_to_distilled_rmse_improvement_pct', ''),
            'student_to_distilled_rmse_improvement': improvements.get('student_to_distilled_rmse_improvement', ''),
            'student_to_distilled_rmse_improvement_pct': improvements.get('student_to_distilled_rmse_improvement_pct', ''),
            
            # Overall pipeline
            'pipeline_status': pipeline_status,
            'total_runtime': total_runtime or '',
            'notes': notes
        }
        
        # Append to CSV
        headers = self.get_csv_headers()
        row_values = [row_data.get(header, '') for header in headers]
        
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_values)
        
        print(f"✓ Pipeline run logged to CSV: {self.csv_file_path}")
        print(f"  📊 Pipeline Status: {pipeline_status}")
        if teacher_rmse:
            print(f"  📊 Teacher RMSE: {teacher_rmse:.4f}")
        if student_rmse:
            print(f"  📊 Student Baseline RMSE: {student_rmse:.4f}")
        if distilled_rmse:
            print(f"  📊 Distilled RMSE: {distilled_rmse:.4f}")
        if improvements.get('teacher_to_distilled_rmse_improvement_pct'):
            print(f"  📊 Improvement vs Teacher: {improvements['teacher_to_distilled_rmse_improvement_pct']:.2f}%")
    
    def get_pipeline_summary(self) -> pd.DataFrame:
        """Get summary statistics of all logged pipeline runs."""
        if not os.path.exists(self.csv_file_path):
            print(f"❌ CSV file not found: {self.csv_file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.csv_file_path)
        return df
    
    def print_recent_runs(self, n: int = 5):
        """Print summary of recent pipeline runs."""
        df = self.get_pipeline_summary()
        if df.empty:
            print("❌ No pipeline runs logged yet.")
            return
        
        print(f"\n📊 Recent {min(n, len(df))} Pipeline Runs:")
        print("=" * 80)
        
        recent = df.tail(n)
        for _, row in recent.iterrows():
            print(f"🔹 {row['timestamp']} | {row['teacher_model']}→{row['student_model']} | Patients: {row['patient_ids']}")
            print(f"   Teacher: {row['teacher_rmse']:.4f} | Student: {row['student_baseline_rmse']:.4f} | Distilled: {row['distilled_rmse']:.4f}")
            if row['teacher_to_distilled_rmse_improvement_pct']:
                print(f"   Improvement: {row['teacher_to_distilled_rmse_improvement_pct']:.2f}% | Status: {row['pipeline_status']}")
            print()


def main():
    """CLI interface for the pipeline CSV logger."""
    parser = argparse.ArgumentParser(description="Log pipeline run results to CSV")
    parser.add_argument("--pipeline-dir", required=True, help="Pipeline directory path")
    parser.add_argument("--csv-file", default="distillation_experiments/pipeline_results.csv", help="CSV file path")
    parser.add_argument("--teacher-metrics", help="Teacher training metrics JSON file")
    parser.add_argument("--student-metrics", help="Student training metrics JSON file")
    parser.add_argument("--distillation-metrics", help="Distillation metrics JSON file")
    parser.add_argument("--total-runtime", type=float, help="Total pipeline runtime in seconds")
    parser.add_argument("--notes", default="", help="Additional notes")
    
    # Run parameters
    parser.add_argument("--patients", help="Patient IDs")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--seed", help="Random seed")
    parser.add_argument("--teacher", help="Teacher model name")
    parser.add_argument("--student", help="Student model name")
    parser.add_argument("--lr", help="Learning rate")
    parser.add_argument("--batch-size", help="Batch size")
    parser.add_argument("--teacher-epochs", help="Teacher epochs")
    parser.add_argument("--student-epochs", help="Student epochs")
    parser.add_argument("--distill-epochs", help="Distillation epochs")
    parser.add_argument("--alpha", help="Distillation alpha")
    parser.add_argument("--beta", help="Distillation beta")
    
    # Actions
    parser.add_argument("--summary", action="store_true", help="Show recent runs summary")
    
    args = parser.parse_args()
    
    logger = PipelineCSVLogger(args.csv_file)
    
    if args.summary:
        logger.print_recent_runs()
        return
    
    # Build run parameters dictionary
    run_params = {
        'patients': args.patients,
        'dataset_name': args.dataset,
        'seed': args.seed,
        'teacher': args.teacher,
        'student': args.student,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'teacher_epochs': args.teacher_epochs,
        'student_epochs': args.student_epochs,
        'distill_epochs': args.distill_epochs,
        'alpha': args.alpha,
        'beta': args.beta,
    }
    
    # Log the pipeline run
    logger.log_pipeline_run(
        pipeline_dir=args.pipeline_dir,
        run_params=run_params,
        teacher_metrics_file=args.teacher_metrics,
        student_metrics_file=args.student_metrics,
        distillation_metrics_file=args.distillation_metrics,
        total_runtime=args.total_runtime,
        notes=args.notes
    )


if __name__ == "__main__":
    main()