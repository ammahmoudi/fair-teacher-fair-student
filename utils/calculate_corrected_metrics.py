#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from datetime import datetime
import glob

# Add the parent directory to Python path to import metrics  
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.path_utils import get_project_root

def process_raw_corrected_csv(csv_path):
    """Process a single raw corrected CSV file and calculate metrics."""
    print(f"Processing: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get all column names
        columns = df.columns.tolist()
        
        # Find true and pred column pairs
        true_cols = [col for col in columns if col.endswith('_true')]
        pred_cols = [col for col in columns if col.endswith('_pred')]
        
        # Sort to ensure matching pairs
        true_cols.sort()
        pred_cols.sort()
        
        print(f"Found {len(true_cols)} true/pred column pairs")
        
        if len(true_cols) != len(pred_cols):
            print(f"Warning: Mismatch in true/pred columns: {len(true_cols)} vs {len(pred_cols)}")
            return None
        
        # Collect all true and predicted values
        all_true_values = []
        all_pred_values = []
        
        for true_col, pred_col in zip(true_cols, pred_cols):
            # Ensure they correspond to the same time step
            true_timestep = true_col.split('_')[1]
            pred_timestep = pred_col.split('_')[1]
            
            if true_timestep != pred_timestep:
                print(f"Warning: Timestep mismatch: {true_col} vs {pred_col}")
                continue
            
            # Collect values
            true_values = df[true_col].values
            pred_values = df[pred_col].values
            
            all_true_values.extend(true_values)
            all_pred_values.extend(pred_values)
        
        # Convert to numpy arrays
        all_true_values = np.array(all_true_values, dtype=float)
        all_pred_values = np.array(all_pred_values, dtype=float)
        
        print(f"Total data points: {len(all_true_values)}")
        
        # Calculate metrics using existing functions
        rmse = calculate_rmse(all_pred_values, all_true_values)
        mae = calculate_mae(all_pred_values, all_true_values)  
        mape = calculate_mape(all_pred_values, all_true_values) * 100  # Convert to percentage
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'total_points': len(all_true_values)
        }
        
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def log_corrected_metrics(log_dir, metrics, experiment_info):
    """Log the corrected metrics to the existing log.log file."""
    log_file = os.path.join(log_dir, 'log.log')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = (
        f"[{timestamp}] CORRECTED_METRICS: "
        f"RMSE={metrics['rmse']:.6f}, "
        f"MAE={metrics['mae']:.6f}, "
        f"MAPE={metrics['mape']:.6f}, "
        f"POINTS={metrics['total_points']}, "
        f"EXPERIMENT={experiment_info['experiment_name']}, "
        f"SEED={experiment_info['seed']}, "
        f"PATIENT={experiment_info['patient']}, "
        f"MODEL={experiment_info['model']}\n"
    )
    
    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(f"Logged corrected metrics to: {log_file}")
    return log_file

def parse_experiment_path(experiment_path):
    """Parse experiment path to extract information."""
    path_parts = Path(experiment_path).parts
    
    # Find experiment directory name
    experiment_name = None
    for part in path_parts:
        if 'chronos' in part or 'time_llm' in part:
            experiment_name = part
            break
    
    # Extract seed information
    seed = None
    model = None
    patient = None
    
    for part in path_parts:
        if part.startswith('seed_'):
            # Parse seed and model info
            seed_part = part.split('_')
            if len(seed_part) >= 2:
                seed = seed_part[1]
            
            # Extract model info from the seed directory name
            if 'model_' in part:
                model_start = part.find('model_') + 6
                model_end = part.find('_', model_start)
                if model_end != -1:
                    model = part[model_start:model_end]
                else:
                    model = part[model_start:]
        
        if part.startswith('patient_'):
            patient = part.split('_')[1]
    
    return {
        'experiment_name': experiment_name or 'unknown',
        'seed': seed or 'unknown',
        'model': model or 'unknown', 
        'patient': patient or 'unknown'
    }

def process_all_experiments(experiments_root=None):
    """Process all experiments with raw corrected CSV files."""
    
    if experiments_root is None:
        experiments_dir = str(get_project_root() / "experiments")
    else:
        experiments_dir = experiments_root
    
    # Find all raw corrected CSV files
    chronos_files = glob.glob(os.path.join(experiments_dir, '*chronos*', '*', '*', 'logs', '*', 'raw_corrected_final_results.csv'))
    time_llm_files = glob.glob(os.path.join(experiments_dir, '*time_llm*', '*', '*', 'logs', '*', 'raw_corrected_inference_results_reformatted.csv'))
    
    all_files = chronos_files + time_llm_files
    
    print(f"Found {len(chronos_files)} Chronos raw corrected files")
    print(f"Found {len(time_llm_files)} Time-LLM raw corrected files")
    print(f"Total: {len(all_files)} files to process")
    
    processed_count = 0
    
    for csv_path in all_files:
        print(f"\n{'='*80}")
        print(f"Processing file {processed_count + 1}/{len(all_files)}")
        
        # Parse experiment information
        experiment_info = parse_experiment_path(csv_path)
        print(f"Experiment: {experiment_info}")
        
        # Process the CSV file
        metrics = process_raw_corrected_csv(csv_path)
        
        if metrics is not None:
            # Get the log directory (same as CSV directory)
            log_dir = os.path.dirname(csv_path)
            
            # Log the metrics
            log_file = log_corrected_metrics(log_dir, metrics, experiment_info)
            
            print(f"Corrected Metrics:")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MAPE: {metrics['mape']:.6f}")
            print(f"  Points: {metrics['total_points']}")
            
            processed_count += 1
        else:
            print(f"Failed to process: {csv_path}")
    
    print(f"\n{'='*80}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count}/{len(all_files)} files")
    
    return processed_count

def process_single_experiment(experiment_dir):
    """Process corrected metrics for a single experiment directory."""
    
    print(f"Processing single experiment: {experiment_dir}")
    
    # Find raw corrected CSV files in this experiment
    chronos_files = glob.glob(os.path.join(experiment_dir, '*', '*', 'logs', '*', 'raw_corrected_final_results.csv'))
    time_llm_files = glob.glob(os.path.join(experiment_dir, '*', '*', 'logs', '*', 'raw_corrected_inference_results_reformatted.csv'))
    
    all_files = chronos_files + time_llm_files
    
    if not all_files:
        print(f"No raw corrected CSV files found in {experiment_dir}")
        return 0
    
    print(f"Found {len(all_files)} raw corrected files to process")
    
    processed_count = 0
    
    for csv_path in all_files:
        # Parse experiment information
        experiment_info = parse_experiment_path(csv_path)
        
        # Process the CSV file
        metrics = process_raw_corrected_csv(csv_path)
        
        if metrics is not None:
            # Get the log directory (same as CSV directory)
            log_dir = os.path.dirname(csv_path)
            
            # Log the metrics
            log_file = log_corrected_metrics(log_dir, metrics, experiment_info)
            
            processed_count += 1
        else:
            print(f"Failed to process: {csv_path}")
    
    print(f"Successfully processed: {processed_count}/{len(all_files)} files in {experiment_dir}")
    return processed_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate corrected metrics for experiments')
    parser.add_argument('--experiment_dir', type=str, help='Specific experiment directory to process')
    parser.add_argument('--experiments_root', type=str, default=None,
                       help='Root experiments directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set default experiments root if not provided
    if not args.experiments_root:
        args.experiments_root = str(get_project_root() / "experiments")
    
    if args.experiment_dir:
        # Process single experiment directory
        print(f"Starting corrected metrics calculation for experiment: {args.experiment_dir}")
        processed = process_single_experiment(args.experiment_dir)
        print(f"\nProcessing complete! Calculated corrected metrics for {processed} files.")
    else:
        # Process all experiments
        print("Starting corrected metrics calculation for all experiments...")
        print(f"Experiments root: {args.experiments_root}")
        processed = process_all_experiments(args.experiments_root)
        print(f"\nProcessing complete! Calculated corrected metrics for {processed} total files.")