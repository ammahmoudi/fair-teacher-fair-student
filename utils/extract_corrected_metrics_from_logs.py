#!/usr/bin/env python3

import os
import sys
import re
import csv
import logging
import ast
import math
from pathlib import Path
import glob

# Add project root to path for path utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils.path_utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def sanitize_metric(value):
    """Sanitize metric values to handle inf, nan, and numpy types."""
    try:
        # Handle string representations of numpy values like np.float32(13.443349)
        if isinstance(value, str) and "np.float" in value:
            # Extract the numeric value from inside parentheses
            match = re.search(r"\((.*?)\)", value)
            if match:
                value = match.group(1)

        val = float(value)
        if math.isinf(val) or math.isnan(val):
            return "NaN"
        return str(val)
    except (ValueError, TypeError):
        return "NaN"

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
            # Parse seed and model info from the seed directory name
            seed_parts = part.split('_')
            if len(seed_parts) >= 2:
                seed = seed_parts[1]
            
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

def extract_corrected_metrics_from_logs(base_dir):
    """Extract corrected metrics from log files and save to separate CSV files per experiment."""
    
    logging.info("Starting corrected metrics extraction from log files.")
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            experiment_dirs.append(item)
    
    logging.info(f"Found {len(experiment_dirs)} experiment directories: {experiment_dirs}")
    
    # Pattern to extract corrected metrics from log lines
    corrected_pattern = re.compile(
        r'\[([^\]]+)\] CORRECTED_METRICS: RMSE=([0-9.-]+), MAE=([0-9.-]+), MAPE=([0-9.-]+), POINTS=([0-9]+), EXPERIMENT=([^,]+), SEED=([^,]+), PATIENT=([^,]+), MODEL=(.+)'
    )
    log_datetime_pattern = re.compile(r"logs_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    
    csv_header = [
        'experiment_name', 'seed', 'model', 'patient', 'log_datetime', 
        'timestamp', 'rmse', 'mae', 'mape', 'total_points', 'log_file_path'
    ]
    
    results_per_experiment = {}
    created_files = []
    
    # Process each experiment directory
    for experiment_dir in experiment_dirs:
        experiment_path = os.path.join(base_dir, experiment_dir)
        output_csv = os.path.join(experiment_path, f"{experiment_dir}_corrected_metrics.csv")
        
        logging.info(f"Processing experiment: {experiment_dir}")
        
        # Find all log.log files in this experiment
        log_files = []
        for root, dirs, files in os.walk(experiment_path):
            if 'log.log' in files:
                log_files.append(os.path.join(root, 'log.log'))
        
        if not log_files:
            logging.warning(f"No log files found in {experiment_dir}")
            continue
            
        logging.info(f"Found {len(log_files)} log.log files in {experiment_dir}")
        
        # Track existing rows for this experiment
        existing_rows = set()
        file_exists = os.path.exists(output_csv)
        if file_exists and os.stat(output_csv).st_size > 0:
            with open(output_csv, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Read the header
                for row in reader:
                    existing_rows.add(tuple(row))  # Store existing rows
            logging.info(f"Loaded {len(existing_rows)} existing rows from {output_csv}.")
        
        # Extract results for this experiment
        results = []
        
        for log_file in log_files:
            # Extract log datetime from the log folder path
            log_datetime_match = log_datetime_pattern.search(log_file)
            log_datetime = log_datetime_match.group(1) if log_datetime_match else "Unknown"
            
            # Parse experiment information from path
            experiment_info = parse_experiment_path(log_file)
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if 'CORRECTED_METRICS:' in line:
                            match = corrected_pattern.search(line.strip())
                            if match:
                                timestamp, rmse, mae, mape, points, experiment, seed, patient, model = match.groups()
                                
                                # Create result row
                                result_row = (
                                    str(experiment_info['experiment_name']),
                                    str(experiment_info['seed']),
                                    str(experiment_info['model']),
                                    str(experiment_info['patient']),
                                    str(log_datetime),
                                    str(timestamp),
                                    sanitize_metric(rmse),
                                    sanitize_metric(mae),
                                    sanitize_metric(mape),
                                    str(points),
                                    str(log_file)
                                )
                                
                                if result_row not in existing_rows:  # Avoid duplicates
                                    results.append(result_row)
                                    existing_rows.add(result_row)
                                else:
                                    logging.debug("Duplicate entry found, skipping.")
                            else:
                                logging.warning(f"Could not parse CORRECTED_METRICS line in {log_file}:{line_num}: {line.strip()}")
                                
            except Exception as e:
                logging.error(f"Error processing {log_file}: {str(e)}")
        
        # Save results to CSV for this experiment
        if results or not file_exists:
            with open(output_csv, "w" if not file_exists else "a", newline="") as f:
                writer = csv.writer(f)
                
                # Write header only if file is new or empty
                if not file_exists or os.stat(output_csv).st_size == 0:
                    writer.writerow(csv_header)
                    logging.info(f"CSV header written for {experiment_dir}")
                
                if results:  # Only write if there are new results
                    writer.writerows(results)
                    logging.info(f"Appended {len(results)} new rows to {output_csv}")
                else:
                    logging.info(f"No new data to append for {experiment_dir}")
            
            results_per_experiment[experiment_dir] = len(results)
            created_files.append(output_csv)
            logging.info(f"Results for {experiment_dir} saved to {output_csv}")
        else:
            logging.warning(f"No results found for {experiment_dir}")
    
    return created_files, results_per_experiment

def main():
    """Main function to extract corrected metrics from experiments."""
    
    experiments_dir = str(get_project_root() / "experiments")
    
    if not os.path.exists(experiments_dir):
        logging.error(f"Experiments directory not found: {experiments_dir}")
        return
    
    print("Extracting corrected metrics from log files...")
    print("Generating CSV files in each experiment folder...")
    
    created_files, results_per_experiment = extract_corrected_metrics_from_logs(experiments_dir)
    
    if created_files:
        print(f"\n📁 Created {len(created_files)} CSV files:")
        for file_path in created_files:
            filename = os.path.basename(file_path)
            experiment_name = filename.replace('_corrected_metrics.csv', '')
            record_count = results_per_experiment.get(experiment_name, 0)
            print(f"  📄 {filename}: {record_count} records")
        
        # Display overall summary statistics
        import pandas as pd
        try:
            all_dfs = []
            total_records = 0
            
            for file_path in created_files:
                if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                    total_records += len(df)
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                print(f"\n📊 Overall Summary Statistics:")
                print(f"Total records across all experiments: {total_records}")
                print(f"Unique experiments: {combined_df['experiment_name'].nunique()}")
                print(f"Unique seeds: {combined_df['seed'].nunique()}")
                print(f"Unique patients: {combined_df['patient'].nunique()}")
                print(f"Unique models: {combined_df['model'].nunique()}")
                
                print(f"\n📈 Experiment breakdown:")
                exp_counts = combined_df['experiment_name'].value_counts()
                for exp, count in exp_counts.items():
                    print(f"  {exp}: {count} records")
                
                # Convert numeric columns for statistics
                numeric_cols = ['rmse', 'mae', 'mape']
                for col in numeric_cols:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                
                print(f"\n📏 Metrics overview:")
                for col in numeric_cols:
                    valid_values = combined_df[col].dropna()
                    if len(valid_values) > 0:
                        print(f"  {col.upper()}: {valid_values.min():.3f} - {valid_values.max():.3f} (avg: {valid_values.mean():.3f})")
                    else:
                        print(f"  {col.upper()}: No valid values")
            
        except Exception as e:
            print(f"Error generating summary: {e}")
        
        print(f"\n✅ Extraction complete! CSV files saved in their respective experiment folders")
        print(f"📂 Experiments directory: {experiments_dir}")
    else:
        print("❌ No corrected metrics found to extract.")

if __name__ == "__main__":
    main()