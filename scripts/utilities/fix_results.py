"""
Result correction utilities for fixing outliers in prediction results.
Extracted from fix_chronos.py to provide modular outlier detection and correction.
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

def calculate_rmse(prediction, ground_truth):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))

def calculate_mae(prediction, ground_truth):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(prediction - ground_truth))

def calculate_mape(prediction, ground_truth):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((prediction - ground_truth) / ground_truth)) * 100

def replace_bad_predictions(df, pred_columns, file_name, threshold_factor=3):
    """
    Detects and replaces extremely bad predictions using a threshold factor.

    Parameters:
        df (pd.DataFrame): The dataframe containing prediction values.
        pred_columns (list): List of column names with predicted values.
        file_name (str): Name of the CSV file being processed.
        threshold_factor (float): Multiplier for defining a bad prediction threshold.

    Returns:
        tuple: (corrected_df, corrections_made) where corrections_made is a list of correction details
    """
    corrections_made = []
    
    for col in pred_columns:
        median_val = np.median(df[col].values)
        threshold = median_val * threshold_factor  # Define the extreme value threshold

        for i in range(1, len(df[col]) - 1):  # Avoid first and last row
            if df.at[i, col] > threshold:  # If the value is way too high
                original_value = df.at[i, col]
                new_value = (df.at[i - 1, col] + df.at[i + 1, col]) / 2
                
                correction_detail = {
                    'file': file_name,
                    'column': col,
                    'row': i,
                    'original': original_value,
                    'corrected': new_value,
                    'threshold': threshold
                }
                corrections_made.append(correction_detail)
                
                logging.info(f"Fixed bad prediction in {file_name} | Column: {col} | Row: {i} | "
                             f"Original: {original_value:.4f} -> New: {new_value:.4f} (Threshold: {threshold:.4f})")
                df.at[i, col] = new_value  # Replace with average of neighbors

    return df, corrections_made

def fix_experiment_results(experiment_dir, threshold_factor=3):
    """
    Fixes outliers in experiment results and returns corrected metrics.
    
    Parameters:
        experiment_dir (str): Directory containing experiment results
        threshold_factor (float): Multiplier for outlier detection threshold
        
    Returns:
        dict: Dictionary containing original metrics, corrected metrics, and correction details
    """
    experiment_path = Path(experiment_dir)
    results = {
        'original_metrics': None,
        'corrected_metrics': None,
        'corrections_made': [],
        'files_processed': [],
        'success': False
    }
    
    try:
        # Find CSV files that match the pattern
        csv_files = []
        for csv_file in experiment_path.rglob("*_reformatted.csv"):
            if not csv_file.name.startswith("smoothed_"):
                csv_files.append(csv_file)
        
        if not csv_files:
            logging.warning(f"No suitable CSV files found in {experiment_dir}")
            return results
        
        all_original_metrics = []
        all_corrected_metrics = []
        
        for csv_path in csv_files:
            logging.info(f"Processing {csv_path}")
            
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Identify true and predicted columns
            true_columns = [col for col in df.columns if "_true" in col]
            pred_columns = [col for col in df.columns if "_pred" in col and not col.startswith("smoothed_")]
            
            if not true_columns or not pred_columns:
                logging.warning(f"Skipping {csv_path}: Missing required columns.")
                continue
            
            # Compute original metrics
            original_rmse = calculate_rmse(df[pred_columns].values, df[true_columns].values)
            original_mae = calculate_mae(df[pred_columns].values, df[true_columns].values)
            original_mape = calculate_mape(df[pred_columns].values, df[true_columns].values)
            
            original_metrics = {
                "rmse": round(original_rmse, 5),
                "mae": round(original_mae, 5),
                "mape": round(original_mape, 5),
            }
            
            # Correct bad predictions
            df_corrected, corrections = replace_bad_predictions(
                df.copy(), pred_columns, csv_path.name, threshold_factor
            )
            
            # Compute corrected metrics
            corrected_rmse = calculate_rmse(df_corrected[pred_columns].values, df_corrected[true_columns].values)
            corrected_mae = calculate_mae(df_corrected[pred_columns].values, df_corrected[true_columns].values)
            corrected_mape = calculate_mape(df_corrected[pred_columns].values, df_corrected[true_columns].values)
            
            corrected_metrics = {
                "rmse": round(corrected_rmse, 5),
                "mae": round(corrected_mae, 5),
                "mape": round(corrected_mape, 5),
            }
            
            # Save corrected data
            final_results_path = csv_path.parent / "final_results.csv"
            df_corrected.to_csv(final_results_path, index=False)
            logging.info(f"Saved corrected results to {final_results_path}")
            
            # Store results
            all_original_metrics.append(original_metrics)
            all_corrected_metrics.append(corrected_metrics)
            results['corrections_made'].extend(corrections)
            results['files_processed'].append(str(csv_path))
            
            # Update experiment log with both original and corrected metrics
            log_file = csv_path.parent / "log.log"
            update_experiment_log(log_file, original_metrics, corrected_metrics)
        
        if all_original_metrics:
            # Average metrics across all processed files
            results['original_metrics'] = {
                'rmse': round(np.mean([m['rmse'] for m in all_original_metrics]), 5),
                'mae': round(np.mean([m['mae'] for m in all_original_metrics]), 5),
                'mape': round(np.mean([m['mape'] for m in all_original_metrics]), 5)
            }
            
            results['corrected_metrics'] = {
                'rmse': round(np.mean([m['rmse'] for m in all_corrected_metrics]), 5),
                'mae': round(np.mean([m['mae'] for m in all_corrected_metrics]), 5),
                'mape': round(np.mean([m['mape'] for m in all_corrected_metrics]), 5)
            }
            
            results['success'] = True
            
            logging.info(f"Outlier correction completed for {experiment_dir}")
            logging.info(f"Original metrics: {results['original_metrics']}")
            logging.info(f"Corrected metrics: {results['corrected_metrics']}")
            logging.info(f"Total corrections made: {len(results['corrections_made'])}")
    
    except Exception as e:
        logging.error(f"Error fixing results in {experiment_dir}: {str(e)}")
        results['error'] = str(e)
    
    return results

def update_experiment_log(log_file, original_metrics, corrected_metrics):
    """
    Updates the experiment log file with metric results for both original and corrected predictions.
    
    Parameters:
        log_file (Path): Path to the log file.
        original_metrics (dict): Metrics for original predictions.
        corrected_metrics (dict): Metrics for corrected predictions.
    """
    try:
        if not log_file.exists():
            logging.warning(f"Log file {log_file} not found. Creating a new one.")
            log_file.touch()
        
        with open(log_file, "a") as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"{timestamp} | INFO | Outlier Correction Results | "
                f"Original Metrics: {original_metrics} | Corrected Metrics: {corrected_metrics}\n"
            )
            f.write(log_entry)
        
        logging.info(f"Updated log file: {log_file} with correction metrics.")
        
    except Exception as e:
        logging.error(f"Failed to update log file {log_file}: {str(e)}")
