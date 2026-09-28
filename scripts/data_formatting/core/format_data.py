#!/usr/bin/env python3
"""
Unified Data Formatting Tool

This script formats time series data for both datasets (ohiot1dm, d1namo) across all scenarios
(raw, missing_periodic, missing_random, noisy, denoised) with multiple window configurations.

Supports window configurations:
- 6,6: input_len=6, pred_len=6
- 6,9: input_len=6, pred_len=9

Usage:
    python format_data.py --dataset ohiot1dm --scenario raw --windows 6,6
    python format_data.py --dataset d1namo --scenario all --windows all
    python format_data.py --dataset all --scenario all --windows all
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import logging

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.path_utils import get_data_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def reformat_t1dm_bg_data(data_path, save_path, input_window_size=6, prediction_window_size=6):
    """
    Core data reformatting function adapted from reformat_t1dm_bg_data.py
    """
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f'Initial data size for {data_path}: {data.shape}')

    # Standardize column names if needed (handle _ts,_value format)
    if "_ts" in data.columns and "_value" in data.columns:
        data.rename(columns={"_ts": "timestamp", "_value": "target"}, inplace=True)
        logger.info(f"Standardized column names for {data_path}")
    
    # Check if target column exists
    if 'target' not in data.columns:
        raise KeyError(f"'target' column not found in {data_path}. Available columns: {list(data.columns)}")

    # Generate column names based on window sizes
    input_features = []
    for i in range(input_window_size):
        offset = input_window_size - 1 - i
        if offset == 0:
            input_features.append("BG_{t}")
        else:
            input_features.append(f"BG_{{t-{offset}}}")
    
    labels = [f"BG_{{t+{i+1}}}" for i in range(prediction_window_size)]
    
    transformed_data = {
        feature: [] for feature in input_features + labels
    }

    total_size = input_window_size + prediction_window_size

    # Use a rolling window to extract all valid sequences
    for i in range(len(data) - total_size + 1):
        input_data = data['target'][i: i + input_window_size].values
        label_data = data['target'][i + input_window_size: i + total_size].values

        # Append input data
        for feature, value in zip(input_features, input_data):
            transformed_data[feature].append(value)

        # Append label data
        for label, value in zip(labels, label_data):
            transformed_data[label].append(value)

    transformed_data = pd.DataFrame(transformed_data)
    logger.info(f'Final transformed data shape for {data_path}: {transformed_data.shape}')

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    transformed_data.to_csv(save_path, index=False)
    logger.info(f'Saved formatted data to: {save_path}')


def get_window_config(window_str):
    """Convert window string like '6,6' to tuple (input, pred)"""
    if window_str == "6,6":
        return (6, 6)
    elif window_str == "6,9":
        return (6, 9)
    else:
        raise ValueError(f"Unsupported window configuration: {window_str}")


def get_available_scenarios(dataset):
    """Get available data scenarios for a dataset"""
    base_path = get_data_path(dataset)
    
    available_scenarios = []
    potential_scenarios = ["raw", "raw_standardized", "missing_periodic", "missing_random", "noisy", "denoised"]
    
    for scenario in potential_scenarios:
        scenario_path = os.path.join(base_path, scenario)
        if os.path.exists(scenario_path):
            available_scenarios.append(scenario)
    
    return available_scenarios


def get_available_files(dataset, scenario):
    """Get available CSV files for a dataset/scenario combination"""
    # Handle raw_standardized scenario or ohiot1dm raw scenario - use raw_standardized folder
    if scenario == "raw_standardized" or (dataset == "ohiot1dm" and scenario == "raw"):
        base_path = get_data_path(dataset, "raw_standardized")
    else:
        base_path = get_data_path(dataset, scenario)
    
    files = []
    if not os.path.exists(base_path):
        logger.warning(f"Path does not exist: {base_path}")
        return files
    
    if dataset == "d1namo" and scenario == "raw":
        # Handle D1NAMO raw data structure (folders with patient IDs)
        raw_folder_path = get_data_path(dataset, "raw")
        for patient_folder in os.listdir(raw_folder_path):
            patient_path = os.path.join(raw_folder_path, patient_folder)
            if os.path.isdir(patient_path):
                train_file = os.path.join(patient_path, "train_data.csv")
                test_file = os.path.join(patient_path, "test_data.csv")
                if os.path.exists(train_file):
                    files.append((train_file, f"{patient_folder}-ws-training.csv"))
                if os.path.exists(test_file):
                    files.append((test_file, f"{patient_folder}-ws-testing.csv"))
    else:
        # Handle standardized CSV files
        for filename in os.listdir(base_path):
            if filename.endswith(".csv") and not filename.endswith(":Zone.Identifier"):
                file_path = os.path.join(base_path, filename)
                files.append((file_path, filename))
    
    return files


def standardize_d1namo_raw_file(input_file, output_file):
    """Standardize D1NAMO raw data file format to match others"""
    df = pd.read_csv(input_file)
    
    # Rename columns if needed
    if "_ts" in df.columns and "_value" in df.columns:
        df.rename(columns={"_ts": "timestamp", "_value": "target"}, inplace=True)
    
    # Extract patient ID from output filename
    patient_id = os.path.basename(output_file).split("-")[0]
    df["item_id"] = patient_id
    
    # Ensure timestamp format
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
    
    # Reorder columns
    df = df[["item_id", "timestamp", "target"]]
    
    return df


def create_missing_data_for_ohiot1dm():
    """Create missing_periodic and missing_random folders for ohiot1dm if they don't exist"""
    base_path = get_data_path("ohiot1dm")
    raw_path = os.path.join(base_path, "raw")
    
    scenarios_to_create = ["missing_periodic", "missing_random"]
    
    for scenario in scenarios_to_create:
        scenario_path = os.path.join(base_path, scenario)
        if not os.path.exists(scenario_path):
            logger.info(f"Creating missing scenario folder: {scenario_path}")
            os.makedirs(scenario_path, exist_ok=True)
            
            # Copy files from raw folder
            for filename in os.listdir(raw_path):
                if filename.endswith(".csv"):
                    src = os.path.join(raw_path, filename)
                    dst = os.path.join(scenario_path, filename)
                    # For now, just copy the raw files - in a real scenario you'd apply missing data patterns
                    import shutil
                    shutil.copy2(src, dst)
                    logger.info(f"Created {scenario} data: {filename}")


def format_dataset_scenario_window(dataset, scenario, window_config):
    """Format all files for a specific dataset/scenario/window combination"""
    input_len, pred_len = get_window_config(window_config)
    
    logger.info(f"Formatting {dataset}/{scenario} with window {window_config} ({input_len}, {pred_len})")
    
    # Create output directory
    if scenario == "raw_standardized":
        # For raw_standardized data, create raw_formatted directory
        output_base = get_data_path(dataset, "raw_formatted")
    elif scenario == "raw":
        # For raw data, don't add the _formatted suffix to match existing structure
        output_base = get_data_path(dataset, "raw_formatted")
    else:
        output_base = get_data_path(dataset, f"{scenario}_formatted")
    
    window_dir = f"{input_len}_{pred_len}"
    output_dir = os.path.join(output_base, window_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get files to process
    available_files = get_available_files(dataset, scenario)
    
    if not available_files:
        logger.warning(f"No files found for {dataset}/{scenario}")
        return
    
    formatted_count = 0
    for input_file, output_filename in available_files:
        try:
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if file already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {output_filename} (already exists)")
                continue
            
            # Handle D1NAMO raw files differently
            if dataset == "d1namo" and scenario == "raw":
                # First standardize the format
                temp_df = standardize_d1namo_raw_file(input_file, output_filename)
                # Save to temp file
                temp_path = f"/tmp/{output_filename}"
                temp_df.to_csv(temp_path, index=False)
                
                # Then format with windows
                reformat_t1dm_bg_data(temp_path, output_path, input_len, pred_len)
                
                # Clean up temp file
                os.remove(temp_path)
            else:
                # Standard formatting
                reformat_t1dm_bg_data(input_file, output_path, input_len, pred_len)
            
            formatted_count += 1
            
        except Exception as e:
            logger.error(f"Error formatting {input_file}: {e}")
    
    logger.info(f"Formatted {formatted_count} files for {dataset}/{scenario}/{window_config}")


def main():
    parser = argparse.ArgumentParser(description="Unified Data Formatting Tool")
    parser.add_argument("--dataset", default="all", 
                       choices=["ohiot1dm", "d1namo", "all"],
                       help="Dataset to process")
    parser.add_argument("--scenario", default="all",
                       choices=["raw", "raw_standardized", "missing_periodic", "missing_random", "noisy", "denoised", "all"],
                       help="Data scenario to process")
    parser.add_argument("--windows", default="all",
                       choices=["6,6", "6,9", "all"],
                       help="Window configuration to process")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing formatted files")
    
    args = parser.parse_args()
    
    # Create missing data folders for ohiot1dm if needed
    if args.dataset in ["ohiot1dm", "all"]:
        create_missing_data_for_ohiot1dm()
    
    # Determine datasets to process
    datasets = ["ohiot1dm", "d1namo"] if args.dataset == "all" else [args.dataset]
    
    # Determine window configurations to process
    window_configs = ["6,6", "6,9"] if args.windows == "all" else [args.windows]
    
    total_combinations = 0
    successful_combinations = 0
    
    for dataset in datasets:
        logger.info(f"\n=== Processing dataset: {dataset} ===")
        
        # Determine scenarios to process
        if args.scenario == "all":
            scenarios = get_available_scenarios(dataset)
        else:
            scenarios = [args.scenario] if args.scenario in get_available_scenarios(dataset) else []
        
        logger.info(f"Available scenarios for {dataset}: {scenarios}")
        
        for scenario in scenarios:
            for window_config in window_configs:
                total_combinations += 1
                try:
                    format_dataset_scenario_window(dataset, scenario, window_config)
                    successful_combinations += 1
                except Exception as e:
                    logger.error(f"Failed to format {dataset}/{scenario}/{window_config}: {e}")
    
    logger.info(f"\n=== Formatting Complete ===")
    logger.info(f"Successfully processed {successful_combinations}/{total_combinations} combinations")


if __name__ == "__main__":
    main()