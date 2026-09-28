#!/usr/bin/env python3
"""
Unified Data Standardization Script for Fair Teacher–Fair Student Project

This script standardizes data files from different scenarios (raw, missing, noisy, denoised)
for both d1namo and ohiot1dm datasets. It converts various column formats to the standard
format: item_id, timestamp, target

Usage:
    python standardize_data.py --dataset all --scenario all
    python standardize_data.py --dataset d1namo --scenario missing_periodic
    python standardize_data.py --dataset ohiot1dm --scenario noisy,denoised
"""

import os
import sys
import pandas as pd
import argparse
import logging
from pathlib import Path

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.path_utils import get_data_path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.path_utils import get_data_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def standardize_file(input_file, output_file, patient_id=None):
    """
    Standardize a single CSV file to the format: item_id, timestamp, target
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output standardized CSV file  
        patient_id (str): Patient ID (extracted from filename if None)
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Initial data shape for {input_file}: {df.shape}")
        
        # Extract patient ID from filename if not provided
        if patient_id is None:
            filename = os.path.basename(input_file)
            patient_id = filename.split("-")[0]
        
        # Handle different column formats
        if '_ts' in df.columns and '_value' in df.columns:
            # Format: _ts, _value -> timestamp, target
            df.rename(columns={"_ts": "timestamp", "_value": "target"}, inplace=True)
            logger.info(f"Renamed _ts,_value columns for {input_file}")
        elif 'timestamp' in df.columns and 'target' in df.columns:
            # Already in standard format
            pass
        else:
            # Try to identify timestamp and value columns
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            value_cols = [col for col in df.columns if 'value' in col.lower() or 'bg' in col.lower() or 'glucose' in col.lower()]
            
            if timestamp_cols and value_cols:
                df.rename(columns={timestamp_cols[0]: "timestamp", value_cols[0]: "target"}, inplace=True)
                logger.info(f"Auto-detected and renamed columns for {input_file}")
            else:
                logger.warning(f"Could not identify timestamp/value columns in {input_file}. Columns: {df.columns.tolist()}")
                return False
        
        # Ensure we have the required columns
        if 'timestamp' not in df.columns or 'target' not in df.columns:
            logger.error(f"Missing required columns in {input_file}. Columns: {df.columns.tolist()}")
            return False
        
        # Add patient ID column
        df["item_id"] = patient_id
        
        # Standardize timestamp format if it's not already datetime
        try:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
            df["timestamp"] = df["timestamp"].dt.strftime("%d-%m-%Y %H:%M:%S")
        except Exception as e:
            logger.warning(f"Could not standardize timestamp format for {input_file}: {e}")
        
        # Reorder columns to standard format
        df = df[["item_id", "timestamp", "target"]]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save standardized file
        df.to_csv(output_file, index=False)
        logger.info(f"Standardized data shape for {output_file}: {df.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return False

def get_scenarios_for_dataset(dataset):
    """Get available scenarios for a dataset"""
    base_path = get_data_path(dataset)
    
    if not os.path.exists(base_path):
        logger.warning(f"Dataset path does not exist: {base_path}")
        return []
    
    scenarios = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and not item.endswith('_formatted') and not item.endswith('_standardized'):
            scenarios.append(item)
    
    return sorted(scenarios)

def standardize_dataset_scenario(dataset, scenario):
    """
    Standardize all files for a specific dataset and scenario
    
    Args:
        dataset (str): Dataset name (d1namo, ohiot1dm)
        scenario (str): Scenario name (raw, missing_periodic, missing_random, noisy, denoised)
    """
    input_folder = get_data_path(dataset, scenario)
    output_folder = get_data_path(dataset, f"{scenario}_standardized")
    
    if not os.path.exists(input_folder):
        logger.warning(f"Input folder does not exist: {input_folder}")
        return 0
    
    # Check if already standardized
    if os.path.exists(output_folder):
        logger.info(f"Standardized folder already exists: {output_folder}")
        # Ask user if they want to overwrite or skip
        # For now, we'll continue to overwrite
    
    logger.info(f"Standardizing {dataset}/{scenario}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy prompt file if it exists
    prompt_file = os.path.join(input_folder, "t1dm_prompt.txt")
    if os.path.exists(prompt_file):
        import shutil
        shutil.copy2(prompt_file, os.path.join(output_folder, "t1dm_prompt.txt"))
        logger.info(f"Copied prompt file to {output_folder}")
    
    # Process all CSV files
    processed_count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            
            if standardize_file(input_file, output_file):
                processed_count += 1
                logger.info(f"✓ Processed: {filename}")
            else:
                logger.error(f"✗ Failed: {filename}")
    
    logger.info(f"Standardized {processed_count} files for {dataset}/{scenario}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Unified data standardization for Fair Teacher–Fair Student datasets")
    parser.add_argument("--dataset", required=True, 
                       choices=["d1namo", "ohiot1dm", "all"],
                       help="Dataset to standardize")
    parser.add_argument("--scenario", required=True,
                       help="Scenario(s) to standardize (comma-separated or 'all')")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing standardized folders")
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset == "all":
        datasets = ["d1namo", "ohiot1dm"]
    else:
        datasets = [args.dataset]
    
    total_processed = 0
    
    for dataset in datasets:
        logger.info(f"\n=== Processing dataset: {dataset} ===")
        
        # Get available scenarios for this dataset
        available_scenarios = get_scenarios_for_dataset(dataset)
        logger.info(f"Available scenarios for {dataset}: {available_scenarios}")
        
        # Determine scenarios to process
        if args.scenario == "all":
            scenarios = available_scenarios
        else:
            scenarios = [s.strip() for s in args.scenario.split(",")]
            # Filter to only available scenarios
            scenarios = [s for s in scenarios if s in available_scenarios]
        
        if not scenarios:
            logger.warning(f"No valid scenarios found for {dataset}")
            continue
        
        # Process each scenario
        for scenario in scenarios:
            count = standardize_dataset_scenario(dataset, scenario)
            total_processed += count
    
    logger.info(f"\n=== Standardization Complete ===")
    logger.info(f"Total files processed: {total_processed}")

if __name__ == "__main__":
    main()