#!/usr/bin/env python3
"""
Unified Arrow Converter for Fair Teacher–Fair Student Project

This script converts standardized CSV files to Arrow format for use with GluonTS.
It processes all standardized data scenarios for both d1namo and ohiot1dm datasets.

Usage:
    python convert_to_arrow.py --dataset all --scenario all
    python convert_to_arrow.py --dataset d1namo --scenario missing_periodic
    python convert_to_arrow.py --dataset ohiot1dm --scenario raw
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import List, Union
from gluonts.dataset.arrow import ArrowWriter

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.path_utils import get_data_path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    timestamps: List[np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Prepare dataset with the actual timestamp
    dataset = [
        {"start": timestamps[i][0], "target": ts} 
        for i, ts in enumerate(time_series)
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )

def load_csv_and_prepare_data(csv_path: str):
    """
    Load a CSV file, parse it, and prepare the data in the format
    suitable for GluonTS ArrowWriter.

    The expected CSV format:
    item_id,timestamp,target
    540,04-07-2027 00:01:44,254
    """
    try:
        # Read CSV data into a pandas DataFrame
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        logger.info(f"Loaded CSV {csv_path} with shape: {df.shape}")

        # Ensure data is sorted by timestamp (if it's not already)
        df = df.sort_values("timestamp")

        # Group by 'item_id' to create separate time series for each item
        time_series = []
        timestamps = []
        for item_id, group in df.groupby("item_id"):
            # Extract the 'target' values and the 'timestamp' values
            target_values = group["target"].values
            timestamp_values = group["timestamp"].values

            # Store the time series and corresponding timestamps
            time_series.append(target_values)
            timestamps.append(timestamp_values)

        logger.info(f"Prepared {len(time_series)} time series from {csv_path}")
        return time_series, timestamps
    
    except Exception as e:
        logger.error(f"Error processing {csv_path}: {e}")
        return None, None

def convert_csv_to_arrow(csv_file_path: str, arrow_file_path: str):
    """
    Convert a single CSV file to Arrow format
    
    Args:
        csv_file_path (str): Path to input CSV file
        arrow_file_path (str): Path to output Arrow file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load and prepare data from CSV
        time_series, timestamps = load_csv_and_prepare_data(csv_file_path)
        
        if time_series is None or timestamps is None:
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(arrow_file_path), exist_ok=True)
        
        # Convert to Arrow format
        convert_to_arrow(arrow_file_path, time_series=time_series, timestamps=timestamps)
        logger.info(f"✓ Converted: {csv_file_path} -> {arrow_file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to convert {csv_file_path}: {e}")
        return False

def get_standardized_scenarios_for_dataset(dataset):
    """Get available standardized scenarios for a dataset"""
    base_path = get_data_path(dataset)
    
    if not os.path.exists(base_path):
        logger.warning(f"Dataset path does not exist: {base_path}")
        return []
    
    scenarios = []
    for item in os.listdir(base_path):
        if item.endswith('_standardized') and os.path.isdir(os.path.join(base_path, item)):
            # Remove '_standardized' suffix to get scenario name
            scenario = item.replace('_standardized', '')
            scenarios.append(scenario)
    
    return sorted(scenarios)

def convert_dataset_scenario(dataset, scenario):
    """
    Convert all CSV files in a standardized scenario folder to Arrow format
    
    Args:
        dataset (str): Dataset name (d1namo, ohiot1dm, or standardized for global)
        scenario (str): Scenario name (raw, missing_periodic, missing_random, noisy, denoised)
    
    Returns:
        int: Number of files successfully converted
    """
    if dataset == "standardized":
        # Handle global standardized folder
        input_folder = get_data_path("standardized")
        output_folder = get_data_path("standardized")
        scenario_name = "standardized"
    else:
        input_folder = get_data_path(dataset, f"{scenario}_standardized")
        output_folder = get_data_path(dataset, f"{scenario}_standardized")
        scenario_name = f"{dataset}/{scenario}"
    
    if not os.path.exists(input_folder):
        logger.warning(f"Input folder does not exist: {input_folder}")
        return 0
    
    logger.info(f"Converting Arrow files for {scenario_name}")
    
    # Process all CSV files in the folder
    converted_count = 0
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for csv_filename in csv_files:
        csv_file_path = os.path.join(input_folder, csv_filename)
        
        # Create corresponding arrow filename
        arrow_filename = csv_filename.replace('.csv', '.arrow')
        arrow_file_path = os.path.join(output_folder, arrow_filename)
        
        # Skip if arrow file already exists and is newer than CSV
        if os.path.exists(arrow_file_path):
            csv_mtime = os.path.getmtime(csv_file_path)
            arrow_mtime = os.path.getmtime(arrow_file_path)
            if arrow_mtime > csv_mtime:
                logger.info(f"⏭ Skipping {csv_filename} (Arrow file is newer)")
                continue
        
        if convert_csv_to_arrow(csv_file_path, arrow_file_path):
            converted_count += 1
    
    logger.info(f"Converted {converted_count} files for {scenario_name}")
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="Unified Arrow converter for Fair Teacher–Fair Student datasets")
    parser.add_argument("--dataset", required=True, 
                       choices=["d1namo", "ohiot1dm", "standardized", "all"],
                       help="Dataset to convert")
    parser.add_argument("--scenario", required=True,
                       help="Scenario(s) to convert (comma-separated or 'all')")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing Arrow files")
    
    args = parser.parse_args()
    
    # Determine datasets to process
    if args.dataset == "all":
        datasets = ["d1namo", "ohiot1dm", "standardized"]
    else:
        datasets = [args.dataset]
    
    total_converted = 0
    
    for dataset in datasets:
        logger.info(f"\n=== Processing dataset: {dataset} ===")
        
        if dataset == "standardized":
            # Handle global standardized folder specially
            if args.scenario == "all" or "standardized" in args.scenario:
                count = convert_dataset_scenario("standardized", "")
                total_converted += count
        else:
            # Get available standardized scenarios for this dataset
            available_scenarios = get_standardized_scenarios_for_dataset(dataset)
            logger.info(f"Available standardized scenarios for {dataset}: {available_scenarios}")
            
            # Determine scenarios to process
            if args.scenario == "all":
                scenarios = available_scenarios
            else:
                scenarios = [s.strip() for s in args.scenario.split(",")]
                # Filter to only available scenarios
                scenarios = [s for s in scenarios if s in available_scenarios]
            
            if not scenarios:
                logger.warning(f"No valid standardized scenarios found for {dataset}")
                continue
            
            # Process each scenario
            for scenario in scenarios:
                count = convert_dataset_scenario(dataset, scenario)
                total_converted += count
    
    logger.info(f"\n=== Arrow Conversion Complete ===")
    logger.info(f"Total files converted: {total_converted}")

if __name__ == "__main__":
    main()