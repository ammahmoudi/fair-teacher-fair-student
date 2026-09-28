import os
import numpy as np
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s")

def calculate_rmse(prediction, ground_truth):
    squared_error = (prediction - ground_truth) ** 2
    mean_squared_error = np.mean(squared_error)
    rmse = np.sqrt(mean_squared_error)
    return rmse


def calculate_mae(prediction, ground_truth):
    return np.mean(np.abs(prediction - ground_truth))

def calculate_mape(prediction, ground_truth):
    return np.mean(np.abs((prediction - ground_truth) / ground_truth))

def update_log_file(log_file, original_metrics, smoothed_metrics):
    """
    Finds the last line in log.log with metric results and appends new metrics
    for both original and smoothed data.

    Parameters:
        log_file (str): Path to the log file.
        original_metrics (dict): Metrics calculated using original predictions.
        smoothed_metrics (dict): Metrics calculated using smoothed predictions.
    """
    # Read existing log file
    if not os.path.exists(log_file):
        logging.warning(f"Log file {log_file} not found. Creating a new one.")
        open(log_file, 'w').close()  # Create empty log file if not exists

    with open(log_file, "r") as f:
        lines = f.readlines()

    # Find last metric results line
    for i in range(len(lines) - 1, -1, -1):
        if "Metric results" in lines[i]:
            last_metrics_index = i
            break
    else:
        last_metrics_index = len(lines)  # If not found, append at the end

    # Format new metric lines
    new_metric_lines = [
        f"{logging.Formatter().formatTime(logging.makeLogRecord({}))} | INFO | Metric results (original): {original_metrics}\n",
        f"{logging.Formatter().formatTime(logging.makeLogRecord({}))} | INFO | Metric results (smoothed): {smoothed_metrics}\n"
    ]

    # Append the new metrics lines
    lines.insert(last_metrics_index + 1, "".join(new_metric_lines))

    # Write back to log file
    with open(log_file, "w") as f:
        f.writelines(lines)

    logging.info(f"Updated log file with original and smoothed metrics.")

def process_metrics_and_update_log(base_dir):
    """
    Processes all smoothed CSV files, computes new metrics for both original
    and smoothed data, and updates the log file.

    Parameters:
        base_dir (str): Base directory containing multiple experiment folders.
    """
    logging.info("Starting metric calculation and log update process.")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("smoothed_") and file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                log_file = os.path.join(root, "log.log")  # Log file path

                # Load CSV
                df = pd.read_csv(csv_path)

                # Ensure required columns exist
                if 'true' not in df.columns or 'smoothed_pred' not in df.columns or 'pred' not in df.columns:
                    logging.warning(f"Skipping {csv_path}: Missing required columns.")
                    continue

                # Compute metrics for original predictions
                original_rmse = calculate_rmse(df['pred'], df['true'])
                original_mae = calculate_mae(df['pred'], df['true'])
                original_mape = calculate_mape(df['pred'], df['true'])

                original_metrics = {
                    "rmse": round(original_rmse, 5),
                    "mae": round(original_mae, 5),
                    "mape": round(original_mape, 5),
                }

                # Compute metrics for smoothed predictions
                smoothed_rmse = calculate_rmse(df['smoothed_pred'], df['true'])
                smoothed_mae = calculate_mae(df['smoothed_pred'], df['true'])
                smoothed_mape = calculate_mape(df['smoothed_pred'], df['true'])

                smoothed_metrics = {
                    "rmse": round(smoothed_rmse, 5),
                    "mae": round(smoothed_mae, 5),
                    "mape": round(smoothed_mape, 5),
                }

                # Update log file
                update_log_file(log_file, original_metrics, smoothed_metrics)

    logging.info("Metric calculation and log update process completed.")

# Example Usage:
# process_metrics_and_update_log("./experiment_configs_chronos_inference/")

# Example Usage:
process_metrics_and_update_log("./experiment_configs_time_llm_training/")
