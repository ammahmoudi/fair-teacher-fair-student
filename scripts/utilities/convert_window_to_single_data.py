import logging
import sys
import os

import pandas as pd

import pandas as pd

def convert_window_to_single_prediction(input_file, output_file="single_step_predictions.csv"):
    """
    Converts windowed time-series predictions into a single-step format by:
    1. Extracting the first step's true and predicted values.
    2. Extracting the last row's true and predicted values, ensuring all steps are correctly paired.
    3. Removing the first two columns from the last row to avoid duplication.
    
    Saves the result to a new CSV file with "true" and "pred" columns.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file (default: "single_step_predictions.csv").
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Identify true and pred columns
    true_cols = [col for col in df.columns if "_true" in col]
    pred_cols = [col for col in df.columns if "_pred" in col]

    # Ensure there are true and pred columns
    if not true_cols or not pred_cols:
        print("No valid columns found in the CSV.")
        return

    # Select only the first true and predicted column (first step)
    selected_true = true_cols[0]
    selected_pred = pred_cols[0]

    # Extract the first step's true and predicted values
    first_step_df = df[[selected_true, selected_pred]].copy()
    first_step_df.columns = ["true", "pred"]  # Rename columns

    # Extract the last row
    last_row = df.iloc[-1]

    # Remove first two columns (already included in first_step_df)
    last_true_values = last_row[true_cols[2:]].reset_index(drop=True)
    last_pred_values = last_row[pred_cols[2:]].reset_index(drop=True)

    # Create DataFrame for last row values (properly paired)
    last_step_df = pd.DataFrame({
        "true": last_true_values.values,
        "pred": last_pred_values.values
    })

    # Concatenate first step data with the correctly paired last row data
    result_df = pd.concat([first_step_df, last_step_df], ignore_index=True)

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Converted window predictions saved to {output_file}")

# Example Usage:
# convert_window_to_single_prediction("inference_results_reformatted.csv")



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_all_folders(base_dir):
    """
    Walks through all folders in base_dir, finds 'inference_results_reformatted.csv',
    reformats it, and saves it as 'single_result.csv' in the same folder.
    
    Parameters:
        base_dir (str): Base directory containing multiple experiment folders.
    """
    logging.info("Starting batch processing of windowed predictions.")

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith( "_reformatted.csv"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(root, "single_result.csv")

                logging.info(f"Processing {input_file} ...")
                convert_window_to_single_prediction(input_file, output_file)

    logging.info("Batch processing complete.")

# Example Usage:
# process_all_folders("./experiment_configs_chronos_inference/")
process_all_folders("./experiment_configs_time_llm_training/")

