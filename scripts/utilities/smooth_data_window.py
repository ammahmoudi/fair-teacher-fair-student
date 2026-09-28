import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s")

def smooth_windowed_predictions(csv_path, sigma=None, scale_factor=0.5, plot=True, show_plot=False):
    """
    Applies Gaussian smoothing to each windowed prediction column separately 
    while maintaining the original window format.

    Parameters:
        csv_path (str): Path to the CSV file.
        sigma (float, optional): Standard deviation for Gaussian filter. If None, auto-calculated.
        scale_factor (float, optional): Scale factor for auto sigma calculation. Default is 0.5.
        plot (bool, optional): Whether to generate and save the plot. Default is True.
        show_plot (bool, optional): Whether to display the plot. Default is False.

    Returns:
        str: Path of the saved smoothed CSV file.
    """
    df = pd.read_csv(csv_path)

    # Identify columns
    true_columns = [col for col in df.columns if "_true" in col]
    pred_columns = [col for col in df.columns if "_pred" in col]

    if not pred_columns:
        logging.warning(f"Skipping {csv_path}: No valid prediction columns found.")
        return None

    # Auto-adjust sigma if not provided
    all_errors = []
    for t_true, t_pred in zip(true_columns, pred_columns):
        errors = df[t_pred] - df[t_true]
        all_errors.extend(errors.tolist())  # Collect all errors

    if sigma is None:
        error_std = np.std(all_errors)  # Measure noise level
        sigma = max(0.5, min(error_std * scale_factor, 2))  # Limit sigma between 0.5 and 2

    logging.info(f"Using sigma={sigma:.2f} for {csv_path}")

    # Apply Gaussian smoothing to each prediction column separately
    smoothed_df = df.copy()
    for t_pred in pred_columns:
        smoothed_df[f"smoothed_{t_pred}"] = gaussian_filter1d(df[t_pred], sigma=sigma)

    # Save the smoothed predictions
    folder, filename = os.path.split(csv_path)
    smoothed_filename = f"smoothed_{filename}"
    smoothed_csv_path = os.path.join(folder, smoothed_filename)
    smoothed_df.to_csv(smoothed_csv_path, index=False)

    # Plot first prediction column if enabled
    if plot and pred_columns:
        first_true_col, first_pred_col = true_columns[0], pred_columns[0]
        plt.figure(figsize=(10, 5))
        plt.plot(df[first_true_col], label=f"True Values ({first_true_col})", linestyle="dashed", alpha=0.7, color="black")
        plt.plot(df[first_pred_col], label=f"Original Predictions ({first_pred_col})", alpha=0.5, color="red")
        plt.plot(smoothed_df[f"smoothed_{first_pred_col}"], label=f"Smoothed Predictions ({first_pred_col})", linewidth=2, color="blue")
        plt.legend()
        plt.xlabel("Window Index")
        plt.ylabel("Glucose Level")
        plt.title(f"Glucose Prediction Smoothing (Sigma={sigma:.2f})")

        # Save the plot
        plot_filename = f"smoothed_plot_windows_{filename.replace('.csv', '.png')}"
        plot_path = os.path.join(folder, plot_filename)
        plt.savefig(plot_path, dpi=300)

        # Show the plot if enabled
        if show_plot:
            plt.show()
        else:
            plt.close()

        logging.info(f"Plot saved to: {plot_path}")

    logging.info(f"Smoothed predictions saved to: {smoothed_csv_path}")
    return smoothed_csv_path


def process_all_windowed_folders(base_dir, sigma=None, scale_factor=0.5, plot=True, show_plot=False):
    """
    Walks through all folders in base_dir, finds '*single_result.csv',
    applies Gaussian smoothing to each prediction column, and saves results.

    Parameters:
        base_dir (str): Base directory containing multiple experiment folders.
        sigma (float, optional): Fixed Gaussian smoothing standard deviation. If None, auto-calculated.
        scale_factor (float, optional): Factor to determine sigma from noise level. Default is 0.5.
        plot (bool, optional): Whether to generate and save plots. Default is True.
        show_plot (bool, optional): Whether to display the plot. Default is False.
    """
    logging.info("Starting batch processing of windowed glucose prediction smoothing.")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_reformatted.csv"):
                input_file = os.path.join(root, file)
                logging.info(f"Processing {input_file} ...")
                smooth_windowed_predictions(input_file, sigma=sigma, scale_factor=scale_factor, plot=plot, show_plot=show_plot)

    logging.info("Batch processing complete.")

# Example Usage:
process_all_windowed_folders("./experiment_configs_chronos_inference/", sigma=None, plot=True, show_plot=False)
# process_all_windowed_folders("./experiment_configs_time_llm_inference/", sigma=None, plot=True, show_plot=False)
