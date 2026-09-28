import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def smooth_glucose_predictions(csv_path, sigma=None, scale_factor=0.5, plot=True, show_plot=False):
    """
    Applies a Gaussian filter to smooth glucose level predictions in a CSV file.
    Auto-adjusts sigma based on noise level and saves the smoothed results.

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

    # Ensure required columns exist
    if 'true' not in df.columns or 'pred' not in df.columns:
        logging.warning(f"Skipping {csv_path}: 'true' and 'pred' columns not found.")
        return None

    # Calculate auto sigma if not provided
    if sigma is None:
        errors = df['pred'] - df['true']
        error_std = np.std(errors)  # Measure noise level
        sigma = max(0.5, min(error_std * scale_factor, 2))  # Limit between 0.5 and 2

    logging.info(f"Using sigma={sigma:.2f} for {csv_path}")

    # Apply Gaussian smoothing to the predictions
    df['smoothed_pred'] = gaussian_filter1d(df['pred'], sigma=sigma)

    # Save the smoothed predictions in the same folder as the input file
    folder, filename = os.path.split(csv_path)
    smoothed_filename = f"smoothed_{filename}"
    smoothed_csv_path = os.path.join(folder, smoothed_filename)
    df.to_csv(smoothed_csv_path, index=False)

    # Generate and save the plot if enabled
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(df['true'], label="True Values", linestyle="dashed", alpha=0.7, color="black")
        plt.plot(df['pred'], label="Original Predictions", alpha=0.5, color="red")
        plt.plot(df['smoothed_pred'], label="Smoothed Predictions", linewidth=2, color="blue")
        plt.legend()
        plt.xlabel("Time/Index")
        plt.ylabel("Glucose Level")
        plt.title(f"Glucose Prediction Smoothing (Sigma={sigma:.2f})")

        # Save the plot in the same folder
        plot_filename = f"smoothed_plot_{filename.replace('.csv', '.png')}"
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


def process_all_folders(base_dir, sigma=None, scale_factor=0.5, plot=True, show_plot=False):
    """
    Walks through all folders in base_dir, finds '*_reformatted.csv',
    applies Gaussian smoothing with dynamic sigma, and saves results.

    Parameters:
        base_dir (str): Base directory containing multiple experiment folders.
        sigma (float, optional): Fixed Gaussian smoothing standard deviation. If None, auto-calculated.
        scale_factor (float, optional): Factor to determine sigma from noise level. Default is 0.5.
        plot (bool, optional): Whether to generate and save plots. Default is True.
        show_plot (bool, optional): Whether to display the plot. Default is False.
    """
    logging.info("Starting batch processing of glucose prediction smoothing.")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("single_result.csv"):
                input_file = os.path.join(root, file)
                logging.info(f"Processing {input_file} ...")
                smooth_glucose_predictions(input_file, sigma=sigma, scale_factor=scale_factor, plot=plot, show_plot=show_plot)

    logging.info("Batch processing complete.")

# Example Usage:
# Auto-adjust sigma based on noise level
# process_all_folders("./experiment_configs_chronos_inference/", sigma=None, scale_factor=0.5, plot=True, show_plot=False)

# If you want a fixed sigma value
# process_all_folders("./experiment_configs_chronos_inference/", sigma=1.0, plot=True, show_plot=False)

# Example Usage:
process_all_folders("./experiment_configs_chronos_inference/", sigma=None, plot=True, show_plot=False)
process_all_folders("./experiment_configs_time_llm_training/", sigma=None, plot=True, show_plot=False)
