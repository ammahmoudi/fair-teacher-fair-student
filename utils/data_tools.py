import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def smooth_glucose_predictions(csv_path, sigma=2, plot=True, show_plot=True):
    """
    Applies a Gaussian filter to smooth glucose level predictions in a CSV file.
    Saves the smoothed results and optionally saves & shows a plot.

    Parameters:
        csv_path (str): Path to the CSV file.
        sigma (int, optional): Standard deviation for Gaussian filter. Default is 2.
        plot (bool, optional): Whether to generate and save the plot. Default is True.
        show_plot (bool, optional): Whether to display the plot. Default is True.

    Returns:
        str: Path of the saved smoothed CSV file.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if 'predictions' not in df.columns:
        raise ValueError("CSV must contain a 'predictions' column.")

    # Apply Gaussian smoothing to the predictions
    df['smoothed_predictions'] = gaussian_filter1d(df['predictions'], sigma=sigma)

    # Save the smoothed predictions in the same folder as the input file
    folder, filename = os.path.split(csv_path)
    smoothed_filename = f"smoothed_{filename}"
    smoothed_csv_path = os.path.join(folder, smoothed_filename)
    df.to_csv(smoothed_csv_path, index=False)

    # Generate and save the plot if enabled
    if plot:
        plt.figure(figsize=(10, 5))
        if 'true_values' in df.columns:
            plt.plot(df['true_values'], label="True Values", linestyle="dashed", alpha=0.7)
        plt.plot(df['predictions'], label="Original Predictions", alpha=0.5)
        plt.plot(df['smoothed_predictions'], label="Smoothed Predictions", linewidth=2)
        plt.legend()
        plt.xlabel("Time/Index")
        plt.ylabel("Glucose Level")
        plt.title("Glucose Prediction Smoothing with Gaussian Filter")

        # Save the plot in the same folder
        plot_filename = f"smoothed_plot_{filename.replace('.csv', '.png')}"
        plot_path = os.path.join(folder, plot_filename)
        plt.savefig(plot_path, dpi=300)

        # Show the plot if enabled
        if show_plot:
            plt.show()
        else:
            plt.close()

        print(f"Plot saved to: {plot_path}")

    print(f"Smoothed predictions saved to: {smoothed_csv_path}")
    return smoothed_csv_path

# Example Usage:
# smooth_glucose_predictions("your_file.csv", sigma=2, plot=True, show_plot=True)  # Show & Save Plot
# smooth_glucose_predictions("your_file.csv", sigma=2, plot=True, show_plot=False) # Save but Don't Show Plot
# smooth_glucose_predictions("your_file.csv", sigma=2, plot=False)  # No Plot at All
