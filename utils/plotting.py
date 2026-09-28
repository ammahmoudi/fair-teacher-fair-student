import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import sem


def plot_predictions_plotly(data, steps=None, save_path=None, name=None):
    """
    Plot the ground truth values and specified prediction steps for each timestamp, using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.
        name (str, optional): A custom name to use in the plot title and file names.

    Returns:
        None: Displays the plot or saves it to disk.
    """
    # Extract ground truth and all prediction steps
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    # Iterate through rows and extract true and all pred values for each row
    for _, row in data.iterrows():
        true_values = [
            row[col]
            for col in data.columns
            if col.endswith("_true") and not pd.isna(row[col])
        ]
        pred_values = [
            row[col]
            for col in data.columns
            if col.endswith("_pred") and not pd.isna(row[col])
        ]

        if true_values:
            ground_truth.append(
                true_values[0]
            )  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on the steps parameter
    if steps is not None:
        selected_steps = [
            i - 1 for i in steps if 1 <= i <= len(all_predictions)
        ]  # Convert to 0-based indices
    else:
        selected_steps = range(len(all_predictions))

    # Create a range for the x-axis
    x_values = list(range(len(ground_truth)))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ground_truth,
            mode="lines+markers",
            name="Ground Truth",
            line=dict(dash="dash", color="green"),
            marker=dict(symbol="triangle-up", size=8, color="green"),
        )
    )

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=predictions,
                mode="lines+markers",
                name=f"Predictions (Step {i+1})",
                line=dict(shape="linear"),
            )
        )

    # Update layout with a title
    plot_title = name if name else "Ground Truth vs Predictions (Selected Steps)"
    fig.update_layout(
        title=plot_title,
        xaxis_title="Index",
        yaxis_title="Values",
        legend_title="Legend",
        template="plotly_white",
        height=600,
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"{name or 'ground_truth_vs_predictions'}_steps_{steps_str}"

        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save PNG image: {e}")
        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save SVG image: {e}")
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
        # Show the plot
        fig.show()


import os
import plotly.graph_objects as go
import pandas as pd


def plot_avg_predictions_plotly(data, save_path=None, name=None):
    """
    Plot the ground truth values and the averaged predictions for each timestamp, using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML.
        name (str, optional): A custom name to use in the plot title and file names.

    Returns:
        None: Displays the plot or saves it to disk.
    """
    # Extract ground truth and predictions
    ground_truth = []
    avg_predictions = []  # To store the averaged predictions

    # Iterate through rows and extract true and pred values
    for _, row in data.iterrows():
        true_values = [
            row[col]
            for col in data.columns
            if col.endswith("_true") and not pd.isna(row[col])
        ]
        pred_values = [
            row[col]
            for col in data.columns
            if col.endswith("_pred") and not pd.isna(row[col])
        ]

        if true_values:
            ground_truth.append(
                true_values[0]
            )  # Take the first available ground truth value

        if pred_values:
            avg_predictions.append(
                sum(pred_values) / len(pred_values)
            )  # Average predictions

    # Create a range for the x-axis
    x_values = list(range(len(ground_truth)))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=ground_truth,
            mode="lines+markers",
            name="Ground Truth",
            line=dict(dash="dash", color="green"),
            marker=dict(symbol="triangle-up", size=8, color="green"),
        )
    )

    # Add averaged prediction trace
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=avg_predictions,
            mode="lines+markers",
            name="Averaged Predictions",
            line=dict(shape="linear", color="blue"),
        )
    )

    # Update layout with a title
    plot_title = name if name else "Ground Truth vs Averaged Predictions"
    fig.update_layout(
        title=plot_title,
        xaxis_title="Index",
        yaxis_title="Values",
        legend_title="Legend",
        template="plotly_white",
        height=600,
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        filename_base = f"{name or 'ground_truth_vs_avg_predictions'}"

        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save PNG image: {e}")
        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save SVG image: {e}")
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
        # Show the plot
        fig.show()


def plot_predictions_matplotlib(data):
    """
    Plot the ground truth values and all predictions for each timestamp, accounting for overlapping windows.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.

    Returns:
        None: Displays the plot.
    """
    # Extract ground truth and all prediction steps
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    # Iterate through rows and extract true and all pred values for each row
    for _, row in data.iterrows():
        true_values = [
            row[col]
            for col in data.columns
            if col.endswith("_true") and not pd.isna(row[col])
        ]
        pred_values = [
            row[col]
            for col in data.columns
            if col.endswith("_pred") and not pd.isna(row[col])
        ]

        if true_values:
            ground_truth.append(
                true_values[0]
            )  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Create a range for the x-axis
    x_values = range(len(ground_truth))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        x_values,
        ground_truth,
        label="Ground Truth",
        linestyle="--",
        marker="o",
        color="green",
    )

    # Plot each prediction step
    for i, predictions in enumerate(all_predictions):
        plt.plot(
            x_values,
            predictions,
            label=f"Predictions (Step {i+1})",
            linestyle="-",
            marker="x",
        )

    # Formatting
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Ground Truth vs Predictions (All Steps)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_clarke_error_grid_plotly(data, steps=None, save_path=None):
    """
    Plot a Clarke Error Grid using Plotly to assess the accuracy of predicted values against ground truth.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.

    Returns:
        None: Displays the Clarke Error Grid plot.
    """
    # Extract true and predicted values
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    for _, row in data.iterrows():
        true_values = [
            row[col]
            for col in data.columns
            if col.endswith("_true") and not pd.isna(row[col])
        ]
        pred_values = [
            row[col]
            for col in data.columns
            if col.endswith("_pred") and not pd.isna(row[col])
        ]

        if true_values:
            ground_truth.append(
                true_values[0]
            )  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on steps
    if steps is not None:
        selected_steps = [i - 1 for i in steps if 1 <= i <= len(all_predictions)]
    else:
        selected_steps = range(len(all_predictions))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add Clarke Error Grid regions
    max_val = max(max(ground_truth), max([max(pred) for pred in all_predictions]))
    x = np.linspace(0, max_val, 100)

    # Add region A (within 20% of true values)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1.2 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="green", dash="dash"),
            name="Region A Upper (20%)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.8 * x,
            mode="lines",
            fill="none",
            line=dict(color="green", dash="dash"),
            name="Region A Lower (20%)",
        )
    )

    # Add region B (clinically acceptable outside region A)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1.3 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="yellow", dash="dash"),
            name="Region B Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.7 * x,
            mode="lines",
            fill="none",
            line=dict(color="yellow", dash="dash"),
            name="Region B Lower",
        )
    )

    # Add region C/D (over-treatment and under-treatment)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=max_val * np.ones_like(x),
            mode="lines",
            fill="tonexty",
            line=dict(color="red", dash="dash"),
            name="Region C",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.7 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="red", dash="dash"),
            name="Region D",
            showlegend=False,
        )
    )

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(
            go.Scatter(
                x=ground_truth,
                y=predictions,
                mode="markers",
                name=f"Predictions (Step {i+1})",
                marker=dict(size=8, symbol="circle"),
            )
        )

    # Update layout
    fig.update_layout(
        title="Clarke Error Grid",
        xaxis_title="True Values",
        yaxis_title="Predicted Values",
        template="plotly_white",
        height=800,
        width=800,
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"clarke_error_grid_steps_{steps_str}"

        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save PNG image: {e}")
        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save SVG image: {e}")
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
        # Show the plot
        fig.show()


def plot_surveillance_error_grid(data, steps=None, save_path=None):
    """
    Plot a Surveillance Error Grid (SEG) for blood glucose predictions using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.

    Returns:
        None: Displays the SEG plot.
    """
    # Extract true and predicted values
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    for _, row in data.iterrows():
        true_values = [
            row[col]
            for col in data.columns
            if col.endswith("_true") and not pd.isna(row[col])
        ]
        pred_values = [
            row[col]
            for col in data.columns
            if col.endswith("_pred") and not pd.isna(row[col])
        ]

        if true_values:
            ground_truth.append(
                true_values[0]
            )  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on steps
    if steps is not None:
        selected_steps = [i - 1 for i in steps if 1 <= i <= len(all_predictions)]
    else:
        selected_steps = range(len(all_predictions))

    # Initialize Plotly figure
    fig = go.Figure()

    # Define Surveillance Error Grid regions (specific to blood glucose levels)
    max_val = max(max(ground_truth), max([max(pred) for pred in all_predictions]))
    x = np.linspace(0, max_val, 100)

    # Region A: Clinically insignificant
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1.2 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="green", dash="dash"),
            name="Region A Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.8 * x,
            mode="lines",
            fill="none",
            line=dict(color="green", dash="dash"),
            name="Region A Lower",
        )
    )

    # Region B: Minimal risk
    fig.add_trace(
        go.Scatter(
            x=x,
            y=1.5 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="yellow", dash="dash"),
            name="Region B Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.6 * x,
            mode="lines",
            fill="none",
            line=dict(color="yellow", dash="dash"),
            name="Region B Lower",
        )
    )

    # Region C: Moderate risk
    fig.add_trace(
        go.Scatter(
            x=x,
            y=2 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="orange", dash="dash"),
            name="Region C Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.5 * x,
            mode="lines",
            fill="none",
            line=dict(color="orange", dash="dash"),
            name="Region C Lower",
        )
    )

    # Region D: High risk
    fig.add_trace(
        go.Scatter(
            x=x,
            y=max_val * np.ones_like(x),
            mode="lines",
            fill="tonexty",
            line=dict(color="red", dash="dash"),
            name="Region D Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=0.4 * x,
            mode="lines",
            fill="tonexty",
            line=dict(color="red", dash="dash"),
            name="Region D Lower",
            showlegend=False,
        )
    )

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(
            go.Scatter(
                x=ground_truth,
                y=predictions,
                mode="markers",
                name=f"Predictions (Step {i+1})",
                marker=dict(size=8, symbol="circle"),
            )
        )

    # Update layout
    fig.update_layout(
        title="Surveillance Error Grid for Blood Glucose",
        xaxis_title="True Glucose Values (mg/dL)",
        yaxis_title="Predicted Glucose Values (mg/dL)",
        template="plotly_white",
        height=800,
        width=800,
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"surveillance_error_grid_steps_{steps_str}"

        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save PNG image: {e}")
        try:
            fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        except Exception as e:
            print(f"⚠️  Warning: Could not save SVG image: {e}")
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
        # Show the plot
        fig.show()


def plot_mae_steps(df: pd.DataFrame, title: str, file_path: str):
    """
    Plots the Mean Absolute Error (MAE) for each future timestep and saves the plot.

    This function calculates the absolute errors between true and predicted values
    for multiple timesteps, then visualizes the mean MAE with a shaded ±1 standard deviation
    (SD) range.

    The plot is saved as both `.svg` and `.png` files in the current directory.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted values for multiple timesteps.
        title (str): Title of the plot.
        file_path (str): Path of the input CSV file to derive the output filename.

    Saves:
        - A `.svg` version of the plot.
        - A `.png` version of the plot.

    Example:
        >>> plot_mae_steps(df, "MAE Across Future Timesteps", "results/data.csv")
    """

    # Identify the true and predicted columns
    true_cols = [col for col in df.columns if "true" in col]
    pred_cols = [col for col in df.columns if "pred" in col]

    # Compute absolute errors for each future timestep and store metrics
    error_data = {}
    mean_errors = []
    std_devs = []

    for i, (true_col, pred_col) in enumerate(zip(true_cols, pred_cols)):
        # Absolute errors per timestep
        abs_error = np.abs(df[true_col] - df[pred_col])
        error_data[f"Future_Timestep_t{i+1}"] = abs_error

        # Compute mean and standard deviation
        mean_errors.append(abs_error.mean())
        std_devs.append(abs_error.std())

    # Prepare data for visualization
    future_timesteps = np.arange(1, len(mean_errors) + 1)

    # --- Enhanced Line Plot with SD Range ---
    plt.figure(figsize=(10, 6))

    # Plot Mean Absolute Error
    plt.plot(
        future_timesteps,
        mean_errors,
        marker="o",
        label="Mean Absolute Error",
        color="blue",
    )

    # Add shaded region for ±1 SD
    lower_bound = np.array(mean_errors) - np.array(std_devs)
    upper_bound = np.array(mean_errors) + np.array(std_devs)
    plt.fill_between(
        future_timesteps,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.2,
        label="±1 SD",
    )

    # Additional plot settings
    # plt.title(title, fontsize=14)
    plt.xlabel("Future Timestep $t_k$", fontsize=14)
    plt.ylabel("Mean Absolute Error (mg/dL)", fontsize=14)
    plt.xticks(ticks=future_timesteps)
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Generate file names based on title and save figures
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    svg_filename = f"{base_filename}_{safe_title}.svg"
    png_filename = f"{base_filename}_{safe_title}.png"

    plt.savefig(svg_filename, format="svg")
    plt.savefig(png_filename, format="png")

    plt.show()
def plot_mae_steps_dual(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    title: str,
    file_path: str
):
    """
    Plots the MAE for each future timestep for two datasets and overlays them on the same plot.
    Each dataset keeps its original time steps.
    """
    
    def compute_mae_and_sd(df):
        true_cols = [col for col in df.columns if "true" in col]
        pred_cols = [col for col in df.columns if "pred" in col]

        mean_errors = []
        std_devs = []

        for true_col, pred_col in zip(true_cols, pred_cols):
            abs_error = np.abs(df[true_col] - df[pred_col])
            mean_errors.append(abs_error.mean())
            std_devs.append(abs_error.std())

        return np.array(mean_errors), np.array(std_devs)

    # Compute metrics for both DataFrames
    mean1, std1 = compute_mae_and_sd(df1)
    mean2, std2 = compute_mae_and_sd(df2)

    # Define future timesteps based on the lengths of the respective sequences
    future_timesteps_1 = np.arange(1, len(mean1) + 1)
    future_timesteps_2 = np.arange(1, len(mean2) + 1)

    # Create x-tick labels as t_1, t_2, ..., t_n
    tick_labels_1 = [f"$t_{i}$" for i in future_timesteps_1]
    tick_labels_2 = [f"$t_{i}$" for i in future_timesteps_2]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Dataset 1 (6 timesteps)
    plt.plot(future_timesteps_1, mean1, marker="o", label=f"{label1} MAE", color="blue")
    plt.fill_between(future_timesteps_1, mean1 - std1, mean1 + std1, color="blue", alpha=0.2, label=f"{label1} ±1 SD")

    # Dataset 2 (9 timesteps)
    plt.plot(future_timesteps_2, mean2, marker="o", label=f"{label2} MAE", color="green")
    plt.fill_between(future_timesteps_2, mean2 - std2, mean2 + std2, color="green", alpha=0.2, label=f"{label2} ±1 SD")

    # Set x-ticks with the generated labels
    plt.xticks(future_timesteps_1, tick_labels_1, rotation=0,fontsize=14)
    plt.xlabel("Future Timestep", fontsize=14)
    plt.ylabel("Mean Absolute Error (mg/dL)", fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save files
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    svg_filename = f"{base_filename}_{safe_title}.svg"
    png_filename = f"{base_filename}_{safe_title}.png"

    plt.savefig(svg_filename, format="svg")
    plt.savefig(png_filename, format="png")
    plt.show()



def plot_mae_steps_box_plot(df: pd.DataFrame, file_path: str):
    """
    Generates a boxplot showing the distribution of absolute errors for each prediction order.

    This function calculates the absolute errors between true and predicted values
    and visualizes them as a boxplot, helping to understand error distribution at different
    prediction orders.

    The plot is saved as both `.svg` and `.png` files in the current directory.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted values.
        file_path (str): Path of the input CSV file to derive the output filename.

    Saves:
        - A `.svg` version of the plot.
        - A `.png` version of the plot.

    Example:
        >>> plot_mae_steps_box_plot(df, "results/data.csv")
    """

    # Identify the true and predicted columns
    true_cols = [col for col in df.columns if "true" in col]
    pred_cols = [col for col in df.columns if "pred" in col]

    # Compute absolute errors for each prediction order and store metrics
    error_data = {}

    for i, (true_col, pred_col) in enumerate(zip(true_cols, pred_cols)):
        # Absolute errors per prediction order
        abs_error = np.abs(df[true_col] - df[pred_col])
        error_data[f"Prediction_Order_{i+1}"] = abs_error

    # Prepare data for visualization
    error_melted = pd.DataFrame(error_data).melt(
        var_name="Prediction_Order", value_name="Absolute_Error"
    )

    # --- Boxplot: Distribution of Absolute Errors (MAE) by Prediction Order ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Prediction_Order", y="Absolute_Error", data=error_melted)
    plt.title("Distribution of Absolute Errors by Prediction Order")
    plt.xlabel("Prediction Order (1 = most recent prediction)")
    plt.ylabel("Absolute Error")
    plt.grid(True, alpha=0.5)

    # Generate file names based on input file name
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    svg_filename = f"{base_filename}_mae_boxplot.svg"
    png_filename = f"{base_filename}_mae_boxplot.png"

    plt.savefig(svg_filename, format="svg")
    plt.savefig(png_filename, format="png")

    plt.show()
