import logging
import os
import pandas as pd

from data_processing.refromat_results import reformat_results
from utils.latex_tools import generate_latex_table
from utils.plotting import plot_avg_predictions_plotly, plot_predictions_plotly

def generate_run_title(data_settings, llm_settings):
    # Extract dataset name from the train data path
    train_data_path = data_settings.get('path_to_train_data', '')
    dataset_name = train_data_path.split('/')[-1].split('-')[0] if train_data_path else 'unknown'

    method = llm_settings.get('method', 'unknown')
    model = llm_settings.get('model', llm_settings.get('llm_model', 'unknown')).replace('/','-')
    mode = llm_settings.get('mode', 'unknown')

    return f"patient{dataset_name}_{method}_{model}_{mode}"

def save_results_and_generate_plots(
    output_dir, predictions, targets, inputs, grouped_x_timestamps=None, grouped_y_timestamps=None, name=None, enable_plots=False
):
    """
    Save prediction results, reformat them, and generate plots.

    :param output_dir: Directory to save the predictions, formatted results, and plots.
    :param predictions: Array of predictions.
    :param targets: Array of ground truth values.
    :param inputs: Array of input data.
    :param grouped_x_timestamps: (Optional) List of grouped x timestamps as strings.
    :param grouped_y_timestamps: (Optional) List of grouped y timestamps as strings.
    :param name: (Optional) A name to use in the plot and file names.
    :param enable_plots: (Optional) If False, skip plot generation to save time and space. Default: False
    """
    # Debug: log the shapes of the inputs
    logging.debug(f"Shape of predictions: {predictions.shape}")
    logging.debug(f"Shape of targets: {targets.shape}")
    logging.debug(f"Shape of inputs: {inputs.shape}")

    if grouped_x_timestamps is not None:
        logging.debug(f"Number of grouped_x_timestamps: {len(grouped_x_timestamps)}")
    else:
        logging.debug("grouped_x_timestamps not provided.")

    if grouped_y_timestamps is not None:
        logging.debug(f"Number of grouped_y_timestamps: {len(grouped_y_timestamps)}")
    else:
        logging.debug("grouped_y_timestamps not provided.")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build the results dictionary
    results = {
        "inputs": [
            ", ".join(map(str, inputs[i].flatten())) for i in range(len(inputs))
        ],
        "ground_truth": [
            ", ".join(map(str, targets[i].flatten())) for i in range(len(targets))
        ],
        "predictions": [
            ", ".join(map(str, predictions[i].flatten())) for i in range(len(predictions))
        ],
    }

    # Add timestamps if provided
    if grouped_x_timestamps is not None:
        results["x_timestamps"] = [
            ", ".join(map(str, grouped_x_timestamps[i])) for i in range(len(grouped_x_timestamps))
        ]
    if grouped_y_timestamps is not None:
        results["y_timestamps"] = [
            ", ".join(map(str, grouped_y_timestamps[i])) for i in range(len(grouped_y_timestamps))
        ]

    # Use the provided name or default to "inference"
    file_name_prefix = name if name else "inference"

    # Save results to CSV
    save_path = os.path.join(output_dir, f"{file_name_prefix}_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    logging.info(f"Results saved to {save_path}")

    # Reformat and save the results
    reformatted_path = save_path.replace(".csv", "_reformatted.csv")
    reformated_results_df = reformat_results(save_path, output_csv_path=reformatted_path)
    logging.info(f"Reformatted results saved to {reformatted_path}")

    # Generate and save plots (only if enabled)
    if enable_plots:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_predictions_path = os.path.join(plots_dir, f"{file_name_prefix}_predictions")
        plot_avg_predictions_path = os.path.join(plots_dir, f"{file_name_prefix}_avg_predictions")

        plot_predictions_plotly(reformated_results_df, save_path=plot_predictions_path,name=name)
        plot_avg_predictions_plotly(reformated_results_df, save_path=plot_avg_predictions_path,name=name)

        logging.info(f"Plots saved to {plots_dir}")
    else:
        logging.info("Plot generation disabled (enable_plots=False)")





import pandas as pd

def convert_window_to_single_prediction(input_file, output_file="single_step_predictions.csv"):
    """
    Converts windowed time-series predictions into a single-step format by:
    1. Extracting the first step's true and predicted values.
    2. Extracting the last row's true and predicted values, ensuring all steps are captured.
    
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

    # Separate the last row into true and pred values
    last_true_values = last_row[true_cols].reset_index(drop=True)
    last_pred_values = last_row[pred_cols].reset_index(drop=True)

    # Create DataFrames for true and pred values
    last_true_df = pd.DataFrame(last_true_values, columns=["true"])
    last_pred_df = pd.DataFrame(last_pred_values, columns=["pred"])

    # Concatenate first step data with last row data
    result_df = pd.concat([first_step_df, last_true_df, last_pred_df], ignore_index=True)

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Converted window predictions saved to {output_file}")

# Example Usage:
# convert_window_to_single_prediction("your_file.csv", "single_step_predictions.csv")



def summarize_by_column(df: pd.DataFrame, columns: list[str], filter_seq_pred: bool = False) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Summarizes the given DataFrame by computing the mean and standard deviation 
    of specific metrics while grouping by configuration columns.

    This function removes predefined metric columns (`'rmse'`, `'mae'`, `'mape'`) 
    and user-specified columns from the grouping columns. It then groups the 
    remaining data by the configuration columns and calculates the mean and 
    standard deviation for each metric.

    If `filter_seq_pred` is set to `True`, the function dynamically identifies 
    all unique (`seq`, `pred`) pairs present in the dataset and returns a 
    dictionary of DataFrames, where keys represent (`seq`, `pred`) pairs such as 
    `"6_6"` or `"8_10"`.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to summarize.
        columns (list[str]): A list of column names to exclude from the 
                             configuration columns (e.g., 'seed', 'log_datetime').
        filter_seq_pred (bool, optional): If True, returns a dictionary of DataFrames
                                          filtered by unique (`seq`, `pred`) pairs 
                                          found in the dataset. Defaults to False.

    Returns:
        dict[str, pd.DataFrame] | pd.DataFrame: 
            - If `filter_seq_pred` is False, returns a single summary DataFrame.
            - If `filter_seq_pred` is True, returns a dictionary of DataFrames,
              where keys are `"seq_pred"` pairs (e.g., `"6_6"`, `"8_10"`).

    Example:
        >>> summary_dict = summarize_by_column(df, ["seed", "log_datetime"], filter_seq_pred=True)
        >>> summary_dict.keys()  # Shows all unique ('seq', 'pred') pairs in the data
        >>> summary_dict["6_6"].head()  # Access DataFrame filtered for seq=6, pred=6

    The output DataFrame will have columns formatted as `metric_mean` and `metric_std`.
    """

    # Define the configuration columns (excluding 'seed', 'patient_id', 'log_datetime')
    config_columns = list(df.columns)
    
    # Metrics to summarize
    metrics = ['rmse', 'mae', 'mape']
    
    # Remove metric columns from the grouping columns
    for metric in metrics:
        if metric in config_columns:
            config_columns.remove(metric)
    
    # Remove user-specified columns from the grouping columns
    for column in columns:
        if column in config_columns:
            config_columns.remove(column)

    # Group by configuration columns and calculate mean and standard deviation for each metric
    summary_df = df.groupby(config_columns)[metrics].agg(['mean', 'std']).reset_index()

    # Rename columns for clarity
    summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns]
    if 'context' in summary_df.columns:summary_df['seq']=summary_df['context']

    # Convert 'seq' and 'pred' to integers if they exist in the DataFrame
    if 'seq' in summary_df.columns and 'pred' in summary_df.columns:
        summary_df['seq'] = summary_df['seq'].astype(int)
        summary_df['pred'] = summary_df['pred'].astype(int)

        # If filtering is enabled, dynamically find all unique (seq, pred) pairs
        if filter_seq_pred:
            unique_pairs = summary_df[['seq', 'pred']].drop_duplicates().values.tolist()
            filtered_dfs = {}

            for seq_val, pred_val in unique_pairs:
                filtered_df = summary_df[(summary_df['seq'] == seq_val) & (summary_df['pred'] == pred_val)]
                filtered_dfs[f"{seq_val}_{pred_val}"] = filtered_df

            return filtered_dfs  # Return dictionary of DataFrames

    return summary_df  # Return a single DataFrame if no filtering is applied



def generate_latex_tables(
    df: pd.DataFrame,
    title_template: str = "Zero-shot Performance for Time-LLM Models ({time_horizon}-minute Forecast)",
    label_template: str = "tab:timellm_zero_shot_{time_horizon}min",
    columns: list[str]=['log_datetime','seed'],
    freq: int=5,
    save: bool = False
):
    """
    Automates the process of summarizing data, generating LaTeX tables, and optionally saving them.

    This function:
    1. Summarizes the input DataFrame using `summarize_by_column`, identifying all (`seq`, `pred`) pairs.
    2. Generates LaTeX tables for each (`seq`, `pred`) pair, embedding the correct time horizon 
       in the title and label based on user-defined templates.
    3. Optionally saves each LaTeX table using the LaTeX label as the filename.

    Args:
        df (pd.DataFrame): The input DataFrame to summarize and generate LaTeX tables from.
        columns (list[str]): A list of columns to exclude from the grouping.
        freq (int): The forecasting frequency (e.g., 30 for 30-minute forecasts).
        title_template (str, optional): A template for the LaTeX table title.
                                        Use `{seq}` for sequence value and `{time_horizon}` for computed time horizon.
                                        Defaults to "Zero-shot Performance for Time-LLM Models ({time_horizon}-minute Forecast)".
        label_template (str, optional): A template for the LaTeX table label.
                                        Use `{seq}` for sequence value and `{time_horizon}` for computed time horizon.
                                        Defaults to "tab:timellm_zero_shot_{time_horizon}min".
        save (bool, optional): If True, saves the generated LaTeX tables to `.tex` files using the label as the filename.
                               Defaults to False.

    Returns:
        dict[str, str]: A dictionary where keys are `"seq_pred"` pairs (e.g., `"6_6"`, `"8_10"`) 
                        and values are the corresponding LaTeX table strings.

    Example:
        >>> latex_tables = generate_latex_tables(
                df, ["seed", "log_datetime"], freq=30, save=True,
                title_template="Performance of Time-LLM ({time_horizon} min)",
                label_template="tab:timellm_{time_horizon}min"
            )
        >>> print(latex_tables["6_6"])  # Print LaTeX table for seq=6, pred=6
    """

    # Step 1: Summarize Data and Find Unique Pairs
    summarized_dfs = summarize_by_column(df, columns, filter_seq_pred=True)

    # Step 2: Generate LaTeX Tables for Each (`seq`, `pred`) Pair
    latex_tables = {}
    print(summarized_dfs.keys)
    for seq_pred, summarized_df in summarized_dfs.items():
        seq, pred = map(int, seq_pred.split("_"))  # Extract sequence and prediction values

        # Compute the time horizon based on freq and pred
        time_horizon = pred * freq

        # Generate title and label dynamically using templates
        title = title_template.format(seq=seq, time_horizon=time_horizon)
        label = label_template.format(seq=seq, time_horizon=time_horizon)

        # Generate LaTeX table
        latex_table = generate_latex_table(summarized_df, title, label)
        latex_tables[seq_pred] = latex_table

        # Step 3: Save the LaTeX Table to a File (If Required)
        if save:
            # Use the label as the filename, replacing colons with underscores for compatibility
            filename = f"{label.replace(':', '_')}.tex"
            with open(filename, "w") as f:
                f.write(latex_table)

    return latex_tables

import pandas as pd

def generate_model_comparison_from_dict(
    model_dfs: dict[str, pd.DataFrame],
    seq: int,
    pred: int,
    columns: list[str] = ['log_datetime', 'seed'],
    label: str = "tab:models_performance_comparison",
    title: str = "Model Performance Comparison",
    save: bool = False
) -> tuple[pd.DataFrame, str]:
    """
    Summarizes multiple DataFrames from a dictionary, extracts model performance for a specific (seq, pred) pair, 
    and generates a LaTeX table.

    This function:
    1. Summarizes each input DataFrame using `summarize_by_column()`.
    2. Extracts RMSE & MAE values for the given `(seq, pred, model)` combination.
    3. Computes the model-wise averages.
    4. Creates a performance comparison table (both DataFrame & LaTeX).

    Args:
        model_dfs (dict[str, pd.DataFrame]): Dictionary where keys are model categories (e.g., "TimeLLM") 
                                             and values are DataFrames containing results for sub-models.
        seq (int): Sequence length for which performance is extracted.
        pred (int): Prediction length for which performance is extracted.
        columns (list[str]): Columns to exclude when summarizing.
        label (str): LaTeX label for the table.
        title (str): Title of the LaTeX table.
        save (bool): If True, saves the LaTeX table to a `.tex` file.

    Returns:
        tuple[pd.DataFrame, str]: 
            - The model comparison DataFrame (with RMSE and MAE).
            - The LaTeX formatted table as a string.

    Example:
        >>> model_dfs = {
                "TimeLLM": df1,  # Contains bert, gpt2, etc.
                "Chronos": df2   # Contains transformer, lstm, etc.
            }
        >>> comparison_df, latex_code = generate_model_comparison_from_dict(model_dfs, seq=6, pred=9, save=True)
        >>> print(comparison_df)
        >>> print(latex_code)
    """

    model_performance = []

    # Step 1: Process each DataFrame and extract RMSE & MAE for (seq, pred)
    for model_category, df in model_dfs.items():
        summarized_dfs = summarize_by_column(df, columns, filter_seq_pred=True)

        # Ensure (seq, pred) exists in the summarized results
        key = f"{seq}_{pred}"
        if key not in summarized_dfs:
            print(f"Warning: (seq={seq}, pred={pred}) not found for {model_category}. Skipping...")
            continue

        summary_df = summarized_dfs[key]

        if summary_df.empty:
            print(f"Warning: No data found for {model_category} (seq={seq}, pred={pred}). Skipping...")
            continue

        # Step 2: Extract RMSE & MAE for each sub-model in the category
        for sub_model in summary_df["model"].unique():
            sub_model_df = summary_df[summary_df["model"] == sub_model]

            # Compute mean RMSE & MAE for the sub-model
            mae_avg = sub_model_df['mae_mean'].mean()
            rmse_avg = sub_model_df['rmse_mean'].mean()

            # Store results with full model name
            full_model_name = f"{model_category} ({sub_model})"
            model_performance.append([full_model_name, rmse_avg, mae_avg])
    # **Sort the DataFrame by MAE in ascending order**
    # Step 3: Convert results to DataFrame
    comparison_df = pd.DataFrame(model_performance, columns=["Model", "RMSE", "MAE"])
    comparison_df = comparison_df.sort_values(by="RMSE", ascending=True)

    # Step 4: Generate LaTeX Table Code
    latex_code = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{{title} (Seq {seq}, Pred {pred})}}
    \\begin{{tabular}}{{lcc}}
        \\toprule
        \\textbf{{Model}} & \\textbf{{RMSE}} & \\textbf{{MAE}} \\\\
        \\midrule
"""

    for _, row in comparison_df.iterrows():
        latex_code += f"        {row['Model']} & {row['RMSE']:.2f} & {row['MAE']:.2f} \\\\\n"

    latex_code += """        \\bottomrule
    \\end{tabular}
    \\label{""" + label + """}
\\end{table}
"""

    # Step 5: Save LaTeX Table (if required)
    if save:
        filename = f"{label.replace(':', '_')}_seq{seq}_pred{pred}.tex"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex_code)

    return comparison_df, latex_code



def prepare_chronos_robustness_table(df: pd.DataFrame, model_filter=None):
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    if model_filter:
        df = df[df["model"] == model_filter]

    if "mape" in df.columns and df["mape"].isna().all():
        df = df.drop(columns=["mape"])
    df = df.rename(columns={"context": "seq"})

    grouped = df.groupby(["scenario", "corruption_type", "patient_id"])[["mae", "rmse"]].agg(["mean", "std"]).reset_index()
    grouped.columns = ["scenario", "corruption_type", "patient_id", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]

    grouped["mae_fmt"] = grouped.apply(lambda row: f"{row['mae_mean']:.2f} ± {row['mae_std']:.2f}", axis=1)
    grouped["rmse_fmt"] = grouped.apply(lambda row: f"{row['rmse_mean']:.2f} ± {row['rmse_std']:.2f}", axis=1)

    # Map training/eval config and corruption type
    scenario_map = {
        "fine-tuned": "Noise-aware evaluation",
        "zero-shot": "Test-time missingness"
    }
    corruption_map = {
        "missing_periodic": "Periodic missing",
        "missing_random": "Random missing",
        "noisy": "Noisy"
    }

    grouped["scenario_label"] = grouped["scenario"].map(scenario_map)
    grouped["corruption_label"] = grouped["corruption_type"].map(corruption_map)
    grouped["row_label"] = grouped["corruption_label"] + " — " + grouped["scenario_label"]

    # Sort by corruption and scenario order
    corruption_order = ["Noisy", "Periodic missing", "Random missing"]
    scenario_order = ["Noise-aware evaluation", "Test-time missingness"]

    grouped["corruption_label"] = grouped["corruption_label"].astype(CategoricalDtype(corruption_order, ordered=True))
    grouped["scenario_label"] = grouped["scenario_label"].astype(CategoricalDtype(scenario_order, ordered=True))

    grouped = grouped.sort_values(["corruption_label", "scenario_label"])

    # Create pivoted DataFrame
    mae_pivot = grouped.pivot(index="row_label", columns="patient_id", values="mae_fmt")
    rmse_pivot = grouped.pivot(index="row_label", columns="patient_id", values="rmse_fmt")

    columns = []
    for col in mae_pivot.columns:
        columns.append((col, "MAE"))
        columns.append((col, "RMSE"))

    result = pd.DataFrame(index=mae_pivot.index, columns=pd.MultiIndex.from_tuples(columns))
    for col in mae_pivot.columns:
        result[(col, "MAE")] = mae_pivot[col]
        result[(col, "RMSE")] = rmse_pivot[col]

    return result.sort_index(axis=1, level=0)

def generate_latex_chronos_robustness_table(df: pd.DataFrame, caption: str, label: str):
    import re

    patients = sorted(set(idx for idx, _ in df.columns))
    col_format = "l" + "|".join(["cc" for _ in patients])

    latex = [
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule"
    ]

    header1 = ["\\textbf{Scenario}"]
    for patient in patients:
        header1.append(f"\\multicolumn{{2}}{{c}}{{\\textbf{{Patient {patient}}}}}")
    latex.append(" & ".join(header1) + " \\\\")

    header2 = [""]
    for _ in patients:
        header2.extend(["MAE", "RMSE"])
    latex.append(" & ".join(header2) + " \\\\")
    latex.append("\\cmidrule(lr){2-" + f"{2 * len(patients) + 1}" + "}")

    last_corruption = None
    for row_label, row in df.iterrows():
        corruption = row_label.split(" — ")[0]
        if last_corruption is not None and corruption != last_corruption:
            latex.append("\\midrule")
        last_corruption = corruption

        values = [str(v) if pd.notna(v) else "--" for v in row]
        latex.append(f"{row_label} & " + " & ".join(values) + " \\\\")

    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ]

    return "\n".join(latex)
