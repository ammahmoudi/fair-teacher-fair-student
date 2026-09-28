import pandas as pd

def generate_latex_table(df: pd.DataFrame, title: str, label: str) -> str:
    """
    Generates a properly formatted LaTeX table from a DataFrame containing model evaluation metrics.

    Args:
        df (pd.DataFrame): A DataFrame with columns 'patient_id', 'model', 'mae_mean', 'mae_std', 'rmse_mean', and 'rmse_std'.
        title (str): The title of the LaTeX table.
        label (str): The LaTeX label for referencing the table.

    Returns:
        str: A string containing the LaTeX formatted table.

    The function organizes the data by unique patient IDs and models, displaying Mean Absolute Error (MAE) and
    Root Mean Squared Error (RMSE) with their respective standard deviations. It also computes and appends 
    model-wise average MAE and RMSE values at the end of the table.
    """
    
    models = df['model'].unique()

    # Begin LaTeX table structure
    latex_code = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{{title}}}
    \\resizebox{{\\columnwidth}}{{!}}{{%
    \\begin{{tabular}}{{c|""" + "cc|" * (len(models) - 1) + "cc}\n"

    # Table top rule
    latex_code += "        \\toprule\n"

    # Header row with model names
    model_headers = " & ".join(
        [f"\\multicolumn{{2}}{{c{'|' if i < len(models) - 1 else ''}}}{{\\textbf{{{m}}}}}" for i, m in enumerate(models)]
    )
    latex_code += f"        {{\\textbf{{Patient ID}}}} & {model_headers} \\\\\n"

    # Column-wise rule alignment for metrics
    col_ranges = [f"{2 + i * 2}-{3 + i * 2}" for i in range(len(models))]
    latex_code += "        " + " ".join([f"\\cmidrule(lr){{{r}}}" for r in col_ranges]) + "\n"
    latex_code += "         & " + " & ".join(["MAE & RMSE"] * len(models)) + " \\\\\n"
    latex_code += "         \\midrule\n"

    # Patient-wise data rows
    num_rows = 0
    for patient in df['patient_id'].unique():
        row = [str(patient)]
        for model in models:
            model_data = df[(df['patient_id'] == patient) & (df['model'] == model)]
            if not model_data.empty:
                mae_mean = model_data['mae_mean'].values[0]
                mae_std = model_data['mae_std'].values[0]
                rmse_mean = model_data['rmse_mean'].values[0]
                rmse_std = model_data['rmse_std'].values[0]
                row.append(f"{mae_mean:.2f} $\\pm$ {mae_std:.2f}")
                row.append(f"{rmse_mean:.2f} $\\pm$ {rmse_std:.2f}")
            else:
                row.append("--")  # Placeholder if missing data
                row.append("--")
        
        latex_code += "         " + " & ".join(row) + " \\\\\n"
        num_rows += 1

    # Add midrule only if there are actual data rows
    if num_rows > 0:
        latex_code += "         \\midrule\n"

    # Compute model-wise averages
    avg_row = ["\\textbf{Avg}"]
    for model in models:
        model_data = df[df['model'] == model]
        if not model_data.empty:
            avg_mae = model_data['mae_mean'].mean()
            avg_rmse = model_data['rmse_mean'].mean()
            avg_row.append(f"\\textbf{{{avg_mae:.2f}}}")
            avg_row.append(f"\\textbf{{{avg_rmse:.2f}}}")
        else:
            avg_row.append("--")
            avg_row.append("--")

    latex_code += "         " + " & ".join(avg_row) + " \\\\\n"

    # Table bottom rule
    latex_code += "         \\bottomrule\n"
    latex_code += "    \\end{tabular}%\n"
    latex_code += "    }\n"
    latex_code += f"    \\label{{{label}}}\n"
    latex_code += "\\end{table}\n"

    return latex_code

import pandas as pd

import pandas as pd

def generate_cross_patient_latex_table_from_df(
    df: pd.DataFrame,
    label: str = "tab:cross_patient_performance",
    title: str = "Cross-Patient Model Performance",
    save: bool = False
) -> str:
    # Filter necessary columns
    df = df[['model', 'train_patient_id', 'test_patient_id', 'rmse', 'mae']]

    # Sort for consistent output
    df = df.sort_values(by=['model', 'train_patient_id', 'test_patient_id'])

    # Start LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{{title}}}
    \\begin{{tabular}}{{lcc|cc}}
        \\toprule
        \\textbf{{Model}} & \\textbf{{Train ID}} & \\textbf{{Test ID}} & \\textbf{{RMSE}} & \\textbf{{MAE}} \\\\
        \\midrule
"""

    # Group by model and format rows
    for model, group in df.groupby('model'):
        first_row = True
        for _, row in group.iterrows():
            model_cell = model if first_row else ""
            latex_table += f"        {model_cell} & {row['train_patient_id']} & {row['test_patient_id']} & {row['rmse']:.2f} & {row['mae']:.2f} \\\\\n"
            first_row = False

    # Close the LaTeX table
    latex_table += f"""        \\bottomrule
    \\end{{tabular}}
    \\label{{{label}}}
\\end{{table}}
"""

    # Save to file if requested
    if save:
        filename = f"{label.replace(':', '_')}.tex"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex_table)

    return latex_table
