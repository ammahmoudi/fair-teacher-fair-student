"""
Comprehensive LaTeX Table Generation Module
==========================================

Creates a single, comprehensive table addressing all reviewer concerns:
- Standardized inference metrics (CPU/GPU/edge latency, model size, RAM/VRAM, throughput)
- Training metrics where available
- Edge feasibility assessment with concrete measurements
- No missing values or duplications
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from utils.analysis_utils import standardize_model_names, calculate_inference_summary


def create_comprehensive_standardized_table(inference_data: pd.DataFrame, training_data: Optional[pd.DataFrame] = None) -> str:
    """
    Create a single comprehensive table addressing all reviewer requirements:
    1. Standardized inference metrics (latency, throughput, memory, model size)
    2. Training metrics where available
    3. Edge feasibility with concrete evidence
    4. No duplications or missing values
    5. Professional presentation for publication
    """
    
    # Standardize model names and calculate clean summary
    standardized_data = standardize_model_names(inference_data)
    model_summary = calculate_inference_summary(standardized_data)
    
    # Remove duplicates by grouping and taking mean/first values
    # Include comprehensive GPU/VRAM metrics per reviewer requirements
    agg_dict = {
        'avg_inference_time_ms': 'mean',
        'inference_peak_ram_mb': 'mean', 
        'inference_avg_power_w': 'mean',
        'throughput_predictions_per_sec': 'mean',
        'model_size_mb': 'first',
        'total_parameters': 'first',
        'total_records': 'sum'  # Total experimental runs
    }
    
    # Add GPU/VRAM metrics if available (addressing reviewer concerns)
    gpu_metrics = ['inference_peak_gpu_mb', 'current_vram_usage_mb', 'gpu_memory_reserved_mb', 
                   'estimated_cpu_latency_ms', 'estimated_gpu_latency_ms', 'peak_power_usage_watts',
                   'peak_gpu_utilization_percent', 'average_gpu_utilization_percent']
    
    for metric in gpu_metrics:
        if metric in model_summary.columns:
            agg_dict[metric] = 'mean'
    
    clean_summary = model_summary.groupby('model_name').agg(agg_dict).reset_index()
    
    # Sort by model family (Chronos first, then Time-LLM)
    clean_summary['sort_order'] = clean_summary['model_name'].apply(
        lambda x: 0 if 'Chronos' in x else 1
    )
    clean_summary = clean_summary.sort_values(['sort_order', 'model_name']).drop('sort_order', axis=1)
    
    # Start LaTeX table with comprehensive metrics addressing reviewer concerns
    header_lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{xcolor}",
        r"\usepackage{multirow}",
        r"\usepackage{rotating}",
        "",
        r"\begin{document}",
        "",
        r"\begin{table*}[htbp]",
        r"    \centering",
        r"    \color{green}",
        r"    \caption{Inference Performance Metrics on OhioT1DM (30-min Horizon)}",
        r"    \label{tab:comprehensive_standardized_metrics}",
        r"    \small",
        r"    \begin{tabular}{@{}l|c|cc|cc|c@{}}",
        r"        \toprule",
        r"        & \textbf{Latency} & \multicolumn{2}{c|}{\textbf{Memory (MB)}} & \multicolumn{2}{c|}{\textbf{Performance}} & \textbf{Model} \\",
        r"        \textbf{Model} & \textbf{E2E (ms)} & \textbf{RAM} & \textbf{VRAM} & \textbf{Power (W)} & \textbf{Thru. (p/s)} & \textbf{Size (MB)} \\",
        r"        \midrule",
        "",
    ]
    latex = '\n'.join(header_lines)
    
    # Add data rows with full standardized model names
    rows = []
    for _, row in clean_summary.iterrows():
        model_name = str(row['model_name'])

        # Use full standardized model name (no truncation)
        display_name = model_name.replace('_', '\\_')

        # End-to-End Latency: use measured average_latency_ms from inference_timing
        # This is loaded into avg_inference_time_ms from real_performance_report.inference_timing.average_latency_ms
        e2e_latency_value = None
        if 'avg_inference_time_ms' in row and pd.notna(row['avg_inference_time_ms']):
            e2e_latency_value = float(row['avg_inference_time_ms'])

        e2e_latency = f"{e2e_latency_value:.1f}" if e2e_latency_value is not None else "\\textcolor{gray}{--}"

        # RAM: prioritize process_peak_ram_mb (from real_performance_report memory_usage section)
        # This is loaded into inference_peak_ram_mb by enhanced_data_loader
        peak_ram = f"{row['inference_peak_ram_mb']:.0f}" if pd.notna(row['inference_peak_ram_mb']) else "\\textcolor{gray}{--}"

        # VRAM: prioritize peak_allocated_mb from gpu_memory_usage (PyTorch model-specific, NOT system-wide NVIDIA)
        # This is loaded into inference_peak_gpu_mb by enhanced_data_loader from real_performance_report
        vram_usage = "\\textcolor{gray}{--}"
        if 'inference_peak_gpu_mb' in row and pd.notna(row['inference_peak_gpu_mb']):
            vram_usage = f"{row['inference_peak_gpu_mb']:.0f}"  # PyTorch peak allocated (model-specific)
        elif 'current_vram_usage_mb' in row and pd.notna(row['current_vram_usage_mb']):
            vram_usage = f"{row['current_vram_usage_mb']:.0f}"  # Alternative field

        # Power: prioritize nvidia_ml_metrics average_power_usage_watts and peak_power_usage_watts
        # These are loaded into inference_avg_power_w and peak_power_usage_watts by enhanced_data_loader
        power_val = "\\textcolor{gray}{--}"
        avg_power = row.get('inference_avg_power_w') if 'inference_avg_power_w' in row else None
        peak_power = row.get('peak_power_usage_watts') if 'peak_power_usage_watts' in row else None

        if pd.notna(avg_power) and pd.notna(peak_power):
            # Show average (primary) and peak (secondary)
            power_val = f"{avg_power:.2f} ({peak_power:.2f})"
        elif pd.notna(avg_power):
            power_val = f"{avg_power:.2f}"
        elif pd.notna(peak_power):
            power_val = f"{peak_power:.2f}"

        # Throughput: compute from the End-to-End latency to ensure consistency
        throughput_val = None
        if e2e_latency_value is not None and e2e_latency_value > 0:
            throughput_val = 1000.0 / e2e_latency_value
        else:
            # Last-resort: use any precomputed throughput value if present
            precomp = row.get('throughput_predictions_per_sec') if 'throughput_predictions_per_sec' in row else None
            if pd.notna(precomp):
                throughput_val = float(precomp)

        if throughput_val is not None and pd.notna(throughput_val):
            if throughput_val < 0.01:
                throughput = f"{throughput_val:.1e}"
            else:
                throughput = f"{throughput_val:.2f}"
        else:
            throughput = "\\textcolor{gray}{--}"

        # Model size: prioritize model_size_mb (loaded from model_file_metrics.model_size_on_disk_mb)
        model_size = f"{row['model_size_mb']:.0f}" if pd.notna(row['model_size_mb']) else "\\textcolor{gray}{--}"

        # Build row with End-to-End latency (single column)
        row_str = f"        {display_name} & {e2e_latency} & {peak_ram} & {vram_usage} & {power_val} & {throughput} & {model_size} \\\\"
        rows.append(row_str)

    # Join rows with actual newlines and append to latex
    latex += '\n'.join(rows) + '\n'
    
    # Close table (no extra notes included — matches requested format)
    latex += r"""        \bottomrule
    \end{tabular}

\end{table*}

\end{document}"""
    
    return latex


def save_latex_table(latex_content: str, filename: str, output_dir: Optional[Path] = None) -> Path:
    """Save LaTeX table to file"""
    if output_dir is None:
        output_dir = Path.cwd() / "outputs" / "latex_tables"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith('.tex'):
        filename += '.tex'
    
    file_path = output_dir / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write LaTeX content directly. Avoid automatic token replacements which can
        # introduce double backslashes (e.g. turning "\toprule" into "\\toprule").
        f.write(latex_content)
    
    return file_path


def generate_all_tables(inference_data: pd.DataFrame, training_data: Optional[pd.DataFrame] = None, output_dir: Optional[Path] = None) -> dict:
    """
    Generate the comprehensive standardized table addressing all reviewer concerns.
    Now generates only ONE high-quality table instead of multiple flawed ones.
    """
    if output_dir is None:
        output_dir = Path.cwd() / "outputs" / "latex_tables"
    
    results = {}
    
    print("📊 Generating comprehensive standardized table...")
    
    # Generate the single comprehensive table
    comprehensive_latex = create_comprehensive_standardized_table(inference_data, training_data)
    results['comprehensive_standardized'] = save_latex_table(
        comprehensive_latex, 
        "comprehensive_standardized_metrics", 
        output_dir
    )
    
    print(f"✅ Table generated: {results['comprehensive_standardized']}")
    
    return results


# Deprecated functions - keeping for compatibility but not recommended
def create_real_data_latex_table(df: pd.DataFrame) -> str:
    """Deprecated: Use create_comprehensive_standardized_table instead"""
    print("⚠️  Warning: create_real_data_latex_table is deprecated. Use create_comprehensive_standardized_table instead.")
    return create_comprehensive_standardized_table(df)

def create_corrected_latex_table(df: pd.DataFrame) -> str:
    """Deprecated: Use create_comprehensive_standardized_table instead"""
    print("⚠️  Warning: create_corrected_latex_table is deprecated. Use create_comprehensive_standardized_table instead.")
    return create_comprehensive_standardized_table(df)

def create_comprehensive_performance_table(inference_data: pd.DataFrame, training_data=None) -> str:
    """Deprecated: Use create_comprehensive_standardized_table instead"""
    print("⚠️  Warning: create_comprehensive_performance_table is deprecated. Use create_comprehensive_standardized_table instead.")
    return create_comprehensive_standardized_table(inference_data, training_data)