"""
Enhanced analysis utilities for Fair Teacher–Fair Student project
with improved data handling for both Chronos and Time-LLM models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def standardize_model_names(data):
    """
    Standardize model names for consistent presentation across all analysis.
    
    Chronos models: chronos-t5-base → Chronos-T5-Base, chronos-t5-tiny → Chronos-T5-Tiny
    Time-LLM models: BERT → Time-LLM-BERT, GPT2 → Time-LLM-GPT-2, 
                    LLAMA → Time-LLM-LLaMA, tinybert → Time-LLM-TinyBERT
    """
    if 'model_name' not in data.columns:
        return data
    
    name_mapping = {
        'chronos-t5-base': 'Chronos-T5-Base',
        'chronos-t5-tiny': 'Chronos-T5-Tiny',
        'BERT': 'Time-LLM-BERT',
        'GPT2': 'Time-LLM-GPT-2',
        'LLAMA': 'Time-LLM-LLaMA',
        'tinybert': 'Time-LLM-TinyBERT (Distilled)'
    }
    
    data_copy = data.copy()
    data_copy['model_name'] = data_copy['model_name'].replace(name_mapping)
    return data_copy

def calculate_inference_summary(data):
    """
    Calculate comprehensive inference performance summary
    """
    # Standardize model names first
    data = standardize_model_names(data)
    
    summary_data = []
    
    for model_name in data['model_name'].unique():
        model_data = data[data['model_name'] == model_name]
        
                # Priority strategy for timing records:
        # 1) For Time-LLM models: prefer real_performance_reports (have accurate inference_timing.average_latency_ms)
        # 2) For Chronos models: prefer efficiency_reports (real_performance_reports contain aggregates)
        # Use threshold < 1000ms to ensure we get per-inference timings, not batch/aggregate
        timing_records = pd.DataFrame()
        
        # Try real_performance_reports first if they have clean per-inference timings (< 1s)
        if 'report_type' in model_data.columns:
            real_perf = model_data[model_data['report_type'] == 'real_performance_reports']
            if not real_perf.dropna(subset=['avg_inference_time_ms']).empty:
                # Use stricter threshold < 1000ms to catch true per-inference measurements
                clean_real_perf = real_perf[real_perf['avg_inference_time_ms'] < 1000]
                if not clean_real_perf.empty:
                    timing_records = clean_real_perf
        
        # Fallback: efficiency_reports (often have clean per-inference timings, especially for Chronos)
        if timing_records.empty and 'report_type' in model_data.columns:
            efficiency = model_data[model_data['report_type'] == 'efficiency_reports']
            clean_efficiency = efficiency[(efficiency['avg_inference_time_ms'].notna()) & (efficiency['avg_inference_time_ms'] < 1000)]
            if not clean_efficiency.empty:
                timing_records = clean_efficiency
        
        # Fallback: comprehensive_reports
        if timing_records.empty and 'report_type' in model_data.columns:
            comprehensive = model_data[model_data['report_type'] == 'comprehensive_reports']
            clean_comprehensive = comprehensive[(comprehensive['avg_inference_time_ms'].notna()) & (comprehensive['avg_inference_time_ms'] < 1000)]
            if not clean_comprehensive.empty:
                timing_records = clean_comprehensive
        
        # Fallback: real_performance_reports with slightly larger threshold (< 10s) for slower models
        if timing_records.empty and 'report_type' in model_data.columns:
            real_perf_relaxed = model_data[model_data['report_type'] == 'real_performance_reports']
            clean_relaxed = real_perf_relaxed[(real_perf_relaxed['avg_inference_time_ms'].notna()) & (real_perf_relaxed['avg_inference_time_ms'] < 10000)]
            if not clean_relaxed.empty:
                timing_records = clean_relaxed
        
        # Fallback: inference-mode records with reasonable timing values
        if timing_records.empty and 'mode' in model_data.columns:
            inf_mode = model_data[model_data['mode'] == 'inference']
            clean_inf = inf_mode[(inf_mode['avg_inference_time_ms'].notna()) & (inf_mode['avg_inference_time_ms'] < 10000)]
            if not clean_inf.empty:
                timing_records = clean_inf
        
        # Last resort: any record with timing (but filter out obvious aggregates > 10s)
        if timing_records.empty:
            all_with_timing = model_data.dropna(subset=['avg_inference_time_ms'])
            timing_records = all_with_timing[all_with_timing['avg_inference_time_ms'] < 10000]
            if timing_records.empty:
                timing_records = all_with_timing  # Accept even large values if nothing else
        memory_records = model_data.dropna(subset=['inference_peak_ram_mb'])
        power_records = model_data.dropna(subset=['inference_avg_power_w'])
        
        # Calculate metrics
        metric_record = {
            'model_name': model_name,
            'total_records': len(model_data),
            'records_with_timing': len(timing_records),
            'records_with_memory': len(memory_records),
            'records_with_power': len(power_records),
        }
        
        # Timing metrics
        if not timing_records.empty:
            # Use median to reduce sensitivity to outliers across mixed measurement contexts
            median_time = timing_records['avg_inference_time_ms'].median()
            metric_record.update({
                'avg_inference_time_ms': float(median_time),
                'min_inference_time_ms': float(timing_records['avg_inference_time_ms'].min()),
                'max_inference_time_ms': float(timing_records['avg_inference_time_ms'].max()),
                'std_inference_time_ms': float(timing_records['avg_inference_time_ms'].std()) if timing_records['avg_inference_time_ms'].std() is not None else None,
                'latency_source': 'measured'
            })

            # Calculate throughput from the chosen latency (consistent with displayed CPU latency)
            avg_time = metric_record['avg_inference_time_ms']
            metric_record['throughput_predictions_per_sec'] = 1000.0 / avg_time if avg_time > 0 else 0
        else:
            # No measured timings available: fall back to estimated CPU latency if present
            est_cpu = model_data.get('estimated_cpu_latency_ms') if 'estimated_cpu_latency_ms' in model_data else None
            est_vals = model_data['estimated_cpu_latency_ms'].dropna() if ('estimated_cpu_latency_ms' in model_data and not model_data['estimated_cpu_latency_ms'].dropna().empty) else None
            if est_vals is not None and len(est_vals) > 0:
                est_value = float(est_vals.mean())
                metric_record.update({
                    'avg_inference_time_ms': est_value,
                    'min_inference_time_ms': est_value,
                    'max_inference_time_ms': est_value,
                    'std_inference_time_ms': None,
                    'latency_source': 'estimated'
                })
                metric_record['throughput_predictions_per_sec'] = 1000.0 / est_value if est_value > 0 else 0
        
        # Memory metrics - prioritize real_performance_report records
        if not memory_records.empty:
            # Prefer real_performance_reports for process_peak_ram_mb (most accurate)
            if 'report_type' in memory_records.columns:
                real_perf_mem = memory_records[memory_records['report_type'] == 'real_performance_reports']
                if not real_perf_mem.empty:
                    memory_records = real_perf_mem
            
            metric_record.update({
                'inference_peak_ram_mb': memory_records['inference_peak_ram_mb'].median(),
                'min_ram_mb': memory_records['inference_peak_ram_mb'].min(),
                'max_ram_mb': memory_records['inference_peak_ram_mb'].max(),
            })
        
        # Power metrics - prioritize real_performance_report nvidia_ml_metrics
        if not power_records.empty:
            # Prefer real_performance_reports for nvidia_ml_metrics.average_power_usage_watts
            if 'report_type' in power_records.columns:
                real_perf_power = power_records[power_records['report_type'] == 'real_performance_reports']
                if not real_perf_power.empty:
                    power_records = real_perf_power
            
            metric_record.update({
                'inference_avg_power_w': power_records['inference_avg_power_w'].median(),
                'min_power_w': power_records['inference_avg_power_w'].min(),
                'max_power_w': power_records['inference_avg_power_w'].max(),
            })
        
        # GPU/VRAM metrics (prioritize PyTorch for model-specific, NVIDIA ML for hardware-level)
        gpu_vram_metrics = ['inference_peak_gpu_mb', 'current_vram_usage_mb', 'gpu_avg_allocated_mb', 'gpu_reserved_mb',
                           'estimated_cpu_latency_ms', 'estimated_gpu_latency_ms', 'peak_power_usage_watts',
                           'peak_gpu_utilization_percent', 'average_gpu_utilization_percent',
                           'nvidia_system_vram_mb', 'nvidia_avg_system_vram_mb']  # Keep system-wide for comparison
        
        for gpu_metric in gpu_vram_metrics:
            if gpu_metric in model_data.columns:
                gpu_records = model_data.dropna(subset=[gpu_metric])
                if not gpu_records.empty:
                    # PRIORITY: real_performance_reports > comprehensive_reports > efficiency_reports
                    if gpu_metric in ['current_vram_usage_mb', 'inference_peak_gpu_mb', 'gpu_avg_allocated_mb']:
                        # Prioritize real_performance_reports (gpu_memory_usage.peak_allocated_mb - most accurate)
                        if 'report_type' in gpu_records.columns:
                            real_perf_gpu = gpu_records[gpu_records['report_type'] == 'real_performance_reports']
                            if not real_perf_gpu.empty:
                                metric_record[gpu_metric] = real_perf_gpu[gpu_metric].median()
                                continue
                            # Fallback: comprehensive_reports (PyTorch)
                            pytorch_records = gpu_records[gpu_records['report_type'] == 'comprehensive_reports']
                            if not pytorch_records.empty:
                                metric_record[gpu_metric] = pytorch_records[gpu_metric].median()
                                continue
                        # Last resort: use all records
                        metric_record[gpu_metric] = gpu_records[gpu_metric].median()
                    else:
                        # For other metrics, prefer real_performance_reports then use median
                        if 'report_type' in gpu_records.columns:
                            real_perf = gpu_records[gpu_records['report_type'] == 'real_performance_reports']
                            if not real_perf.empty:
                                gpu_records = real_perf
                        metric_record[gpu_metric] = gpu_records[gpu_metric].median()
        
        # Model characteristics (use first available)
        first_record = model_data.iloc[0]
        metric_record.update({
            'total_parameters': first_record.get('total_parameters'),
            'model_size_mb': first_record.get('model_size_mb'),
            'model_architecture': first_record.get('model_architecture'),
        })
        
        # Edge feasibility assessment
        feasibility_values = model_data['edge_feasibility'].dropna().unique()
        if len(feasibility_values) > 0:
            # Prioritize the most restrictive feasibility assessment
            feasibility_priority = {'highly_feasible': 4, 'feasible': 3, 'challenging': 2, 'unknown': 1}
            best_feasibility = max(feasibility_values, key=lambda x: feasibility_priority.get(str(x), 0))
            metric_record['edge_feasibility'] = best_feasibility
        else:
            metric_record['edge_feasibility'] = 'unknown'
        
        summary_data.append(metric_record)
    
    return pd.DataFrame(summary_data)


def create_inference_plots(data, save_path=None):
    """
    Create comprehensive inference performance plots
    Generates two separate visualizations:
    1. 2x2 Performance Dashboard
    2. 6-subplot Detailed Model Analysis
    """
    # Standardize model names first
    data = standardize_model_names(data)
    
    # Calculate inference summary with standardized names
    inference_summary = calculate_inference_summary(data)
    
    # Group by model_name and aggregate to remove duplicates
    plot_data = inference_summary.groupby('model_name').agg({
        'avg_inference_time_ms': 'mean',
        'inference_peak_ram_mb': 'mean',
        'throughput_predictions_per_sec': 'mean',
        'model_size_mb': 'first',
        'inference_avg_power_w': 'mean',
        'total_parameters': 'first'
    }).reset_index()
    
    print(f"📊 Plotting {len(plot_data)} unique models: {plot_data['model_name'].tolist()}")
    
    # ========== PLOT 1: 2x2 Performance Dashboard ==========
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('LLM Performance Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Sort by inference time for consistent ordering
    plot_data_sorted = plot_data.sort_values('avg_inference_time_ms')
    
    # 1. Inference Time Comparison
    bars1 = ax1.bar(plot_data_sorted['model_name'], plot_data_sorted['avg_inference_time_ms'], 
                   color='skyblue', alpha=0.7, edgecolor='darkblue')
    ax1.set_title('Average Inference Time', fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        if not pd.isna(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    # 2. Memory Usage Comparison
    bars2 = ax2.bar(plot_data_sorted['model_name'], plot_data_sorted['inference_peak_ram_mb'], 
                   color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_title('Peak Memory Usage', fontweight='bold')
    ax2.set_ylabel('Memory (MB)')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        if not pd.isna(height):
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.0f}MB', ha='center', va='bottom', fontsize=9)
    
    # 3. Throughput Comparison
    throughput_data = plot_data_sorted.dropna(subset=['throughput_predictions_per_sec'])
    if not throughput_data.empty:
        bars3 = ax3.bar(throughput_data['model_name'], throughput_data['throughput_predictions_per_sec'], 
                       color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax3.set_title('Prediction Throughput', fontweight='bold')
        ax3.set_ylabel('Predictions/Second')
        ax3.tick_params(axis='x', rotation=45)
        for bar in bars3:
            height = bar.get_height()
            if not pd.isna(height):
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No throughput data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Prediction Throughput (No Data)', fontweight='bold')
    
    # 4. Model Size Comparison
    size_data = plot_data_sorted.dropna(subset=['model_size_mb'])
    if not size_data.empty:
        bars4 = ax4.bar(size_data['model_name'], size_data['model_size_mb'], 
                       color='gold', alpha=0.7, edgecolor='orange')
        ax4.set_title('Model Size', fontweight='bold')
        ax4.set_ylabel('Size (MB)')
        ax4.tick_params(axis='x', rotation=45)
        for bar in bars4:
            height = bar.get_height()
            if not pd.isna(height):
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.0f}MB', ha='center', va='bottom', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No model size data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Size (No Data)', fontweight='bold')
    
    plt.tight_layout()
    
    # ========== PLOT 2: 6-subplot Detailed Model Analysis ==========
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('Detailed Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # 1. Inference Time vs Memory Usage Scatter
    ax = axes[0]
    scatter_data = plot_data.dropna(subset=['avg_inference_time_ms', 'inference_peak_ram_mb'])
    if not scatter_data.empty:
        scatter = ax.scatter(scatter_data['avg_inference_time_ms'], scatter_data['inference_peak_ram_mb'], 
                           c=range(len(scatter_data)), cmap='viridis', s=100, alpha=0.7)
        for i, txt in enumerate(scatter_data['model_name']):
            ax.annotate(txt, (scatter_data.iloc[i]['avg_inference_time_ms'], 
                           scatter_data.iloc[i]['inference_peak_ram_mb']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8, rotation=15)
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Peak RAM (MB)')
    ax.set_title('Time vs Memory Trade-off')
    ax.grid(True, alpha=0.3)
    
    # 2. Power Consumption (if available)
    ax = axes[1]
    power_data = plot_data.dropna(subset=['inference_avg_power_w'])
    if not power_data.empty:
        bars = ax.bar(power_data['model_name'], power_data['inference_avg_power_w'], 
                     color='red', alpha=0.7, edgecolor='darkred')
        ax.set_title('Average Power Consumption')
        ax.set_ylabel('Power (W)')
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.1f}W', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No power data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Power Consumption (No Data)')
    
    # 3. Model Parameters Count (if available)
    ax = axes[2]
    param_data = plot_data.dropna(subset=['total_parameters'])
    if not param_data.empty:
        # Convert to millions for readability
        param_millions = param_data['total_parameters'] / 1e6
        bars = ax.bar(param_data['model_name'], param_millions, 
                     color='purple', alpha=0.7, edgecolor='darkviolet')
        ax.set_title('Model Parameters Count')
        ax.set_ylabel('Parameters (Millions)')
        ax.tick_params(axis='x', rotation=45)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.1f}M', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No parameter count data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Model Parameters (No Data)')
    
    # 4. Efficiency Ratio (Throughput per MB)
    ax = axes[3]
    efficiency_data = plot_data.dropna(subset=['throughput_predictions_per_sec', 'model_size_mb'])
    if not efficiency_data.empty:
        efficiency_ratio = efficiency_data['throughput_predictions_per_sec'] / efficiency_data['model_size_mb']
        bars = ax.bar(efficiency_data['model_name'], efficiency_ratio, 
                     color='cyan', alpha=0.7, edgecolor='darkcyan')
        ax.set_title('Efficiency Ratio (Throughput/Size)')
        ax.set_ylabel('Predictions/sec per MB')
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for efficiency calculation', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Efficiency Ratio (Insufficient Data)')
    
    # 5. Memory vs Model Size Comparison
    ax = axes[4]
    mem_size_data = plot_data.dropna(subset=['inference_peak_ram_mb', 'model_size_mb'])
    if not mem_size_data.empty:
        x_pos = range(len(mem_size_data))
        width = 0.35
        
        ax.bar([x - width/2 for x in x_pos], mem_size_data['inference_peak_ram_mb'], 
               width, label='Runtime RAM', color='lightcoral', alpha=0.7)
        ax.bar([x + width/2 for x in x_pos], mem_size_data['model_size_mb'], 
               width, label='Model Size', color='gold', alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Runtime Memory vs Model Size')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(mem_size_data['model_name'], rotation=45)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No memory/size comparison data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Memory vs Size (No Data)')
    
    # 6. Performance Summary Heatmap
    ax = axes[5]
    heatmap_cols = ['avg_inference_time_ms', 'inference_peak_ram_mb', 'model_size_mb']
    available_cols = [col for col in heatmap_cols if col in plot_data.columns and not plot_data[col].isna().all()]
    
    if len(available_cols) >= 2:
        heatmap_data = plot_data[['model_name'] + available_cols].set_index('model_name')
        # Normalize data for better heatmap visualization
        try:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_data = pd.DataFrame(scaler.fit_transform(heatmap_data), 
                                         columns=heatmap_data.columns, 
                                         index=heatmap_data.index)
        except ImportError:
            # Fallback normalization if sklearn is not available
            normalized_data = heatmap_data.div(heatmap_data.max(), axis=1)
        
        im = ax.imshow(normalized_data.T, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(range(len(normalized_data.index)))
        ax.set_xticklabels(normalized_data.index, rotation=45)
        ax.set_yticks(range(len(normalized_data.columns)))
        ax.set_yticklabels(normalized_data.columns)
        ax.set_title('Performance Metrics Heatmap\n(Normalized)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance Heatmap (No Data)')
    
    plt.tight_layout()
    
    # Handle saving and display
    if save_path:
        # Save both plots
        base_path = save_path.replace('.png', '')
        
        # Save first plot (dashboard)
        dashboard_path = f"{base_path}_dashboard.png"
        fig1.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        print(f"📊 Dashboard plot saved to: {dashboard_path}")
        
        # Save second plot (detailed analysis)
        detailed_path = f"{base_path}_detailed.png"
        fig2.savefig(detailed_path, dpi=150, bbox_inches='tight')
        print(f"📊 Detailed analysis plot saved to: {detailed_path}")
        
        plt.close(fig1)
        plt.close(fig2)
        return None
    else:
        plt.show()
        return fig1, fig2

def calculate_energy_metrics(inference_summary):
    """
    Calculate energy efficiency metrics
    """
    energy_results = []
    
    for _, model in inference_summary.iterrows():
        if not pd.isna(model.get('inference_avg_power_w')) and not pd.isna(model.get('avg_inference_time_ms')):
            power_w = model['inference_avg_power_w']
            time_ms = model['avg_inference_time_ms']
            
            # Calculate energy per prediction (Wh)
            energy_per_pred = (power_w * time_ms) / (1000 * 3600)
            
            # Daily energy for moderate usage (1000 predictions)
            daily_energy = energy_per_pred * 1000
            
            # Carbon footprint (assuming 0.5 kg CO2/kWh)
            carbon_per_pred = energy_per_pred * 0.5
            
            energy_results.append({
                'model_name': model['model_name'],
                'avg_power_w': power_w,
                'energy_per_prediction_wh': energy_per_pred,
                'daily_energy_moderate_wh': daily_energy,
                'carbon_per_prediction_g': carbon_per_pred
            })
    
    return pd.DataFrame(energy_results)

def generate_efficiency_report(efficiency_df):
    """
    Generate comprehensive efficiency analysis report
    """
    # Standardize model names first
    efficiency_df = standardize_model_names(efficiency_df)
    
    report = []
    report.append("# 🚀 LLM Performance Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Dataset Overview
    report.append("## 📊 Dataset Overview")
    report.append(f"- **Total Records**: {len(efficiency_df)}")
    report.append(f"- **Unique Models**: {efficiency_df['model_name'].nunique()}")
    report.append(f"- **Experiment Types**: {len(efficiency_df['experiment_type'].unique())}")
    
    # Data completeness
    timing_data = efficiency_df.dropna(subset=['avg_inference_time_ms'])
    power_data = efficiency_df.dropna(subset=['inference_avg_power_w'])
    report.append(f"- **Records with Timing Data**: {len(timing_data)}")
    report.append(f"- **Records with Power Data**: {len(power_data)}")
    report.append("")
    
    # Model Performance Summary
    inference_summary = calculate_inference_summary(efficiency_df)
    if not inference_summary.empty:
        report.append("## ⚡ Performance Summary")
        complete_data = inference_summary.dropna(subset=['avg_inference_time_ms', 'inference_peak_ram_mb'])
        
        if not complete_data.empty:
            fastest = complete_data.loc[complete_data['avg_inference_time_ms'].idxmin()]
            most_efficient = complete_data.loc[complete_data['inference_peak_ram_mb'].idxmin()]
            
            report.append(f"- **Fastest Model**: {fastest['model_name']} ({fastest['avg_inference_time_ms']:.1f}ms)")
            report.append(f"- **Most Memory Efficient**: {most_efficient['model_name']} ({most_efficient['inference_peak_ram_mb']:.1f}MB)")
        
        # Edge deployment readiness
        report.append("")
        report.append("## 📱 Edge Deployment Readiness")
        feasibility_counts = inference_summary['edge_feasibility'].value_counts()
        
        for category, count in feasibility_counts.items():
            emoji_map = {"highly_feasible": "🟢", "feasible": "🟡", "challenging": "🔴", "unknown": "⚪"}
            emoji = emoji_map.get(str(category), "🔘")
            category_str = str(category).replace('_', ' ').title()
            report.append(f"- {emoji} **{category_str}**: {count} models")
    
    report.append("")
    report.append("---")
    report.append("*Report generated by Enhanced LLM Efficiency Analysis Tool*")
    
    return "\n".join(report)

def create_efficiency_table(data):
    """
    Create efficiency comparison table
    """
    # Standardize model names first
    data = standardize_model_names(data)
    
    columns = [
        'model_name',
        'avg_inference_time_ms',
        'inference_peak_ram_mb', 
        'inference_avg_power_w',
        'total_parameters',
        'model_size_mb',
        'edge_feasibility'
    ]
    
    # Filter available columns
    available_cols = [col for col in columns if col in data.columns]
    table = data[available_cols].copy()
    
    # Round numeric columns
    numeric_cols = table.select_dtypes(include=['float64', 'int64']).columns
    table[numeric_cols] = table[numeric_cols].round(2)
    
    return table