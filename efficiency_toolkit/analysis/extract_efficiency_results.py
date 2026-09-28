#!/usr/bin/env python3
"""
Comprehensive Efficiency Analysis for Completed Experiments
============================================================

Analyzes results from completed Time-LLM and Chronos experiments following
the comprehensive efficiency runner structure.

Searches for:
1. Time-LLM training experiments (BERT, GPT2, LLAMA)
2. Time-LLM inference experiments (BERT, GPT2)
3. Chronos training experiments (T5-base, T5-tiny)
4. Chronos inference experiments (T5-base, T5-tiny)

Extracts efficiency metrics and performance data from comprehensive reports.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

def find_completed_experiments():
    """Find all completed experiments with comprehensive performance reports."""
    print("🔍 Scanning for completed experiments...")
    
    experiments = {
        'time_llm_training': [],
        'time_llm_inference': [],
        'chronos_training': [],
        'chronos_inference': [],
        'distillation': []
    }
    
    # Find comprehensive performance reports
    base_pattern = "experiments/**/comprehensive_performance_report*.json"
    report_files = glob.glob(base_pattern, recursive=True)
    
    # Also find distillation results
    distillation_pattern = "distillation_experiments/**/comprehensive_performance_report*.json"
    distillation_files = glob.glob(distillation_pattern, recursive=True)
    report_files.extend(distillation_files)
    
    for report_file in report_files:
        try:
            # Parse experiment type from path
            path_parts = Path(report_file).parts
            experiment_dir = path_parts[1]  # e.g., time_llm_training_ohiot1dm
            
            # Determine experiment category
            if 'time_llm_training' in experiment_dir:
                category = 'time_llm_training'
            elif 'time_llm_inference' in experiment_dir:
                category = 'time_llm_inference'
            elif 'chronos_training' in experiment_dir:
                category = 'chronos_training'
            elif 'chronos_inference' in experiment_dir:
                category = 'chronos_inference'
            elif 'distillation' in str(report_file):
                category = 'distillation'
            else:
                continue
                
            experiments[category].append(report_file)
            
        except Exception as e:
            print(f"⚠️ Error processing {report_file}: {e}")
    
    # Print summary
    total = sum(len(exps) for exps in experiments.values())
    print(f"📊 Found {total} completed experiments:")
    for category, files in experiments.items():
        if files:
            print(f"  • {category.replace('_', ' ').title()}: {len(files)}")
    
    return experiments

def extract_experiment_metadata(report_file):
    """Extract metadata from experiment path and filename."""
    path_parts = Path(report_file).parts
    
    metadata = {}
    
    # Handle distillation experiments separately
    if 'distillation' in str(report_file):
        # For distillation: pipeline_runs/pipeline_2025-10-21_07-54-46/patient_570/phase_1_teacher/bert_570_5epochs/...
        if 'phase_1_teacher' in str(report_file):
            metadata['model'] = 'teacher-bert'
        elif 'phase_2_student' in str(report_file):
            metadata['model'] = 'student-bert-tiny'
        else:
            metadata['model'] = 'distillation-bert'
        
        # Extract patient ID from path
        for part in path_parts:
            if part.startswith('patient_'):
                metadata['seed'] = part.replace('patient_', '')
                break
        
        return metadata
    
    # Original logic for regular experiments
    experiment_folder = path_parts[2]  # e.g., seed_831363_model_BERT_...
    
    # Extract seed
    if 'seed_' in experiment_folder:
        seed_part = [p for p in experiment_folder.split('_') if p.startswith('seed')][0]
        metadata['seed'] = seed_part.replace('seed_', '')
    
    # Extract model name
    if 'model_' in experiment_folder:
        parts = experiment_folder.split('_')
        model_idx = next(i for i, p in enumerate(parts) if p == 'model') + 1
        if model_idx < len(parts):
            model_name = parts[model_idx]
            # Handle different model naming conventions
            if 'amazon-chronos' in model_name:
                if 'base' in experiment_folder:
                    metadata['model'] = 'chronos-t5-base'
                elif 'tiny' in experiment_folder:
                    metadata['model'] = 'chronos-t5-tiny'
                else:
                    metadata['model'] = 'chronos-unknown'
            else:
                metadata['model'] = model_name
    
    # Extract context and prediction lengths
    if 'context_' in experiment_folder and 'pred_' in experiment_folder:
        parts = experiment_folder.split('_')
        try:
            context_idx = next(i for i, p in enumerate(parts) if p == 'context') + 1
            pred_idx = next(i for i, p in enumerate(parts) if p == 'pred') + 1
            metadata['context_length'] = int(parts[context_idx])
            metadata['prediction_length'] = int(parts[pred_idx])
        except (StopIteration, ValueError, IndexError):
            pass
    
    # Extract epochs for training experiments
    if 'epochs_' in experiment_folder:
        parts = experiment_folder.split('_')
        try:
            epochs_idx = next(i for i, p in enumerate(parts) if p == 'epochs') + 1
            metadata['epochs'] = int(parts[epochs_idx])
        except (StopIteration, ValueError, IndexError):
            pass
    
    return metadata

def load_and_parse_report(report_file):
    """Load and parse a comprehensive performance report."""
    try:
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        # Extract metadata from path
        metadata = extract_experiment_metadata(report_file)
        
        # Initialize result record
        result = {
            'report_file': report_file,
            'timestamp': data.get('timestamp', 'unknown'),
            'model_name': metadata.get('model', 'unknown'),
            'seed': metadata.get('seed', 'unknown'),
            'context_length': metadata.get('context_length'),
            'prediction_length': metadata.get('prediction_length'),
            'epochs': metadata.get('epochs', 0)
        }
        
        # Extract performance metrics from performance_summary
        if 'performance_summary' in data:
            perf_summary = data['performance_summary']
            
            # Training metrics
            if 'training' in perf_summary:
                training = perf_summary['training']
                result.update({
                    'training_latency_ms': training.get('average_latency_ms'),
                    'training_inferences': training.get('total_inferences', 1),
                    'peak_ram_mb': training.get('process_peak_ram_mb'),
                    'avg_ram_mb': training.get('process_average_ram_mb'),
                    'peak_gpu_mb': training.get('peak_gpu_allocated_mb'),
                    'avg_gpu_mb': training.get('average_gpu_allocated_mb'),
                    'peak_gpu_util_%': training.get('peak_gpu_utilization_percent'),
                    'avg_gpu_util_%': training.get('average_gpu_utilization_percent'),
                    'peak_temp_c': training.get('peak_temperature_celsius'),
                    'peak_power_w': training.get('peak_power_usage_watts'),
                    'avg_power_w': training.get('average_power_usage_watts'),
                    'model_size_mb': training.get('model_size_on_disk_mb'),
                    'parameters_count': training.get('parameters_count')
                })
            
            # Training+Inference metrics (for training experiments with inference)
            if 'training_inference' in perf_summary:
                train_inf = perf_summary['training_inference']
                result.update({
                    'inference_latency_ms': train_inf.get('average_latency_ms'),
                    'inference_inferences': train_inf.get('total_inferences'),
                    'inference_peak_ram_mb': train_inf.get('process_peak_ram_mb'),
                    'inference_avg_ram_mb': train_inf.get('process_average_ram_mb'),
                    'inference_peak_gpu_mb': train_inf.get('peak_gpu_allocated_mb'),
                    'inference_avg_gpu_mb': train_inf.get('average_gpu_allocated_mb')
                })
            
            # Pure inference metrics (for inference-only experiments)
            if 'inference' in perf_summary:
                inference = perf_summary['inference']
                result.update({
                    'inference_latency_ms': inference.get('average_latency_ms'),
                    'inference_inferences': inference.get('total_inferences'),
                    'inference_peak_ram_mb': inference.get('process_peak_ram_mb'),
                    'inference_avg_ram_mb': inference.get('process_average_ram_mb'),
                    'inference_peak_gpu_mb': inference.get('peak_gpu_allocated_mb'),
                    'inference_avg_gpu_mb': inference.get('average_gpu_allocated_mb')
                })
        
        # Look for accuracy metrics in detailed_measurements
        if 'detailed_measurements' in data and data['detailed_measurements']:
            detailed = data['detailed_measurements']
            # Handle different structures - could be list or dict
            if isinstance(detailed, list) and len(detailed) > 0:
                # Get the last measurement (final result)
                last_measurement = detailed[-1]
                if 'metrics' in last_measurement:
                    metrics = last_measurement['metrics']
                    result.update({
                        'rmse': metrics.get('rmse'),
                        'mae': metrics.get('mae'),
                        'mape': metrics.get('mape')
                    })
            elif isinstance(detailed, dict):
                # If it's a dict, look for metrics directly or in nested structure
                if 'metrics' in detailed:
                    metrics = detailed['metrics']
                    result.update({
                        'rmse': metrics.get('rmse'),
                        'mae': metrics.get('mae'),
                        'mape': metrics.get('mape')
                    })
                # Also check for direct metric values
                for metric in ['rmse', 'mae', 'mape']:
                    if metric in detailed:
                        result[metric] = detailed[metric]
        
        # Calculate derived efficiency metrics
        if result.get('training_latency_ms') and result.get('rmse'):
            training_time_sec = result['training_latency_ms'] / 1000.0
            result['training_efficiency'] = 1.0 / (training_time_sec * result['rmse'])
        
        if result.get('inference_latency_ms') and result.get('rmse'):
            inference_time_sec = result['inference_latency_ms'] / 1000.0
            result['inference_efficiency'] = 1.0 / (inference_time_sec * result['rmse'])
            
        if result.get('peak_ram_mb') and result.get('rmse'):
            result['memory_efficiency'] = 1.0 / (result['peak_ram_mb'] * result['rmse'])
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading {report_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_experiments():
    """Main analysis function."""
    print("🚀 Comprehensive Efficiency Analysis")
    print("=" * 80)
    
    # Find completed experiments
    experiments = find_completed_experiments()
    
    if not any(experiments.values()):
        print("❌ No completed experiments found!")
        return
    
    # Load and parse all reports
    all_results = []
    for category, report_files in experiments.items():
        print(f"\n📊 Processing {category.replace('_', ' ').title()}...")
        
        for report_file in report_files:
            result = load_and_parse_report(report_file)
            if result:
                result['experiment_type'] = category
                all_results.append(result)
    
    if not all_results:
        print("❌ No valid results extracted!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    print(f"\n✅ Successfully processed {len(df)} experiment results")
    print("=" * 80)
    
    # Analysis by model type
    print("\n📈 RESULTS BY MODEL TYPE")
    print("-" * 50)
    
    time_llm_results = df[df['experiment_type'].str.contains('time_llm')]
    chronos_results = df[df['experiment_type'].str.contains('chronos')]
    distillation_results = df[df['experiment_type'].str.contains('distillation')]
    
    if not time_llm_results.empty:
        print("\n🤖 TIME-LLM MODELS:")
        # Only aggregate columns that exist
        agg_cols = {}
        for col in ['training_latency_ms', 'inference_latency_ms', 'peak_ram_mb', 'peak_gpu_mb', 'rmse', 'mae']:
            if col in time_llm_results.columns:
                agg_cols[col] = 'mean'
        
        if agg_cols:
            time_llm_summary = time_llm_results.groupby(['model_name', 'experiment_type']).agg(agg_cols).round(2)
            print(time_llm_summary.to_string())
        else:
            print("No aggregatable columns found for Time-LLM results")
    
    if not chronos_results.empty:
        print("\n⏰ CHRONOS MODELS:")
        # Only aggregate columns that exist
        agg_cols = {}
        for col in ['training_latency_ms', 'inference_latency_ms', 'peak_ram_mb', 'peak_gpu_mb', 'rmse', 'mae']:
            if col in chronos_results.columns:
                agg_cols[col] = 'mean'
        
        if agg_cols:
            chronos_summary = chronos_results.groupby(['model_name', 'experiment_type']).agg(agg_cols).round(2)
            print(chronos_summary.to_string())
        else:
            print("No aggregatable columns found for Chronos results")

    if not distillation_results.empty:
        print("\n🧠 DISTILLATION MODELS:")
        # Only aggregate columns that exist
        agg_cols = {}
        for col in ['training_latency_ms', 'inference_latency_ms', 'peak_ram_mb', 'peak_gpu_mb', 'rmse', 'mae']:
            if col in distillation_results.columns:
                agg_cols[col] = 'mean'
        
        if agg_cols:
            distillation_summary = distillation_results.groupby(['model_name', 'experiment_type']).agg(agg_cols).round(2)
            print(distillation_summary.to_string())
        else:
            print("No aggregatable columns found for Distillation results")
    
    # Training vs Inference Analysis
    print(f"\n⚖️ TRAINING VS INFERENCE COMPARISON")
    print("-" * 50)
    
    training_results = df[df['experiment_type'].str.contains('training')]
    inference_results = df[df['experiment_type'].str.contains('inference')]
    
    if not training_results.empty and not inference_results.empty:
        comparison_data = []
        
        for model in df['model_name'].unique():
            if model == 'unknown':
                continue
                
            train_data = training_results[training_results['model_name'] == model]
            inf_data = inference_results[inference_results['model_name'] == model]
            
            if not train_data.empty and not inf_data.empty:
                comp_row = {'Model': model}
                
                # Only add columns that exist
                if 'training_latency_ms' in train_data.columns:
                    comp_row['Train_Time_ms'] = train_data['training_latency_ms'].mean()
                if 'inference_latency_ms' in inf_data.columns:
                    comp_row['Inference_Time_ms'] = inf_data['inference_latency_ms'].mean()
                if 'peak_ram_mb' in train_data.columns:
                    comp_row['Train_Peak_RAM_MB'] = train_data['peak_ram_mb'].mean()
                if 'inference_peak_ram_mb' in inf_data.columns:
                    comp_row['Inference_Peak_RAM_MB'] = inf_data['inference_peak_ram_mb'].mean()
                if 'rmse' in train_data.columns:
                    comp_row['Train_RMSE'] = train_data['rmse'].mean()
                if 'rmse' in inf_data.columns:
                    comp_row['Inference_RMSE'] = inf_data['rmse'].mean()
                    
                comparison_data.append(comp_row)
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            # Only show columns that have data
            comp_df = comp_df.dropna(axis=1, how='all')
            print(comp_df.round(2).to_string(index=False))
    
    # Model Efficiency Ranking
    print(f"\n🏆 MODEL EFFICIENCY RANKING")
    print("-" * 50)
    
    # Find efficiency columns that exist
    efficiency_cols = [col for col in ['training_efficiency', 'memory_efficiency', 'inference_efficiency'] if col in df.columns]
    
    if efficiency_cols:
        # Filter out rows with missing efficiency metrics (at least one efficiency metric must exist)
        efficiency_results = df.dropna(subset=efficiency_cols, how='all')
        
        if not efficiency_results.empty:
            # Only aggregate columns that exist
            agg_cols = {}
            for col in ['training_efficiency', 'memory_efficiency', 'rmse', 'peak_ram_mb', 'training_latency_ms']:
                if col in efficiency_results.columns:
                    agg_cols[col] = 'mean'
            
            if agg_cols:
                ranking = efficiency_results.groupby('model_name').agg(agg_cols).round(4)
                
                # Sort by the first available efficiency metric
                sort_col = next((col for col in ['training_efficiency', 'memory_efficiency', 'inference_efficiency'] if col in ranking.columns), None)
                if sort_col:
                    ranking = ranking.sort_values(sort_col, ascending=False)
                
                print("Efficiency Ranking (Higher = Better):")
                print(ranking.to_string())
            else:
                print("No efficiency metrics available for ranking")
        else:
            print("No experiments with efficiency metrics found")
    else:
        print("No efficiency columns found in data")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results
    output_file = f"efficiency_analysis_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Complete results saved to: {output_file}")
    
    # Save summary by model (if created)
    try:
        if not time_llm_results.empty and 'time_llm_summary' in locals():
            time_llm_file = f"time_llm_efficiency_summary_{timestamp}.csv"
            time_llm_summary.to_csv(time_llm_file)
            print(f"💾 Time-LLM summary saved to: {time_llm_file}")
    except:
        pass
    
    try:
        if not chronos_results.empty and 'chronos_summary' in locals():
            chronos_file = f"chronos_efficiency_summary_{timestamp}.csv"
            chronos_summary.to_csv(chronos_file)
            print(f"💾 Chronos summary saved to: {chronos_file}")
    except:
        pass
    
    try:
        if not distillation_results.empty and 'distillation_summary' in locals():
            distillation_file = f"distillation_efficiency_summary_{timestamp}.csv"
            distillation_summary.to_csv(distillation_file)
            print(f"💾 Distillation summary saved to: {distillation_file}")
    except:
        pass
    
    print(f"\n🎉 Analysis Complete!")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    analyze_experiments()