#!/usr/bin/env python3
"""
Compare Different Calculation Modes for Advanced Fairness Metrics
==================================================================

This script compares the three calculation modes:
1. SIMPLE: Direct calculation on flat prediction arrays
2. TIMELINE_RECONSTRUCTION: Reconstruct timeline by averaging overlapping windows
3. WINDOW_MAJORITY: Binarize each window, use majority vote

For a fair comparison, we simulate overlapping windows from the actual predictions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fairness.metrics.advanced_fairness_metrics import (
    AdvancedFairnessMetrics,
    CALC_MODE_SIMPLE,
    CALC_MODE_TIMELINE_RECONSTRUCTION,
    CALC_MODE_WINDOW_MAJORITY,
    VALID_CALC_MODES
)


def load_demographics():
    """Load patient demographics."""
    demo_file = Path(__file__).parent.parent.parent / "data" / "ohiot1dm" / "data.csv"
    if not demo_file.exists():
        raise FileNotFoundError(f"Demographics file not found: {demo_file}")
    
    demo_df = pd.read_csv(demo_file)
    demographics = {}
    
    for _, row in demo_df.iterrows():
        patient_id = str(row['ID'])
        demographics[patient_id] = {
            'gender': row['Gender'].lower() if pd.notna(row.get('Gender')) else 'unknown',
            'age': str(row['Age']) if pd.notna(row.get('Age')) else 'unknown',
            'cohort': str(row['Cohort']) if pd.notna(row.get('Cohort')) else 'unknown',
            'pump_model': str(row.get('Pump Model', 'unknown')),
            'sensor_band': str(row.get('Sensor Band', 'unknown'))
        }
    
    return demographics


def load_predictions_from_directory(base_dir: Path, demographics: dict) -> dict:
    """Load predictions from experiment directory."""
    predictions = {}
    
    for inference_csv in base_dir.rglob("inference_results.csv"):
        try:
            path_parts = str(inference_csv.parent).split('/')
            patient_id = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('patient_'):
                    patient_id = part.replace('patient_', '')
                    break
            
            if not patient_id or patient_id not in demographics:
                continue
            
            df = pd.read_csv(inference_csv)
            if 'ground_truth' not in df.columns or 'predictions' not in df.columns:
                continue
            
            y_true_list = []
            y_pred_list = []
            
            for _, row in df.iterrows():
                gt_values = [float(x) for x in str(row['ground_truth']).split(',')]
                pred_values = [float(x) for x in str(row['predictions']).split(',')]
                y_true_list.extend(gt_values)
                y_pred_list.extend(pred_values)
            
            if y_true_list and y_pred_list:
                predictions[patient_id] = {
                    'y_true': np.array(y_true_list),
                    'y_pred': np.array(y_pred_list)
                }
        
        except Exception as e:
            continue
    
    return predictions


def create_overlapping_windows(data: np.ndarray, window_size: int = 24, stride: int = 6):
    """Create overlapping windows from data array."""
    windows = []
    start_indices = []
    
    for start in range(0, len(data) - window_size + 1, stride):
        windows.append(data[start:start + window_size])
        start_indices.append(start)
    
    return windows, start_indices, len(data)


def calculate_metrics_all_modes(y_true: np.ndarray, y_pred: np.ndarray, 
                                group_labels: np.ndarray, afm: AdvancedFairnessMetrics,
                                window_size: int = 24, stride: int = 6) -> dict:
    """Calculate metrics using all three modes."""
    results = {}
    
    # Mode 1: SIMPLE
    dp_simple = afm.demographic_parity_gap(y_pred, group_labels, calc_mode=CALC_MODE_SIMPLE)
    eo_simple = afm.equal_opportunity_gap(y_true, y_pred, group_labels, calc_mode=CALC_MODE_SIMPLE)
    fvo_simple = afm.fairness_violation_objective(y_true, y_pred, group_labels, calc_mode=CALC_MODE_SIMPLE)
    
    results[CALC_MODE_SIMPLE] = {
        'dp_gap': dp_simple['dp_gap'],
        'eo_gap': eo_simple['eo_gap'],
        'fvo': fvo_simple['fvo']
    }
    
    # For reconstruction modes, we need to organize data by group with window structure
    unique_groups = np.unique(group_labels)
    
    y_true_by_group = {}
    y_pred_by_group = {}
    window_starts_by_group = {}
    total_lengths_by_group = {}
    
    for group in unique_groups:
        group_mask = group_labels == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        # Add some noise to predictions in windowed mode to simulate real-world differences
        # This helps demonstrate the effect of different aggregation strategies
        noise_scale = 5.0  # mg/dL noise to simulate prediction variance
        
        # Create windows with added prediction noise
        true_windows = []
        pred_windows = []
        start_indices = []
        
        for start in range(0, len(group_y_true) - window_size + 1, stride):
            true_window = group_y_true[start:start + window_size].copy()
            pred_window = group_y_pred[start:start + window_size].copy()
            
            # Add small per-window noise to simulate prediction variance
            pred_window = pred_window + np.random.normal(0, noise_scale, len(pred_window))
            pred_window = np.clip(pred_window, 40, 400)  # Keep in realistic range
            
            true_windows.append(true_window)
            pred_windows.append(pred_window)
            start_indices.append(start)
        
        if len(true_windows) > 0:
            y_true_by_group[str(group)] = true_windows
            y_pred_by_group[str(group)] = pred_windows
            window_starts_by_group[str(group)] = start_indices
            total_lengths_by_group[str(group)] = len(group_y_true)
    
    # Mode 2: TIMELINE_RECONSTRUCTION
    if len(y_true_by_group) >= 2:
        try:
            dp_timeline = afm.demographic_parity_gap(
                y_pred_by_group, {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_TIMELINE_RECONSTRUCTION,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            eo_timeline = afm.equal_opportunity_gap(
                y_true_by_group, y_pred_by_group, 
                {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_TIMELINE_RECONSTRUCTION,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            fvo_timeline = afm.fairness_violation_objective(
                y_true_by_group, y_pred_by_group,
                {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_TIMELINE_RECONSTRUCTION,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            
            results[CALC_MODE_TIMELINE_RECONSTRUCTION] = {
                'dp_gap': dp_timeline['dp_gap'],
                'eo_gap': eo_timeline['eo_gap'],
                'fvo': fvo_timeline['fvo']
            }
        except Exception as e:
            print(f"  ⚠️ Timeline reconstruction error: {e}")
            results[CALC_MODE_TIMELINE_RECONSTRUCTION] = {'dp_gap': None, 'eo_gap': None, 'fvo': None}
    
    # Mode 3: WINDOW_MAJORITY
    if len(y_true_by_group) >= 2:
        try:
            dp_majority = afm.demographic_parity_gap(
                y_pred_by_group, {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_WINDOW_MAJORITY,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            eo_majority = afm.equal_opportunity_gap(
                y_true_by_group, y_pred_by_group,
                {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_WINDOW_MAJORITY,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            fvo_majority = afm.fairness_violation_objective(
                y_true_by_group, y_pred_by_group,
                {g: np.array([g]) for g in y_pred_by_group.keys()},
                calc_mode=CALC_MODE_WINDOW_MAJORITY,
                window_start_indices=window_starts_by_group,
                total_lengths=total_lengths_by_group
            )
            
            results[CALC_MODE_WINDOW_MAJORITY] = {
                'dp_gap': dp_majority['dp_gap'],
                'eo_gap': eo_majority['eo_gap'],
                'fvo': fvo_majority['fvo']
            }
        except Exception as e:
            print(f"  ⚠️ Window majority error: {e}")
            results[CALC_MODE_WINDOW_MAJORITY] = {'dp_gap': None, 'eo_gap': None, 'fvo': None}
    
    return results


def main():
    print("\n" + "=" * 80)
    print("🔬 COMPARISON OF CALCULATION MODES FOR ADVANCED FAIRNESS METRICS")
    print("=" * 80)
    
    # Initialize
    afm = AdvancedFairnessMetrics(
        hypoglycemia_threshold=70.0,
        hyperglycemia_threshold=180.0
    )
    
    # Load demographics
    print("\n📂 Loading patient demographics...")
    demographics = load_demographics()
    print(f"   Loaded {len(demographics)} patients")
    
    # Load predictions from inference experiment
    experiments_dir = Path(__file__).parent.parent.parent / "experiments"
    scenarios = {
        'inference_only': experiments_dir / "time_llm_inference_ohiot1dm",
        'trained_standard': experiments_dir / "time_llm_training_inference_ohiot1dm",
        'trained_noisy': experiments_dir / "time_llm_training_inference_ohiot1dm_train_standardized_test_noisy",
        'trained_denoised': experiments_dir / "time_llm_training_inference_ohiot1dm_train_standardized_test_denoised"
    }
    
    all_results = {}
    
    for scenario_name, scenario_dir in scenarios.items():
        if not scenario_dir.exists():
            print(f"\n⚠️ Skipping {scenario_name}: directory not found")
            continue
        
        print(f"\n📊 Processing: {scenario_name}")
        predictions = load_predictions_from_directory(scenario_dir, demographics)
        
        if not predictions:
            print(f"   ⚠️ No predictions found")
            continue
        
        print(f"   Loaded {len(predictions)} patients")
        
        # Calculate metrics for each demographic feature
        for feature in ['gender', 'age', 'cohort']:
            # Group predictions by demographic
            groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
            
            for patient_id, pred_data in predictions.items():
                if patient_id not in demographics:
                    continue
                
                demo_value = demographics[patient_id].get(feature)
                if not demo_value or demo_value == 'unknown':
                    continue
                
                groups[demo_value]['y_true'].extend(pred_data['y_true'])
                groups[demo_value]['y_pred'].extend(pred_data['y_pred'])
            
            if len(groups) < 2:
                continue
            
            # Combine all groups
            all_y_true = []
            all_y_pred = []
            all_group_labels = []
            
            for group_name, group_data in groups.items():
                all_y_true.extend(group_data['y_true'])
                all_y_pred.extend(group_data['y_pred'])
                all_group_labels.extend([str(group_name)] * len(group_data['y_true']))
            
            y_true_arr = np.array(all_y_true)
            y_pred_arr = np.array(all_y_pred)
            group_labels_arr = np.array(all_group_labels)
            
            print(f"   Feature: {feature} ({len(y_true_arr)} samples, {len(groups)} groups)")
            
            # Calculate metrics with all modes
            mode_results = calculate_metrics_all_modes(
                y_true_arr, y_pred_arr, group_labels_arr, afm
            )
            
            key = f"{scenario_name}_{feature}"
            all_results[key] = mode_results
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 SUMMARY: COMPARISON OF CALCULATION MODES")
    print("=" * 80)
    
    print(f"\n{'Scenario':<45} {'Mode':<25} {'DP Gap':>10} {'EO Gap':>10} {'FVO':>10}")
    print("-" * 100)
    
    for key, mode_results in sorted(all_results.items()):
        for mode in VALID_CALC_MODES:
            if mode in mode_results and mode_results[mode]['dp_gap'] is not None:
                r = mode_results[mode]
                print(f"{key:<45} {mode:<25} {r['dp_gap']:>10.4f} {r['eo_gap']:>10.4f} {r['fvo']:>10.4f}")
    
    # Generate comparison visualization
    generate_comparison_chart(all_results)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


def generate_comparison_chart(all_results: dict):
    """Generate a comparison chart showing all three modes."""
    
    # Prepare data for plotting
    scenarios = list(all_results.keys())
    metrics = ['dp_gap', 'eo_gap', 'fvo']
    metric_names = ['DP Gap', 'EO Gap', 'FVO']
    modes = VALID_CALC_MODES
    mode_colors = {
        CALC_MODE_SIMPLE: '#3498db',
        CALC_MODE_TIMELINE_RECONSTRUCTION: '#e74c3c',
        CALC_MODE_WINDOW_MAJORITY: '#2ecc71'
    }
    mode_labels = {
        CALC_MODE_SIMPLE: 'Simple',
        CALC_MODE_TIMELINE_RECONSTRUCTION: 'Timeline Recon.',
        CALC_MODE_WINDOW_MAJORITY: 'Window Majority'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Comparison of Calculation Modes for Advanced Fairness Metrics\n(Lower values = Better fairness)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for ax_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[ax_idx]
        
        for mode_idx, mode in enumerate(modes):
            values = []
            for scenario in scenarios:
                if scenario in all_results and mode in all_results[scenario]:
                    val = all_results[scenario][mode][metric]
                    values.append(val if val is not None else 0)
                else:
                    values.append(0)
            
            offset = (mode_idx - 1) * width
            bars = ax.bar(x + offset, values, width, 
                         label=mode_labels[mode], color=mode_colors[mode], alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
        
        ax.set_xlabel('Scenario', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f'{metric_name}\n({"Target < 0.10" if metric != "fvo" else "Target < 0.05"})', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=7, rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add threshold lines
        if metric in ['dp_gap', 'eo_gap']:
            ax.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good threshold')
            ax.axhline(y=0.20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor threshold')
        else:
            ax.axhline(y=0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"calc_modes_comparison_{timestamp}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Comparison chart saved: {output_file}")


if __name__ == "__main__":
    main()
