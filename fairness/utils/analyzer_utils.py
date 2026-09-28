#!/usr/bin/env python3
"""
Utility Functions for Fairness Analyzers

Common helper functions used across multiple analyzers.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def load_summary_json(file_path: Path) -> Dict:
    """
    Load and parse a summary JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return {}


def extract_performance_metrics(summary_data: Dict) -> Dict:
    """
    Extract performance metrics from summary data.
    
    Args:
        summary_data: Summary JSON data
        
    Returns:
        Dictionary with standardized performance metrics
    """
    perf_metrics = summary_data.get('performance_metrics', {})
    rmse = perf_metrics.get('rmse', 0)
    
    return {
        'rmse': rmse,
        'mse': rmse ** 2,
        'mae': perf_metrics.get('mae', 0),
        'mape': perf_metrics.get('mape', 0)
    }


def load_multi_phase_results(patient_dir: Path) -> Dict:
    """
    Load results from all three distillation phases.
    
    Args:
        patient_dir: Path to patient directory
        
    Returns:
        Dictionary with teacher, student_baseline, and distilled results
    """
    results = {}
    
    # Phase 1: Teacher
    teacher_file = patient_dir / "phase_1_teacher" / "teacher_training_summary.json"
    if teacher_file.exists():
        with open(teacher_file) as f:
            data = json.load(f)
            results['teacher'] = extract_performance_metrics(data)
    
    # Phase 2: Student Baseline
    student_file = patient_dir / "phase_2_student" / "student_baseline_summary.json"
    if student_file.exists():
        with open(student_file) as f:
            data = json.load(f)
            results['student_baseline'] = extract_performance_metrics(data)
    
    # Phase 3: Distilled
    distilled_file = patient_dir / "phase_3_distillation" / "distillation_summary.json"
    if distilled_file.exists():
        with open(distilled_file) as f:
            data = json.load(f)
            results['distilled'] = extract_performance_metrics(data)
    
    return results


def calculate_statistics(values: List[float]) -> Dict:
    """
    Calculate statistical measures for a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


def format_fairness_level(ratio: float) -> Tuple[str, str]:
    """
    Determine fairness level and emoji based on ratio.
    
    Args:
        ratio: Fairness ratio (worst/best performance)
        
    Returns:
        Tuple of (level_text, emoji)
    """
    if ratio < 1.10:
        return "EXCELLENT", "🏆"
    elif ratio < 1.25:
        return "GOOD", "👍"
    elif ratio < 1.50:
        return "ACCEPTABLE", "⚠️ "
    else:
        return "POOR", "💥"


def format_report_header(title: str, width: int = 80) -> str:
    """
    Format a report header with decorative lines.
    
    Args:
        title: Header title
        width: Width of the header line
        
    Returns:
        Formatted header string
    """
    return f"\n{'=' * width}\n{title}\n{'=' * width}\n"


def format_metric(metric_name: str, value: float, decimals: int = 3) -> str:
    """
    Format a metric for display.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{metric_name}: {value:.{decimals}f}"


def get_ohiot1dm_default_data() -> Dict[str, Dict]:
    """
    Get OhioT1DM patient metadata.

    Loads from data/ohiot1dm/data.csv (primary source) with a hardcoded
    fallback matching Table 1 of the OhioT1DM paper (Marling & Bunescu, 2020).

    CSV format expected:
        ID,Gender,Age,Pump Model,Sensor Band,Cohort

    Returns:
        Dictionary mapping patient_id (str) -> demographic dict with keys:
        gender, age, pump, sensor, cohort
    """
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent / "data" / "ohiot1dm" / "data.csv"
    if csv_path.exists():
        result = {}
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                pid = str(row['ID']).strip()
                result[pid] = {
                    'gender': row['Gender'].strip().capitalize(),
                    'age':    row['Age'].strip(),
                    'pump':   row['Pump Model'].strip(),
                    'sensor': row['Sensor Band'].strip(),
                    'cohort': str(row['Cohort']).strip(),
                }
        return result

    # Fallback: hardcoded from Table 1, OhioT1DM paper
    return {
        '540': {'gender': 'Male',   'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '544': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
        '552': {'gender': 'Male',   'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '559': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '563': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '567': {'gender': 'Female', 'age': '20-40', 'pump': '630G', 'sensor': 'Empatica', 'cohort': '2020'},
        '570': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '575': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '584': {'gender': 'Male',   'age': '40-60', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
        '588': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '591': {'gender': 'Female', 'age': '40-60', 'pump': '530G', 'sensor': 'Basis',    'cohort': '2018'},
        '596': {'gender': 'Male',   'age': '60-80', 'pump': '530G', 'sensor': 'Empatica', 'cohort': '2020'},
    }


def extract_feature_from_default_data(feature: str) -> Dict[str, str]:
    """
    Extract a specific feature from default data.
    
    Args:
        feature: Feature name ('gender', 'age', 'pump', 'sensor', 'cohort')
        
    Returns:
        Dictionary mapping patient_id to feature value
    """
    default_data = get_ohiot1dm_default_data()
    return {pid: data[feature] for pid, data in default_data.items()}


def print_group_summary(group_name: str, statistics: Dict, phase: str = 'distilled'):
    """
    Print summary for a demographic group (supports multi-phase data).
    
    Args:
        group_name: Name of the group
        statistics: Statistics dictionary (can be multi-phase or single-phase)
        phase: Which phase to display ('teacher', 'student_baseline', or 'distilled')
    """
    print(f"\n  {group_name}:")
    print(f"    Patients: {statistics['count']}")
    
    # Check if this is multi-phase data
    if phase in statistics:
        phase_stats = statistics[phase]
        print(f"    Avg RMSE ({phase}): {phase_stats['rmse_mean']:.3f}")
        print(f"    Avg MAE ({phase}): {phase_stats['mae_mean']:.3f}")
    elif 'rmse_mean' in statistics:
        # Old single-phase format
        print(f"    Avg RMSE: {statistics['rmse_mean']:.3f}")
        print(f"    Avg MAE: {statistics['mae_mean']:.3f}")
    else:
        print(f"    No data for phase: {phase}")


def print_multi_phase_summary(group_name: str, statistics: Dict):
    """
    Print summary for all phases of a demographic group.
    
    Args:
        group_name: Name of the group
        statistics: Statistics dictionary with multi-phase data
    """
    print(f"\n  {group_name} ({statistics['count']} patients):")
    
    for phase in ['teacher', 'student_baseline', 'distilled']:
        if phase in statistics:
            phase_stats = statistics[phase]
            phase_label = phase.replace('_', ' ').title()
            print(f"    {phase_label:20s}: RMSE={phase_stats['rmse_mean']:.3f}, MAE={phase_stats['mae_mean']:.3f}")


def create_comparison_text(groups: List[str], ratios: List[float]) -> str:
    """
    Create comparison text for multiple groups.
    
    Args:
        groups: List of group names
        ratios: List of performance ratios
        
    Returns:
        Formatted comparison text
    """
    lines = []
    for i, (group, ratio) in enumerate(zip(groups, ratios)):
        level, emoji = format_fairness_level(ratio)
        lines.append(f"{i+1}. {group}: {ratio:.2f}x - {emoji} {level}")
    return "\n".join(lines)


def calculate_distillation_impact(teacher_score: float, distilled_score: float) -> Dict:
    """
    Calculate the impact of distillation on fairness.
    
    Args:
        teacher_score: Fairness score for teacher model
        distilled_score: Fairness score for distilled model
        
    Returns:
        Dictionary with impact analysis
    """
    change = distilled_score - teacher_score
    percent_change = (change / teacher_score * 100) if teacher_score > 0 else 0
    
    makes_worse = distilled_score > teacher_score
    
    if makes_worse:
        if abs(change) > 0.5:  # Using ratio-based threshold
            conclusion = "🚨 DISTILLATION SIGNIFICANTLY WORSENS FAIRNESS"
            severity = "CRITICAL"
        elif abs(change) > 0.2:
            conclusion = "⚠️  DISTILLATION MODERATELY WORSENS FAIRNESS"
            severity = "MODERATE"
        else:
            conclusion = "➖ DISTILLATION SLIGHTLY WORSENS FAIRNESS"
            severity = "MINOR"
    else:
        conclusion = "✅ DISTILLATION MAINTAINS OR IMPROVES FAIRNESS"
        severity = "GOOD"
    
    return {
        'makes_worse': makes_worse,
        'change': change,
        'percent_change': percent_change,
        'conclusion': conclusion,
        'severity': severity,
        'teacher_score': teacher_score,
        'distilled_score': distilled_score
    }


def analyze_per_group_distillation_impact(statistics: Dict) -> Dict:
    """
    Analyze distillation impact for each individual group.
    
    Args:
        statistics: Group statistics with multi-phase data
        
    Returns:
        Dictionary mapping group names to their distillation impact
    """
    group_impacts = {}
    
    for group_name, stats in statistics.items():
        if 'teacher' in stats and 'distilled' in stats:
            teacher_rmse = stats['teacher']['rmse_mean']
            distilled_rmse = stats['distilled']['rmse_mean']
            
            # Calculate change (positive = worse performance)
            rmse_change = distilled_rmse - teacher_rmse
            percent_change = (rmse_change / teacher_rmse * 100) if teacher_rmse > 0 else 0
            
            # Determine if better or worse
            if rmse_change < -0.5:  # Significant improvement (RMSE decreased)
                status = "IMPROVED"
            elif rmse_change < 0:  # Slight improvement
                status = "Slightly Improved"
            elif rmse_change < 0.5:  # Slight degradation
                status = "Slightly Worse"
            else:  # Significant degradation
                status = "WORSE"
            
            group_impacts[group_name] = {
                'teacher_rmse': teacher_rmse,
                'distilled_rmse': distilled_rmse,
                'rmse_change': rmse_change,
                'percent_change': percent_change,
                'status': status
            }
    
    return group_impacts
