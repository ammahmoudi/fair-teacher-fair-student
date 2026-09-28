"""
Edge deployment analysis functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path


# Edge device specifications
EDGE_DEVICES = {
    'raspberry_pi_4': {
        'name': 'Raspberry Pi 4B',
        'ram_gb': 8,
        'cpu_cores': 4,
        'gpu': False,
        'max_model_size_mb': 2000,
        'cpu_performance_factor': 0.1,
        'cost_usd': 75
    },
    'jetson_nano': {
        'name': 'Jetson Nano',
        'ram_gb': 4,
        'cpu_cores': 4,
        'gpu': True,
        'max_model_size_mb': 3000,
        'cpu_performance_factor': 0.3,
        'cost_usd': 149
    },
    'jetson_xavier': {
        'name': 'Jetson Xavier NX',
        'ram_gb': 8,
        'cpu_cores': 6,
        'gpu': True,
        'max_model_size_mb': 6000,
        'cpu_performance_factor': 0.6,
        'cost_usd': 399
    },
    'intel_nuc': {
        'name': 'Intel NUC i5',
        'ram_gb': 16,
        'cpu_cores': 4,
        'gpu': False,
        'max_model_size_mb': 8000,
        'cpu_performance_factor': 0.8,
        'cost_usd': 600
    },
    'coral_dev': {
        'name': 'Coral Dev Board',
        'ram_gb': 4,
        'cpu_cores': 4,
        'gpu': False,
        'max_model_size_mb': 1000,
        'cpu_performance_factor': 0.2,
        'cost_usd': 175
    }
}


def assess_edge_compatibility(inference_summary: pd.DataFrame) -> pd.DataFrame:
    """Assess edge deployment compatibility for all models."""
    results = []
    
    for _, model in inference_summary.iterrows():
        model_name = model['model_name']
        model_size_mb = model.get('model_size_mb', 0)
        base_latency_ms = model.get('avg_inference_time_ms', np.nan)
        
        for device_id, specs in EDGE_DEVICES.items():
            # Check compatibility
            size_compatible = pd.isna(model_size_mb) or model_size_mb <= specs['max_model_size_mb']
            
            # Estimate latency on edge device
            if not pd.isna(base_latency_ms):
                estimated_latency = base_latency_ms / specs['cpu_performance_factor']
                throughput = 1000.0 / estimated_latency if estimated_latency > 0 else 0
            else:
                estimated_latency = np.nan
                throughput = 0
            
            results.append({
                'model_name': model_name,
                'device_id': device_id,
                'device_name': specs['name'],
                'compatible': size_compatible,
                'estimated_latency_ms': estimated_latency,
                'estimated_throughput': throughput,
                'device_cost_usd': specs['cost_usd'],
                'model_size_mb': model_size_mb
            })
    
    return pd.DataFrame(results)


def create_edge_compatibility_heatmap(edge_df: pd.DataFrame, save_path: str = None) -> None:
    """Create heatmap of model-device compatibility."""
    # Create pivot table
    pivot = edge_df.pivot(index='model_name', columns='device_name', values='compatible')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Compatible')
    
    # Set labels
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title('Edge Device Compatibility Matrix')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = '✓' if pivot.iloc[i, j] else '✗'
            plt.text(j, i, text, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def get_edge_recommendations(edge_df: pd.DataFrame) -> Dict:
    """Generate edge deployment recommendations."""
    recommendations = {}
    
    for model_name in edge_df['model_name'].unique():
        model_data = edge_df[edge_df['model_name'] == model_name]
        compatible_devices = model_data[model_data['compatible'] == True]
        
        if not compatible_devices.empty:
            # Sort by throughput (higher is better)
            best_devices = compatible_devices.nlargest(3, 'estimated_throughput')
            recommendations[model_name] = best_devices[['device_name', 'estimated_latency_ms', 'estimated_throughput', 'device_cost_usd']].to_dict('records')
        else:
            recommendations[model_name] = []
    
    return recommendations