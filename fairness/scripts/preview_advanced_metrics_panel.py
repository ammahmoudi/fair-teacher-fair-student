"""
Quick visualization of what the Advanced Metrics Panel looks like
==================================================================

Run this to see a standalone version of the advanced metrics panel.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Generate sample data
np.random.seed(42)
all_dp_gaps = np.random.beta(2, 8, 30) * 0.3  # Skewed towards lower values
all_eo_gaps = np.random.beta(2, 5, 30) * 0.4  # Some concerning values
all_fvos = np.random.beta(3, 10, 30) * 0.2    # Mostly good

# Create figure
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(1, 3, left=0.05, right=0.95, bottom=0.2, top=0.85, wspace=0.3)

fig.suptitle('🔬 ADVANCED FAIRNESS METRICS SUMMARY\n' + 
             'Example Visualization from Comprehensive Report',
             fontsize=16, fontweight='bold')

def create_metric_panel(ax, values, metric_name, threshold_good, threshold_concern, 
                       description, color_scheme='lightblue'):
    """Create a metric panel with box plot and statistics."""
    
    # Box plot
    bp = ax.boxplot([values], positions=[0], widths=0.4, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][0].set_facecolor(color_scheme)
    
    # Add threshold lines
    ax.axhline(y=threshold_good, color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Good (<{threshold_good})')
    ax.axhline(y=threshold_concern, color='orange', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'Concern (<{threshold_concern})')
    
    # Add statistics text
    mean_val = np.mean(values)
    max_val = np.max(values)
    n_concern = sum(1 for x in values if x > threshold_concern)
    
    stats_text = f"Mean: {mean_val:.3f}\nMax: {max_val:.3f}\n>{threshold_concern}: {n_concern}/{len(values)}"
    ax.text(0.65, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'{metric_name}\n{description}', fontweight='bold', fontsize=11)
    ax.set_xticks([])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    # Status indicator
    if mean_val < threshold_good:
        status = '✅ Good'
        color = 'green'
    elif mean_val < threshold_concern:
        status = '⚠️ Monitor'
        color = 'orange'
    else:
        status = '❌ Action Needed'
        color = 'red'
    
    ax.text(0.5, -0.12, status, transform=ax.transAxes,
           ha='center', fontsize=11, fontweight='bold', color=color)

# Create three panels
ax1 = fig.add_subplot(gs[0, 0])
create_metric_panel(ax1, all_dp_gaps, 'DP Gap', 0.10, 0.20, 
                   'Alert Distribution Equality', 'lightblue')

ax2 = fig.add_subplot(gs[0, 1])
create_metric_panel(ax2, all_eo_gaps, 'EO Gap', 0.10, 0.20,
                   'Detection Equality\n⚠️ CRITICAL FOR SAFETY', 'lightcoral')

ax3 = fig.add_subplot(gs[0, 2])
create_metric_panel(ax3, all_fvos, 'FVO', 0.05, 0.10,
                   'Accuracy Disparity', 'lightgreen')

# Add interpretation text at bottom
interp_text = (
    "Interpretation: Lower values = better fairness. "
    "EO Gap is MOST CRITICAL for patient safety - high values mean some groups miss critical warnings!"
)
fig.text(0.5, 0.08, interp_text, ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
        style='italic', wrap=True)

# Add info text
info_text = "This panel appears in the comprehensive fairness report (Row 5, full width)"
fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=8,
        style='italic', color='gray')

output_path = PROJECT_ROOT / "fairness" / "analysis_results" / "advanced_metrics_panel_preview.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Preview saved to: {output_path}")

plt.show()
