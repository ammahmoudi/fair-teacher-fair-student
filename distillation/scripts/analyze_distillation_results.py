#!/usr/bin/env python3
"""
Distillation Results Analyzer and Visualizer
===========================================

Analyzes distillation comparison results and creates visualizations.

Usage:
    python scripts/analyze_distillation_results.py /path/to/results.json
    python scripts/analyze_distillation_results.py --latest
"""

import argparse
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def find_latest_results(results_dir="distillation_experiments/comparison_results"):
    """Find the most recent comparison results file."""
    pattern = os.path.join(results_dir, "comparison_report_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No comparison results found in {results_dir}")
    
    latest = max(files, key=os.path.getctime)
    return latest

def load_results(file_path):
    """Load and parse results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_performance_visualization(results, output_dir):
    """Create performance comparison visualizations."""
    
    # Extract successful results for plotting
    successful = [r for r in results['all_results'] if r['success'] and r['improvement_ratio'] > 0]
    
    if not successful:
        print("⚠️  No successful results to visualize")
        return
    
    # Create DataFrame for easier plotting
    data = []
    for result in successful:
        data.append({
            'Teacher': result['teacher'].split('/')[-1],  # Get model name without prefix
            'Student': result['student'].split('/')[-1],
            'Improvement_Ratio': result['improvement_ratio'] * 100,
            'Duration_Minutes': result['duration_minutes'],
            'Final_Loss': result['distilled_final_loss'],
            'Pair': f"{result['teacher'].split('/')[-1]} → {result['student'].split('/')[-1]}"
        })
    
    df = pd.DataFrame(data)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distillation Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Improvement Ratio Bar Chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(df)), df['Improvement_Ratio'], color=sns.color_palette("viridis", len(df)))
    ax1.set_xlabel('Teacher-Student Pairs')
    ax1.set_ylabel('Improvement Ratio (%)')
    ax1.set_title('Performance Improvement by Model Pair')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Pair'], rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Training Duration
    ax2 = axes[0, 1]
    ax2.scatter(df['Improvement_Ratio'], df['Duration_Minutes'], 
               s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    ax2.set_xlabel('Improvement Ratio (%)')
    ax2.set_ylabel('Training Duration (minutes)')
    ax2.set_title('Improvement vs Training Time')
    
    # Add annotations for outliers
    for i, row in df.iterrows():
        if row['Improvement_Ratio'] > df['Improvement_Ratio'].mean() + df['Improvement_Ratio'].std():
            ax2.annotate(row['Pair'], (row['Improvement_Ratio'], row['Duration_Minutes']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Final Loss Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(df)), df['Final_Loss'], color=sns.color_palette("plasma", len(df)))
    ax3.set_xlabel('Teacher-Student Pairs')
    ax3.set_ylabel('Final Distilled Loss')
    ax3.set_title('Final Loss by Model Pair')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['Pair'], rotation=45, ha='right')
    
    # 4. Efficiency Plot (Improvement vs Duration)
    ax4 = axes[1, 1]
    bubble_sizes = df['Improvement_Ratio'] * 5  # Scale for visibility
    scatter = ax4.scatter(df['Duration_Minutes'], df['Improvement_Ratio'], 
                         s=bubble_sizes, alpha=0.6, c=df['Final_Loss'], 
                         cmap='RdYlBu_r')
    ax4.set_xlabel('Training Duration (minutes)')
    ax4.set_ylabel('Improvement Ratio (%)')
    ax4.set_title('Efficiency: Improvement vs Time (size=improvement, color=loss)')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Final Loss')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'distillation_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Performance visualization saved to: {plot_path}")
    
    return plot_path

def create_ranking_table(results, output_dir):
    """Create a detailed ranking table."""
    successful = [r for r in results['all_results'] if r['success'] and r['improvement_ratio'] > 0]
    successful.sort(key=lambda x: x['improvement_ratio'], reverse=True)
    
    # Create ranking table
    table_data = []
    for i, result in enumerate(successful, 1):
        table_data.append({
            'Rank': i,
            'Teacher': result['teacher'],
            'Student': result['student'],
            'Improvement (%)': f"{result['improvement_ratio']*100:.1f}%",
            'Final Loss': f"{result['distilled_final_loss']:.4f}",
            'Duration (min)': f"{result['duration_minutes']:.1f}",
            'Efficiency': f"{result['improvement_ratio']/result['duration_minutes']*100:.2f}",  # Improvement per minute
        })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, 'distillation_rankings.csv')
    df.to_csv(csv_path, index=False)
    print(f"📋 Ranking table saved to: {csv_path}")
    
    # Print top 5 to console
    print("\n🏆 TOP 5 PERFORMING PAIRS:")
    print("=" * 100)
    for _, row in df.head(5).iterrows():
        print(f"{row['Rank']}. {row['Teacher']} → {row['Student']}")
        print(f"   📈 Improvement: {row['Improvement (%)']} | ⏱️ Duration: {row['Duration (min)']}min | 📊 Loss: {row['Final Loss']}")
    
    return csv_path

def generate_recommendations(results):
    """Generate specific recommendations based on results."""
    successful = [r for r in results['all_results'] if r['success'] and r['improvement_ratio'] > 0]
    
    if not successful:
        return ["No successful experiments to analyze"]
    
    successful.sort(key=lambda x: x['improvement_ratio'], reverse=True)
    
    recommendations = []
    
    # Best overall
    best = successful[0]
    recommendations.append(f"🎯 **Best Overall**: {best['teacher']} → {best['student']} "
                          f"({best['improvement_ratio']*100:.1f}% improvement)")
    
    # Most efficient (improvement per minute)
    efficiency_sorted = sorted(successful, 
                              key=lambda x: x['improvement_ratio']/x['duration_minutes'], 
                              reverse=True)
    most_efficient = efficiency_sorted[0]
    recommendations.append(f"⚡ **Most Efficient**: {most_efficient['teacher']} → {most_efficient['student']} "
                          f"({most_efficient['improvement_ratio']/most_efficient['duration_minutes']*100:.2f} improvement/min)")
    
    # Fastest training
    fastest = min(successful, key=lambda x: x['duration_minutes'])
    recommendations.append(f"🏃 **Fastest Training**: {fastest['teacher']} → {fastest['student']} "
                          f"({fastest['duration_minutes']:.1f} minutes)")
    
    # Best for different student categories
    tiny_students = [r for r in successful if 'bert-tiny' in r['student']]
    if tiny_students:
        best_tiny = max(tiny_students, key=lambda x: x['improvement_ratio'])
        recommendations.append(f"🔬 **Best for Ultra-Tiny Models**: {best_tiny['teacher']} → {best_tiny['student']} "
                              f"({best_tiny['improvement_ratio']*100:.1f}% improvement)")
    
    small_students = [r for r in successful if 'TinyBERT' in r['student']]
    if small_students:
        best_small = max(small_students, key=lambda x: x['improvement_ratio'])
        recommendations.append(f"📱 **Best for Small Models**: {best_small['teacher']} → {best_small['student']} "
                              f"({best_small['improvement_ratio']*100:.1f}% improvement)")
    
    return recommendations

def analyze_results(file_path):
    """Main analysis function."""
    print(f"📊 Analyzing distillation results from: {file_path}")
    
    # Load results
    results = load_results(file_path)
    
    # Create output directory
    output_dir = os.path.dirname(file_path)
    analysis_dir = os.path.join(output_dir, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\n📈 Experiment Summary:")
    print(f"   🧪 Total Experiments: {results['total_experiments']}")
    print(f"   ✅ Successful: {results['successful_experiments']}")
    print(f"   📅 Test Date: {results['timestamp']}")
    print(f"   🎯 Test Mode: {results['test_mode']}")
    
    # Create visualizations
    if results['successful_experiments'] > 0:
        try:
            create_performance_visualization(results, analysis_dir)
            create_ranking_table(results, analysis_dir)
            
            # Generate and save recommendations
            recommendations = generate_recommendations(results)
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")
            
            # Save recommendations to file
            rec_path = os.path.join(analysis_dir, 'recommendations.txt')
            with open(rec_path, 'w') as f:
                f.write("Distillation Experiment Recommendations\n")
                f.write("=" * 50 + "\n\n")
                for rec in recommendations:
                    f.write(rec + "\n")
            
            print(f"\n📁 Analysis saved to: {analysis_dir}")
            
        except ImportError:
            print("⚠️  matplotlib/seaborn not available. Install with: pip install matplotlib seaborn pandas")
            print("📋 Creating text-only analysis...")
            create_ranking_table(results, analysis_dir)
    
    else:
        print("❌ No successful experiments to analyze")

def main():
    parser = argparse.ArgumentParser(description="Analyze distillation comparison results")
    parser.add_argument("--latest", action="store_true", help="Analyze the latest results file")
    parser.add_argument("file_path", nargs="?", help="Path to results JSON file")
    
    args = parser.parse_args()
    
    if args.latest:
        try:
            file_path = find_latest_results()
            print(f"📂 Found latest results: {file_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    else:
        if not args.file_path:
            print("❌ Please provide a file path or use --latest")
            return
        file_path = args.file_path
    
    analyze_results(file_path)

if __name__ == "__main__":
    main()