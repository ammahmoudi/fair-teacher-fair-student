"""Script to combine multiple performance reports into one comprehensive report."""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional


def find_performance_reports(log_dir: str) -> List[str]:
    """Find all performance report JSON files in the log directory."""
    pattern = os.path.join(log_dir, "**/real_performance_report_*.json")
    return glob.glob(pattern, recursive=True)


def load_report(filepath: str) -> Optional[Dict]:
    """Load a JSON report file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def categorize_reports(reports: List[Dict]) -> Dict[str, Dict]:
    """Categorize reports by type (training, inference, training_inference)."""
    categorized = {
        'training': None,
        'inference': None, 
        'training_inference': None
    }
    
    for report in reports:
        model_name = report.get('model_name', '').lower()
        
        if 'training_inference' in model_name:
            categorized['training_inference'] = report
        elif 'training' in model_name:
            categorized['training'] = report
        elif 'inference' in model_name or any(x in model_name for x in ['chronos', 'time_llm']):
            categorized['inference'] = report
    
    return categorized


def create_comprehensive_report(categorized_reports: Dict[str, Dict], base_model_name: str) -> Dict[str, Any]:
    """Create a comprehensive report combining all performance data."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Get system info from any available report
    system_info = None
    for report in categorized_reports.values():
        if report and 'system_info' in report:
            system_info = report['system_info']
            break
    
    comprehensive_report = {
        "model_name": f"{base_model_name}_comprehensive",
        "timestamp": timestamp,
        "system_info": system_info,
        "performance_summary": {},
        "detailed_measurements": {}
    }
    
    # Process each category
    for mode, report in categorized_reports.items():
        if report is None:
            continue
            
        # Store detailed measurements
        comprehensive_report["detailed_measurements"][mode] = report
        
        # Extract key metrics for summary
        if "real_performance_measurements" in report:
            perf_data = report["real_performance_measurements"]
            summary = {}
            
            if "inference_timing" in perf_data:
                timing = perf_data["inference_timing"]
                summary["average_latency_ms"] = timing.get("average_latency_ms", 0)
                summary["total_inferences"] = timing.get("total_inferences_measured", 0)
                summary["median_latency_ms"] = timing.get("median_latency_ms", 0)
                summary["p95_latency_ms"] = timing.get("p95_latency_ms", 0)
            
            if "memory_usage" in perf_data:
                memory = perf_data["memory_usage"]
                # Prioritize process-specific metrics, fallback to system metrics
                summary["process_peak_ram_mb"] = memory.get("process_peak_ram_mb", memory.get("peak_ram_mb", 0))
                summary["process_average_ram_mb"] = memory.get("process_average_ram_mb", memory.get("average_ram_mb", 0))
                summary["system_peak_ram_mb"] = memory.get("system_peak_ram_mb", 0)
                summary["process_peak_delta_mb"] = memory.get("process_peak_delta_mb", 0)
            
            if "gpu_memory_usage" in perf_data:
                gpu_mem = perf_data["gpu_memory_usage"]
                summary["peak_gpu_allocated_mb"] = gpu_mem.get("peak_allocated_mb", 0)
                summary["average_gpu_allocated_mb"] = gpu_mem.get("average_allocated_mb", 0)
                summary["peak_gpu_reserved_mb"] = gpu_mem.get("peak_reserved_mb", 0)
            
            if "nvidia_ml_metrics" in perf_data:
                nvidia = perf_data["nvidia_ml_metrics"]
                summary["peak_gpu_utilization_percent"] = nvidia.get("peak_gpu_utilization_percent", 0)
                summary["average_gpu_utilization_percent"] = nvidia.get("average_gpu_utilization_percent", 0)
                summary["peak_temperature_celsius"] = nvidia.get("peak_temperature_celsius", 0)
                summary["peak_power_usage_watts"] = nvidia.get("peak_power_usage_watts", 0)
                summary["average_power_usage_watts"] = nvidia.get("average_power_usage_watts", 0)
            
            if "model_file_metrics" in perf_data:
                model_metrics = perf_data["model_file_metrics"]
                summary["model_size_on_disk_mb"] = model_metrics.get("model_size_on_disk_mb", 0)
                summary["parameters_count"] = model_metrics.get("parameters_count", 0)
            
            if "edge_deployment_analysis" in report:
                edge = report["edge_deployment_analysis"]
                summary["edge_feasibility"] = edge.get("feasibility_assessment", "unknown")
                summary["feasible_edge_devices"] = edge.get("feasible_edge_devices", [])
            
            comprehensive_report["performance_summary"][mode] = summary
    
    return comprehensive_report


def save_comprehensive_report(report: Dict[str, Any], output_dir: str) -> str:
    """Save the comprehensive report to a JSON file."""
    timestamp = report["timestamp"]
    model_name = report["model_name"]
    filename = f"comprehensive_performance_report_{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return filepath


def combine_performance_reports(log_dir: str, base_model_name: str = "model") -> str:
    """Main function to combine all performance reports in a directory."""
    print(f"🔍 Searching for performance reports in: {log_dir}")
    
    # Find all performance report files
    report_files = find_performance_reports(log_dir)
    
    if not report_files:
        print("❌ No performance reports found!")
        return None
    
    print(f"📊 Found {len(report_files)} performance reports:")
    for file in report_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load all reports
    reports = []
    for filepath in report_files:
        report = load_report(filepath)
        if report:
            reports.append(report)
    
    if not reports:
        print("❌ No valid reports could be loaded!")
        return None
    
    # Categorize reports
    categorized = categorize_reports(reports)
    print(f"\n📋 Categorized reports:")
    for mode, report in categorized.items():
        status = "✅ Found" if report else "❌ Missing"
        print(f"  - {mode}: {status}")
    
    # Create comprehensive report
    comprehensive_report = create_comprehensive_report(categorized, base_model_name)
    
    # Save comprehensive report
    output_path = save_comprehensive_report(comprehensive_report, log_dir)
    
    print(f"\n✅ Comprehensive performance report saved to: {output_path}")
    
    # Print summary
    print(f"\n📊 Performance Summary:")
    for mode, summary in comprehensive_report["performance_summary"].items():
        print(f"\n{mode.upper()}:")
        if "average_latency_ms" in summary:
            print(f"  - Average Latency: {summary['average_latency_ms']:.2f}ms")
        if "process_peak_ram_mb" in summary:
            print(f"  - Process Peak RAM: {summary['process_peak_ram_mb']:.1f}MB (ML model only)")
        if "system_peak_ram_mb" in summary:
            print(f"  - System Peak RAM: {summary['system_peak_ram_mb']:.1f}MB (all processes)")
        if "process_peak_delta_mb" in summary:
            print(f"  - ML Model RAM Usage: {summary['process_peak_delta_mb']:.1f}MB (delta from baseline)")
        if "peak_gpu_allocated_mb" in summary:
            print(f"  - Peak GPU Memory: {summary['peak_gpu_allocated_mb']:.1f}MB")
        if "peak_gpu_utilization_percent" in summary:
            print(f"  - Peak GPU Utilization: {summary['peak_gpu_utilization_percent']}%")
        if "peak_temperature_celsius" in summary:
            print(f"  - Peak Temperature: {summary['peak_temperature_celsius']}°C")
        if "peak_power_usage_watts" in summary:
            print(f"  - Peak Power: {summary['peak_power_usage_watts']:.1f}W")
        if "edge_feasibility" in summary:
            print(f"  - Edge Feasibility: {summary['edge_feasibility']}")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python combine_reports.py <log_directory> [model_name]")
        sys.exit(1)
    
    log_directory = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "model"
    
    combine_performance_reports(log_directory, model_name)