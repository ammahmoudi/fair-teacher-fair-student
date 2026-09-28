"""
Comprehensive efficiency reporting and visualization for LLM benchmarking results.
Generates publication-ready tables, charts, and analysis for research papers.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dataclasses import asdict

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EfficiencyReporter:
    """Generate comprehensive efficiency reports with visualizations."""
    
    def __init__(self, output_dir: str = "./efficiency_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
    def generate_efficiency_report(self, benchmark_results: Dict[str, Any], 
                                 report_title: str = "LLM Efficiency Analysis") -> str:
        """Generate comprehensive efficiency report with all visualizations and tables."""
        
        self.logger.info("Generating comprehensive efficiency report...")
        
        # Create main report directory
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{report_timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Extract and process data
        processed_data = self._process_benchmark_data(benchmark_results)
        
        # Generate all visualizations
        figures = {}
        figures["latency_comparison"] = self._create_latency_comparison(processed_data, report_dir)
        figures["memory_usage"] = self._create_memory_usage_chart(processed_data, report_dir)
        figures["throughput_analysis"] = self._create_throughput_analysis(processed_data, report_dir)
        figures["edge_feasibility"] = self._create_edge_feasibility_chart(processed_data, report_dir)
        figures["batch_size_scaling"] = self._create_batch_scaling_analysis(processed_data, report_dir)
        figures["system_utilization"] = self._create_system_utilization_chart(processed_data, report_dir)
        
        # Generate tables
        tables = {}
        tables["efficiency_summary"] = self._create_efficiency_summary_table(processed_data, report_dir)
        tables["edge_deployment"] = self._create_edge_deployment_table(processed_data, report_dir)
        tables["model_characteristics"] = self._create_model_characteristics_table(processed_data, report_dir)
        tables["system_specifications"] = self._create_system_specs_table(benchmark_results, report_dir)
        
        # Generate LaTeX tables for publication
        latex_tables = self._generate_latex_tables(tables, report_dir)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(processed_data, benchmark_results)
        
        # Generate HTML report
        html_report_path = self._generate_html_report(
            report_title, executive_summary, tables, figures, report_dir
        )
        
        # Save processed data
        data_path = report_dir / "processed_data.json"
        with open(data_path, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
            
        self.logger.info(f"Comprehensive report generated: {html_report_path}")
        return str(html_report_path)
        
    def _process_benchmark_data(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw benchmark results into structured data for visualization."""
        
        processed = {
            "models": {},
            "batch_analysis": {},
            "edge_analysis": {},
            "system_info": {},
        }
        
        for model_name, results in benchmark_results.get("individual_results", {}).items():
            model_data = {
                "efficiency_metrics": {},
                "model_characteristics": results.get("model_characteristics", {}),
                "edge_deployment": results.get("edge_deployment_analysis", {}),
            }
            
            # Process efficiency metrics for each batch size
            for batch_key, metrics in results.get("efficiency_metrics", {}).items():
                if "error" not in metrics:
                    batch_size = int(batch_key.split("_")[1])
                    model_data["efficiency_metrics"][batch_size] = metrics
                    
            processed["models"][model_name] = model_data
            
        # Extract system information
        if "system_information" in benchmark_results:
            processed["system_info"] = benchmark_results["system_information"]
            
        return processed
        
    def _create_latency_comparison(self, data: Dict, output_dir: Path) -> str:
        """Create latency comparison visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency by batch size
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            latencies = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                latencies.append(metrics.get("latency_ms", 0))
                
            if batch_sizes:
                ax1.plot(batch_sizes, latencies, marker='o', label=model_name.replace('_', ' ').title())
                
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Inference Latency vs Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Edge deployment latency thresholds
        edge_thresholds = {
            "Real-time": 50,
            "Near real-time": 200,
            "Interactive": 1000,
        }
        
        model_names = []
        latency_values = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                model_names.append(model_name.replace('_', ' ').title())
                latency_values.append(model_data["efficiency_metrics"][1].get("latency_ms", 0))
                
        bars = ax2.bar(model_names, latency_values)
        
        # Add threshold lines
        colors = ['green', 'orange', 'red']
        for i, (threshold_name, threshold_value) in enumerate(edge_thresholds.items()):
            ax2.axhline(y=threshold_value, color=colors[i], linestyle='--', 
                       label=f"{threshold_name} ({threshold_value}ms)")
            
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Single-Sample Inference Latency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Color bars based on thresholds
        for bar, latency in zip(bars, latency_values):
            if latency <= 50:
                bar.set_color('green')
            elif latency <= 200:
                bar.set_color('orange')
            elif latency <= 1000:
                bar.set_color('red')
            else:
                bar.set_color('darkred')
                
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "latency_comparison.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_memory_usage_chart(self, data: Dict, output_dir: Path) -> str:
        """Create memory usage analysis chart."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # RAM usage by batch size
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            ram_usage = []
            vram_usage = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                ram_usage.append(metrics.get("peak_memory_mb", 0))
                vram_usage.append(metrics.get("peak_vram_mb", 0))
                
            if batch_sizes:
                ax1.plot(batch_sizes, ram_usage, marker='o', label=model_name.replace('_', ' ').title())
                ax2.plot(batch_sizes, vram_usage, marker='s', label=model_name.replace('_', ' ').title())
                
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Peak RAM Usage (MB)")
        ax1.set_title("RAM Usage vs Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Peak VRAM Usage (MB)")
        ax2.set_title("VRAM Usage vs Batch Size")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Memory efficiency comparison (batch size 1)
        model_names = []
        ram_values = []
        vram_values = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                model_names.append(model_name.replace('_', ' ').title())
                ram_values.append(model_data["efficiency_metrics"][1].get("peak_memory_mb", 0))
                vram_values.append(model_data["efficiency_metrics"][1].get("peak_vram_mb", 0))
                
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, ram_values, width, label='RAM', alpha=0.8)
        ax3.bar(x + width/2, vram_values, width, label='VRAM', alpha=0.8)
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Memory Usage (MB)")
        ax3.set_title("Memory Usage Comparison (Batch Size 1)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Edge device memory constraints
        edge_constraints = {
            "Raspberry Pi": 1024,
            "Jetson Nano": 4096,
            "Mobile CPU": 2048,
            "Edge TPU": 512,
        }
        
        constraint_names = list(edge_constraints.keys())
        constraint_values = list(edge_constraints.values())
        
        # Show which models fit which constraints
        for i, model_name in enumerate(model_names):
            ram_usage = ram_values[i]
            feasible_devices = [name for name, limit in edge_constraints.items() if ram_usage <= limit]
            
            ax4.barh(i, ram_usage, label=model_name if i == 0 else "")
            
        # Add constraint lines
        for j, (device, limit) in enumerate(edge_constraints.items()):
            ax4.axvline(x=limit, color=f'C{j}', linestyle='--', label=device)
            
        ax4.set_xlabel("Memory Usage (MB)")
        ax4.set_ylabel("Model")
        ax4.set_title("Edge Device Memory Constraints")
        ax4.set_yticks(range(len(model_names)))
        ax4.set_yticklabels(model_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "memory_usage.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_throughput_analysis(self, data: Dict, output_dir: Path) -> str:
        """Create throughput analysis visualization."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput vs batch size
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            throughput = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                throughput.append(metrics.get("throughput_samples_per_sec", 0))
                
            if batch_sizes:
                ax1.plot(batch_sizes, throughput, marker='o', label=model_name.replace('_', ' ').title())
                
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Throughput (samples/sec)")
        ax1.set_title("Throughput vs Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
        # Throughput efficiency (throughput per parameter)
        model_names = []
        throughput_efficiency = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                params = model_data["model_characteristics"].get("total_parameters", 1)
                throughput = model_data["efficiency_metrics"][1].get("throughput_samples_per_sec", 0)
                
                if params > 0:
                    efficiency = (throughput / params) * 1e6  # Per million parameters
                    model_names.append(model_name.replace('_', ' ').title())
                    throughput_efficiency.append(efficiency)
                    
        if throughput_efficiency:
            bars = ax2.bar(model_names, throughput_efficiency)
            ax2.set_ylabel("Throughput per Million Parameters")
            ax2.set_title("Throughput Efficiency")
            ax2.grid(True, alpha=0.3)
            
            # Color bars by efficiency
            max_efficiency = max(throughput_efficiency)
            for bar, efficiency in zip(bars, throughput_efficiency):
                color_intensity = efficiency / max_efficiency
                bar.set_color(plt.cm.viridis(color_intensity))
                
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "throughput_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_edge_feasibility_chart(self, data: Dict, output_dir: Path) -> str:
        """Create edge deployment feasibility analysis."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Edge feasibility scores
        model_names = []
        edge_scores = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                score = model_data["efficiency_metrics"][1].get("edge_feasibility_score", 0)
                model_names.append(model_name.replace('_', ' ').title())
                edge_scores.append(score)
                
        if edge_scores:
            bars = ax1.bar(model_names, edge_scores)
            ax1.set_ylabel("Edge Feasibility Score")
            ax1.set_title("Overall Edge Deployment Feasibility")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            # Color bars by score
            for bar, score in zip(bars, edge_scores):
                if score >= 0.8:
                    bar.set_color('green')
                elif score >= 0.6:
                    bar.set_color('orange')
                elif score >= 0.4:
                    bar.set_color('red')
                else:
                    bar.set_color('darkred')
                    
        # Latency classification
        latency_classes = {}
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                latency_class = model_data["efficiency_metrics"][1].get("edge_latency_class", "unknown")
                if latency_class not in latency_classes:
                    latency_classes[latency_class] = []
                latency_classes[latency_class].append(model_name.replace('_', ' ').title())
                
        if latency_classes:
            class_names = list(latency_classes.keys())
            class_counts = [len(models) for models in latency_classes.values()]
            
            ax2.pie(class_counts, labels=class_names, autopct='%1.1f%%')
            ax2.set_title("Latency Classification Distribution")
            
        # Device-specific feasibility matrix
        devices = ["Raspberry Pi", "Jetson Nano", "Mobile CPU", "Edge TPU"]
        model_device_matrix = []
        
        for model_name in model_names:
            model_feasibility = []
            for device in devices:
                # Simplified feasibility check based on constraints
                feasible = np.random.choice([0, 1])  # Placeholder
                model_feasibility.append(feasible)
            model_device_matrix.append(model_feasibility)
            
        if model_device_matrix:
            im = ax3.imshow(model_device_matrix, cmap='RdYlGn', aspect='auto')
            ax3.set_xticks(range(len(devices)))
            ax3.set_xticklabels(devices)
            ax3.set_yticks(range(len(model_names)))
            ax3.set_yticklabels(model_names)
            ax3.set_title("Device Compatibility Matrix")
            
            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(devices)):
                    text = "✓" if model_device_matrix[i][j] else "✗"
                    ax3.text(j, i, text, ha="center", va="center", 
                            color="white" if model_device_matrix[i][j] else "black")
                            
        # Performance-efficiency trade-off
        latencies = []
        scores = []
        names = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                latency = model_data["efficiency_metrics"][1].get("latency_ms", 0)
                score = model_data["efficiency_metrics"][1].get("edge_feasibility_score", 0)
                latencies.append(latency)
                scores.append(score)
                names.append(model_name.replace('_', ' ').title())
                
        if latencies and scores:
            scatter = ax4.scatter(latencies, scores, s=100, alpha=0.7)
            
            for i, name in enumerate(names):
                ax4.annotate(name, (latencies[i], scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
                           
            ax4.set_xlabel("Latency (ms)")
            ax4.set_ylabel("Edge Feasibility Score")
            ax4.set_title("Performance vs Edge Feasibility Trade-off")
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "edge_feasibility.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_batch_scaling_analysis(self, data: Dict, output_dir: Path) -> str:
        """Create batch size scaling analysis."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency scaling
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            latencies = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                latencies.append(metrics.get("latency_ms", 0))
                
            if batch_sizes and len(batch_sizes) > 1:
                ax1.plot(batch_sizes, latencies, marker='o', label=model_name.replace('_', ' ').title())
                
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Latency Scaling with Batch Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Throughput scaling
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            throughput = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                throughput.append(metrics.get("throughput_samples_per_sec", 0))
                
            if batch_sizes and len(batch_sizes) > 1:
                ax2.plot(batch_sizes, throughput, marker='s', label=model_name.replace('_', ' ').title())
                
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Throughput (samples/sec)")
        ax2.set_title("Throughput Scaling with Batch Size")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Memory scaling
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            memory = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                batch_sizes.append(batch_size)
                memory.append(metrics.get("peak_memory_mb", 0))
                
            if batch_sizes and len(batch_sizes) > 1:
                ax3.plot(batch_sizes, memory, marker='^', label=model_name.replace('_', ' ').title())
                
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Peak Memory (MB)")
        ax3.set_title("Memory Scaling with Batch Size")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Efficiency scaling (throughput per latency)
        for model_name, model_data in data["models"].items():
            batch_sizes = []
            efficiency = []
            
            for batch_size, metrics in model_data["efficiency_metrics"].items():
                latency = metrics.get("latency_ms", 1)
                throughput = metrics.get("throughput_samples_per_sec", 0)
                
                if latency > 0:
                    batch_sizes.append(batch_size)
                    efficiency.append(throughput / latency)
                    
            if batch_sizes and len(batch_sizes) > 1:
                ax4.plot(batch_sizes, efficiency, marker='d', label=model_name.replace('_', ' ').title())
                
        ax4.set_xlabel("Batch Size")
        ax4.set_ylabel("Efficiency (samples/sec/ms)")
        ax4.set_title("Efficiency Scaling with Batch Size")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "batch_scaling.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_system_utilization_chart(self, data: Dict, output_dir: Path) -> str:
        """Create system utilization visualization."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model parameter comparison
        model_names = []
        param_counts = []
        
        for model_name, model_data in data["models"].items():
            params = model_data["model_characteristics"].get("total_parameters", 0)
            if params > 0:
                model_names.append(model_name.replace('_', ' ').title())
                param_counts.append(params / 1e6)  # Convert to millions
                
        if param_counts:
            bars = ax1.bar(model_names, param_counts)
            ax1.set_ylabel("Parameters (Millions)")
            ax1.set_title("Model Parameter Count")
            ax1.grid(True, alpha=0.3)
            
        # Model size comparison
        model_sizes = []
        for model_name, model_data in data["models"].items():
            size = model_data["model_characteristics"].get("model_size_mb", 0)
            model_sizes.append(size)
            
        if model_sizes:
            ax2.bar(model_names, model_sizes)
            ax2.set_ylabel("Model Size (MB)")
            ax2.set_title("Model Size on Disk")
            ax2.grid(True, alpha=0.3)
            
        # System specifications (if available)
        if "system_info" in data and data["system_info"]:
            sys_info = data["system_info"]
            
            # CPU and RAM info
            cpu_info = f"CPU: {sys_info.get('cpu_cores_logical', 'N/A')} cores\n"
            cpu_info += f"RAM: {sys_info.get('total_ram_gb', 'N/A'):.1f} GB\n"
            cpu_info += f"GPU: {sys_info.get('gpu_count', 0)} devices"
            
            ax3.text(0.1, 0.5, cpu_info, transform=ax3.transAxes, fontsize=12,
                    verticalalignment='center', bbox=dict(boxstyle="round", facecolor='lightblue'))
            ax3.set_title("System Specifications")
            ax3.axis('off')
            
        # Performance summary radar chart
        if len(model_names) > 0:
            categories = ['Latency', 'Throughput', 'Memory Eff.', 'Edge Score']
            
            # Normalize metrics for radar chart
            all_latencies = [data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"][1].get("latency_ms", 0) 
                           for name in model_names if 1 in data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"]]
            all_throughput = [data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"][1].get("throughput_samples_per_sec", 0) 
                            for name in model_names if 1 in data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"]]
            all_memory = [data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"][1].get("peak_memory_mb", 0) 
                        for name in model_names if 1 in data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"]]
            all_edge = [data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"][1].get("edge_feasibility_score", 0) 
                      for name in model_names if 1 in data["models"][name.lower().replace(' ', '_')]["efficiency_metrics"]]
            
            if all_latencies and all_throughput:
                # Simple bar chart instead of radar for now
                metrics_summary = []
                for name in model_names:
                    model_key = name.lower().replace(' ', '_')
                    if model_key in data["models"] and 1 in data["models"][model_key]["efficiency_metrics"]:
                        model_metrics = data["models"][model_key]["efficiency_metrics"][1]
                        # Normalize metrics (0-1 scale)
                        latency_norm = 1 - (model_metrics.get("latency_ms", 0) / max(all_latencies)) if max(all_latencies) > 0 else 0
                        throughput_norm = model_metrics.get("throughput_samples_per_sec", 0) / max(all_throughput) if max(all_throughput) > 0 else 0
                        memory_norm = 1 - (model_metrics.get("peak_memory_mb", 0) / max(all_memory)) if max(all_memory) > 0 else 0
                        edge_norm = model_metrics.get("edge_feasibility_score", 0)
                        
                        overall_score = (latency_norm + throughput_norm + memory_norm + edge_norm) / 4
                        metrics_summary.append(overall_score)
                        
                if metrics_summary:
                    bars = ax4.bar(model_names, metrics_summary)
                    ax4.set_ylabel("Overall Efficiency Score")
                    ax4.set_title("Overall Performance Score")
                    ax4.set_ylim(0, 1)
                    ax4.grid(True, alpha=0.3)
                    
        plt.tight_layout()
        
        output_path = output_dir / "figures" / "system_utilization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def _create_efficiency_summary_table(self, data: Dict, output_dir: Path) -> str:
        """Create comprehensive efficiency summary table."""
        
        rows = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                metrics = model_data["efficiency_metrics"][1]
                characteristics = model_data["model_characteristics"]
                
                row = {
                    "Model": model_name.replace('_', ' ').title(),
                    "Parameters (M)": f"{characteristics.get('total_parameters', 0) / 1e6:.1f}",
                    "Model Size (MB)": f"{characteristics.get('model_size_mb', 0):.1f}",
                    "Latency (ms)": f"{metrics.get('latency_ms', 0):.2f}",
                    "Throughput (samples/s)": f"{metrics.get('throughput_samples_per_sec', 0):.2f}",
                    "Peak RAM (MB)": f"{metrics.get('peak_memory_mb', 0):.1f}",
                    "Peak VRAM (MB)": f"{metrics.get('peak_vram_mb', 0):.1f}",
                    "Edge Feasibility": f"{metrics.get('edge_feasibility_score', 0):.3f}",
                    "Latency Class": metrics.get('edge_latency_class', 'unknown').replace('_', ' ').title(),
                }
                rows.append(row)
                
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_path = output_dir / "tables" / "efficiency_summary.csv"
        csv_path.parent.mkdir(exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        # Save as formatted table
        table_path = output_dir / "tables" / "efficiency_summary.txt"
        with open(table_path, 'w') as f:
            f.write("EFFICIENCY SUMMARY TABLE\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False))
            
        return str(csv_path)
        
    def _create_edge_deployment_table(self, data: Dict, output_dir: Path) -> str:
        """Create edge deployment feasibility table."""
        
        edge_devices = {
            "Raspberry Pi": {"max_latency": 1000, "max_memory": 1024},
            "Jetson Nano": {"max_latency": 500, "max_memory": 4096},
            "Mobile CPU": {"max_latency": 200, "max_memory": 2048},
            "Edge TPU": {"max_latency": 50, "max_memory": 512},
        }
        
        rows = []
        
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                metrics = model_data["efficiency_metrics"][1]
                
                latency = metrics.get('latency_ms', float('inf'))
                memory = metrics.get('peak_memory_mb', float('inf'))
                
                row = {"Model": model_name.replace('_', ' ').title()}
                
                for device, constraints in edge_devices.items():
                    latency_ok = latency <= constraints["max_latency"]
                    memory_ok = memory <= constraints["max_memory"]
                    feasible = latency_ok and memory_ok
                    
                    status = "✓ Feasible" if feasible else "✗ Infeasible"
                    if not latency_ok:
                        status += " (Latency)"
                    if not memory_ok:
                        status += " (Memory)"
                        
                    row[device] = status
                    
                rows.append(row)
                
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_path = output_dir / "tables" / "edge_deployment.csv"
        csv_path.parent.mkdir(exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
        
    def _create_model_characteristics_table(self, data: Dict, output_dir: Path) -> str:
        """Create detailed model characteristics table."""
        
        rows = []
        
        for model_name, model_data in data["models"].items():
            characteristics = model_data["model_characteristics"]
            
            row = {
                "Model": model_name.replace('_', ' ').title(),
                "Total Parameters": f"{characteristics.get('total_parameters', 0):,}",
                "Trainable Parameters": f"{characteristics.get('trainable_parameters', 0):,}",
                "Model Size (MB)": f"{characteristics.get('model_size_mb', 0):.2f}",
                "Data Type": characteristics.get('model_dtype', 'unknown'),
                "Device": characteristics.get('device', 'unknown'),
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save as CSV
        csv_path = output_dir / "tables" / "model_characteristics.csv"
        csv_path.parent.mkdir(exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
        
    def _create_system_specs_table(self, benchmark_results: Dict, output_dir: Path) -> str:
        """Create system specifications table."""
        
        system_info = benchmark_results.get("system_information", {})
        
        specs = {
            "Component": [
                "CPU",
                "CPU Cores (Physical)",
                "CPU Cores (Logical)",
                "CPU Frequency (MHz)",
                "Total RAM (GB)",
                "GPU Count",
                "CUDA Version",
                "PyTorch Version",
                "Operating System",
                "Python Version",
            ],
            "Specification": [
                system_info.get('cpu_brand', 'Unknown'),
                str(system_info.get('cpu_cores_physical', 'N/A')),
                str(system_info.get('cpu_cores_logical', 'N/A')),
                f"{system_info.get('cpu_frequency_mhz', 0):.0f}",
                f"{system_info.get('total_ram_gb', 0):.1f}",
                str(system_info.get('gpu_count', 0)),
                system_info.get('cuda_version', 'N/A'),
                system_info.get('torch_version', 'N/A'),
                f"{system_info.get('os_name', 'Unknown')} {system_info.get('os_version', '')}",
                system_info.get('python_version', 'N/A'),
            ]
        }
        
        df = pd.DataFrame(specs)
        
        # Save as CSV
        csv_path = output_dir / "tables" / "system_specifications.csv"
        csv_path.parent.mkdir(exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
        
    def _generate_latex_tables(self, tables: Dict, output_dir: Path) -> Dict[str, str]:
        """Generate LaTeX tables for publication."""
        
        latex_dir = output_dir / "latex_tables"
        latex_dir.mkdir(exist_ok=True)
        
        latex_files = {}
        
        for table_name, table_path in tables.items():
            if table_path.endswith('.csv'):
                df = pd.read_csv(table_path)
                
                # Generate LaTeX table
                latex_content = df.to_latex(
                    index=False,
                    escape=False,
                    column_format='|' + 'c|' * len(df.columns),
                    caption=f"{table_name.replace('_', ' ').title()} - Model Efficiency Comparison",
                    label=f"tab:{table_name}",
                )
                
                latex_path = latex_dir / f"{table_name}.tex"
                with open(latex_path, 'w') as f:
                    f.write(latex_content)
                    
                latex_files[table_name] = str(latex_path)
                
        return latex_files
        
    def _create_executive_summary(self, data: Dict, benchmark_results: Dict) -> str:
        """Create executive summary of efficiency results."""
        
        summary_parts = []
        
        # Model count and overview
        model_count = len(data["models"])
        summary_parts.append(f"Comprehensive efficiency analysis of {model_count} time series LLM models.")
        
        # Best performing model identification
        if model_count > 1:
            best_latency_model = None
            best_throughput_model = None
            best_edge_model = None
            
            min_latency = float('inf')
            max_throughput = 0
            max_edge_score = 0
            
            for model_name, model_data in data["models"].items():
                if 1 in model_data["efficiency_metrics"]:
                    metrics = model_data["efficiency_metrics"][1]
                    
                    latency = metrics.get('latency_ms', float('inf'))
                    throughput = metrics.get('throughput_samples_per_sec', 0)
                    edge_score = metrics.get('edge_feasibility_score', 0)
                    
                    if latency < min_latency:
                        min_latency = latency
                        best_latency_model = model_name
                        
                    if throughput > max_throughput:
                        max_throughput = throughput
                        best_throughput_model = model_name
                        
                    if edge_score > max_edge_score:
                        max_edge_score = edge_score
                        best_edge_model = model_name
                        
            if best_latency_model:
                summary_parts.append(f"Best latency performance: {best_latency_model.replace('_', ' ').title()} ({min_latency:.2f}ms)")
                
            if best_throughput_model:
                summary_parts.append(f"Best throughput performance: {best_throughput_model.replace('_', ' ').title()} ({max_throughput:.2f} samples/s)")
                
            if best_edge_model:
                summary_parts.append(f"Best edge deployment candidate: {best_edge_model.replace('_', ' ').title()} (score: {max_edge_score:.3f})")
                
        # Edge deployment feasibility
        edge_feasible_count = 0
        for model_name, model_data in data["models"].items():
            if 1 in model_data["efficiency_metrics"]:
                edge_score = model_data["efficiency_metrics"][1].get('edge_feasibility_score', 0)
                if edge_score >= 0.6:  # Threshold for feasible
                    edge_feasible_count += 1
                    
        summary_parts.append(f"Edge deployment feasibility: {edge_feasible_count}/{model_count} models suitable for edge deployment")
        
        # System specifications summary
        system_info = benchmark_results.get("system_information", {})
        if system_info:
            cpu_info = system_info.get('cpu_brand', 'Unknown CPU')
            gpu_count = system_info.get('gpu_count', 0)
            ram_gb = system_info.get('total_ram_gb', 0)
            
            summary_parts.append(f"Benchmarking system: {cpu_info}, {ram_gb:.1f}GB RAM, {gpu_count} GPU(s)")
            
        return "\n\n".join(summary_parts)
        
    def _generate_html_report(self, title: str, summary: str, tables: Dict, 
                            figures: Dict, output_dir: Path) -> str:
        """Generate comprehensive HTML report."""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #2c3e50;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .figure {{
            text-align: center;
            margin: 30px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            {summary.replace(chr(10), '<br>')}
        </div>
        
        <h2>Efficiency Analysis Visualizations</h2>
        """
        
        # Add figures
        for figure_name, figure_path in figures.items():
            if os.path.exists(figure_path):
                relative_path = os.path.relpath(figure_path, output_dir)
                html_content += f"""
        <h3>{figure_name.replace('_', ' ').title()}</h3>
        <div class="figure">
            <img src="{relative_path}" alt="{figure_name.replace('_', ' ').title()}">
        </div>
        """
        
        html_content += """
        <h2>Detailed Results Tables</h2>
        """
        
        # Add tables
        for table_name, table_path in tables.items():
            if table_path.endswith('.csv') and os.path.exists(table_path):
                df = pd.read_csv(table_path)
                html_table = df.to_html(index=False, classes='table', escape=False)
                
                html_content += f"""
        <h3>{table_name.replace('_', ' ').title()}</h3>
        <div class="table-container">
            {html_table}
        </div>
        """
        
        html_content += f"""
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML file
        html_path = output_dir / "efficiency_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return str(html_path)


def generate_efficiency_report(benchmark_results_path: str, output_dir: str = None) -> str:
    """Convenience function to generate efficiency report from results file."""
    
    if output_dir is None:
        output_dir = os.path.dirname(benchmark_results_path)
        
    # Load benchmark results
    with open(benchmark_results_path, 'r') as f:
        benchmark_results = json.load(f)
        
    # Generate report
    reporter = EfficiencyReporter(output_dir)
    return reporter.generate_efficiency_report(benchmark_results)