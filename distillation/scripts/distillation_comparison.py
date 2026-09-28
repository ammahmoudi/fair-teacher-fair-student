#!/usr/bin/env python3
"""
Comprehensive Distillation Comparison Framework
==============================================

This script systematically tests different teacher-student model pairs
to find the optimal combinations for knowledge distillation.

Usage:
    python scripts/distillation_comparison.py --mode quick
    python scripts/distillation_comparison.py --mode comprehensive
    python scripts/distillation_comparison.py --custom-pairs bert-base-uncased,prajjwal1/bert-tiny
"""

import argparse
import subprocess
import json
import csv
import os
from datetime import datetime
from itertools import product
import time

class DistillationTester:
    def __init__(self):
        self.results_dir = "distillation_experiments/comparison_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define model categories based on our ecosystem analysis
        self.teacher_models = {
            # Large, capable models (good teachers)
            "bert-base-uncased": {"size": "110M", "dim": 768, "category": "large"},
            "bert-large-uncased": {"size": "340M", "dim": 1024, "category": "very_large"},
            "distilbert-base-uncased": {"size": "66M", "dim": 768, "category": "medium"},
            "gpt2": {"size": "117M", "dim": 768, "category": "generative"},
            "huggyllama/llama-7b": {"size": "7B", "dim": 4096, "category": "huge"},
            "google/mobilebert-uncased": {"size": "25M", "dim": 512, "category": "mobile"},
            "albert/albert-base-v2": {"size": "12M", "dim": 768, "category": "efficient"},
            "facebook/opt-125m": {"size": "125M", "dim": 768, "category": "large"},
        }
        
        self.student_models = {
            # Small, fast models (good students)
            "prajjwal1/bert-tiny": {"size": "4M", "dim": 128, "category": "tiny"},
            "huawei-noah/TinyBERT_General_4L_312D": {"size": "14M", "dim": 312, "category": "small"},
            "prajjwal1/bert-mini": {"size": "11M", "dim": 256, "category": "small"},
            "prajjwal1/bert-small": {"size": "29M", "dim": 512, "category": "small"},
            "prajjwal1/bert-medium": {"size": "41M", "dim": 512, "category": "medium"},
            "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large": {"size": "22M", "dim": 384, "category": "small"},
        }
        
        self.test_params = {
            "dataset": "ohiot1dm",
            "patients": ["570"],  # Start with one patient for speed
            "seed": 42,
            "teacher_epochs": 2,  # Quick training for comparison
            "student_epochs": 2,
            "distill_epochs": 2,
        }

    def get_model_pairs(self, mode="balanced"):
        """Generate teacher-student pairs based on strategy."""
        pairs = []
        
        if mode == "all":
            # Test all combinations
            for teacher in self.teacher_models.keys():
                for student in self.student_models.keys():
                    pairs.append((teacher, student))
                    
        elif mode == "balanced":
            # Strategic combinations based on model characteristics
            strategic_pairs = [
                # Strong teacher -> tiny student (max compression)
                ("bert-base-uncased", "prajjwal1/bert-tiny"),
                ("bert-large-uncased", "prajjwal1/bert-tiny"),
                ("facebook/opt-125m", "prajjwal1/bert-tiny"),
                
                # Strong teacher -> small student (good balance)
                ("bert-base-uncased", "huawei-noah/TinyBERT_General_4L_312D"),
                ("gpt2", "prajjwal1/bert-mini"),
                
                # Medium teacher -> small student (efficient)
                ("distilbert-base-uncased", "prajjwal1/bert-small"),
                ("google/mobilebert-uncased", "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"),
                
                # Extreme compression experiments
                ("huggyllama/llama-7b", "prajjwal1/bert-tiny"),
                ("albert/albert-base-v2", "prajjwal1/bert-mini"),
            ]
            pairs = strategic_pairs
            
        elif mode == "quick":
            # Just a few key comparisons for rapid testing
            quick_pairs = [
                ("bert-base-uncased", "prajjwal1/bert-tiny"),
                ("bert-base-uncased", "huawei-noah/TinyBERT_General_4L_312D"),
                ("distilbert-base-uncased", "prajjwal1/bert-small"),
            ]
            pairs = quick_pairs
            
        return pairs

    def run_distillation_experiment(self, teacher, student, experiment_id):
        """Run a single teacher-student distillation experiment."""
        print(f"\n🧪 Running Experiment {experiment_id}: {teacher} -> {student}")
        
        # Create unique output directory for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"{self.results_dir}/exp_{experiment_id}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "bash", "scripts/pipelines/distill_pipeline.sh",
            "--teacher", teacher,
            "--student", student,
            "--patients", ",".join(self.test_params["patients"]),
            "--dataset", self.test_params["dataset"],
            "--seed", str(self.test_params["seed"]),
            "--teacher-epochs", str(self.test_params["teacher_epochs"]),
            "--student-epochs", str(self.test_params["student_epochs"]),
            "--distill-epochs", str(self.test_params["distill_epochs"]),
            "--output-dir", exp_dir
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        # Run experiment and capture results
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            end_time = time.time()
            
            return {
                "experiment_id": experiment_id,
                "teacher": teacher,
                "student": student,
                "success": result.returncode == 0,
                "duration_minutes": (end_time - start_time) / 60,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_dir": exp_dir,
                "timestamp": timestamp
            }
            
        except subprocess.TimeoutExpired:
            return {
                "experiment_id": experiment_id,
                "teacher": teacher,
                "student": student,
                "success": False,
                "duration_minutes": 30,
                "error": "Timeout after 30 minutes",
                "output_dir": exp_dir,
                "timestamp": timestamp
            }

    def extract_metrics_from_logs(self, result):
        """Extract performance metrics from experiment logs."""
        metrics = {
            "teacher_final_loss": 0.0,
            "student_final_loss": 0.0,
            "distilled_final_loss": 0.0,
            "improvement_ratio": 0.0,
        }
        
        if not result["success"]:
            return metrics
            
        # Look for CSV log files in the output directory
        csv_files = []
        if os.path.exists(result["output_dir"]):
            for root, dirs, files in os.walk(result["output_dir"]):
                csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
        
        # Extract metrics from the most recent CSV log
        if csv_files:
            latest_csv = max(csv_files, key=os.path.getctime)
            try:
                with open(latest_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        metrics["teacher_final_loss"] = float(last_row.get("teacher_final_loss", 0))
                        metrics["student_final_loss"] = float(last_row.get("student_final_loss", 0))
                        metrics["distilled_final_loss"] = float(last_row.get("distilled_final_loss", 0))
                        
                        # Calculate improvement ratio
                        if metrics["student_final_loss"] and metrics["distilled_final_loss"]:
                            metrics["improvement_ratio"] = (
                                metrics["student_final_loss"] - metrics["distilled_final_loss"]
                            ) / metrics["student_final_loss"]
                            
            except Exception as e:
                print(f"   ⚠️  Could not parse metrics from {latest_csv}: {e}")
        
        return metrics

    def run_comparison(self, mode="balanced", custom_pairs=None):
        """Run full comparison of teacher-student pairs."""
        print("🚀 Starting Distillation Comparison Framework")
        print(f"📊 Mode: {mode}")
        print(f"🗃️  Dataset: {self.test_params['dataset']}")
        print(f"👥 Patients: {self.test_params['patients']}")
        print(f"🎲 Seed: {self.test_params['seed']}")
        
        # Get model pairs to test
        if custom_pairs:
            pairs = [(pair.split(',')[0].strip(), pair.split(',')[1].strip()) for pair in custom_pairs]
        else:
            pairs = self.get_model_pairs(mode)
        
        print(f"\n🔬 Testing {len(pairs)} teacher-student combinations:")
        for i, (teacher, student) in enumerate(pairs, 1):
            teacher_info = self.teacher_models.get(teacher, {"size": "Unknown"})
            student_info = self.student_models.get(student, {"size": "Unknown"})
            print(f"   {i}. {teacher} ({teacher_info['size']}) -> {student} ({student_info['size']})")
        
        # Run experiments
        all_results = []
        for i, (teacher, student) in enumerate(pairs, 1):
            result = self.run_distillation_experiment(teacher, student, i)
            metrics = self.extract_metrics_from_logs(result)
            result.update(metrics)
            all_results.append(result)
            
            # Print quick status
            status = "✅" if result["success"] else "❌"
            duration = f"{result['duration_minutes']:.1f}min"
            print(f"   {status} Experiment {i}: {duration}")
        
        # Generate report
        self.generate_comparison_report(all_results, mode)
        return all_results

    def generate_comparison_report(self, results, mode):
        """Generate comprehensive comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.results_dir}/comparison_report_{mode}_{timestamp}.json"
        
        # Calculate rankings
        successful_results = [r for r in results if r["success"] and r["improvement_ratio"] > 0]
        successful_results.sort(key=lambda x: x["improvement_ratio"], reverse=True)
        
        # Generate summary
        summary = {
            "test_mode": mode,
            "timestamp": timestamp,
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "test_parameters": self.test_params,
            "top_performers": successful_results[:3] if successful_results else [],
            "all_results": results
        }
        
        # Save detailed results
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary report
        self.print_summary_report(summary)
        
        return report_file

    def print_summary_report(self, summary):
        """Print human-readable summary report."""
        print(f"\n{'='*80}")
        print(f"📊 DISTILLATION COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"📅 Timestamp: {summary['timestamp']}")
        print(f"🧪 Total Experiments: {summary['total_experiments']}")
        print(f"✅ Successful: {summary['successful_experiments']}")
        print(f"❌ Failed: {summary['total_experiments'] - summary['successful_experiments']}")
        
        if summary['top_performers']:
            print(f"\n🏆 TOP PERFORMING TEACHER-STUDENT PAIRS:")
            for i, result in enumerate(summary['top_performers'], 1):
                teacher_info = self.teacher_models.get(result['teacher'], {"size": "Unknown"})
                student_info = self.student_models.get(result['student'], {"size": "Unknown"})
                improvement = result['improvement_ratio'] * 100
                
                print(f"\n   {i}. 🥇 {result['teacher']} -> {result['student']}")
                print(f"      📈 Improvement: {improvement:.1f}%")
                print(f"      📏 Compression: {teacher_info['size']} -> {student_info['size']}")
                print(f"      ⏱️  Duration: {result['duration_minutes']:.1f} minutes")
                
                if result.get('distilled_final_loss'):
                    print(f"      📉 Final Loss: {result['distilled_final_loss']:.4f}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if summary['top_performers']:
            best = summary['top_performers'][0]
            print(f"   🎯 Best Overall: {best['teacher']} -> {best['student']}")
            print(f"   📊 Achieved {best['improvement_ratio']*100:.1f}% improvement over baseline")
            
            # Find best by category
            tiny_models = [r for r in summary['top_performers'] 
                          if self.student_models.get(r['student'], {}).get('category') == 'tiny']
            if tiny_models:
                print(f"   🏃 Best for Ultra-Fast Inference: {tiny_models[0]['teacher']} -> {tiny_models[0]['student']}")
                
        print(f"\n📁 Detailed results saved to: {self.results_dir}/")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Distillation Model Comparison Framework")
    parser.add_argument("--mode", choices=["quick", "balanced", "all"], default="balanced",
                       help="Testing mode: quick (3 pairs), balanced (8 pairs), all (16 pairs)")
    parser.add_argument("--custom-pairs", nargs="+", help="Custom teacher,student pairs to test")
    parser.add_argument("--patients", default="570", help="Comma-separated patient IDs")
    parser.add_argument("--dataset", default="ohiot1dm", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs for each phase")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DistillationTester()
    
    # Update parameters if provided
    if args.patients != "570":
        tester.test_params["patients"] = args.patients.split(",")
    if args.dataset != "ohiot1dm":
        tester.test_params["dataset"] = args.dataset
    if args.epochs != 2:
        tester.test_params.update({
            "teacher_epochs": args.epochs,
            "student_epochs": args.epochs,
            "distill_epochs": args.epochs
        })
    
    # Run comparison
    results = tester.run_comparison(mode=args.mode, custom_pairs=args.custom_pairs)
    
    print(f"\n🎉 Comparison complete! Check {tester.results_dir}/ for detailed results.")


if __name__ == "__main__":
    main()