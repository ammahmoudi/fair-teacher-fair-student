#!/usr/bin/env python3
"""
Comprehensive Efficiency Testing Runner
========================================

This script runs efficiency tests for all major model types:
1. Time-LLM models: BERT, GPT2, LLAMA (training + inference)
2. Chronos models: T5-base, T5-tiny (training + inference) 
3. Distillation pipeline: BERT→TinyBERT (training + inference)

For efficiency analysis, we test with:
- One patient (570 for ohiot1dm)
- One seed (831363)
- Standardized data scenario
- Full train+inference cycle to capture both training and inference efficiency

The script automatically:
- Generates configurations using unified config generators
- Runs experiments with efficiency monitoring enabled
- Collects comprehensive efficiency reports (memory, GPU, timing, power)
- Organizes results in structured folders for analysis

Usage:
    python comprehensive_efficiency_runner.py [--dry-run]
    
Examples:
    python comprehensive_efficiency_runner.py --dry-run          # Preview focused experiments (16)
    python comprehensive_efficiency_runner.py                    # Run focused experiments
"""

import os
import sys
import subprocess
import argparse
import time
import json
import pandas as pd
import numpy as np
import glob
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parents[2]))  # Add Fair Teacher–Fair Student root to path
from utils.path_utils import get_project_root, get_scripts_path

# Fixed seeds for consistency
FIXED_SEEDS = [831363, 809906, 427368, 238822, 247659]

class ComprehensiveEfficiencyRunner:
    """Comprehensive efficiency testing for all model types."""
    
    def __init__(self, base_dir=None, dry_run=False):
        """Initialize the efficiency runner."""
        if base_dir is None:
            self.base_dir = get_project_root()
        else:
            self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        
        # Test parameters for efficiency focus
        self.test_patient = "570"  # Single patient for efficiency testing
        self.test_seed = str(FIXED_SEEDS[0])  # Use first seed: 831363
        self.data_scenario = "standardized"  # Clean data for consistent efficiency measurement
        self.dataset = "ohiot1dm"  # Primary dataset
        
        # Model configurations - Using separate modes for cleaner efficiency measurement
        self.models_config = {
            "time_llm": {
                "script": "scripts/time_llm/config_generator.py",
                "models": ["BERT", "GPT2", "LLAMA"],
                "modes": {
                    "train": {"epochs": 10, "mode": "train"},
                    "inference": {"epochs": 0, "mode": "inference"}
                },
                "window_config": "6_9"  # 6 context, 9 prediction
            },
            "chronos": {
                "script": "scripts/chronos/config_generator.py", 
                "models": ["amazon/chronos-t5-base", "amazon/chronos-t5-tiny"],
                "modes": {
                    "train": {"mode": "train"},
                    "inference": {"mode": "inference"}
                },
                "window_config": "6_9"  # 6 context, 9 prediction
            },
            "distillation": {
                "script": "distill_pipeline.sh",
                "pairs": [("bert-base-uncased", "prajjwal1/bert-tiny")],  # BERT → TinyBERT
                "teacher_epochs": 5,
                "student_epochs": 5, 
                "distill_epochs": 5,
                "modes": {
                    "train": {"mode": "train"},
                    "inference": {"mode": "inference"}
                }
            },
            "distillation_inference": {
                "script": "main.py",
                "checkpoint_path": "efficiency_experiments/distillation_experiments/pipeline_runs/*/patient_*/phase_3_distillation/*/logs/*/student_distilled.pth",
                "model": "prajjwal1/bert-tiny",
                "modes": {
                    "inference": {"epochs": 0, "mode": "inference"}
                }
            }
        }
        
        # Output directory for efficiency experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / f"efficiency_experiments_{timestamp}"
        
        # Analysis results directory
        self.analysis_dir = self.base_dir / "efficiency_toolkit" / "results" / "efficiency_analysis_results"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped analysis subdirectory
        self.analysis_timestamp_dir = self.analysis_dir / f"analysis_{timestamp}"
        self.analysis_timestamp_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Comprehensive Efficiency Testing Setup")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Test patient: {self.test_patient}")
        print(f"🎲 Test seed: {self.test_seed}")
        print(f"📈 Data scenario: {self.data_scenario}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🎯 Comprehensive Efficiency Testing Setup")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🔍 Dry run: {self.dry_run}")
        
        # Fixed seeds and focused testing parameters
        self.test_patient = "570"
        self.test_seed = str(FIXED_SEEDS[0])  # Use first seed: 831363
        self.data_scenario = "standardized" 
        self.dataset = "ohiot1dm"
        
    def run_command(self, cmd, description, cwd=None, timeout_hours=6):
        """Run a command with proper error handling and logging for long experiments."""
        if cwd is None:
            cwd = self.base_dir
            
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"📂 Working directory: {cwd}")
        print(f"⚡ Command: {cmd}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("🔍 DRY RUN - Command would be executed here")
            return True
        
        # For long experiments, we need to show live output and have longer timeout
        timeout_seconds = timeout_hours * 3600
        
        try:
            # Run command with live output (no capture_output for long experiments)
            if "run_all" in cmd or "distill_pipeline" in cmd:
                print(f"🏃 Starting long-running experiment (timeout: {timeout_hours}h)...")
                print(f"📺 Live output will be shown below:")
                print("-" * 60)
                
                # Run without capturing output so we can see what's happening
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    timeout=timeout_seconds,
                    text=True
                )
                
                print("-" * 60)
                if result.returncode == 0:
                    print(f"✅ SUCCESS: {description}")
                    return True
                else:
                    print(f"❌ FAILED: {description}")
                    print(f"💥 Error code: {result.returncode}")
                    return False
            else:
                # For quick commands (config generation), capture output
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    cwd=cwd,
                    capture_output=True, 
                    text=True,
                    timeout=300  # 5 minutes for config generation
                )
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS: {description}")
                    if result.stdout.strip():
                        print(f"📝 Output:\n{result.stdout}")
                    return True
                else:
                    print(f"❌ FAILED: {description}")
                    print(f"💥 Error code: {result.returncode}")
                    if result.stderr.strip():
                        print(f"🚨 Error output:\n{result.stderr}")
                    if result.stdout.strip():
                        print(f"📝 Standard output:\n{result.stdout}")
                    return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {description} exceeded {timeout_hours} hour limit")
            return False
        except Exception as e:
            print(f"💥 EXCEPTION: {description} failed with: {str(e)}")
            return False
    

    
    def run_generated_experiments(self, experiment_type):
        """Run the generated experiment configurations using proper experiment runners."""
        print(f"\n🏃 RUNNING {experiment_type.upper()} EFFICIENCY EXPERIMENTS")
        print(f"{'='*80}")
        
        results = []
        
        if experiment_type == "time_llm":
            # Use the Time-LLM experiment runner with specific filters for our efficiency configs
            cmd = [
                "python", "scripts/time_llm/run_experiments.py",
                "--modes", "train,inference", 
                "--datasets", self.dataset,
                "--models", "BERT,GPT2,LLAMA",
                "--log_level", "INFO",
                # Force a fresh run by removing any resume file
                "--resume_file", f"time_llm_efficiency_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ]
            
            cmd_str = " ".join(cmd)
            print(f"🚀 Time-LLM Command: {cmd_str}")
            
            # Important: Remove any existing resume files to force fresh execution
            resume_files = list(self.base_dir.glob("time_llm_experiments_progress.json"))
            for resume_file in resume_files:
                print(f"🗑️ Removing old resume file: {resume_file}")
                resume_file.unlink(missing_ok=True)
            
            # Wrap with efficiency monitoring
            efficiency_cmd = [
                "python", "efficiency/real_time_profiler.py",
                "--output_file", f"time_llm_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "--"
            ] + cmd
            
            efficiency_cmd_str = " ".join(efficiency_cmd)
            description = f"Run Time-LLM experiments with efficiency monitoring (BERT, GPT2, LLAMA)"
            success = self.run_command(efficiency_cmd_str, description)
            results.append(("time_llm_all_models", success))
            
        elif experiment_type == "chronos":
            # Use the Chronos experiment runner with specific filters for our efficiency configs
            cmd = [
                "python", "scripts/chronos/run_experiments.py", 
                "--modes", "training,inference",
                "--datasets", self.dataset,
                "--log_level", "INFO",
                # Force a fresh run by removing any resume file
                "--resume_file", f"chronos_efficiency_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ]
            
            cmd_str = " ".join(cmd)
            print(f"⏰ Chronos Command: {cmd_str}")
            
            # Important: Remove any existing resume files to force fresh execution
            resume_files = list(self.base_dir.glob("chronos_experiments_progress.json"))
            for resume_file in resume_files:
                print(f"🗑️ Removing old resume file: {resume_file}")
                resume_file.unlink(missing_ok=True)
            
            # Wrap with efficiency monitoring
            efficiency_cmd = [
                "python", "efficiency/real_time_profiler.py",
                "--output_file", f"chronos_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "--"
            ] + cmd
            
            efficiency_cmd_str = " ".join(efficiency_cmd)
            description = f"Run Chronos experiments with efficiency monitoring (T5-base, T5-tiny)"
            success = self.run_command(efficiency_cmd_str, description)
            results.append(("chronos_all_models", success))
        
        return results
    
    def collect_efficiency_reports(self):
        """Collect and summarize efficiency reports from all experiments."""
        print(f"\n📊 COLLECTING EFFICIENCY REPORTS")
        print(f"{'='*80}")
        
        # Find all efficiency reports
        report_patterns = [
            "**/efficiency_report_*.json",
            "**/real_performance_report_*.json", 
            "**/comprehensive_performance_report_*.json"
        ]
        
        all_reports = []
        for pattern in report_patterns:
            reports = list(self.base_dir.glob(pattern))
            all_reports.extend(reports)
        
        print(f"📋 Found {len(all_reports)} efficiency reports")
        
        # Group reports by experiment type
        report_groups = {
            "time_llm": [],
            "chronos": [],
            "distillation": []
        }
        
        for report in all_reports:
            report_path = str(report)
            if "time_llm" in report_path:
                report_groups["time_llm"].append(report)
            elif "chronos" in report_path:
                report_groups["chronos"].append(report)
            elif "distillation" in report_path:
                report_groups["distillation"].append(report)
        
        # Create summary
        summary_file = self.output_dir / "efficiency_summary.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE EFFICIENCY TESTING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Configuration:\n")
            f.write(f"- Patient: {self.test_patient}\n")
            f.write(f"- Seed: {self.test_seed}\n")
            f.write(f"- Data Scenario: {self.data_scenario}\n")
            f.write(f"- Dataset: {self.dataset}\n\n")
            
            for exp_type, reports in report_groups.items():
                f.write(f"{exp_type.upper()} EFFICIENCY REPORTS:\n")
                f.write("-" * 40 + "\n")
                for report in reports:
                    rel_path = report.relative_to(self.base_dir)
                    f.write(f"  {rel_path}\n")
                f.write(f"Total: {len(reports)} reports\n\n")
        
        print(f"📝 Efficiency summary saved to: {summary_file}")
        return all_reports
    

    
    def find_focused_configs(self, model_types=None):
        """Find specific configs for focused efficiency testing."""
        if model_types is None:
            model_types = ['time_llm', 'chronos']
            
        configs = []
        
        # Time-LLM configs (training and inference)
        if 'time_llm' in model_types:
            time_llm_patterns = [
                f"efficiency_experiments/experiments/time_llm_training_{self.dataset}/seed_{self.test_seed}_model_*_epochs_10/patient_{self.test_patient}/config.gin",
                f"efficiency_experiments/experiments/time_llm_inference_{self.dataset}/seed_{self.test_seed}_model_*_epochs_0/patient_{self.test_patient}/config.gin"
            ]
            
            for pattern in time_llm_patterns:
                matches = list(self.base_dir.glob(pattern))
                for config_file in matches:
                    if config_file.exists():
                        configs.append({
                            'type': 'time_llm',
                            'path': config_file,
                            'relative_path': config_file.relative_to(self.base_dir)
                        })
        
        # Chronos configs (training and inference)
        if 'chronos' in model_types:
            chronos_patterns = [
                f"efficiency_experiments/experiments/chronos_training_{self.dataset}/seed_{self.test_seed}_model_*_mode_train_*/patient_{self.test_patient}/config.gin",
                f"efficiency_experiments/experiments/chronos_inference_{self.dataset}/seed_{self.test_seed}_model_*_mode_inference_*/patient_{self.test_patient}/config.gin"
            ]
            
            for pattern in chronos_patterns:
                matches = list(self.base_dir.glob(pattern))
                for config_file in matches:
                    if config_file.exists():
                        configs.append({
                            'type': 'chronos',
                            'path': config_file,
                            'relative_path': config_file.relative_to(self.base_dir)
                        })
        
        # Distillation configs
        if 'distillation' in model_types:
            distillation_patterns = [
                f"efficiency_experiments/experiments/distillation_{self.dataset}/seed_{self.test_seed}_teacher_*_student_*/patient_{self.test_patient}/config.gin"
            ]
            
            for pattern in distillation_patterns:
                matches = list(self.base_dir.glob(pattern))
                for config_file in matches:
                    if config_file.exists():
                        configs.append({
                            'type': 'distillation',
                            'path': config_file,
                            'relative_path': config_file.relative_to(self.base_dir)
                        })
        
        # Distillation inference configs  
        if 'distillation_inference' in model_types:
            distillation_inf_patterns = [
                f"efficiency_experiments/experiments/distillation_inference_{self.dataset}/seed_{self.test_seed}_model_tinybert/patient_{self.test_patient}/config.gin"
            ]
            
            for pattern in distillation_inf_patterns:
                matches = list(self.base_dir.glob(pattern))
                for config_file in matches:
                    if config_file.exists():
                        configs.append({
                            'type': 'distillation_inference',
                            'path': config_file,
                            'relative_path': config_file.relative_to(self.base_dir)
                        })
        
        print(f"🔍 Found {len(configs)} configs:")
        for config in configs:
            print(f"  {config['type']}: {config['relative_path']}")
        
        return configs
    
    def generate_all_configs(self, model_types=None):
        """Generate all configs using the appropriate generators."""
        if model_types is None:
            model_types = ['time_llm', 'chronos']
            
        print(f"🔧 Generating fresh configs for: {', '.join(model_types)}...")
        
        # Time-LLM config generation
        if 'time_llm' in model_types:
            venv_python = str(self.base_dir / "venv" / "bin" / "python")
            time_llm_script = str(get_scripts_path("time_llm", "config_generator.py"))
        
            # Generate training configs
            time_llm_train_cmd = [
                venv_python, time_llm_script,
                "--mode", "train",
                "--patients", self.test_patient,
                "--llm_models", "BERT,GPT2,LLAMA", 
                "--seeds", self.test_seed,
                "--epochs", "1",  # Reduced from 10 to 1 for faster efficiency testing
                "--dataset", self.dataset
            ]
            
            # Generate inference configs
            time_llm_inference_cmd = [
                venv_python, time_llm_script,
                "--mode", "inference",
                "--patients", self.test_patient,
                "--llm_models", "BERT,GPT2,LLAMA", 
                "--seeds", self.test_seed,
                "--epochs", "0",
                "--dataset", self.dataset
            ]
            
            print(f"🤖 Running Time-LLM training config generation...")
            result = subprocess.run(time_llm_train_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"❌ Time-LLM training config generation failed: {result.stderr}")
                return False
            
            print(f"🤖 Running Time-LLM inference config generation...")
            result = subprocess.run(time_llm_inference_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"❌ Time-LLM inference config generation failed: {result.stderr}")
                return False
            
            print("✅ Time-LLM configs generated successfully")
        
        # Chronos config generation  
        if 'chronos' in model_types:
            chronos_script = str(get_scripts_path("chronos", "config_generator.py"))
            
            # Generate training configs
            chronos_train_cmd = [
                venv_python, chronos_script,
                "--mode", "train",
                "--patients", self.test_patient, 
                "--models", "amazon/chronos-t5-base,amazon/chronos-t5-tiny",
                "--seeds", self.test_seed,
                "--dataset", self.dataset
            ]
            
            # Generate inference configs
            chronos_inference_cmd = [
                venv_python, chronos_script,
                "--mode", "inference",
                "--patients", self.test_patient, 
                "--models", "amazon/chronos-t5-base,amazon/chronos-t5-tiny",
                "--seeds", self.test_seed,
                "--dataset", self.dataset
            ]
            
            print(f"⏰ Running Chronos training config generation...")
            result = subprocess.run(chronos_train_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"❌ Chronos training config generation failed: {result.stderr}")
                return False
                
            print(f"⏰ Running Chronos inference config generation...")
            result = subprocess.run(chronos_inference_cmd, capture_output=True, text=True, cwd=self.base_dir)
            if result.returncode != 0:
                print(f"❌ Chronos inference config generation failed: {result.stderr}")
                return False
                
            print("✅ Chronos configs generated successfully")
        
        # Distillation config generation
        if 'distillation' in model_types:
            venv_python = str(self.base_dir / "venv" / "bin" / "python")
            
            # Use the existing distillation config as a template
            distill_config_template = self.base_dir / "configs" / "distillation" / "config_distill_bert_to_tinybert_570.gin"
            
            if distill_config_template.exists():
                # Copy the existing config to experiments directory for execution
                distill_experiment_dir = self.base_dir / "efficiency_experiments" / "experiments" / "distillation_ohiot1dm" / f"seed_{self.test_seed}_teacher_bert_student_tinybert"
                distill_experiment_dir.mkdir(parents=True, exist_ok=True)
                
                patient_dir = distill_experiment_dir / f"patient_{self.test_patient}"
                patient_dir.mkdir(parents=True, exist_ok=True)
                
                import shutil
                target_config = patient_dir / "config.gin"
                shutil.copy2(distill_config_template, target_config)
                
                print("✅ Distillation config prepared successfully")
            else:
                # Generate a simple distillation config
                print("🔧 Generating distillation config...")
                
                distill_config_content = f"""
# Distillation Configuration for Efficiency Testing
# Teacher: BERT, Student: TinyBERT
# Dataset: {self.dataset}, Patient: {self.test_patient}, Seed: {self.test_seed}

# Data configuration
DataLoader.dataset_name = '{self.dataset}'
DataLoader.patient_ids = [{self.test_patient}]
DataLoader.seed = {self.test_seed}
DataLoader.data_scenario = '{self.data_scenario}'

# Model configuration
DistillationTrainer.teacher_model = 'bert-base-uncased'
DistillationTrainer.student_model = 'prajjwal1/bert-tiny'
DistillationTrainer.epochs = 5
DistillationTrainer.batch_size = 16
DistillationTrainer.learning_rate = 5e-5

# Efficiency monitoring
DistillationTrainer.enable_efficiency_monitoring = True
DistillationTrainer.log_interval = 10
"""
                
                distill_experiment_dir = self.base_dir / "efficiency_experiments" / "experiments" / "distillation_ohiot1dm" / f"seed_{self.test_seed}_teacher_bert_student_tinybert"
                distill_experiment_dir.mkdir(parents=True, exist_ok=True)
                
                patient_dir = distill_experiment_dir / f"patient_{self.test_patient}"
                patient_dir.mkdir(parents=True, exist_ok=True)
                
                target_config = patient_dir / "config.gin"
                with open(target_config, 'w') as f:
                    f.write(distill_config_content.strip())
                
                print("✅ Distillation config generated successfully")
        
        # Distillation inference config generation
        if 'distillation_inference' in model_types:
            print("🔧 Generating distillation inference config...")
            
            # Find the distilled model checkpoint
            import glob
            checkpoint_pattern = str(self.base_dir / "efficiency_experiments" / "distillation_experiments" / "pipeline_runs" / "*" / f"patient_{self.test_patient}" / "phase_3_distillation" / "*" / "logs" / "*" / "student_distilled.pth")
            checkpoints = glob.glob(checkpoint_pattern)
            
            if checkpoints:
                checkpoint_path = checkpoints[0]  # Use the first found checkpoint
                print(f"📁 Found distilled checkpoint: {checkpoint_path}")
                
                # Create experiment directory
                distill_inf_experiment_dir = self.base_dir / "efficiency_experiments" / "experiments" / "distillation_inference_ohiot1dm" / f"seed_{self.test_seed}_model_tinybert"
                distill_inf_experiment_dir.mkdir(parents=True, exist_ok=True)
                
                patient_dir = distill_inf_experiment_dir / f"patient_{self.test_patient}"
                patient_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate inference config for distilled model (using TinyBERT as proxy)
                distill_inf_config_content = f"""# Parameters for run:
# ==============================================================================
run.chronos_dir = '.'
run.data_settings = \\
    {{'frequency': '5min',
     'input_features': ['target'],
     'labels': ['target'],
     'path_to_test_data': './data/{self.dataset}/raw_standardized/{self.test_patient}-ws-testing.csv',
     'path_to_train_data': './data/{self.dataset}/raw_standardized/{self.test_patient}-ws-training.csv',
     'percent': 100,
     'preprocess_input_features': False,
     'preprocess_label': False,
     'preprocessing_method': 'min_max',
     'prompt_path': './data/{self.dataset}/raw_standardized/t1dm_prompt.txt',
     'val_split': 0}}

run.llm_settings = \\
    {{'activation': 'gelu',
     'c_out': 1,
     'context_length': 6,
     'd_ff': 32,
     'd_layers': 1,
     'd_model': 32,
     'dec_in': 1,
     'des': 'tinybert_{self.test_patient}_inference',
     'dropout': 0.1,
     'e_layers': 2,
     'embed': 'timeF',
     'enc_in': 1,
     'eval_metrics': ['rmse', 'mae', 'mape'],
     'factor': 1,
     'features': 'S',
     'learning_rate': 0.001,
     'llm_dim': 312,
     'llm_layers': 4,
     'llm_model': 'TinyBERT',
     'lradj': 'COS',
     'method': 'time_llm',
     'mode': 'training+inference',
     'model_comment': 'tinybert_{self.test_patient}_inference',
     'model_id': 'tinybert_inference',
     'moving_avg': 25,
     'n_heads': 4,
     'num_workers': 1,
     'patch_len': 6,
     'patience': 10,
     'prediction_batch_size': 32,
     'prediction_length': 6,
     'prompt_domain': 0,
     'restore_from_checkpoint': False,
     'seed': {self.test_seed},
     'sequence_length': 6,
     'stride': 6,
     'task_name': 'long_term_forecast',
     'timeenc': 0,
     'torch_dtype': 'bfloat16',
     'train_batch_size': 32,
     'train_epochs': 0}}
run.log_dir = \\
    './efficiency_experiments/experiments/distillation_inference_{self.dataset}/seed_{self.test_seed}_model_tinybert/patient_{self.test_patient}/logs'
"""
                
                target_config = patient_dir / "config.gin"
                with open(target_config, 'w') as f:
                    f.write(distill_inf_config_content.strip())
                
                print("✅ Distillation inference config generated successfully")
            else:
                print(f"⚠️ No distilled checkpoint found at pattern: {checkpoint_pattern}")
                print("🔍 Make sure to run distillation training first")
        
        return True

    def analyze_all_experiments(self, save_results=True):
        """Comprehensive analysis of all completed experiments."""
        print("\n🔍 COMPREHENSIVE EFFICIENCY ANALYSIS")
        print("=" * 80)
        
        # Find all completed experiments
        experiments = self._find_completed_experiments()
        
        if not any(experiments.values()):
            print("❌ No completed experiments found!")
            return None
            
        # Load and parse all reports
        all_results = []
        for category, report_files in experiments.items():
            if report_files:
                print(f"\n📊 Processing {category.replace('_', ' ').title()}...")
                
                for report_file in report_files:
                    result = self._load_and_parse_report(report_file)
                    if result:
                        result['experiment_type'] = category
                        all_results.append(result)
        
        if not all_results:
            print("❌ No valid results extracted!")
            return None
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_results)
        
        print(f"\n✅ Successfully processed {len(df)} experiment results")
        print("=" * 80)
        
        # Perform comprehensive analysis
        analysis_results = self._perform_comprehensive_analysis(df)
        
        if save_results:
            self._save_analysis_results(df, analysis_results)
            
        return df, analysis_results

    def _find_completed_experiments(self):
        """Find all completed experiments with comprehensive performance reports."""
        print("🔍 Scanning for completed experiments...")
        
        experiments = {
            'time_llm_training': [],
            'time_llm_inference': [],
            'chronos_training': [],
            'chronos_inference': [],
            'distillation': [],
            'distillation_inference': []
        }
        
        # Find comprehensive performance reports
        base_pattern = "efficiency_experiments/experiments/**/comprehensive_performance_report*.json"
        report_files = glob.glob(str(self.base_dir / base_pattern), recursive=True)
        
        # Also find distillation results
        distillation_pattern = "efficiency_experiments/distillation_experiments/**/comprehensive_performance_report*.json"
        distillation_files = glob.glob(str(self.base_dir / distillation_pattern), recursive=True)
        report_files.extend(distillation_files)
        
        for report_file in report_files:
            try:
                # Parse experiment type from path
                path_parts = Path(report_file).parts
                # Find experiment directory by looking for directories containing model type names
                experiment_dir = "unknown"
                for part in path_parts:
                    if any(exp_type in part for exp_type in ['time_llm_inference', 'time_llm_training', 'chronos_inference', 'chronos_training', 'distillation_inference', 'distillation']):
                        experiment_dir = part
                        break
                
                # Determine experiment category
                if 'time_llm_training' in experiment_dir:
                    category = 'time_llm_training'
                elif 'time_llm_inference' in experiment_dir:
                    category = 'time_llm_inference'
                elif 'chronos_training' in experiment_dir:
                    category = 'chronos_training'
                elif 'chronos_inference' in experiment_dir:
                    category = 'chronos_inference'
                elif 'distillation_inference' in experiment_dir:
                    category = 'distillation_inference'
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

    def _load_and_parse_report(self, report_file):
        """Load and parse a comprehensive performance report."""
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            # Extract metadata from path and filename
            metadata = self._extract_experiment_metadata(report_file)
            
            # Build result dictionary
            result = {
                'report_file': report_file,
                'timestamp': data.get('timestamp', 'unknown'),
                'model_name': metadata.get('model', 'unknown'),
                'seed': metadata.get('seed', 'unknown'),
            }
            
            # Extract performance metrics from the performance_summary section
            performance_summary = data.get('performance_summary', {})
            
            # Parse different modes
            for mode in ['training', 'inference', 'training_inference']:
                mode_data = performance_summary.get(mode, {})
                if mode_data:
                    # Basic metrics
                    result[f'{mode}_latency_ms'] = mode_data.get('average_latency_ms')
                    result[f'{mode}_inferences'] = mode_data.get('total_inferences')
                    
                    # Memory metrics
                    result[f'{mode}_peak_ram_mb'] = mode_data.get('process_peak_ram_mb')
                    result[f'{mode}_avg_ram_mb'] = mode_data.get('process_average_ram_mb')
                    
                    # GPU metrics
                    result[f'{mode}_peak_gpu_mb'] = mode_data.get('peak_gpu_allocated_mb')
                    result[f'{mode}_avg_gpu_mb'] = mode_data.get('average_gpu_allocated_mb')
                    
                    # Performance metrics
                    result[f'{mode}_peak_gpu_util_%'] = mode_data.get('peak_gpu_utilization_percent')
                    result[f'{mode}_avg_gpu_util_%'] = mode_data.get('average_gpu_utilization_percent')
                    result[f'{mode}_peak_temp_c'] = mode_data.get('peak_temperature_celsius')
                    result[f'{mode}_peak_power_w'] = mode_data.get('peak_power_usage_watts')
                    result[f'{mode}_avg_power_w'] = mode_data.get('average_power_usage_watts')
                    
                    # Model metrics
                    result[f'{mode}_model_size_mb'] = mode_data.get('model_size_on_disk_mb')
                    result[f'{mode}_parameters_count'] = mode_data.get('parameters_count')
            
            # Extract context and prediction lengths
            result['context_length'] = metadata.get('context_length')
            result['prediction_length'] = metadata.get('prediction_length')
            result['epochs'] = metadata.get('epochs', 0)
            
            # Simplify for main metrics
            result['peak_ram_mb'] = result.get('training_peak_ram_mb') or result.get('inference_peak_ram_mb') or result.get('training_inference_peak_ram_mb')
            result['peak_gpu_mb'] = result.get('training_peak_gpu_mb') or result.get('inference_peak_gpu_mb') or result.get('training_inference_peak_gpu_mb')
            result['training_latency_ms'] = result.get('training_latency_ms')
            result['inference_latency_ms'] = result.get('inference_latency_ms') or result.get('training_inference_latency_ms')
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error loading {report_file}: {e}")
            return None

    def _extract_experiment_metadata(self, report_file):
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
            elif 'distillation_inference' in str(report_file):
                metadata['model'] = 'distilled-bert-tiny'
            else:
                metadata['model'] = 'distillation-bert'
            
            # Extract patient ID from path
            for part in path_parts:
                if part.startswith('patient_'):
                    metadata['seed'] = part.replace('patient_', '')
                    break
            
            return metadata
        
        # Find the experiment folder containing seed and model info
        experiment_folder = None
        for part in path_parts:
            if 'seed_' in part and 'model_' in part:
                experiment_folder = part
                break
        
        if not experiment_folder:
            return metadata
        
        # Extract seed
        if 'seed_' in experiment_folder:
            try:
                parts = experiment_folder.split('_')
                seed_idx = next(i for i, p in enumerate(parts) if p == 'seed') + 1
                if seed_idx < len(parts):
                    metadata['seed'] = parts[seed_idx]
            except (StopIteration, ValueError, IndexError):
                metadata['seed'] = 'unknown'
        
        # Extract model name with improved logic
        if 'model_' in experiment_folder:
            try:
                parts = experiment_folder.split('_')
                model_idx = next(i for i, p in enumerate(parts) if p == 'model') + 1
                if model_idx < len(parts):
                    model_name = parts[model_idx]
                    
                    # Handle Chronos models
                    if 'amazon-chronos-t5-base' in experiment_folder:
                        metadata['model'] = 'chronos-t5-base'
                    elif 'amazon-chronos-t5-tiny' in experiment_folder:
                        metadata['model'] = 'chronos-t5-tiny'
                    # Handle Time-LLM models
                    elif model_name in ['BERT', 'GPT2', 'LLAMA']:
                        metadata['model'] = model_name
                    elif 'TinyBERT' in model_name or 'tiny' in model_name.lower():
                        metadata['model'] = 'TinyBERT'
                    else:
                        metadata['model'] = model_name
            except (StopIteration, ValueError, IndexError):
                metadata['model'] = 'unknown'
        
        # Extract context and prediction lengths
        if 'context_' in experiment_folder and 'pred_' in experiment_folder:
            try:
                parts = experiment_folder.split('_')
                context_idx = next(i for i, p in enumerate(parts) if p == 'context') + 1
                pred_idx = next(i for i, p in enumerate(parts) if p == 'pred') + 1
                metadata['context_length'] = int(parts[context_idx])
                metadata['prediction_length'] = int(parts[pred_idx])
            except (StopIteration, ValueError, IndexError):
                pass
        
        # Extract epochs for training experiments
        if 'epochs_' in experiment_folder:
            try:
                parts = experiment_folder.split('_')
                epochs_idx = next(i for i, p in enumerate(parts) if p == 'epochs') + 1
                metadata['epochs'] = int(parts[epochs_idx])
            except (StopIteration, ValueError, IndexError):
                metadata['epochs'] = 0
        
        return metadata

    def _perform_comprehensive_analysis(self, df):
        """Perform comprehensive efficiency analysis."""
        analysis_results = {}
        
        print("\n📈 RESULTS BY MODEL TYPE")
        print("-" * 50)
        
        # Analysis by model type
        time_llm_results = df[df['experiment_type'].str.contains('time_llm')]
        chronos_results = df[df['experiment_type'].str.contains('chronos')]
        distillation_results = df[df['experiment_type'].str.contains('distillation')]
        
        if not time_llm_results.empty:
            print("\n🤖 TIME-LLM MODELS:")
            time_llm_summary = self._create_model_summary(time_llm_results)
            print(time_llm_summary.to_string())
            analysis_results['time_llm_summary'] = time_llm_summary
        
        if not chronos_results.empty:
            print("\n⏰ CHRONOS MODELS:")
            chronos_summary = self._create_model_summary(chronos_results)
            print(chronos_summary.to_string())
            analysis_results['chronos_summary'] = chronos_summary
        
        if not distillation_results.empty:
            print("\n🧠 DISTILLATION MODELS:")
            distillation_summary = self._create_model_summary(distillation_results)
            print(distillation_summary.to_string())
            analysis_results['distillation_summary'] = distillation_summary
        
        # Training vs Inference Analysis
        print(f"\n⚖️ TRAINING VS INFERENCE COMPARISON")
        print("-" * 50)
        
        comparison_results = self._create_training_inference_comparison(df)
        if not comparison_results.empty:
            print(comparison_results.to_string(index=False))
            analysis_results['training_inference_comparison'] = comparison_results
        
        # Efficiency Rankings
        print(f"\n🏆 MODEL EFFICIENCY RANKING")
        print("-" * 50)
        
        efficiency_ranking = self._create_efficiency_ranking(df)
        if efficiency_ranking is not None:
            print(efficiency_ranking.to_string())
            analysis_results['efficiency_ranking'] = efficiency_ranking
        else:
            print("No efficiency metrics available for ranking")
        
        return analysis_results

    def _create_model_summary(self, results_df):
        """Create summary statistics for a model type."""
        # Only aggregate columns that exist
        agg_cols = {}
        for col in ['training_latency_ms', 'inference_latency_ms', 'peak_ram_mb', 'peak_gpu_mb', 'rmse', 'mae']:
            if col in results_df.columns:
                agg_cols[col] = 'mean'
        
        if agg_cols:
            summary = results_df.groupby(['model_name', 'experiment_type']).agg(agg_cols).round(2)
            return summary
        else:
            return pd.DataFrame()

    def _create_training_inference_comparison(self, df):
        """Create training vs inference comparison."""
        training_results = df[df['experiment_type'].str.contains('training')]
        inference_results = df[df['experiment_type'].str.contains('inference')]
        
        if training_results.empty or inference_results.empty:
            return pd.DataFrame()
        
        # Group by model for comparison
        comparison_data = []
        
        for model in training_results['model_name'].unique():
            train_data = training_results[training_results['model_name'] == model]
            infer_data = inference_results[inference_results['model_name'] == model]
            
            if not train_data.empty and not infer_data.empty:
                row = {
                    'Model': model,
                    'Train_Time_ms': train_data['training_latency_ms'].mean(),
                    'Inference_Time_ms': infer_data['inference_latency_ms'].mean(),
                    'Train_Peak_RAM_MB': train_data['peak_ram_mb'].mean(),
                    'Inference_Peak_RAM_MB': infer_data['peak_ram_mb'].mean()
                }
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).round(2)

    def _create_efficiency_ranking(self, df):
        """Create efficiency ranking based on multiple metrics."""
        # Calculate efficiency metrics
        efficiency_cols = ['training_efficiency', 'memory_efficiency', 'inference_efficiency']
        
        # Check if we can calculate efficiency metrics
        if 'training_latency_ms' in df.columns and 'peak_ram_mb' in df.columns:
            # Simple efficiency metrics (lower is better, so invert)
            df_efficiency = df.copy()
            
            # Convert to numeric, handling any non-numeric values
            numeric_cols = ['training_latency_ms', 'inference_latency_ms', 'peak_ram_mb', 'peak_gpu_mb']
            for col in numeric_cols:
                if col in df_efficiency.columns:
                    df_efficiency[col] = pd.to_numeric(df_efficiency[col], errors='coerce')
            
            # Training efficiency (inversely related to time and memory)
            if 'training_latency_ms' in df_efficiency.columns:
                df_efficiency['training_efficiency'] = 1000000 / (df_efficiency['training_latency_ms'].fillna(float('inf')) + df_efficiency['peak_ram_mb'].fillna(0))
            
            # Memory efficiency (inversely related to memory usage)
            if 'peak_ram_mb' in df_efficiency.columns:
                df_efficiency['memory_efficiency'] = 10000 / df_efficiency['peak_ram_mb'].fillna(float('inf'))
            
            # Inference efficiency (inversely related to inference time)
            if 'inference_latency_ms' in df_efficiency.columns:
                df_efficiency['inference_efficiency'] = 10000 / df_efficiency['inference_latency_ms'].fillna(float('inf'))
            
            # Filter out rows with missing efficiency metrics
            efficiency_results = df_efficiency.dropna(subset=[col for col in efficiency_cols if col in df_efficiency.columns], how='all')
            
            if not efficiency_results.empty:
                # Only aggregate columns that exist
                agg_cols = {}
                for col in ['training_efficiency', 'memory_efficiency', 'inference_efficiency', 'peak_ram_mb', 'training_latency_ms']:
                    if col in efficiency_results.columns:
                        agg_cols[col] = 'mean'
                
                if agg_cols:
                    ranking = efficiency_results.groupby('model_name').agg(agg_cols).round(4)
                    
                    # Sort by the first available efficiency metric
                    sort_col = next((col for col in ['training_efficiency', 'memory_efficiency', 'inference_efficiency'] if col in ranking.columns), None)
                    if sort_col:
                        ranking = ranking.sort_values(sort_col, ascending=False)
                    
                    return ranking
        
        return None

    def _save_analysis_results(self, df, analysis_results):
        """Save comprehensive analysis results to organized folder structure."""
        print(f"\n💾 SAVING ANALYSIS RESULTS")
        print("-" * 50)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized folder structure
        folders = {
            'raw_data': self.analysis_timestamp_dir / 'raw_data',
            'summaries': self.analysis_timestamp_dir / 'summaries', 
            'comparisons': self.analysis_timestamp_dir / 'comparisons',
            'rankings': self.analysis_timestamp_dir / 'rankings',
            'reports': self.analysis_timestamp_dir / 'reports'
        }
        
        for folder in folders.values():
            folder.mkdir(exist_ok=True)
        
        # Save raw data
        raw_data_file = folders['raw_data'] / f"all_experiments_raw_{timestamp}.csv"
        df.to_csv(raw_data_file, index=False)
        print(f"💾 Raw data saved: {raw_data_file.name}")
        
        # Save individual summaries
        for summary_name, summary_df in analysis_results.items():
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                summary_file = folders['summaries'] / f"{summary_name}_{timestamp}.csv"
                summary_df.to_csv(summary_file)
                print(f"💾 {summary_name.replace('_', ' ').title()} saved: {summary_file.name}")
        
        # Create comprehensive markdown report
        self._create_markdown_report(df, analysis_results, folders['reports'], timestamp)
        
        # Create a summary index file
        self._create_summary_index(folders, timestamp)
        
        print(f"\n📁 All results saved to: {self.analysis_timestamp_dir}")
        print(f"🎉 Analysis Complete!")
        print("=" * 80)
        
        return self.analysis_timestamp_dir

    def _create_markdown_report(self, df, analysis_results, reports_folder, timestamp):
        """Create a comprehensive markdown report."""
        report_file = reports_folder / f"comprehensive_efficiency_report_{timestamp}.md"
        
        total_experiments = len(df)
        model_types = df['experiment_type'].value_counts()
        
        report_content = f"""# Comprehensive Efficiency Analysis Report
*Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}*

## Executive Summary

**📊 Total Experiments Analyzed:** {total_experiments}

**Experiment Breakdown:**
"""
        
        for exp_type, count in model_types.items():
            report_content += f"- {exp_type.replace('_', ' ').title()}: {count} experiments\n"
        
        report_content += "\n---\n\n## Model Performance Analysis\n\n"
        
        # Add model summaries
        for summary_name, summary_df in analysis_results.items():
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                report_content += f"### {summary_name.replace('_', ' ').title()}\n\n"
                report_content += summary_df.to_markdown() + "\n\n"
        
        # Add efficiency insights
        report_content += """## Key Insights

### Performance Rankings

**Fastest Training:**
"""
        
        # Convert numeric columns and find fastest training models
        if 'training_latency_ms' in df.columns:
            df['training_latency_ms'] = pd.to_numeric(df['training_latency_ms'], errors='coerce')
            training_models = df[df['training_latency_ms'].notna()].nsmallest(3, 'training_latency_ms')
            for i, (_, row) in enumerate(training_models.iterrows(), 1):
                report_content += f"{i}. **{row['model_name']}**: {row['training_latency_ms']:.0f}ms\n"
        else:
            report_content += "No training latency data available.\n"
        
        report_content += "\n**Fastest Inference:**\n"
        
        # Find fastest inference models
        if 'inference_latency_ms' in df.columns:
            df['inference_latency_ms'] = pd.to_numeric(df['inference_latency_ms'], errors='coerce')
            inference_models = df[df['inference_latency_ms'].notna()].nsmallest(3, 'inference_latency_ms')
            for i, (_, row) in enumerate(inference_models.iterrows(), 1):
                report_content += f"{i}. **{row['model_name']}**: {row['inference_latency_ms']:.0f}ms\n"
        else:
            report_content += "No inference latency data available.\n"
        
        report_content += "\n**Most Memory Efficient:**\n"
        
        # Find most memory efficient models
        if 'peak_ram_mb' in df.columns:
            df['peak_ram_mb'] = pd.to_numeric(df['peak_ram_mb'], errors='coerce')
            memory_efficient = df[df['peak_ram_mb'].notna()].nsmallest(3, 'peak_ram_mb')
            for i, (_, row) in enumerate(memory_efficient.iterrows(), 1):
                report_content += f"{i}. **{row['model_name']}**: {row['peak_ram_mb']:.0f}MB\n"
        else:
            report_content += "No memory usage data available.\n"
        
        report_content += f"\n---\n\n*This analysis provides comprehensive insights for model selection based on specific deployment requirements and resource constraints.*\n"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"💾 Markdown report saved: {report_file.name}")

    def _create_summary_index(self, folders, timestamp):
        """Create an index file listing all generated files."""
        index_file = self.analysis_timestamp_dir / f"analysis_index_{timestamp}.txt"
        
        index_content = f"""Efficiency Analysis Results Index
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Folder Structure:
================
"""
        
        for folder_name, folder_path in folders.items():
            index_content += f"\n{folder_name.upper()}:\n"
            if folder_path.exists():
                files = list(folder_path.glob("*"))
                for file in sorted(files):
                    index_content += f"  - {file.name}\n"
            else:
                index_content += "  (empty)\n"
        
        index_content += f"\nTotal Files Generated: {sum(len(list(folder.glob('*'))) for folder in folders.values() if folder.exists())}\n"
        
        with open(index_file, 'w') as f:
            f.write(index_content)
        
        print(f"💾 Analysis index saved: {index_file.name}")

    def run_single_experiment(self, config_path, experiment_type):
        """Run a single experiment with efficiency monitoring."""
        print(f"\n{'='*60}")
        print(f"🚀 Running {experiment_type.upper()} Efficiency Experiment")
        print(f"📄 Config: {config_path.relative_to(self.base_dir)}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("🔍 DRY RUN - Would execute experiment here")
            return True
        
        # Choose the appropriate runner based on experiment type
        import random
        
        if experiment_type == 'distillation':
            # Use distillation pipeline for distillation experiments
            cmd = [
                "./distill_pipeline.sh",
                "--teacher", "bert-base-uncased",
                "--student", "prajjwal1/bert-tiny", 
                "--patients", self.test_patient,
                "--dataset", self.dataset,
                "--seed", self.test_seed,
                "--teacher-epochs", "5",
                "--student-epochs", "5",
                "--distill-epochs", "5"
            ]
        elif experiment_type == 'distillation_inference':
            # Use main.py for distillation inference experiments
            import random
            master_port = random.randint(20000, 30000)
            cmd = [
                "./run_main.sh",
                "--config_path", str(config_path),
                "--log_level", "INFO", 
                "--remove_checkpoints", "True",
                "--master_port", str(master_port)
            ]
        else:
            # Use run_main.sh like the working Time-LLM script to avoid dtype issues
            master_port = random.randint(20000, 30000)
            cmd = [
                "./run_main.sh",
                "--config_path", str(config_path),
                "--log_level", "INFO", 
                "--remove_checkpoints", "True",
                "--master_port", str(master_port)
            ]
        
        cmd_str = " ".join(cmd)
        print(f"⚡ Command: {cmd_str}")
        print(f"🏃 Starting experiment with proper environment setup...")
        
        # Retry logic for port conflicts (same as working Time-LLM script)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = subprocess.run(cmd, cwd=self.base_dir, timeout=7200)  # 2 hour timeout
                end_time = time.time()
                duration = end_time - start_time
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS: Completed in {duration:.2f}s")
                    return True
                else:
                    # Check if it's a port conflict and retry with new port (only for non-distillation experiments)
                    if attempt < max_retries - 1 and experiment_type != 'distillation':
                        master_port = random.randint(20000, 30000)
                        cmd[-1] = str(master_port)  # Update port in command
                        print(f"🔄 Retrying with port {master_port} (attempt {attempt + 2}/{max_retries})")
                        continue
                    elif attempt < max_retries - 1:
                        print(f"🔄 Retrying (attempt {attempt + 2}/{max_retries})")
                        continue
                    
                    print(f"❌ FAILED: Exit code {result.returncode}")
                    return False
                    
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    print(f"⏰ Timeout occurred, retrying (attempt {attempt + 2}/{max_retries})")
                    if experiment_type != 'distillation':
                        master_port = random.randint(20000, 30000)
                    cmd[-1] = str(master_port)  # Update port in command
                    continue
                else:
                    print(f"⏰ TIMEOUT: Experiment exceeded 2 hour limit")
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"� Exception occurred, retrying (attempt {attempt + 2}/{max_retries}): {e}")
                    master_port = random.randint(20000, 30000)
                    cmd[-1] = str(master_port)  # Update port in command
                    continue
                else:
                    print(f"�💥 ERROR: {str(e)}")
                    return False
        
        return False
    
    def run_focused_experiments(self, model_types=None):
        """Run focused efficiency tests on specific configs."""
        if model_types is None:
            model_types = ['time_llm', 'chronos']
            
        print(f"\n🎯 STARTING FOCUSED EFFICIENCY TESTING")
        print(f"{'='*80}")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Test patient: {self.test_patient}")
        print(f"🎲 Test seed: {self.test_seed}")
        print(f"📈 Data scenario: {self.data_scenario}")
        print(f"🔬 Model types: {', '.join(model_types)}")
        
        # Always generate fresh configs for each run
        print("📝 Generating fresh experiment configurations...")
        
        # Generate configs using the actual working scripts
        success = self.generate_all_configs(model_types=model_types)
        
        if not success:
            print("❌ Failed to generate configs!")
            return False
        
        # Find the generated configs
        configs = self.find_focused_configs(model_types=model_types)
        
        if not configs:
            print("❌ No configs found after generation!")
            return False
        
        print(f"✅ Generated {len(configs)} experiment configurations")
        
        print(f"\n📋 Found {len(configs)} focused test configs:")
        for config in configs:
            print(f"  {config['type']}: {config['relative_path']}")
        
        if self.dry_run:
            print("\n🔍 DRY RUN MODE - No experiments will be executed")
            return True
        
        # Run each experiment
        results = {'success': [], 'failed': []}
        for config in configs:
            success = self.run_single_experiment(config['path'], config['type'])
            if success:
                results['success'].append(config['relative_path'])
            else:
                results['failed'].append(config['relative_path'])
        
        # Summary
        total = len(configs)
        success_count = len(results['success'])
        
        print(f"\n🎯 FOCUSED EFFICIENCY TESTING COMPLETE")
        print(f"{'='*80}")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Successful: {success_count}/{total}")
        print(f"❌ Failed: {len(results['failed'])}/{total}")
        
        if results['failed']:
            print(f"\n❌ Failed experiments:")
            for failure in results['failed']:
                print(f"  {failure}")
        
        return success_count == total

def main():
    """Main entry point for the comprehensive efficiency runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Efficiency Testing for Fair Teacher–Fair Student Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_efficiency_runner.py --dry-run          # Preview focused experiments (16)
  python comprehensive_efficiency_runner.py                    # Run focused experiments
  python comprehensive_efficiency_runner.py --models time_llm  # Run only Time-LLM experiments
  python comprehensive_efficiency_runner.py --models chronos   # Run only Chronos experiments
  python comprehensive_efficiency_runner.py --models time_llm,chronos  # Run Time-LLM and Chronos
        """
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview experiments without executing them"
    )
    
    parser.add_argument(
        "--models",
        default="time_llm,chronos",
        help="Comma-separated list of model types to run (time_llm,chronos,distillation,distillation_inference). Default: time_llm,chronos"
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help="Run comprehensive analysis of all completed experiments after running experiments"
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true', 
        help="Only run analysis without experiment execution"
    )
    

    
    args = parser.parse_args()
    
    # Parse model types to run
    model_types = [model.strip() for model in args.models.split(',')]
    valid_models = ['time_llm', 'chronos', 'distillation', 'distillation_inference']
    
    for model_type in model_types:
        if model_type not in valid_models:
            print(f"❌ Invalid model type: {model_type}")
            print(f"Valid options: {', '.join(valid_models)}")
            return 1
    
    # Run focused efficiency testing with selected models
    runner = ComprehensiveEfficiencyRunner(dry_run=args.dry_run)
    
    if args.analyze_only:
        print("🔍 ANALYSIS ONLY MODE - Running comprehensive efficiency analysis")
        analysis_results = runner.analyze_all_experiments(save_results=True)
        if analysis_results:
            print(f"\n✅ Analysis completed successfully!")
            df, results = analysis_results
            print(f"📊 Analyzed {len(df)} experiments")
            return 0
        else:
            print("❌ Analysis failed - no experiments found")
            return 1
    elif args.dry_run:
        print("🔍 DRY RUN MODE - No experiments will be executed")
        print("\\nThis would run experiments for model types:", model_types)
        success = runner.run_focused_experiments(model_types=model_types)
        return 0 if success else 1
    else:
        success = runner.run_focused_experiments(model_types=model_types)
        
        # Auto-run analysis after successful experiments if analyze flag is set
        if success and args.analyze:
            print("\\n🔍 Running post-experiment analysis...")
            runner.analyze_all_experiments(save_results=True)
        
        return 0 if success else 1

if __name__ == "__main__":
    exit(main())