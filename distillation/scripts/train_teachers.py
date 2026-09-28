#!/usr/bin/env python3
"""
Automated Teacher Training Script for Time-LLM Models
Trains BERT and Tiny BERT teacher models with configurable parameters.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_configs_path


class TeacherTrainer:
    """Class to handle automated training of teacher models."""
    
    def __init__(self, base_dir=None, output_dir=None, config_dir=None, dataset_name="ohiot1dm", seed=238822, lr=0.001, batch_size=32, remove_checkpoints=False, fair_teacher=False, fair_teacher_feature="gender", train_data_override=None, test_data_override=None):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.data_path = f"./data/{dataset_name}"  # Auto-detect data path
        # T2: explicit train/test CSV overrides (e.g. male-only / female-only splits)
        self.train_data_override = train_data_override
        self.test_data_override = test_data_override
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.remove_checkpoints = remove_checkpoints
        self.fair_teacher = fair_teacher
        self.fair_teacher_feature = fair_teacher_feature
        
        # Set config directory - use pipeline config dir if provided, otherwise root configs
        if config_dir:
            self.configs_dir = Path(config_dir)
        else:
            self.configs_dir = get_configs_path()
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set results directory
        if output_dir:
            self.results_dir = Path(output_dir)
        else:
            self.results_dir = self.base_dir / "results" / "teacher_models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define teacher model configurations
        # Model name mappings from HuggingFace names to internal config names
        self.model_name_mapping = {
            # BERT variants
            "bert": "bert",
            "bert-base-uncased": "bert",
            "bert-base-cased": "bert",
            "bert-large-uncased": "bert-large",
            "bert-large-cased": "bert-large",
            # DistilBERT variants
            "distilbert": "distilbert", 
            "distilbert-base-uncased": "distilbert",
            "distilbert-base-cased": "distilbert",
            # TinyBERT and small model variants
            "tinybert": "tinybert",
            "huawei-noah/TinyBERT_General_4L_312D": "tinybert",
            "prajjwal1/bert-tiny": "bert-tiny",
            "prajjwal1/bert-mini": "bert-mini",
            "prajjwal1/bert-small": "bert-small",
            "prajjwal1/bert-medium": "bert-medium",
            # Additional models from Time-LLM
            "gpt2": "gpt2",
            "openai-community/gpt2": "gpt2",
            "llama": "llama",
            "huggyllama/llama-7b": "llama",
            "minilm": "minilm",
            "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large": "minilm",
            "mobilebert": "mobilebert",
            "google/mobilebert-uncased": "mobilebert",
            "albert": "albert",
            "albert/albert-base-v2": "albert",
            "opt-125m": "opt-125m",
            "facebook/opt-125m": "opt-125m",
            # Meta option
            "all": "all"
        }
        
        self.teacher_models = {
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "time_llm_BERT_768_6_6_9_6"
            },
            "bert-large": {
                "llm_model": "BERT",
                "llm_layers": 24,
                "llm_dim": 1024,
                "model_comment": "time_llm_BERT_1024_6_6_9_6"
            },
            "distilbert": {
                "llm_model": "DistilBERT", 
                "llm_layers": 6,
                "llm_dim": 768,
                "model_comment": "time_llm_DistilBERT_768_6_6_9_6"
            },
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "model_comment": "time_llm_TinyBERT_312_6_6_9_6"
            },
            "bert-tiny": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "model_comment": "time_llm_BERT-tiny_128_6_6_9_6"
            },
            "bert-mini": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 256,
                "model_comment": "time_llm_BERT_256_6_6_9_6"
            },
            "bert-small": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 512,
                "model_comment": "time_llm_BERT_512_6_6_9_6"
            },
            "bert-medium": {
                "llm_model": "BERT",
                "llm_layers": 8,
                "llm_dim": 512,
                "model_comment": "time_llm_BERT_512_6_6_9_6"
            },
            "gpt2": {
                "llm_model": "GPT2",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "time_llm_GPT2_768_6_6_9_6"
            },
            "llama": {
                "llm_model": "LLAMA",
                "llm_layers": 32,
                "llm_dim": 4096,
                "model_comment": "time_llm_LLAMA_4096_6_6_9_6"
            },
            "minilm": {
                "llm_model": "MiniLM",
                "llm_layers": 6,
                "llm_dim": 384,
                "model_comment": "time_llm_MiniLM_384_6_6_9_6"
            },
            "mobilebert": {
                "llm_model": "MobileBERT",
                "llm_layers": 24,
                "llm_dim": 512,
                "model_comment": "time_llm_MobileBERT_512_6_6_9_6"
            },
            "albert": {
                "llm_model": "ALBERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "time_llm_ALBERT_768_6_6_9_6"
            },
            "opt-125m": {
                "llm_model": "OPT-125M",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "time_llm_OPT-125M_768_6_6_9_6"
            }
        }
        
        # Base training configuration template - will be updated based on dataset
        self.base_config = {
            "data_settings": {
                "path_to_train_data": "",  # Will be set dynamically
                "path_to_test_data": "",   # Will be set dynamically
                "input_features": ["target"],
                "labels": ["target"],
                "prompt_path": "",  # Will be set dynamically
                "preprocessing_method": "min_max",
                "preprocess_input_features": False,
                "preprocess_label": False,
                "frequency": "5min",
                "percent": 100,
                "val_split": 0
            },
            "llm_settings": {
                "task_name": "long_term_forecast",
                "mode": "training+inference",
                "method": "time_llm",
                "num_workers": 1,
                "torch_dtype": "float32",
                "model_id": "teacher",
                "sequence_length": 6,
                "context_length": 6,
                "prediction_length": 9,
                "patch_len": 6,
                "stride": 8,
                "prediction_batch_size": 64,
                "train_batch_size": 32,
                "learning_rate": 0.001,
                "train_epochs": 20,
                "features": "S",
                "d_model": 32,
                "d_ff": 32,
                "factor": 1,
                "enc_in": 1,
                "dec_in": 1,
                "c_out": 1,
                "e_layers": 2,
                "d_layers": 1,
                "n_heads": 8,
                "dropout": 0.1,
                "moving_avg": 25,
                "activation": "gelu",
                "embed": "timeF",
                "patience": 10,
                "lradj": "COS",
                "des": "teacher_training",
                "prompt_domain": 0,
                "timeenc": 0,
                "eval_metrics": ["rmse", "mae", "mape"],
                "seed": 238822  # Will be updated in generate_config
            }
        }

    def normalize_model_name(self, model_name):
        """Normalize HuggingFace model names to internal config names."""
        if model_name in self.model_name_mapping:
            return self.model_name_mapping[model_name]
        else:
            # If not in mapping, try to extract base model name
            model_lower = model_name.lower()
            if "bert-tiny" in model_lower or "tiny" in model_lower:
                return "bert-tiny"
            elif "bert-mini" in model_lower or "mini" in model_lower:
                return "bert-mini"
            elif "bert-small" in model_lower or "small" in model_lower:
                return "bert-small"
            elif "bert-medium" in model_lower or "medium" in model_lower:
                return "bert-medium"
            elif "bert-large" in model_lower:
                return "bert-large"
            elif "bert" in model_lower:
                return "bert"
            elif "distilbert" in model_lower:
                return "distilbert" 
            elif "tinybert" in model_lower:
                return "tinybert"
            else:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.model_name_mapping.keys())}")

    def sanitize_filename(self, name):
        """Sanitize model names for safe file/directory usage."""
        return name.replace('/', '_').replace('\\', '_')

    def generate_config(self, model_name, dataset="584", epochs=20, original_name=None):
        """Generate gin config file for a specific teacher model."""
        # If original_name is provided, use it for file paths, otherwise use model_name
        display_name = original_name if original_name else model_name
        
        # If model_name is already normalized, use it directly, otherwise normalize it
        if model_name in self.teacher_models:
            normalized_name = model_name
        else:
            normalized_name = self.normalize_model_name(model_name)
            
        if normalized_name not in self.teacher_models:
            raise ValueError(f"Model {normalized_name} not supported. Choose from: {list(self.teacher_models.keys())}")
        
        # Build data paths using flexible parameters
        # Remove leading ./ from data_path if present to avoid double slashes
        clean_data_path = self.data_path.lstrip('./')
        
        # Check if using all_patients combined data
        if dataset == "all_patients":
            data_dir = f"{clean_data_path}/all_patients_combined"
            train_file = f"{data_dir}/all_patients_training.csv"
            test_file = f"{data_dir}/all_patients_testing.csv"
            prompt_file = f"{clean_data_path}/raw_standardized/t1dm_prompt.txt"
            # T2: override the train/test CSVs with group-filtered splits when provided
            if self.train_data_override:
                train_file = self.train_data_override
            if self.test_data_override:
                test_file = self.test_data_override
        else:
            data_dir = f"{clean_data_path}/raw_standardized"
            train_file = f"{data_dir}/{dataset}-ws-training.csv"
            test_file = f"{data_dir}/{dataset}-ws-testing.csv"
            prompt_file = f"{data_dir}/t1dm_prompt.txt"
        
        config = self.base_config.copy()
        model_config = self.teacher_models[normalized_name]
        
        # Update data paths
        config["data_settings"]["path_to_train_data"] = train_file
        config["data_settings"]["path_to_test_data"] = test_file
        config["data_settings"]["prompt_path"] = prompt_file
        # Fair teacher sampling (T1 fix)
        if self.fair_teacher:
            config["data_settings"]["fair_teacher_sampling"] = True
            config["data_settings"]["fair_teacher_feature"] = self.fair_teacher_feature
        
        # Update model-specific settings
        config["llm_settings"].update(model_config)
        config["llm_settings"]["train_epochs"] = epochs
        config["llm_settings"]["seed"] = self.seed
        config["llm_settings"]["learning_rate"] = self.lr
        
        # Use model-specific batch sizes for large models
        if normalized_name == "llama":
            # LLAMA-7B is much larger, use smaller batch size to avoid OOM
            config["llm_settings"]["train_batch_size"] = max(1, self.batch_size // 8)  # 32->4, 16->2, 8->1
            config["llm_settings"]["prediction_batch_size"] = max(1, 64 // 8)  # 64->8
            print(f"Using reduced batch size for LLAMA-7B: train={config['llm_settings']['train_batch_size']}, pred={config['llm_settings']['prediction_batch_size']}")
        else:
            config["llm_settings"]["train_batch_size"] = self.batch_size
        
        # Generate log directory path using output directory if provided
        if hasattr(self, 'results_dir') and str(self.results_dir) != str(self.base_dir / "results" / "teacher_models"):
            log_dir = f"{self.results_dir}/{model_name}_{dataset}_{epochs}epochs/logs"
        else:
            log_dir = f"./results/teacher_models/{model_name}_{dataset}_{epochs}epochs/logs"
        config_content = f'run.log_dir = "{log_dir}"\n'
        
        # Convert data_settings to gin format
        config_content += "run.data_settings = {\n"
        for key, value in config["data_settings"].items():
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n\n"
        
        # Convert llm_settings to gin format  
        config_content += "run.llm_settings = {\n"
        for key, value in config["llm_settings"].items():
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            elif isinstance(value, list):
                config_content += f"    '{key}': {value},\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n"
        
        return config_content

    def save_config(self, config_content, model_name, dataset="584", epochs=20):
        """Save config content to a gin file."""
        # Sanitize model name for filename
        safe_model_name = self.sanitize_filename(model_name)
        config_filename = f"config_teacher_{safe_model_name}_{dataset}_{epochs}epochs.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def train_model(self, model_name, dataset="584", epochs=20, dry_run=False):
        """Train a specific teacher model."""
        # Normalize the model name
        normalized_name = self.normalize_model_name(model_name)
        print(f"\n{'='*60}")
        print(f"Training Teacher Model: {model_name} ({normalized_name.upper()})")
        print(f"Dataset: {dataset}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Generate and save config
        config_content = self.generate_config(normalized_name, dataset, epochs, original_name=model_name)
        config_path = self.save_config(config_content, model_name, dataset, epochs)
        
        print(f"✓ Config generated: {config_path}")
        
        if dry_run:
            print("DRY RUN: Would execute training command")
            return config_path
        
        # Execute training command
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        # Add remove_checkpoints flag if requested
        if hasattr(self, 'remove_checkpoints') and self.remove_checkpoints:
            cmd.append("--remove_checkpoints")
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Change to base directory for execution
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir)
            
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully trained {model_name} teacher model")
                
                # Create training summary
                self.create_training_summary(model_name, dataset, epochs, config_path, normalized_name)
                
                return config_path
            else:
                print(f"✗ Training failed with return code: {result.returncode}")
                return None
        
        except Exception as e:
            print(f"✗ Error during training: {str(e)}")
            return None
        finally:
            os.chdir(original_dir)

    def create_training_summary(self, model_name, dataset, epochs, config_path, normalized_name):
        """Create a comprehensive summary of the training run."""
        # Find the latest model directory and results
        model_dir = None
        performance_metrics = None
        training_info = None
        
        # Look for model directory pattern: {model}_{dataset}_{epochs}epochs
        # Use normalized model name for directory pattern matching (as that's what's actually created)
        model_pattern = f"{normalized_name}_{dataset}_{epochs}epochs"
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith(model_pattern):
                model_dir = item
                break
        
        if model_dir:
            # Look for logs directory
            logs_dirs = list(model_dir.glob("logs/logs_*"))
            if logs_dirs:
                latest_log_dir = max(logs_dirs, key=lambda x: x.name)
                
                # Try to extract performance metrics from various sources
                # 1. Look for performance reports
                perf_reports = list(latest_log_dir.glob("*performance_report*.json"))
                if perf_reports:
                    try:
                        with open(perf_reports[0], 'r') as f:
                            perf_data = json.load(f)
                            if 'evaluation_metrics' in perf_data:
                                performance_metrics = perf_data['evaluation_metrics']
                    except:
                        pass
                
                # 2. Look for inference results
                if not performance_metrics:
                    inference_file = latest_log_dir / "inference_results.csv"
                    if inference_file.exists():
                        try:
                            import pandas as pd
                            import numpy as np
                            df = pd.read_csv(inference_file)
                            if 'ground_truth' in df.columns and 'predictions' in df.columns:
                                # Calculate basic metrics from flattened arrays
                                y_true_list = []
                                y_pred_list = []
                                
                                for _, row in df.iterrows():
                                    # Parse comma-separated values
                                    true_vals = [float(x.strip()) for x in str(row['ground_truth']).split(',')]
                                    pred_vals = [float(x.strip()) for x in str(row['predictions']).split(',')]
                                    y_true_list.extend(true_vals)
                                    y_pred_list.extend(pred_vals)
                                
                                y_true = np.array(y_true_list)
                                y_pred = np.array(y_pred_list)
                                
                                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                                mae = np.mean(np.abs(y_true - y_pred))
                                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                                
                                performance_metrics = {
                                    "rmse": float(rmse),
                                    "mae": float(mae), 
                                    "mape": float(mape)
                                }
                        except Exception as e:
                            print(f"Warning: Could not calculate metrics from inference results: {e}")
                
                # Extract training information
                training_info = {
                    "model_directory": str(model_dir),
                    "logs_directory": str(latest_log_dir),
                    "checkpoint_available": (latest_log_dir / "checkpoints" / "checkpoint.pth").exists()
                }
        
        summary = {
            "model_name": model_name,
            "dataset": dataset,
            "epochs": epochs,
            "config_path": str(config_path),
            "model_config": self.teacher_models[normalized_name],
            "training_info": training_info,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Sanitize model name for filename
        safe_model_name = self.sanitize_filename(model_name)
        summary_file = self.results_dir / f"{safe_model_name}_{dataset}_{epochs}epochs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save with standardized name for pipeline CSV logging
        standard_summary_file = self.results_dir / "teacher_training_summary.json"
        with open(standard_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Enhanced training summary saved: {summary_file}")
        print(f"✓ Standardized summary saved: {standard_summary_file}")
        if performance_metrics:
            print(f"  📊 Performance: RMSE={performance_metrics.get('rmse', 'N/A'):.3f}, MAE={performance_metrics.get('mae', 'N/A'):.3f}, MAPE={performance_metrics.get('mape', 'N/A'):.3f}%")

    def train_all_teachers(self, dataset="584", epochs=20, dry_run=False):
        """Train all teacher models."""
        print(f"\n{'='*80}")
        print("AUTOMATED TEACHER MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name in self.teacher_models.keys():
            try:
                config_path = self.train_model(model_name, dataset, epochs, dry_run)
                results[model_name] = {
                    "status": "success" if config_path else "failed",
                    "config_path": str(config_path) if config_path else None
                }
            except Exception as e:
                print(f"✗ Failed to train {model_name}: {str(e)}")
                results[model_name] = {"status": "error", "error": str(e)}
        
        # Save overall results
        results_file = self.results_dir / f"teacher_training_results_{dataset}_{epochs}epochs.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("TEACHER TRAINING COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return results

    def list_available_checkpoints(self):
        """List available teacher model checkpoints."""
        print("\nAvailable Teacher Model Checkpoints:")
        print("-" * 50)
        
        for model_dir in self.results_dir.glob("*"):
            if model_dir.is_dir():
                checkpoint_files = list(model_dir.glob("**/*.pth"))
                if checkpoint_files:
                    print(f"\n{model_dir.name}:")
                    for checkpoint in checkpoint_files:
                        print(f"  - {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train Time-LLM Teacher Models")
    parser.add_argument("--model", choices=["bert", "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", 
                                           "distilbert", "distilbert-base-uncased", "distilbert-base-cased", 
                                           "tinybert", "huawei-noah/TinyBERT_General_4L_312D",
                                           "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "prajjwal1/bert-medium",
                                           "gpt2", "openai-community/gpt2", "llama", "huggyllama/llama-7b",
                                           "minilm", "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
                                           "mobilebert", "google/mobilebert-uncased", 
                                           "albert", "albert/albert-base-v2",
                                           "opt-125m", "facebook/opt-125m", "all"], 
                       default="all", help="Model to train")
    parser.add_argument("--patients", default="584", help="Patient IDs (comma-separated or single)")
    parser.add_argument("--all-patients", action="store_true", help="Train on ALL patients combined data (overrides --patients)")
    parser.add_argument("--dataset", default="ohiot1dm", help="Dataset name (ohiot1dm, d1namo)")
    parser.add_argument("--seed", type=int, default=238822, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't train")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    parser.add_argument("--remove-checkpoints", action="store_true", help="Remove checkpoint files after training to save disk space")
    parser.add_argument("--output-dir", help="Output directory for trained models (default: results/teacher_models)")
    parser.add_argument("--config-dir", help="Directory for saving config files (default: configs)")
    parser.add_argument("--fair-teacher", action="store_true",
                       help="Enable fair teacher training: oversample minority-group hypo windows (T1 fix)")
    parser.add_argument("--fair-teacher-feature", default="gender",
                       choices=["gender", "age", "pump", "sensor", "cohort"],
                       help="Demographic feature for fair teacher sampling (default: gender)")
    parser.add_argument("--train-data-override", default=None,
                       help="T2: explicit training CSV (e.g. a group-filtered split); used with --all-patients")
    parser.add_argument("--test-data-override", default=None,
                       help="T2: explicit testing CSV; used with --all-patients")

    args = parser.parse_args()

    trainer = TeacherTrainer(
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        dataset_name=args.dataset,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        remove_checkpoints=args.remove_checkpoints,
        fair_teacher=args.fair_teacher,
        fair_teacher_feature=args.fair_teacher_feature,
        train_data_override=args.train_data_override,
        test_data_override=args.test_data_override,
    )
    
    if args.list_checkpoints:
        trainer.list_available_checkpoints()
        return
    
    # If --all-patients flag is set, use "all_patients" as the patient ID
    if args.all_patients:
        print("🌍 Training on ALL PATIENTS COMBINED")
        patient_id = "all_patients"
    else:
        patient_id = args.patients
    
    if args.model == "all":
        result = trainer.train_all_teachers(patient_id, args.epochs, args.dry_run)
    else:
        result = trainer.train_model(args.model, patient_id, args.epochs, args.dry_run)
    
    # Exit with appropriate code
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()