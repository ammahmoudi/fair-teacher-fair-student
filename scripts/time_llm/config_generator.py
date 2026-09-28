#!/usr/bin/env python3
"""
Unified Time-LLM Configuration Generator

This script combines all time-llm config generators into one unified tool with different modes:
- train: Generate training configurations  
- inference: Generate inference configurations (epochs=0)
- train_inference: Generate combined training+inference configurations

Supports multiple datasets:
- ohiot1dm: OhioT1DM dataset (default)
- d1namo: D1NAMO dataset

Supports multiple data scenarios:
- standardized: Clean/raw data (default)
- noisy: Data with added noise
- denoised: Data that has been denoised
- missing_periodic: Data with periodic missing values
- missing_random: Data with random missing values

Usage:
    python config_generator_time_llm.py --mode train
    python config_generator_time_llm.py --mode train --dataset d1namo
    python config_generator_time_llm.py --mode train --dataset ohiot1dm --data_scenario noisy
    python config_generator_time_llm.py --mode inference --dataset d1namo --data_scenario missing_periodic
    
    # Cross-scenario evaluation (train on clean, test on missing data):
    python config_generator_time_llm.py --mode train_inference --data_scenario missing_periodic --train_data_scenario standardized
    
Options:
    --mode: Operation mode (train, inference, train_inference)
    --dataset: Dataset type (ohiot1dm, d1namo)
    --data_scenario: Data type for inference/testing (standardized, noisy, denoised, missing_periodic, missing_random)
    --train_data_scenario: Data type used for training (optional, for cross-scenario evaluation)
    --patients: Comma-separated list of patient IDs (default: 570,584)
    --llm_models: Comma-separated list of LLM models (default: GPT2,LLAMA)
    --seeds: Comma-separated list of seeds (default: from utilities.seeds)
    --epochs: Number of training epochs (default: 10 for train, 0 for inference)
    --output_dir: Output directory (default: auto-generated based on mode, dataset, and data_scenario)
    --help: Show this help message
"""

import os
import sys
import json
import glob
import argparse
from itertools import product
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.seeds import fixed_seeds


def get_llm_config(llm_model):
    """Get LLM model configuration - updated to support all Time-LLM models."""
    
    # First map HuggingFace model names to our short names
    hf_to_short_name = {
        "bert-base-uncased": "BERT",
        "bert-base-cased": "BERT",
        "bert-large-uncased": "BERT",
        "bert-large-cased": "BERT",
        "distilbert-base-uncased": "DistilBERT",
        "distilbert-base-cased": "DistilBERT", 
        "huawei-noah/TinyBERT_General_4L_312D": "TinyBERT",
        "prajjwal1/bert-tiny": "BERT-tiny",
        "prajjwal1/bert-mini": "BERT-mini",
        "prajjwal1/bert-small": "BERT-small",
        "prajjwal1/bert-medium": "BERT-medium",
        "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large": "MiniLM",
        "google/mobilebert-uncased": "MobileBERT",
        "albert/albert-base-v2": "ALBERT",
        "gpt2": "GPT2",
        "openai-community/gpt2": "GPT2",
        "facebook/opt-125m": "OPT-125M",
        "huggyllama/llama-7b": "LLAMA",
    }
    
    # Convert HuggingFace name to short name if needed
    short_name = hf_to_short_name.get(llm_model, llm_model)
    
    llm_configs = {
        # Large teacher models
        "BERT": {"llm_model": "BERT", "llm_dim": 768, "llm_layers": 12},
        "GPT2": {"llm_model": "GPT2", "llm_dim": 768, "llm_layers": 12},
        "LLAMA": {"llm_model": "LLAMA", "llm_dim": 4096, "llm_layers": 32},
        "DistilBERT": {"llm_model": "DistilBERT", "llm_dim": 768, "llm_layers": 6},
        # Small student models
        "TinyBERT": {"llm_model": "TinyBERT", "llm_dim": 312, "llm_layers": 4},
        "BERT-tiny": {"llm_model": "BERT-tiny", "llm_dim": 128, "llm_layers": 2},
        "BERT-mini": {"llm_model": "BERT", "llm_dim": 256, "llm_layers": 4},
        "BERT-small": {"llm_model": "BERT", "llm_dim": 512, "llm_layers": 4},
        "BERT-medium": {"llm_model": "BERT", "llm_dim": 512, "llm_layers": 8},
        "MiniLM": {"llm_model": "MiniLM", "llm_dim": 384, "llm_layers": 6},
        "MobileBERT": {"llm_model": "MobileBERT", "llm_dim": 512, "llm_layers": 24},
        "ALBERT": {"llm_model": "ALBERT", "llm_dim": 768, "llm_layers": 12},
        "OPT-125M": {"llm_model": "OPT-125M", "llm_dim": 768, "llm_layers": 12},
    }
    # Ensure we always have a valid short name
    final_short_name = short_name if short_name else "BERT"
    return llm_configs.get(final_short_name, llm_configs["BERT"])


def get_length_sets(mode):
    """Get appropriate length sets based on mode."""
    # Common length configurations for Time-LLM
    return [
        {"sequence_length": 6, "context_length": 6, "prediction_length": 6, "patch_len": 6},
        {"sequence_length": 6, "context_length": 6, "prediction_length": 9, "patch_len": 6},
    ]


# Map scenarios to subfolder names in datasets - always use standardized versions
SCENARIO_MAP = {
    "standardized": "raw_standardized",  
    "noisy": "noisy_standardized",
    "denoised": "denoised_standardized", 
    "missing_periodic": "missing_periodic_standardized",
    "missing_random": "missing_random_standardized"
}

def get_data_file_path(mode, patient_id, data_scenario="standardized", dataset="ohiot1dm", is_train_data=True):
    """Get appropriate data file path based on mode, data scenario, and dataset."""
    
    scenario_folder = SCENARIO_MAP[data_scenario]
    base_path = f"./data/{dataset}/{scenario_folder}"
    
    # For Time-LLM, we use CSV files
    if is_train_data:
        return f"{base_path}/{patient_id}-ws-training.csv"
    else:
        return f"{base_path}/{patient_id}-ws-testing.csv"


def get_model_batch_sizes(llm_model):
    """Get appropriate batch sizes based on the LLM model type."""
    # LLAMA models require significantly more memory, use very small batch sizes
    if llm_model == "LLAMA":
        return {
            "train_batch_size": 2,      # Further reduced from 8 to 2 for LLAMA
            "prediction_batch_size": 4  # Further reduced from 16 to 4 for LLAMA
        }
    elif llm_model == "GPT2":
        return {
            "train_batch_size": 8,      # Further reduced from 16 to 8 for GPT2
            "prediction_batch_size": 16  # Reduced from 32 to 16 for GPT2
        }
    else:  # BERT and other models
        return {
            "train_batch_size": 32,     # Default batch size for BERT  
            "prediction_batch_size": 64  # Default batch size for BERT
        }

def generate_config_content(mode, seed, llm_config, length_set, patient_id, train_epochs=10,
                          data_scenario="standardized", dataset="ohiot1dm", train_data_scenario=None,
                          checkpoint_path=None, torch_dtype="bfloat16", fair_teacher=False,
                          fair_teacher_feature="gender"):
    """Generate the configuration content based on parameters."""
    
    # Use train_data_scenario for training data, data_scenario for test data
    actual_train_scenario = train_data_scenario if train_data_scenario else data_scenario
    
    # For per_patient_inference mode, use empty training data path
    if mode == "per_patient_inference":
        train_data_path = ""
    else:
        train_data_path = get_data_file_path(mode, patient_id, actual_train_scenario, dataset, is_train_data=True)
    
    test_data_path = get_data_file_path(mode, patient_id, data_scenario, dataset, is_train_data=False)
    
    # Get prompt file path
    prompt_path = f"./data/{dataset}/{SCENARIO_MAP.get(data_scenario, 'raw_standardized')}/t1dm_prompt.txt"
    
    # Get model-specific batch sizes
    batch_sizes = get_model_batch_sizes(llm_config["llm_model"])
    
    log_folder_placeholder = "LOGS_PLACEHOLDER"  # Will be replaced with actual log folder
    
    # Set mode string
    if mode == "per_patient_inference":
        mode_str = "inference"
    else:
        mode_str = "training+inference"

    # Fair teacher sampling flags
    fair_teacher_line = (
        f"\n     'fair_teacher_sampling': True,"
        f"\n     'fair_teacher_feature': '{fair_teacher_feature}',"
        if fair_teacher and mode != "per_patient_inference" else ""
    )
    
    config_content = f'''# Parameters for run:
# ==============================================================================
run.chronos_dir = '.'
run.data_settings = \\
    {{'frequency': '5min',
     'input_features': ['target'],
     'labels': ['target'],
     'path_to_test_data': '{test_data_path}',
     'path_to_train_data': '{train_data_path}',
     'percent': 100,
     'preprocess_input_features': False,
     'preprocess_label': False,
     'preprocessing_method': 'min_max',
     'prompt_path': '{prompt_path}',
     'val_split': 0{fair_teacher_line}}}

run.llm_settings = \\
    {{'activation': 'gelu',
     'c_out': 1,
     'context_length': {length_set["context_length"]},
     'd_ff': 32,
     'd_layers': 1,
     'd_model': 32,
     'dec_in': 1,
     'des': 'test',
     'dropout': 0.1,
     'e_layers': 2,
     'embed': 'timeF',
     'enc_in': 1,
     'eval_metrics': ['rmse', 'mae', 'mape'],
     'factor': 1,
     'features': 'S',
     'learning_rate': 0.001,
     'llm_dim': {llm_config["llm_dim"]},
     'llm_layers': {llm_config["llm_layers"]},
     'llm_model': '{llm_config["llm_model"]}',
     'lradj': 'COS',
     'method': 'time_llm',
     'mode': '{mode_str}',
     'model_comment': 'time_llm_{llm_config["llm_model"]}_{llm_config["llm_dim"]}_{length_set["sequence_length"]}_{length_set["context_length"]}_{length_set["prediction_length"]}_{length_set["patch_len"]}',
     'model_id': 'test',
     'moving_avg': 25,
     'n_heads': 8,
     'num_workers': 1,
     'patch_len': {length_set["patch_len"]},
     'patience': 10,
     'prediction_batch_size': {batch_sizes["prediction_batch_size"]},
     'prediction_length': {length_set["prediction_length"]},
     'prompt_domain': 0,
     'seed': {seed},
     'sequence_length': {length_set["sequence_length"]},
     'stride': 8,
     'task_name': 'long_term_forecast',
     'timeenc': 0,
     'torch_dtype': '{torch_dtype}',
     'train_batch_size': {batch_sizes["train_batch_size"]},
     'train_epochs': {train_epochs}'''
    
    # Add checkpoint restoration for per_patient_inference mode
    if mode == "per_patient_inference" and checkpoint_path:
        config_content += f''',
     'restore_from_checkpoint': True,
     'restore_checkpoint_path': '{checkpoint_path}' '''
    
    config_content += f'''}}
run.log_dir = \\
    '{log_folder_placeholder}'
'''
    
    return config_content


def main():
    parser = argparse.ArgumentParser(description="Unified Time-LLM Configuration Generator")
    parser.add_argument("--mode", required=True, 
                       choices=["train", "inference", "train_inference", "per_patient_inference"],
                       help="Operation mode (per_patient_inference: inference on individual patients using all-patients checkpoint)")
    parser.add_argument("--patients", default="570,584", 
                       help="Comma-separated patient IDs")
    parser.add_argument("--llm_models", default="GPT2,LLAMA",
                       help="Comma-separated LLM model names") 
    parser.add_argument("--seeds", default=None,
                       help="Comma-separated seeds (default: use fixed_seeds)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (default: 10 for train modes, 0 for inference)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--data_scenario", default="standardized",
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Data scenario type for inference/testing (default: standardized)")
    parser.add_argument("--dataset", default="ohiot1dm",
                       choices=["ohiot1dm", "d1namo"],
                       help="Dataset type (default: ohiot1dm)")
    parser.add_argument("--train_data_scenario", default=None,
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Training data scenario (for cross-scenario evaluation). If not specified, uses --data_scenario")
    parser.add_argument("--checkpoint-path", default=None,
                       help="Path to checkpoint for per_patient_inference mode (required for per_patient_inference)")
    parser.add_argument("--checkpoint-dir", default=None,
                       help="Directory to search for checkpoint (alternative to --checkpoint-path)")
    parser.add_argument("--pred-lengths", default=None,
                       help="Comma-separated prediction lengths to use (e.g., '6,9'). If not specified, uses all available: 6,9")
    parser.add_argument("--torch-dtype", default="bfloat16",
                       choices=["float32", "bfloat16", "float16"],
                       help="Torch dtype for model (default: bfloat16). Use float32 for loading checkpoints trained with float32")
    parser.add_argument("--fair-teacher", action="store_true",
                       help="Enable fair teacher training: oversample minority-group hypo windows (T1 fix)")
    parser.add_argument("--fair-teacher-feature", default="gender",
                       choices=["gender", "age", "pump", "sensor", "cohort"],
                       help="Demographic feature for fair teacher sampling (default: gender)")
    
    args = parser.parse_args()
    
    # Parse inputs
    patients = [p.strip() for p in args.patients.split(",")]
    llm_models = [m.strip() for m in args.llm_models.split(",")]
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds[:2]  # Use first 2 seeds by default
    
    # Set epochs based on mode
    if args.epochs is not None:
        train_epochs = args.epochs
    else:
        train_epochs = 0 if args.mode == "inference" else 10
    
    # Validation for per_patient_inference
    if args.mode == "per_patient_inference":
        if not args.checkpoint_path and not args.checkpoint_dir:
            raise ValueError("--checkpoint-path or --checkpoint-dir required for per_patient_inference mode")
    
    # Handle cross-scenario evaluation
    train_scenario = args.train_data_scenario if args.train_data_scenario else args.data_scenario
    inference_scenario = args.data_scenario
    
    # Set output directory based on mode
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Include dataset and scenario in directory names
        dataset_suffix = f"_{args.dataset}"
        
        if args.mode == "train":
            # For training, only use the data scenario
            scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_training"
        elif args.mode == "inference":
            scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_inference"
        elif args.mode == "per_patient_inference":
            scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_per_patient_inference"
        elif args.mode == "train_inference":
            if args.train_data_scenario:
                # Cross-scenario case
                scenario_suffix = f"_train_{train_scenario}_test_{inference_scenario}" if train_scenario != inference_scenario else f"_{inference_scenario}"
            else:
                scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_training_inference"
            
        combined_suffix = f"{dataset_suffix}{scenario_suffix}"
        base_output_dir = f"./experiments/{dir_name}{combined_suffix}/"
    
    # Get length sets
    length_sets = get_length_sets(args.mode)
    
    # Filter by prediction lengths if specified
    if args.pred_lengths:
        pred_lengths_filter = [int(p.strip()) for p in args.pred_lengths.split(",")]
        length_sets = [ls for ls in length_sets if ls["prediction_length"] in pred_lengths_filter]
        if not length_sets:
            raise ValueError(f"No length sets match prediction lengths: {pred_lengths_filter}")
    
    torch_dtypes = [args.torch_dtype]  # Use the user-specified torch_dtype
    model_ids = ["test"]
    
    print(f"🚀 Starting {args.mode} config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"🗃️  Dataset: {args.dataset}")
    if args.train_data_scenario:
        print(f"🏋️  Training scenario: {train_scenario}")
        print(f"📊 Inference scenario: {inference_scenario}")
    else:
        print(f"📊 Data scenario: {args.data_scenario}")
    print(f"👥 Patients: {patients}")
    print(f"🤖 LLM Models: {llm_models}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📈 Epochs: {train_epochs}")
    
    if args.mode == "per_patient_inference":
        if args.checkpoint_path:
            print(f"📦 Checkpoint: {args.checkpoint_path}")
        elif args.checkpoint_dir:
            print(f"📂 Checkpoint directory: {args.checkpoint_dir}")
    
    if args.fair_teacher:
        print(f"⚖️  Fair teacher sampling: enabled (feature={args.fair_teacher_feature})")

    # Generate configurations
    config_count = 0
    for seed, llm_model_name, length_set, torch_dtype, model_id in product(seeds, llm_models, length_sets, torch_dtypes, model_ids):
        
        llm_config = get_llm_config(llm_model_name)
        seq_len = length_set["sequence_length"]
        context_len = length_set["context_length"]
        pred_len = length_set["prediction_length"]
        patch_len = length_set["patch_len"]
        
        # Create experiment folder
        folder_name = f"seed_{seed}_model_{llm_config['llm_model']}_dim_{llm_config['llm_dim']}_seq_{seq_len}_context_{context_len}_pred_{pred_len}_patch_{patch_len}_epochs_{train_epochs}"
        experiment_folder = os.path.join(base_output_dir, folder_name)
        
        # Resolve checkpoint path for per_patient_inference mode
        checkpoint_path = None
        if args.mode == "per_patient_inference":
            if args.checkpoint_path:
                checkpoint_path = args.checkpoint_path
            elif args.checkpoint_dir:
                # Search for checkpoint in the directory matching this experiment's parameters
                search_pattern = f"*seed_{seed}*model_{llm_config['llm_model']}*dim_{llm_config['llm_dim']}*seq_{seq_len}*context_{context_len}*pred_{pred_len}*patch_{patch_len}*/checkpoint.pth"
                import glob
                matches = glob.glob(os.path.join(args.checkpoint_dir, "**", search_pattern), recursive=True)
                if matches:
                    checkpoint_path = matches[0]
                    print(f"Found checkpoint: {checkpoint_path}")
                else:
                    print(f"Warning: No checkpoint found matching pattern: {search_pattern}")
                    continue
        
        for patient_id in patients:
            patient_folder = os.path.join(experiment_folder, f"patient_{patient_id}")
            log_folder = os.path.join(patient_folder, "logs")
            
            # Create directories
            os.makedirs(patient_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)
            
            # Generate config content
            config_content = generate_config_content(
                args.mode, seed, llm_config, length_set, patient_id, train_epochs,
                data_scenario=args.data_scenario, dataset=args.dataset, train_data_scenario=args.train_data_scenario,
                checkpoint_path=checkpoint_path, torch_dtype=torch_dtype,
                fair_teacher=args.fair_teacher, fair_teacher_feature=args.fair_teacher_feature,
            )
            
            # Replace log folder placeholder with actual path
            config_content = config_content.replace("LOGS_PLACEHOLDER", log_folder)
            
            # Save config file
            config_path = os.path.join(patient_folder, "config.gin")
            with open(config_path, "w") as f:
                f.write(config_content)
            
            print(f"Generated: {config_path}")
            config_count += 1
    
    print(f"✅ Generated {config_count} configuration files in {base_output_dir}")


# Add scenario_map to global scope for generate_config_content
scenario_map = {
    "standardized": "raw_standardized",  
    "noisy": "noisy",
    "denoised": "denoised", 
    "missing_periodic": "missing_periodic",
    "missing_random": "missing_random"
}


if __name__ == "__main__":
    main()