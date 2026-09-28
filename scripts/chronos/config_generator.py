#!/usr/bin/env python3
"""
Unified Chronos Configuration Generator

This script combines all chronos config generators into one unified tool with different modes:
- train: Generate training configurations  
- inference: Generate inference configurations with pretrained models
- trained_inference: Generate inference configurations using trained checkpoints
- lora_inference: Generate inference configurations using trained LoRA checkpoints

Supports multiple datasets:
- standardized: Legacy format (default)
- d1namo: D1NAMO dataset
- ohiot1dm: OhioT1DM dataset

Supports multiple data scenarios:
- standardized: Clean/raw data (default)
- noisy: Data with added noise
- denoised: Data that has been denoised
- missing_periodic: Data with periodic missing values
- missing_random: Data with random missing values

Usage:
    python config_generator.py --mode train
    python config_generator.py --mode train --dataset d1namo
    python config_generator.py --mode train --dataset ohiot1dm --data_scenario noisy
    python config_generator.py --mode trained_inference --dataset d1namo --data_scenario missing_periodic
    
Options:
    --mode: Operation mode (train, inference, trained_inference, lora_inference)
    --dataset: Dataset type (standardized, d1namo, ohiot1dm)
    --data_scenario: Data type (standardized, noisy, denoised, missing_periodic, missing_random)
    --patients: Comma-separated list of patient IDs (default: 570,584)
    --models: Comma-separated list of models (default: amazon/chronos-t5-tiny,amazon/chronos-t5-base)
    --seeds: Comma-separated list of seeds (default: from utilities.seeds)
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

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_data_path, get_models_path, get_arrow_data_path, get_formatted_data_path


def find_latest_checkpoint(training_folder_pattern, patient_id):
    """
    Finds the latest 'checkpoint-final' inside the correct patient folder.
    Falls back to any available seed if the specific seed is not found.
    Example path:
    experiments/chronos_training/seed_*_model_amazon_chronos-t5-*/patient_575/logs/logs_YYYY-MM-DD_HH-MM-SS/chronos-t5-*/run-0/checkpoint-final
    """
    print(f"\n🔍 Searching for checkpoint in: {training_folder_pattern} for patient {patient_id}")

    try:
        # Expand wildcard pattern (match specific seed first)
        training_folders = glob.glob(training_folder_pattern)
        if not training_folders:
            # Fallback: Try to find ANY available checkpoint for this patient
            base_pattern = training_folder_pattern.replace("seed_*", "seed_*")
            fallback_pattern = "/".join(training_folder_pattern.split("/")[:-1]) + "/seed_*_model_*"
            print(f"🔄 Trying fallback pattern: {fallback_pattern}")
            training_folders = glob.glob(fallback_pattern)
            
            if not training_folders:
                print(f"⚠️ No training folders found even with fallback pattern!")
                return ""
            else:
                print(f"✅ Found {len(training_folders)} training folders with fallback pattern")

        # Find all checkpoint-final files in the correct patient logs  
        checkpoint_paths = []
        for training_folder in training_folders:
            patient_logs_path = os.path.join(training_folder, f"patient_{patient_id}")
            checkpoints = glob.glob(
                os.path.join(patient_logs_path, "**", "checkpoint-final"),
                recursive=True
            )
            if checkpoints:
                checkpoint_paths.extend(checkpoints)

        if not checkpoint_paths:
            print(f"⚠️ No 'checkpoint-final' found for patient {patient_id} in any training folder!")
            return ""

        # Sort checkpoints by modification time (latest first)
        checkpoint_paths.sort(key=os.path.getmtime, reverse=True)

        latest_checkpoint = checkpoint_paths[0]  # Pick the most recent one
        print(f"✅ Found latest checkpoint for patient {patient_id}: {latest_checkpoint}")

        return latest_checkpoint

    except Exception as e:
        print(f"❌ Error finding checkpoint for patient {patient_id} in {training_folder_pattern}: {e}")
        return ""


def find_checkpoint_with_fallback(args, seed, model, torch_dtype, train_scenario, patient_id):
    """
    Finds checkpoint with priority on matching seed, but falls back to any available seed.
    
    Priority:
    1. Try to find checkpoint with exact seed match
    2. If not found, try to find checkpoint with any seed for the same model/scenario
    
    Returns: (checkpoint_path, used_seed) or (None, None) if no checkpoint found
    """
    dataset_suffix = f"_{args.dataset}"
    scenario_suffix = "" if train_scenario == "standardized" else f"_{train_scenario}"
    combined_suffix = f"{dataset_suffix}{scenario_suffix}"
    model_pattern = model.replace('/', '-').replace('_', '-')
    
    # First, try with the specific seed
    training_pattern_specific = f"./experiments/chronos_training{combined_suffix}/seed_{seed}_model_{model_pattern}_dtype_{torch_dtype}_mode_train_*"
    checkpoint_path = find_latest_checkpoint(training_pattern_specific, patient_id)
    
    if checkpoint_path:
        print(f"✅ Found checkpoint with matching seed {seed}")
        return checkpoint_path, seed
    
    # If no checkpoint found with specific seed, try with any seed
    print(f"⚠️ No checkpoint found with seed {seed}, trying any available seed...")
    training_pattern_any = f"./experiments/chronos_training{combined_suffix}/seed_*_model_{model_pattern}_dtype_{torch_dtype}_mode_train_*"
    checkpoint_path = find_latest_checkpoint(training_pattern_any, patient_id)
    
    if checkpoint_path:
        # Extract the seed from the found checkpoint path
        import re
        seed_match = re.search(r'/seed_(\d+)_model_', checkpoint_path)
        used_seed = seed_match.group(1) if seed_match else "unknown"
        print(f"✅ Found checkpoint with different seed {used_seed} (requested: {seed})")
        return checkpoint_path, used_seed
    
    print(f"❌ No checkpoint found for model {model} with train_scenario {train_scenario}")
    return None, None


def get_feature_label_sets(mode, window_config=None):
    """Get appropriate feature/label sets based on mode and window configuration."""
    if mode == "train":
        return [{
            "input_features": ["target"],
            "labels": ["target"], 
            "prediction_length": 64,
            "context_length": 512,
        }]
    else:  # inference modes
        # Define both window configurations
        window_6_6 = {
            "input_features": [
                "BG_{t-5}", "BG_{t-4}", "BG_{t-3}",
                "BG_{t-2}", "BG_{t-1}", "BG_{t}"
            ],
            "labels": [
                "BG_{t+1}", "BG_{t+2}", "BG_{t+3}",
                "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"
            ],
            "prediction_length": 6,
            "context_length": 6,
        }
        
        window_6_9 = {
            "input_features": [
                "BG_{t-8}", "BG_{t-7}", "BG_{t-6}",
                "BG_{t-5}", "BG_{t-4}", "BG_{t-3}",
                "BG_{t-2}", "BG_{t-1}", "BG_{t}"
            ],
            "labels": [
                "BG_{t+1}", "BG_{t+2}", "BG_{t+3}",
                "BG_{t+4}", "BG_{t+5}", "BG_{t+6}",
                "BG_{t+7}", "BG_{t+8}", "BG_{t+9}"
            ],
            "prediction_length": 9,
            "context_length": 9,
        }
        
        # Return based on window_config parameter
        if window_config == "6_6":
            return [window_6_6]
        elif window_config == "6_9":
            return [window_6_9]
        else:  # "both" or None (default)
            return [window_6_6, window_6_9]


def get_max_train_steps(model):
    """Get max training steps based on model."""
    max_steps_map = {
        "amazon/chronos-t5-tiny": 2000,
        "amazon/chronos-t5-mini": 10000,
        "amazon/chronos-t5-small": 10000, 
        "amazon/chronos-t5-base": 2000,
        "amazon/chronos-t5-large": 1000,
        "google/t5-efficient-tiny": 2000,
        "google/t5-efficient-small": 10000,
        "google/t5-efficient-base": 2000,
    }
    return max_steps_map.get(model, 200000)


def get_data_file_path(mode, patient_id, data_scenario="standardized", dataset="ohiot1dm", prediction_length=None, context_length=None):
    """Get appropriate data file path based on mode, data scenario, and dataset."""
    
    if mode == "train":
        # For training: Use .arrow files from standardized folders
        return str(get_arrow_data_path(dataset, data_scenario, patient_id))
    else:
        # For inference: Use scenario-specific formatted data
        # Format: ./data/{dataset}/{scenario}_formatted/{context_len}_{pred_len}
        scenario_suffix = "raw" if data_scenario == "standardized" else data_scenario
        return f"./data/{dataset}/{scenario_suffix}_formatted/{context_length}_{prediction_length}"


def generate_config_content(mode, seed, model, torch_dtype, feature_set, patient_id, 
                          max_steps=None, checkpoint_path=None, use_lora=False, data_scenario="standardized", dataset="ohiot1dm", log_folder="./logs/", requested_seed=None, train_scenario=None):
    """Generate the configuration content based on parameters."""
    
    pred_len = feature_set["prediction_length"]
    context_len = feature_set["context_length"] 
    
    # Handle data paths based on mode
    if mode == 'train':
        # Training mode: data_folder is the full .arrow file path
        data_folder = get_data_file_path(mode, patient_id, data_scenario, dataset, pred_len, context_len)
        train_data_path = data_folder
        test_data_path = f"./data/{dataset}/raw_standardized/{patient_id}-ws-testing.csv"
    else:
        # Inference mode: Use train_scenario for training data, data_scenario for test data
        if train_scenario and mode in ['trained_inference', 'lora_inference']:
            # Cross-scenario inference: training data from train_scenario, test data from data_scenario
            train_data_folder = get_data_file_path(mode, patient_id, train_scenario, dataset, pred_len, context_len)
            test_data_folder = get_data_file_path(mode, patient_id, data_scenario, dataset, pred_len, context_len)
            train_data_path = f"{train_data_folder}/{patient_id}-ws-training.csv"
            test_data_path = f"{test_data_folder}/{patient_id}-ws-testing.csv"
        else:
            # Regular inference: both from same scenario
            data_folder = get_data_file_path(mode, patient_id, data_scenario, dataset, pred_len, context_len)
            train_data_path = f"{data_folder}/{patient_id}-ws-training.csv"
            test_data_path = f"{data_folder}/{patient_id}-ws-testing.csv"
    
    # Prepare .gin configuration content in the correct format
    seed_comment = ""
    if requested_seed is not None and str(seed) != str(requested_seed):
        seed_comment = f"# NOTE: Using seed {seed} (requested seed {requested_seed} not found in training)\n"
    
    config_content = f'''{seed_comment}run.log_dir = "{log_folder}"
run.chronos_dir = "{get_models_path()}/"

run.data_settings = {{
    'path_to_train_data': '{train_data_path}',
    'path_to_test_data': '{test_data_path}',
    'input_features': {feature_set["input_features"]},
    'labels': {feature_set["labels"]},
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'percent': 100
}}

run.llm_settings = {{
    'mode': '{"training" if mode == "train" else "inference"}',    
    'method': 'chronos',    
    'model': '{model}',  
    'torch_dtype': '{torch_dtype}',   
    'ntokens': 4096,
    'tokenizer_kwargs': "{{'low_limit': -30,'high_limit': 30}}",
    'prediction_length': {pred_len},    
    'num_samples': {1 if mode != 'train' else 20},
    'context_length': {context_len},
    'min_past': {context_len if mode != 'train' else 60},
    'prediction_batch_size': 64,
    'prediction_use_auto_split': False,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'learning_rate': 0.001,
    'max_train_steps': {max_steps or get_max_train_steps(model)},
    'save_steps': 1000,
    'log_steps': 200,
    'train_batch_size': 8,
    'random_init': False,
    'seed': {seed}'''
    
    # Add checkpoint restoration settings if needed
    if mode in ['trained_inference', 'lora_inference'] and checkpoint_path:
        config_content += f''',
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{checkpoint_path}' '''
    
    # Add LoRA settings
    if use_lora:
        config_content += f''',
    'use_peft': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05'''
    else:
        config_content += f''',
    'use_peft': False,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05'''
    
    config_content += '''
}'''
    
    return config_content
    
    return "\n".join(gin_content)


def main():
    parser = argparse.ArgumentParser(description="Unified Chronos Configuration Generator")
    parser.add_argument("--mode", required=True, 
                       choices=["train", "inference", "trained_inference", "lora_inference"],
                       help="Operation mode")
    parser.add_argument("--patients", default="570,584", 
                       help="Comma-separated patient IDs")
    parser.add_argument("--models", default="amazon/chronos-t5-tiny,amazon/chronos-t5-base",
                       help="Comma-separated model names") 
    parser.add_argument("--seeds", default=None,
                       help="Comma-separated seeds (default: use fixed_seeds)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--data_scenario", default="standardized",
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Data scenario type (default: standardized)")
    parser.add_argument("--dataset", default="ohiot1dm",
                       choices=["ohiot1dm", "d1namo"],
                       help="Dataset type (default: ohiot1dm)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Enable LoRA (PEFT) for training mode")
    parser.add_argument("--train_scenario", default=None,
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Training scenario for trained_inference mode (default: same as data_scenario)")
    parser.add_argument("--window_config", default=None,
                       choices=["6_6", "6_9", "both"],
                       help="Window configuration: 6_6, 6_9, or both (default: both for inference, N/A for training)")
    
    args = parser.parse_args()
    
    # Parse inputs
    patients = [p.strip() for p in args.patients.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds
    
    # Set training scenario (for cross-scenario inference)
    train_scenario = args.train_scenario if args.train_scenario else args.data_scenario
    
    # Set output directory based on mode
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Include dataset and scenario in directory names
        dataset_suffix = f"_{args.dataset}"
        scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
        combined_suffix = f"{dataset_suffix}{scenario_suffix}"
        
        dir_map = {
            "train": f"./experiments/chronos_training{combined_suffix}/",
            "inference": f"./experiments/chronos_inference{combined_suffix}/", 
            "trained_inference": f"./experiments/chronos_trained_inference{combined_suffix}/",
            "lora_inference": f"./experiments/chronos_lora_inference{combined_suffix}/"
        }
        base_output_dir = dir_map[args.mode]
    
    # Get feature sets and other parameters
    feature_label_sets = get_feature_label_sets(args.mode, args.window_config)
    torch_dtypes = ["float32"]
    
    print(f"🚀 Starting {args.mode} config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"📊 Data scenario: {args.data_scenario}")
    if args.mode in ["trained_inference", "lora_inference"] and args.train_scenario:
        print(f"🏋️ Train scenario: {train_scenario}")
    if args.window_config:
        print(f"🪟 Window config: {args.window_config}")
    print(f"👥 Patients: {patients}")
    print(f"🤖 Models: {models}")
    print(f"🎲 Seeds: {seeds}")
    
    # Generate configurations
    config_count = 0
    for seed, feature_set, model, torch_dtype in product(seeds, feature_label_sets, models, torch_dtypes):
        
        pred_len = feature_set["prediction_length"]
        context_len = feature_set["context_length"]
        
        # Create experiment folder
        folder_name = f"seed_{seed}_model_{model.replace('/', '-').replace('_','-')}_dtype_{torch_dtype}_mode_{args.mode}_context_{context_len}_pred_{pred_len}"
        experiment_folder = os.path.join(base_output_dir, folder_name)
        
        for patient_id in patients:
            patient_folder = os.path.join(experiment_folder, f"patient_{patient_id}")
            log_folder = os.path.join(patient_folder, "logs")
            
            # Create directories
            os.makedirs(patient_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)
            
            # Handle checkpoint finding for trained inference modes
            checkpoint_path = None
            used_seed = seed
            if args.mode in ["trained_inference", "lora_inference"]:
                # Use train_scenario for finding training checkpoints (supports cross-scenario inference)
                # Priority: matching seed first, then any available seed
                checkpoint_path, used_seed = find_checkpoint_with_fallback(
                    args, seed, model, torch_dtype, train_scenario, patient_id
                )
                if not checkpoint_path:
                    print(f"⚠️ Skipping {patient_id} - no checkpoint found for train_scenario: {train_scenario}")
                    continue
            
            # Generate config content
            use_lora = (args.mode == "lora_inference") or (args.mode == "train" and getattr(args, 'use_lora', False))
            config_content = generate_config_content(
                args.mode, used_seed, model, torch_dtype, feature_set, patient_id,
                checkpoint_path=checkpoint_path, use_lora=use_lora, data_scenario=args.data_scenario, 
                dataset=args.dataset, log_folder=log_folder, requested_seed=seed, train_scenario=train_scenario
            )
            
            # Save config file
            config_path = os.path.join(patient_folder, "config.gin")
            with open(config_path, "w") as f:
                f.write(config_content)
            
            print(f"Generated: {config_path}")
            config_count += 1
    
    print(f"✅ Generated {config_count} configuration files in {base_output_dir}")


if __name__ == "__main__":
    main()