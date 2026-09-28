#!/usr/bin/env python3
"""
Chronos Trained Inference Configuration Generator

This script generates trained_inference configurations that use checkpoints from completed training runs.
This should be run AFTER training experiments have been completed and checkpoints are available.

The script generates:
- trained_inference configs for all datasets and scenarios
- Cross-scenario trained_inference configs (train on standardized, test on corrupted data)

Usage:
    python helpers/batch_generate_trained_inference.py

Prerequisites:
- Training experiments must be completed with available checkpoints
- Checkpoint paths should follow the expected structure in ./experiments/chronos_training_*/
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_config_generation(command_args, description):
    """Run a config generation command with error handling."""
    print(f"\n🔧 {description}")
    command = ["python", "scripts/chronos/config_generator.py"] + command_args
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def main():
    """Generate trained_inference Chronos configurations using existing checkpoints."""
    
    # Import seeds here to ensure it's available when main() is called
    from utilities.seeds import fixed_seeds
    print(f"✅ Using seeds from utilities/seeds.py: {fixed_seeds}")
    
    # Configuration parameters
    chronos_models = "amazon/chronos-t5-tiny,amazon/chronos-t5-base"
    seeds_str = ",".join(map(str, fixed_seeds))
    
    # Dataset configurations
    datasets = {
        "d1namo": "001,002,003,004,005,006,007",
        "ohiot1dm": "540,544,552,559,563,567,570,575,584,588,591,596"
    }
    
    # Data scenarios
    scenarios = ["standardized", "missing_periodic", "missing_random", "noisy", "denoised"]
    
    # Track success/failure
    success_count = 0
    total_operations = 0
    
    print("🚀 Generating Chronos Trained Inference Configurations")
    print("=" * 60)
    print("⚠️  PREREQUISITE: Training experiments must be completed with available checkpoints")
    print(f"🤖 Models: {chronos_models}")
    print(f"🎲 Seeds: {seeds_str}")
    print(f"📊 Datasets: {list(datasets.keys())}")
    print(f"🔬 Scenarios: {scenarios}")
    print("=" * 60)
    
    # 1. Generate trained inference configurations for all datasets and scenarios
    print("\n" + "="*60)
    print("PHASE 1: CHRONOS TRAINED INFERENCE CONFIGURATIONS")
    print("="*60)
    
    for dataset, patients in datasets.items():
        for scenario in scenarios:
            total_operations += 1
            description = f"Generating {dataset} trained inference configs for {scenario} scenario"
            
            command_args = [
                "--mode", "trained_inference",
                "--dataset", dataset, 
                "--data_scenario", scenario,
                "--patients", patients,
                "--models", chronos_models,
                "--seeds", seeds_str
            ]
            
            if run_config_generation(command_args, description):
                success_count += 1
    
    # 2. Generate cross-scenario trained inference experiments (train on standardized, test on corrupted scenarios)
    print("\n" + "="*60) 
    print("PHASE 2: CHRONOS CROSS-SCENARIO TRAINED INFERENCE")
    print("="*60)
    
    # Cross-scenario: Train on clean/standardized data, test on corrupted scenarios
    corrupted_scenarios = ["missing_periodic", "missing_random", "noisy", "denoised"]
    
    for dataset, patients in datasets.items():
        for test_scenario in corrupted_scenarios:
            total_operations += 1
            description = f"Generating {dataset} cross-scenario trained inference: train on standardized, test on {test_scenario}"
            
            # Generate trained_inference configs that use standardized training checkpoints
            # but test on the corrupted scenario data
            command_args = [
                "--mode", "trained_inference",
                "--dataset", dataset,
                "--data_scenario", test_scenario,  # Test on corrupted scenario
                "--patients", patients,
                "--models", chronos_models, 
                "--seeds", seeds_str,
                "--output_dir", f"./experiments/chronos_cross_scenario_{dataset}_train_standardized_test_{test_scenario}_trained/"
            ]
            
            if run_config_generation(command_args, description):
                success_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("CHRONOS TRAINED INFERENCE CONFIGURATION GENERATION COMPLETE!")
    print("="*60)
    print(f"📈 Success: {success_count}/{total_operations} operations")
    print(f"📁 Experiments folder: {os.path.abspath('./experiments/')}")
    
    # Count generated experiment directories
    experiments_path = Path("./experiments/")
    if experiments_path.exists():
        trained_inference_dirs = [d for d in experiments_path.iterdir() 
                                if d.is_dir() and "trained_inference" in d.name.lower()]
        print(f"📂 Created {len(trained_inference_dirs)} trained inference experiment directories")
        
        if trained_inference_dirs:
            print(f"   🔮 Trained Inference directories:")
            for d in sorted([d.name for d in trained_inference_dirs])[:5]:  # Show first 5
                print(f"      - {d}")
            if len(trained_inference_dirs) > 5:
                print(f"      ... and {len(trained_inference_dirs) - 5} more")
    
    print("\n🎉 All Chronos trained inference configurations generated successfully!")
    print("💡 These configs will use checkpoints from completed training experiments")
    
    if success_count < total_operations:
        print(f"\n⚠️  Note: {total_operations - success_count} operations had issues.")
        print("   This is expected if training checkpoints don't exist yet.")
        return 1
    
    return 0

if __name__ == "__main__":
    # Change to project root directory
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    exit_code = main()
    sys.exit(exit_code)