#!/bin/bash

# ==============================================================================
# Chronos Unified Configuration Generator - Examples
# ==============================================================================
# This script demonstrates how to use the unified config_generator.py
# for different modes and scenarios.
#
# Usage: ./chronos_config_examples.sh [mode]
# Modes: train, inference, trained_inference, lora_inference, all
# ==============================================================================

# Set script directory and config generator path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_GEN="python3 ${SCRIPT_DIR}/config_generator.py"

# Default parameters
DEFAULT_PATIENTS="570,584"
DEFAULT_MODELS="amazon/chronos-t5-tiny,amazon/chronos-t5-base"
DEFAULT_SEEDS="831363,809906" # Using first 2 seeds from fixed_seeds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
# for different modes and scenarios.
#
# Usage: ./chronos_config_examples.sh [mode]
# Modes: train, inference, trained_inference, lora_inference, all
# ==============================================================================

# Set script directory and config generator path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_GEN="python3 ${SCRIPT_DIR}/config_generator.py"

# Default parameters
DEFAULT_PATIENTS="570,584"
DEFAULT_MODELS="amazon/chronos-t5-tiny,amazon/chronos-t5-base"
DEFAULT_SEEDS="831363,809906" # Using first 2 seeds from fixed_seeds

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to run training config generation
run_training_examples() {
    print_header "TRAINING MODE EXAMPLES"
    
    echo "🔥 Basic Training Configuration"
    echo "Command: $CONFIG_GEN --mode train"
    $CONFIG_GEN --mode train --patients $DEFAULT_PATIENTS --models $DEFAULT_MODELS
    print_success "Basic training configs generated!"
    echo
    
    echo "🔥 Training with Custom Parameters"
    echo "Command: $CONFIG_GEN --mode train --patients 570 --models amazon/chronos-t5-tiny --seeds 831363"
    $CONFIG_GEN --mode train --patients "570" --models "amazon/chronos-t5-tiny" --seeds "831363"
    print_success "Custom training configs generated!"
    echo
    
    echo "🔥 Training for Multiple Patients"
    echo "Command: $CONFIG_GEN --mode train --patients 540,544,552,570,584"
    $CONFIG_GEN --mode train --patients "540,544,552,570,584" --models "amazon/chronos-t5-tiny"
    print_success "Multi-patient training configs generated!"
    echo
    
    echo "🏋️ LoRA Training (PEFT Enabled)"
    echo "Command: $CONFIG_GEN --mode train --use_lora --patients 570 --models amazon/chronos-t5-tiny"
    $CONFIG_GEN --mode train --use_lora --patients "570" --models "amazon/chronos-t5-tiny" --seeds "831363"
    print_success "LoRA training configs generated!"
    echo
    
    echo "🚀 LoRA Training with Noisy Data"
    echo "Command: $CONFIG_GEN --mode train --use_lora --data_scenario noisy --patients 570"
    $CONFIG_GEN --mode train --use_lora --data_scenario noisy --patients "570" --models "amazon/chronos-t5-tiny" --seeds "831363"
    print_success "LoRA training configs with noisy data generated!"
    echo
}

# Function to run inference config generation  
run_inference_examples() {
    print_header "INFERENCE MODE EXAMPLES"
    
    echo "🔮 Basic Inference Configuration (Pretrained Models)"
    echo "Command: $CONFIG_GEN --mode inference"
    $CONFIG_GEN --mode inference --patients $DEFAULT_PATIENTS --models $DEFAULT_MODELS
    print_success "Basic inference configs generated!"
    echo
    
    echo "🔮 Inference with Single Model"
    echo "Command: $CONFIG_GEN --mode inference --models amazon/chronos-t5-base --patients 570,584"
    $CONFIG_GEN --mode inference --models "amazon/chronos-t5-base" --patients "570,584"
    print_success "Single model inference configs generated!"
    echo
    
    echo "🔮 Inference with All Available Models"
    ALL_MODELS="amazon/chronos-t5-tiny,amazon/chronos-t5-small,amazon/chronos-t5-base"
    echo "Command: $CONFIG_GEN --mode inference --models $ALL_MODELS"
    $CONFIG_GEN --mode inference --models "$ALL_MODELS" --patients "570"
    print_success "Multi-model inference configs generated!"
    echo
}

# Function to run trained inference config generation
run_trained_inference_examples() {
    print_header "TRAINED INFERENCE MODE EXAMPLES"
    
    print_warning "Note: Trained inference requires existing training checkpoints!"
    print_warning "Make sure you've run training configs first and have checkpoint-final files."
    echo
    
    echo "🎯 Inference on Trained Checkpoints"
    echo "Command: $CONFIG_GEN --mode trained_inference"
    $CONFIG_GEN --mode trained_inference --patients $DEFAULT_PATIENTS --models $DEFAULT_MODELS
    echo
    
    echo "🎯 Trained Inference for Specific Patient"
    echo "Command: $CONFIG_GEN --mode trained_inference --patients 570 --models amazon/chronos-t5-tiny"
    $CONFIG_GEN --mode trained_inference --patients "570" --models "amazon/chronos-t5-tiny"
    echo
}

# Function to run LoRA inference config generation
run_lora_inference_examples() {
    print_header "LORA INFERENCE MODE EXAMPLES"
    
    print_warning "Note: LoRA inference requires existing training checkpoints with LoRA adapters!"
    echo
    
    echo "🧬 LoRA Inference on Trained Checkpoints"
    echo "Command: $CONFIG_GEN --mode lora_inference"
    $CONFIG_GEN --mode lora_inference --patients $DEFAULT_PATIENTS --models $DEFAULT_MODELS
    echo
    
    echo "🧬 LoRA Inference with Custom Output Directory"
    echo "Command: $CONFIG_GEN --mode lora_inference --output_dir ./my_lora_configs/"
    $CONFIG_GEN --mode lora_inference --output_dir "./my_lora_configs/" --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "LoRA configs with custom output generated!"
    echo
}

# Function to show advanced usage examples
show_advanced_examples() {
    print_header "ADVANCED USAGE EXAMPLES"
    
    echo "🔧 Custom Seeds and Output Directory"
    echo "Command: $CONFIG_GEN --mode train --seeds 123,456,789 --output_dir ./custom_training/"
    echo "$CONFIG_GEN --mode train --seeds \"123,456,789\" --output_dir \"./custom_training/\" --patients \"570\""
    echo
    
    echo "🔧 All Models with Specific Patients"
    ALL_MODELS="amazon/chronos-t5-tiny,amazon/chronos-t5-small,amazon/chronos-t5-base,amazon/chronos-t5-large"
    echo "Command: $CONFIG_GEN --mode inference --models $ALL_MODELS --patients 540,570,584"
    echo "$CONFIG_GEN --mode inference --models \"$ALL_MODELS\" --patients \"540,570,584\""
    echo
    
    echo "🔧 Help and Usage Information"
    echo "Command: $CONFIG_GEN --help"
    $CONFIG_GEN --help
    echo
}

# Function to show data scenario examples
run_data_scenario_examples() {
    print_header "DATA SCENARIO EXAMPLES"
    
    echo "🧪 Training with Different Data Scenarios and Datasets"
    
    echo "📊 Standard/Clean Data (default)"
    echo "Command: $CONFIG_GEN --mode train --data_scenario standardized --patients 570"
    $CONFIG_GEN --mode train --data_scenario standardized --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "Standardized data training configs generated!"
    echo
    
    echo "🔊 Noisy Data Training"
    echo "Command: $CONFIG_GEN --mode train --data_scenario noisy --patients 570"
    $CONFIG_GEN --mode train --data_scenario noisy --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "Noisy data training configs generated!"
    echo
    
    echo "🧹 Denoised Data Training"
    echo "Command: $CONFIG_GEN --mode train --data_scenario denoised --patients 570"
    $CONFIG_GEN --mode train --data_scenario denoised --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "Denoised data training configs generated!"
    echo
    
    echo "📉 Missing Data - Periodic Pattern"
    echo "Command: $CONFIG_GEN --mode train --data_scenario missing_periodic --patients 570"
    $CONFIG_GEN --mode train --data_scenario missing_periodic --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "Missing periodic data training configs generated!"
    echo
    
    echo "🎲 Missing Data - Random Pattern" 
    echo "Command: $CONFIG_GEN --mode train --data_scenario missing_random --patients 570"
    $CONFIG_GEN --mode train --data_scenario missing_random --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "Missing random data training configs generated!"
    echo
    
    echo "�️  Dataset Examples"
    
    echo "📈 D1NAMO Dataset - Standard Data"
    echo "Command: $CONFIG_GEN --mode train --dataset d1namo --patients 570"
    $CONFIG_GEN --mode train --dataset d1namo --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "D1NAMO standard data training configs generated!"
    echo
    
    echo "📈 D1NAMO Dataset - Noisy Data"
    echo "Command: $CONFIG_GEN --mode train --dataset d1namo --data_scenario noisy --patients 570"
    $CONFIG_GEN --mode train --dataset d1namo --data_scenario noisy --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "D1NAMO noisy data training configs generated!"
    echo
    
    echo "🩺 OhioT1DM Dataset - Standard Data"
    echo "Command: $CONFIG_GEN --mode train --dataset ohiot1dm --patients 570"
    $CONFIG_GEN --mode train --dataset ohiot1dm --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "OhioT1DM standard data training configs generated!"
    echo
    
    echo "🩺 OhioT1DM Dataset - Denoised Data"
    echo "Command: $CONFIG_GEN --mode train --dataset ohiot1dm --data_scenario denoised --patients 570"
    $CONFIG_GEN --mode train --dataset ohiot1dm --data_scenario denoised --patients "570" --models "amazon/chronos-t5-tiny"
    print_success "OhioT1DM denoised data training configs generated!"
    echo
    
    echo "�🔄 Inference on Different Datasets and Scenarios"
    echo "Command: $CONFIG_GEN --mode trained_inference --dataset d1namo --data_scenario missing_periodic --patients 570"
    echo "(Note: This requires training on d1namo missing_periodic data first)"
    echo
}

# Function to run a complete pipeline example
run_complete_pipeline() {
    print_header "COMPLETE PIPELINE EXAMPLE"
    
    PIPELINE_PATIENTS="570"
    PIPELINE_MODELS="amazon/chronos-t5-tiny"
    PIPELINE_SEEDS="831363"
    
    echo "This example shows a complete workflow:"
    echo "1️⃣ Generate training configs"
    echo "2️⃣ Generate inference configs (pretrained)"
    echo "3️⃣ Generate trained inference configs (after training)"
    echo "4️⃣ Generate LoRA inference configs"
    echo
    
    print_warning "Step 1: Training Configurations"
    $CONFIG_GEN --mode train --patients $PIPELINE_PATIENTS --models $PIPELINE_MODELS --seeds $PIPELINE_SEEDS
    
    print_warning "Step 2: Pretrained Inference Configurations"  
    $CONFIG_GEN --mode inference --patients $PIPELINE_PATIENTS --models $PIPELINE_MODELS --seeds $PIPELINE_SEEDS
    
    print_warning "Step 3: Trained Inference Configurations (requires checkpoints)"
    $CONFIG_GEN --mode trained_inference --patients $PIPELINE_PATIENTS --models $PIPELINE_MODELS --seeds $PIPELINE_SEEDS
    
    print_warning "Step 4: LoRA Inference Configurations (requires checkpoints)"
    $CONFIG_GEN --mode lora_inference --patients $PIPELINE_PATIENTS --models $PIPELINE_MODELS --seeds $PIPELINE_SEEDS
    
    print_success "Complete pipeline configurations generated!"
}

# Function to show directory structure
show_directory_structure() {
    print_header "EXPECTED OUTPUT DIRECTORY STRUCTURE"
    
    echo "After running the config generator, you'll see:"
    echo
    echo "📁 experiments/chronos_training/"
    echo "   └── seed_831363_model_amazon-chronos-t5-tiny_dtype_float32_mode_train_context_512_pred_64/"
    echo "       ├── patient_570/"
    echo "       │   ├── config.gin"
    echo "       │   └── logs/"
    echo "       └── patient_584/"
    echo "           ├── config.gin"
    echo "           └── logs/"
    echo
    echo "📁 experiments/chronos_inference/"
    echo "   └── seed_831363_model_amazon-chronos-t5-tiny_dtype_float32_mode_inference_context_6_pred_6/"
    echo "       ├── patient_570/"
    echo "       │   ├── config.gin"
    echo "       │   └── logs/"
    echo "       └── patient_584/"
    echo "           ├── config.gin"
    echo "           └── logs/"
    echo
    echo "📁 experiments/chronos_trained_inference/"
    echo "📁 experiments/chronos_lora_inference/"
    echo
}

# Main execution logic
case "${1:-all}" in
    "train")
        run_training_examples
        ;;
    "inference")
        run_inference_examples
        ;;
    "trained_inference")
        run_trained_inference_examples
        ;;
    "lora_inference")
        run_lora_inference_examples
        ;;
    "data_scenarios")
        run_data_scenario_examples
        ;;
    "advanced")
        show_advanced_examples
        ;;
    "pipeline")
        run_complete_pipeline
        ;;
    "structure")
        show_directory_structure
        ;;
    "all")
        echo "🚀 Running all Chronos configuration examples..."
        echo
        run_training_examples
        run_inference_examples
        run_trained_inference_examples
        run_lora_inference_examples
        run_data_scenario_examples
        show_advanced_examples
        show_directory_structure
        ;;
    *)
        print_header "CHRONOS CONFIG GENERATOR - USAGE"
        echo "Usage: $0 [mode]"
        echo
        echo "Available modes:"
        echo "  train              - Generate training configuration examples"
        echo "  inference          - Generate inference configuration examples"
        echo "  trained_inference  - Generate trained inference configuration examples"
        echo "  lora_inference     - Generate LoRA inference configuration examples"
        echo "  advanced           - Show advanced usage examples"
        echo "  pipeline           - Run complete pipeline example"
        echo "  structure          - Show expected directory structure"
        echo "  all                - Run all examples (default)"
        echo
        echo "Examples:"
        echo "  $0 train"
        echo "  $0 inference"
        echo "  $0 all"
        ;;
esac