#!/bin/bash
# Time-LLM Configuration Generator - Quick Commands
# Edit this file and uncomment the lines you want to run, then execute: bash quick_config_time_llm.sh

# Set the script path
SCRIPT_PATH="./config_generator.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}Time-LLM Configuration Generator - Quick Commands${NC}"
echo "============================================================"
echo -e "${YELLOW}Edit this file and uncomment the lines you want to run${NC}"
echo ""

# Basic Training Configurations
echo -e "${BLUE}# Basic Training Configurations${NC}"
# python $SCRIPT_PATH --mode train
# python $SCRIPT_PATH --mode train --dataset d1namo
# python $SCRIPT_PATH --mode train --llm_models GPT2
# python $SCRIPT_PATH --mode train --llm_models LLAMA
# python $SCRIPT_PATH --mode train --llm_models BERT
# python $SCRIPT_PATH --mode train --llm_models GPT2,LLAMA,BERT

# Basic Inference Configurations
echo -e "${BLUE}# Basic Inference Configurations${NC}"
# python $SCRIPT_PATH --mode inference
# python $SCRIPT_PATH --mode inference --dataset d1namo
# python $SCRIPT_PATH --mode inference --llm_models GPT2
# python $SCRIPT_PATH --mode inference --llm_models LLAMA
# python $SCRIPT_PATH --mode inference --llm_models BERT

# Combined Training + Inference
echo -e "${BLUE}# Combined Training + Inference${NC}"
# python $SCRIPT_PATH --mode train_inference
# python $SCRIPT_PATH --mode train_inference --dataset d1namo
# python $SCRIPT_PATH --mode train_inference --llm_models GPT2,LLAMA

# Data Scenarios - OhioT1DM
echo -e "${PURPLE}# Data Scenarios - OhioT1DM Dataset${NC}"
# python $SCRIPT_PATH --mode train --data_scenario noisy
# python $SCRIPT_PATH --mode train --data_scenario denoised
# python $SCRIPT_PATH --mode train --data_scenario missing_periodic
# python $SCRIPT_PATH --mode train --data_scenario missing_random
# python $SCRIPT_PATH --mode inference --data_scenario noisy
# python $SCRIPT_PATH --mode inference --data_scenario denoised
# python $SCRIPT_PATH --mode inference --data_scenario missing_periodic
# python $SCRIPT_PATH --mode inference --data_scenario missing_random

# Data Scenarios - D1NAMO Dataset
echo -e "${PURPLE}# Data Scenarios - D1NAMO Dataset${NC}"
# python $SCRIPT_PATH --mode train --dataset d1namo --data_scenario noisy
# python $SCRIPT_PATH --mode train --dataset d1namo --data_scenario denoised
# python $SCRIPT_PATH --mode train --dataset d1namo --data_scenario missing_periodic
# python $SCRIPT_PATH --mode train --dataset d1namo --data_scenario missing_random
# python $SCRIPT_PATH --mode inference --dataset d1namo --data_scenario noisy
# python $SCRIPT_PATH --mode inference --dataset d1namo --data_scenario denoised
# python $SCRIPT_PATH --mode inference --dataset d1namo --data_scenario missing_periodic
# python $SCRIPT_PATH --mode inference --dataset d1namo --data_scenario missing_random

# Cross-Scenario Evaluation (Train on one scenario, test on another)
echo -e "${CYAN}# Cross-Scenario Evaluation${NC}"
# python $SCRIPT_PATH --mode train_inference --data_scenario noisy --train_data_scenario standardized
# python $SCRIPT_PATH --mode train_inference --data_scenario missing_periodic --train_data_scenario standardized
# python $SCRIPT_PATH --mode train_inference --data_scenario missing_random --train_data_scenario standardized
# python $SCRIPT_PATH --mode train_inference --data_scenario denoised --train_data_scenario noisy
# python $SCRIPT_PATH --mode train_inference --dataset d1namo --data_scenario noisy --train_data_scenario standardized
# python $SCRIPT_PATH --mode train_inference --dataset d1namo --data_scenario missing_periodic --train_data_scenario standardized

# Specific Patient and Model Combinations
echo -e "${YELLOW}# Specific Patient and Model Combinations${NC}"
# python $SCRIPT_PATH --mode train --patients 570 --llm_models GPT2
# python $SCRIPT_PATH --mode train --patients 584 --llm_models LLAMA
# python $SCRIPT_PATH --mode train --patients 570,584 --llm_models GPT2,LLAMA --data_scenario missing_periodic
# python $SCRIPT_PATH --mode inference --patients 570 --llm_models BERT --dataset d1namo

# Custom Epochs and Seeds
echo -e "${RED}# Custom Epochs and Seeds${NC}"
# python $SCRIPT_PATH --mode train --epochs 20 --seeds 1,2,3
# python $SCRIPT_PATH --mode train --epochs 5 --seeds 42 --llm_models GPT2
# python $SCRIPT_PATH --mode train_inference --epochs 15 --seeds 1,5 --data_scenario missing_periodic

# All Model Types
echo -e "${GREEN}# All Model Types${NC}"
# python $SCRIPT_PATH --mode train --llm_models GPT2,LLAMA,BERT --data_scenario standardized
# python $SCRIPT_PATH --mode train --llm_models GPT2,LLAMA,BERT --data_scenario noisy
# python $SCRIPT_PATH --mode train --llm_models GPT2,LLAMA,BERT --data_scenario missing_periodic
# python $SCRIPT_PATH --mode inference --llm_models GPT2,LLAMA,BERT --dataset d1namo

echo ""
echo -e "${GREEN}Usage Instructions:${NC}"
echo "1. Edit this file and uncomment the commands you want to run"
echo "2. Run: bash quick_config_time_llm.sh"
echo "3. Or copy individual commands to run them separately"
echo ""
echo -e "${YELLOW}For help with parameters: python $SCRIPT_PATH --help${NC}"