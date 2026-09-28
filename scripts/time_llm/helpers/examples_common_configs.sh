#!/bin/bash
# Time-LLM Configuration Generator - Auto-Run Common Configurations
# This script automatically generates common Time-LLM research configurations

# Set the script path
SCRIPT_PATH="./config_generator.py"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${GREEN}Time-LLM Auto-Config Generator - Running Common Configurations${NC}"
echo "=================================================================="
echo ""

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: $SCRIPT_PATH not found!${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Generating Standard Training Configurations...${NC}"

# Standard training configurations
echo -e "${YELLOW}Training - OhioT1DM Standardized Data${NC}"
python $SCRIPT_PATH --mode train --dataset ohiot1dm --data_scenario standardized

echo -e "${YELLOW}Training - OhioT1DM Missing Periodic Data${NC}"
python $SCRIPT_PATH --mode train --dataset ohiot1dm --data_scenario missing_periodic

echo -e "${YELLOW}Training - D1NAMO Standardized Data${NC}"
python $SCRIPT_PATH --mode train --dataset d1namo --data_scenario standardized

echo ""
echo -e "${BLUE}🔍 Generating Standard Inference Configurations...${NC}"

# Standard inference configurations
echo -e "${YELLOW}Inference - OhioT1DM Standardized Data${NC}"
python $SCRIPT_PATH --mode inference --dataset ohiot1dm --data_scenario standardized

echo -e "${YELLOW}Inference - OhioT1DM Missing Random Data${NC}"
python $SCRIPT_PATH --mode inference --dataset ohiot1dm --data_scenario missing_random

echo -e "${YELLOW}Inference - D1NAMO Noisy Data${NC}"
python $SCRIPT_PATH --mode inference --dataset d1namo --data_scenario noisy

echo ""
echo -e "${BLUE}🔄 Generating Cross-Scenario Evaluation Configurations...${NC}"

# Cross-scenario evaluations
echo -e "${YELLOW}Cross-Scenario - Train Standardized, Test Missing Periodic${NC}"
python $SCRIPT_PATH --mode train_inference --data_scenario missing_periodic --train_data_scenario standardized

echo -e "${YELLOW}Cross-Scenario - Train Standardized, Test Noisy${NC}"
python $SCRIPT_PATH --mode train_inference --data_scenario noisy --train_data_scenario standardized

echo ""
echo -e "${PURPLE}🤖 Generating LLM Model Specific Configurations...${NC}"

# Model-specific configurations
echo -e "${YELLOW}GPT2 Model - Training and Inference${NC}"
python $SCRIPT_PATH --mode train_inference --llm_models GPT2 --data_scenario standardized

echo -e "${YELLOW}LLAMA Model - Training and Inference${NC}"
python $SCRIPT_PATH --mode train_inference --llm_models LLAMA --data_scenario standardized

echo ""
echo -e "${GREEN}✅ Auto-configuration complete!${NC}"
echo ""
echo -e "${BLUE}Generated configurations for:${NC}"
echo "• Standard training (OhioT1DM & D1NAMO)"
echo "• Standard inference (multiple scenarios)"  
echo "• Cross-scenario evaluations"
echo "• Model-specific configurations (GPT2 & LLAMA)"
echo ""
echo -e "${YELLOW}To generate additional configurations, use:${NC}"
echo "• bash quick_config_time_llm.sh (edit and uncomment desired lines)"
echo "• python $SCRIPT_PATH --help (see all options)"