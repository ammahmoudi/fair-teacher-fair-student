#!/bin/bash
# Quick Efficiency Test Launcher
# ===============================
# This script provides quick commands to run efficiency tests for different model types.
# All tests use: Patient 570, Seed 831363, Standardized data, OhioT1DM dataset

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m' 
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${GREEN}🔧 Fair Teacher–Fair Student Comprehensive Efficiency Testing${NC}"
echo "=============================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment not detected. Activating...${NC}"
    source venv/bin/activate
fi

echo ""

# Usage function
show_usage() {
    echo -e "${BLUE}Usage: $0 [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  all                    - Run all efficiency tests (Time-LLM + Chronos + Distillation)"
    echo "  time_llm              - Run Time-LLM efficiency tests (BERT, GPT2, LLAMA)"
    echo "  chronos               - Run Chronos efficiency tests (T5-base, T5-tiny)"
    echo "  distillation          - Run distillation efficiency tests (BERT→TinyBERT)"
    echo "  preview               - Preview all commands (dry-run)"
    echo "  help                  - Show this help message"
    echo ""
    echo -e "${PURPLE}Examples:${NC}"
    echo "  $0 preview            # See what will be run without executing"
    echo "  $0 time_llm           # Run only Time-LLM efficiency tests"
    echo "  $0 chronos            # Run only Chronos efficiency tests"
    echo "  $0 all                # Run everything"
    echo ""
}

# Main execution
case "${1:-help}" in
    "all")
        echo -e "${BLUE}🚀 Running ALL efficiency tests...${NC}"
        echo "This includes Time-LLM (BERT, GPT2, LLAMA), Chronos (T5-base, T5-tiny), and Distillation (BERT→TinyBERT)"
        echo ""
        python comprehensive_efficiency_runner.py --models time_llm,chronos,distillation
        ;;
    "time_llm")
        echo -e "${BLUE}🤖 Running Time-LLM efficiency tests...${NC}"
        echo "Testing BERT, GPT2, and LLAMA models with train+inference efficiency measurement"
        echo ""
        python comprehensive_efficiency_runner.py --models time_llm
        ;;
    "chronos")
        echo -e "${BLUE}⏰ Running Chronos efficiency tests...${NC}"
        echo "Testing amazon/chronos-t5-base and amazon/chronos-t5-tiny models"
        echo ""
        python comprehensive_efficiency_runner.py --models chronos
        ;;
    "distillation")
        echo -e "${BLUE}🧠 Running Distillation efficiency tests...${NC}"
        echo "Testing knowledge distillation from BERT to TinyBERT"
        echo ""
        python comprehensive_efficiency_runner.py --models distillation
        ;;
    "preview")
        echo -e "${BLUE}🔍 PREVIEW MODE - Showing what would be executed...${NC}"
        echo ""
        python comprehensive_efficiency_runner.py --dry-run
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}🎉 Efficiency testing completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}📊 Next Steps:${NC}"
    echo "1. Open the experiment_efficiency_analysis.ipynb notebook"
    echo "2. Run the notebook to analyze efficiency results"
    echo "3. Check experiments/ folder for detailed efficiency reports"
    echo "4. Look for JSON files with performance metrics"
else
    echo -e "${RED}❌ Some efficiency tests failed. Check the output above for details.${NC}"
fi

echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "- Use 'preview' option first to see what will be executed"
echo "- Efficiency reports are saved as JSON files in experiment logs/"
echo "- Each test uses patient 570, seed 831363 for consistency"
echo "- Training+inference efficiency is measured for comprehensive analysis"