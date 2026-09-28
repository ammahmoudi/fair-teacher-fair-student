#!/bin/bash
#
# Quick Distillation Testing Script
# ================================
#
# This script provides easy shortcuts for common distillation testing scenarios.
# 
# Usage:
#   ./test_distillation_pairs.sh quick          # Test 3 best pairs (~30 min)
#   ./test_distillation_pairs.sh balanced       # Test 8 strategic pairs (~2 hours)
#   ./test_distillation_pairs.sh best           # Test only the #1 recommended pair (~20 min)
#   ./test_distillation_pairs.sh tiny           # Test ultra-tiny model pairs (~45 min)
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🔬 Distillation Testing Framework${NC}"
    echo -e "${BLUE}========================================${NC}"
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

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if distillation pipeline exists
check_dependencies() {
    if [[ ! -f "distill_pipeline.sh" ]]; then
        print_error "distill_pipeline.sh not found! Run from Fair Teacher–Fair Student root directory."
        exit 1
    fi
    
    if [[ ! -f "scripts/distillation_comparison.py" ]]; then
        print_error "distillation_comparison.py not found!"
        exit 1
    fi
    
    print_success "Dependencies found"
}

# Test presets
test_quick() {
    print_info "Running quick test (3 pairs, ~30 minutes)"
    python3 scripts/distillation_comparison.py --mode quick --epochs 1
}

test_balanced() {
    print_info "Running balanced test (8 pairs, ~2 hours)"
    python3 scripts/distillation_comparison.py --mode balanced --epochs 2
}

test_best() {
    print_info "Testing the #1 recommended pair: BERT -> TinyBERT"
    python3 scripts/distillation_comparison.py \
        --custom-pairs "bert-base-uncased,huawei-noah/TinyBERT_General_4L_312D" \
        --epochs 3
}

test_tiny() {
    print_info "Testing ultra-tiny model combinations"
    python3 scripts/distillation_comparison.py \
        --custom-pairs \
        "bert-base-uncased,prajjwal1/bert-tiny" \
        "albert-base-v2,prajjwal1/bert-tiny" \
        "distilbert-base-uncased,prajjwal1/bert-tiny" \
        --epochs 2
}

test_comprehensive() {
    print_info "Running comprehensive test (all pairs, ~4-6 hours)"
    print_warning "This will take a long time! Consider running in the background."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/distillation_comparison.py --mode all --epochs 2
    else
        print_info "Cancelled"
        exit 0
    fi
}

# Analyze results after testing
analyze_results() {
    print_info "Analyzing latest results..."
    if python3 scripts/analyze_distillation_results.py --latest 2>/dev/null; then
        print_success "Analysis complete!"
    else
        print_warning "Analysis script requires matplotlib/seaborn/pandas"
        print_info "Install with: pip install matplotlib seaborn pandas"
    fi
}

# Main script
main() {
    print_header
    
    check_dependencies
    
    case "${1:-help}" in
        "quick")
            test_quick
            ;;
        "balanced")
            test_balanced
            ;;
        "best")
            test_best
            ;;
        "tiny")
            test_tiny
            ;;
        "all"|"comprehensive")
            test_comprehensive
            ;;
        "analyze")
            analyze_results
            exit 0
            ;;
        "help"|*)
            echo "🔬 Distillation Testing Framework"
            echo ""
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  quick        Test 3 best pairs (~30 min)"
            echo "  balanced     Test 8 strategic pairs (~2 hours)" 
            echo "  best         Test only #1 recommended pair (~20 min)"
            echo "  tiny         Test ultra-tiny model pairs (~45 min)"
            echo "  all          Test all possible pairs (~4-6 hours)"
            echo "  analyze      Analyze latest test results"
            echo "  help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 quick           # Quick test for immediate results"
            echo "  $0 best            # Test the single best combination"
            echo "  $0 balanced        # Strategic testing (recommended)"
            echo "  $0 analyze         # Analyze previous results"
            echo ""
            echo "📖 For more info: docs/DISTILLATION_MODEL_PAIRS.md"
            exit 0
            ;;
    esac
    
    # Auto-analyze results after successful testing
    if [[ $? -eq 0 ]]; then
        echo ""
        print_success "Testing completed successfully!"
        analyze_results
        
        echo ""
        print_info "Next steps:"
        echo "  • Check the generated reports in distillation_experiments/comparison_results/"
        echo "  • Use the best performing pair for your production pipeline"
        echo "  • Run '$0 analyze' to re-analyze results anytime"
    else
        print_error "Testing failed. Check the logs above."
        exit 1
    fi
}

main "$@"