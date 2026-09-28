# Distillation Analysis Script

## 📊 Overview
`analyze_distillation_results.py` - Comprehensive analysis tool for knowledge distillation experiments.

## 🚀 Usage

### Basic Analysis
```bash
python scripts/analysis/analyze_distillation_results.py \
  --results-dir distillation_pairs_comparison
```

### Full Analysis with All Exports
```bash
python scripts/analysis/analyze_distillation_results.py \
  --results-dir distillation_pairs_comparison \
  --export-json \
  --export-summary \
  --export-diagrams
```

### Options
- `--results-dir DIR` - Directory containing pipeline_results.csv (default: distillation_pairs_comparison)
- `--top N` - Number of top results to show (default: 10)
- `--export-json` - Export detailed analysis to JSON
- `--export-summary` - Export simplified summary CSV
- `--export-diagrams` - Generate visualization diagrams
- `--output-dir DIR` - Output directory for exports (default: same as results-dir)

## 📁 Outputs Generated

### Always Generated (Automatic)
1. **Console Report** - Comprehensive text report printed to terminal
2. **Grouped Report (TXT)** - `distilled_not_beating_teacher_report_*.txt`
   - GROUP 1: Distilled improved student but didn't beat teacher (partial success)
   - GROUP 2: Distilled failed to improve student AND didn't beat teacher (complete failure)
3. **Grouped CSV** - `distilled_not_beating_teacher_*.csv`
   - Machine-readable data for the two groups

### Optional Exports
4. **JSON Analysis** - `analysis_results_*.json` (with --export-json)
5. **Summary CSV** - `analysis_summary_*.csv` (with --export-summary)
6. **Diagrams** (with --export-diagrams):
   - `comprehensive_dashboard.png` - ⭐ ALL-IN-ONE main dashboard
   - `distillation_improvement_comparison.png` - Bar chart of improvements
   - `rmse_performance_comparison.png` - RMSE comparison
   - `improvement_heatmap.png` - Teacher-student matrix
   - `teacher_analysis.png` - Teacher model performance
   - `student_analysis.png` - Student model performance
   - `training_efficiency.png` - Efficiency scores
   - `summary_statistics.png` - Overall statistics

## 📊 Analysis Categories

### Console Report Shows:
- Overall statistics (total experiments, success rate)
- Top pairs by distillation effectiveness
- Top pairs by absolute performance
- Most efficient pairs (improvement per minute)
- Teacher model analysis
- Student model analysis
- Students beating teachers (baseline comparison)
- Negative distillations (degraded performance)
- Improvement matrix
- Key recommendations

### Grouped Report Shows:
**GROUP 1 - Partial Success:**
- Distilled RMSE < Student Baseline RMSE ✅
- Distilled RMSE ≥ Teacher RMSE ❌
- Knowledge transfer worked but couldn't match teacher

**GROUP 2 - Complete Failure:**
- Distilled RMSE ≥ Student Baseline RMSE ❌
- Distilled RMSE ≥ Teacher RMSE ❌
- Knowledge transfer failed or caused negative transfer

## 🎯 Key Metrics

- **Distillation Effectiveness**: Student baseline RMSE vs Distilled RMSE
- **Absolute Performance**: Lowest RMSE achieved
- **Efficiency**: Improvement % per minute of training
- **Success Rate**: Percentage with positive improvement
- **Teacher Superiority**: Cases where teacher beats distilled student

## 💡 Interpretation

### Successful Distillation
- Distilled RMSE < Student Baseline RMSE
- Knowledge successfully transferred from teacher to student

### Failed Distillation
- Distilled RMSE ≥ Student Baseline RMSE
- Knowledge transfer failed or degraded performance

### Student Beats Teacher
- Student Baseline RMSE < Teacher RMSE
- Smaller model inherently better for the task

## 📝 Example Output Files

After running with all options:
```
distillation_pairs_comparison/
├── pipeline_results.csv
├── analysis_results_20251105_173836.json
├── analysis_summary_20251105_173836.csv
├── distilled_not_beating_teacher_report_20251105_173836.txt  ← Always generated
├── distilled_not_beating_teacher_20251105_173836.csv         ← Always generated
└── diagrams/
    ├── comprehensive_dashboard.png  ← Main visualization
    ├── distillation_improvement_comparison.png
    ├── rmse_performance_comparison.png
    ├── improvement_heatmap.png
    ├── teacher_analysis.png
    ├── student_analysis.png
    ├── training_efficiency.png
    └── summary_statistics.png
```

## 🔄 Workflow

1. Run distillation experiments → generates `pipeline_results.csv`
2. Run analysis script → gets comprehensive analysis + grouped reports
3. Review grouped reports to understand:
   - Which distillations partially succeeded (improved student, didn't beat teacher)
   - Which distillations completely failed (didn't improve student at all)
4. Export diagrams for presentations/papers
5. Use CSV/JSON for further analysis or plotting

---
*Script automatically generates grouped reports on every run!*
