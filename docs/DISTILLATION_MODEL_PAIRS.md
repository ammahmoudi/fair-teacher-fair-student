# Distillation Model Pair Recommendations

Based on our comprehensive Time-LLM model ecosystem analysis, here are the recommended teacher-student pairs for knowledge distillation:

## Classification Scope

The recommendations below were originally written for forecasting. The ECG
classification path can construct BERT, TinyBERT, BERT-tiny, DistilBERT,
MiniLM, MobileBERT, ALBERT, GPT-2, and OPT-125M backbones, but only
**BERT -> TinyBERT** has completed the locked five-seed MIT-BIH protocol.

| ECG pair | Status | Supported endpoints |
| --- | --- | --- |
| BERT -> TinyBERT | Locked five-seed validated | AAMI-5 and binary ectopy |
| BERT -> BERT-tiny | Config/protocol tested | AAMI-5 and binary modes |
| Other implemented backbone pairs | Engineering support only | Require controlled validation |

ECG classification uses `alpha * cross_entropy + beta * T^2 * KL` rather than
the forecasting MSE-to-ground-truth plus MSE-to-teacher objective. Do not carry
forecasting pair rankings, expected retention percentages, or KD temperatures
into ECG without validation.

## 🎯 Recommended Teacher-Student Combinations

### 🏆 **High-Performance Pairs** (Best Overall)
1. **BERT -> TinyBERT**: `bert-base-uncased` -> `huawei-noah/TinyBERT_General_4L_312D`
   - **Compression**: 110M -> 14M (87% reduction)
   - **Speed**: ~8x faster inference
   - **Use Case**: Production systems needing good accuracy with fast inference

2. **ALBERT -> BERT-tiny**: `albert-base-v2` -> `prajjwal1/bert-tiny`
   - **Compression**: 12M -> 4M (67% reduction)
   - **Speed**: ~15x faster inference  
   - **Use Case**: Ultra-low resource environments

3. **BERT -> MiniLM**: `bert-base-uncased` -> `microsoft/MiniLM-L12-H384-A12`
   - **Compression**: 110M -> 33M (70% reduction)
   - **Speed**: ~4x faster inference
   - **Use Case**: Balanced performance and efficiency

### ⚡ **Ultra-Fast Pairs** (Maximum Speed)
4. **BERT -> BERT-tiny**: `bert-base-uncased` -> `prajjwal1/bert-tiny`
   - **Compression**: 110M -> 4M (96% reduction)
   - **Speed**: ~20x faster inference
   - **Use Case**: Real-time applications, mobile deployment

5. **DistilBERT -> BERT-tiny**: `distilbert-base-uncased` -> `prajjwal1/bert-tiny`
   - **Compression**: 66M -> 4M (94% reduction)  
   - **Speed**: ~15x faster inference
   - **Use Case**: Quick prototyping, resource-constrained environments

### 📱 **Mobile-Optimized Pairs**
6. **DistilBERT -> MobileBERT**: `distilbert-base-uncased` -> `google/mobilebert-uncased`
   - **Compression**: 66M -> 25M (62% reduction)
   - **Speed**: ~3x faster inference
   - **Use Case**: Mobile applications, edge deployment

7. **ALBERT -> MobileBERT**: `albert-base-v2` -> `google/mobilebert-uncased`
   - **Compression**: 12M -> 25M (teacher actually smaller!)
   - **Speed**: Optimized for mobile inference patterns
   - **Use Case**: Knowledge transfer for mobile-specific optimizations

### 🧪 **Experimental Pairs** (Cross-Architecture)
8. **GPT2 -> TinyBERT**: `gpt2` -> `huawei-noah/TinyBERT_General_4L_312D`
   - **Compression**: 117M -> 14M (88% reduction)
   - **Speed**: ~8x faster inference
   - **Use Case**: Transfer generative knowledge to encoder-only model

## 🔬 Testing Strategy

### Quick Test (3 pairs, ~30 minutes)
```bash
python scripts/distillation_comparison.py --mode quick
```
Tests: BERT->BERT-tiny, BERT->TinyBERT, DistilBERT->MiniLM

### Balanced Test (8 pairs, ~2 hours)
```bash
python scripts/distillation_comparison.py --mode balanced
```
Tests all recommended high-performance and ultra-fast pairs

### Comprehensive Test (16 pairs, ~4-6 hours)
```bash
python scripts/distillation_comparison.py --mode all
```
Tests all possible teacher-student combinations

### Custom Test
```bash
python scripts/distillation_comparison.py --custom-pairs "bert-base-uncased,prajjwal1/bert-tiny" "albert-base-v2,huawei-noah/TinyBERT_General_4L_312D"
```

## 📊 Expected Performance Characteristics

| Teacher | Student | Size Reduction | Speed Gain | Accuracy Retention |
|---------|---------|---------------|------------|-------------------|
| BERT | TinyBERT | 87% | 8x | ~95% |
| BERT | BERT-tiny | 96% | 20x | ~85% |
| BERT | MiniLM | 70% | 4x | ~97% |
| ALBERT | BERT-tiny | 67% | 15x | ~90% |
| DistilBERT | MobileBERT | 62% | 3x | ~98% |

## 🎯 Selection Guidelines

**Choose based on your priority:**

- **🎯 Best Overall Balance**: BERT -> TinyBERT
- **⚡ Maximum Speed**: BERT -> BERT-tiny  
- **📊 Best Accuracy Retention**: BERT -> MiniLM
- **📱 Mobile Deployment**: DistilBERT -> MobileBERT
- **🔋 Ultra-Low Resource**: ALBERT -> BERT-tiny

## 🚀 Quick Start Example

```bash
# Test the top 3 recommended pairs
python scripts/distillation_comparison.py --mode quick --epochs 1

# Run a specific high-performance pair
bash distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student huawei-noah/TinyBERT_General_4L_312D \
  --patients 570 --teacher-epochs 2 --student-epochs 2 --distill-epochs 2
```

The comparison script will automatically generate performance reports and recommend the best pairs for your specific use case!

### MIT-BIH ECG Classification

Generate a standalone classifier configuration:

```bash
python scripts/time_llm/config_generator_mitbih.py \
   --mode train_inference \
   --llm_models BERT,TinyBERT \
   --label-mode aami5 \
   --class-balanced
```

Generate BERT-to-TinyBERT ECG KD configuration after training the teacher:

```bash
python scripts/time_llm/config_generator_mitbih_distillation.py \
   --mode train_inference \
   --teacher-model BERT \
   --student-models TinyBERT \
   --teacher-checkpoint-path /path/to/checkpoint_best.pth \
   --label-mode binary_ectopy \
   --use-rr-features \
   --teacher-use-rr-features \
   --class-balanced
```

For the reproducible locked experiment, use
`scripts/pipelines/run_mitbih_binary_ectopy_five_seed.sh` rather than manually
selecting settings.