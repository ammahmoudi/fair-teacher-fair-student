# 🤖 Supported Models in Time-LLM Distillation Pipeline

This document lists models supported by the original forecasting path and the
separate MIT-BIH ECG classification path. A model being constructible does not
mean that it has completed the locked five-seed classification protocol.

## Task Support

| Backbone | Forecasting | ECG classifier | ECG KD role | Evidence level |
| --- | --- | --- | --- | --- |
| BERT | Yes | Yes | Teacher or student | Five-seed validated as Teacher |
| TinyBERT | Yes | Yes | Teacher or student | Five-seed validated as Student |
| BERT-tiny | Yes | Yes | Teacher or student | Protocol/smoke tested |
| DistilBERT | Yes | Yes | Teacher or student | Implemented, not five-seed validated |
| MiniLM | Yes | Yes | Teacher or student | Implemented, not five-seed validated for ECG |
| MobileBERT | Yes | Yes | Teacher or student | Implemented, not five-seed validated for ECG |
| ALBERT | Yes | Yes | Teacher or student | Implemented, not five-seed validated for ECG |
| GPT-2 | Yes | Yes | Teacher or student | Implemented, not five-seed validated for ECG |
| OPT-125M | Yes | Yes | Teacher or student | Implemented, not five-seed validated for ECG |
| LLaMA-7B | Yes | Yes | Standalone backbone only in practice | Constructible; not validated for ECG KD |
| BERT mini/small/medium aliases | Partial | No distinct ECG backbone | No distinct ECG role | Generator aliases currently resolve to BERT |

The ECG implementation is `models/ecg/time_llm_classifier.py`, exposed through
`llms/time_llm_ecg.py` and the `time_llm_ecg_classifier` method in `main.py`.
It supports AAMI-5, binary ectopy (N versus S/V/F with Q excluded), and binary
non-N label modes. Previous/next RR timing features are optional.

## 📋 Legacy Forecasting Compatibility Matrix

This original matrix describes the general forecasting ecosystem. Use the task
support table above for ECG classification claims.

| Model Name | HuggingFace ID | Parameters | Time-LLM | Teacher | Student | Distillation |
|------------|----------------|------------|----------|---------|---------|--------------|
| **LLAMA** | `huggyllama/llama-7b` | ~6.7B | ✅ | ✅ | ❌ | ❌ |
| **GPT2** | `openai-community/gpt2` | ~117M | ✅ | ✅ | ❌ | ✅ |
| **BERT** | `google-bert/bert-base-uncased` | ~110M | ✅ | ✅ | ✅ | ✅ |
| **BERT-Large** | `bert-large-uncased` | ~340M | ❌ | ✅ | ❌ | ❌ |
| **DistilBERT** | `distilbert/distilbert-base-uncased` | ~66M | ✅ | ✅ | ✅ | ✅ |
| **TinyBERT** | `huawei-noah/TinyBERT_General_4L_312D` | ~14M | ✅ | ✅ | ✅ | ✅ |
| **BERT-tiny** | `prajjwal1/bert-tiny` | ~4.4M | ✅ | ✅ | ✅ | ✅ |
| **BERT-mini** | `prajjwal1/bert-mini` | ~11M | ❌ | ✅ | ✅ | ✅ |
| **BERT-small** | `prajjwal1/bert-small` | ~29M | ❌ | ✅ | ✅ | ✅ |
| **BERT-medium** | `prajjwal1/bert-medium` | ~41M | ❌ | ✅ | ✅ | ✅ |
| **MiniLM** | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` | ~33M | ✅ | ✅ | ✅ | ✅ |
| **MobileBERT** | `google/mobilebert-uncased` | ~25M | ✅ | ✅ | ✅ | ✅ |
| **ALBERT** | `albert/albert-base-v2` | ~12-18M | ✅ | ✅ | ✅ | ✅ |
| **OPT-125M** | `facebook/opt-125m` | ~125M | ✅ | ✅ | ✅ | ✅ |

## 🎯 Recommended Teacher-Student Pairs

For ECG classification, the only locked five-seed pair is **BERT -> TinyBERT**.
Other pairs below are general forecasting recommendations or unvalidated ECG
options and should not be described as classification results without a new
controlled run.

### High Performance Pairs
```bash
# BERT → TinyBERT (Most tested)
--teacher bert-base-uncased --student huawei-noah/TinyBERT_General_4L_312D

# BERT → DistilBERT (Balanced)
--teacher bert-base-uncased --student distilbert-base-uncased

# DistilBERT → BERT-tiny (Good compression)
--teacher distilbert-base-uncased --student prajjwal1/bert-tiny
```

### Experimental Pairs
```bash
# MobileBERT → BERT-tiny (Mobile optimized)
--teacher google/mobilebert-uncased --student prajjwal1/bert-tiny

# ALBERT → MiniLM (Efficient pair)
--teacher albert/albert-base-v2 --student nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large
```

## 🔧 Model Name Mappings

### Input Formats Accepted
The distillation pipeline accepts both short names and full HuggingFace model IDs:

**Short Names:**
- `bert`, `distilbert`, `tinybert`, `minilm`, `mobilebert`, `albert`

**Full HuggingFace IDs:**
- `google-bert/bert-base-uncased`
- `distilbert/distilbert-base-uncased`
- `prajjwal1/bert-tiny`
- `huawei-noah/TinyBERT_General_4L_312D`
- `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large`
- `google/mobilebert-uncased`
- `albert/albert-base-v2`
- `facebook/opt-125m`

### Example Usage
```bash
# Using short names
bash distill_pipeline.sh --teacher bert --student tinybert --patients 570 --dataset ohiot1dm

# Using full HuggingFace IDs
bash distill_pipeline.sh \
  --teacher google-bert/bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570 --dataset ohiot1dm
```

## 📊 Model Characteristics

### Teacher Models (Large, High Accuracy)
- **BERT** (110M): Best general performance
- **DistilBERT** (66M): Good balance of size/performance
- **GPT2** (117M): Decoder-only architecture
- **OPT-125M** (125M): Meta's efficient decoder

### Student Models (Small, Fast Inference)
- **BERT-tiny** (4.4M): Smallest, fastest
- **TinyBERT** (14M): Purpose-built for distillation
- **MiniLM** (33M): Good performance/size trade-off
- **MobileBERT** (25M): Mobile-optimized

## ⚠️ Important Notes

1. **LLAMA models** are only supported in the base Time-LLM model, not in distillation (too large for typical distillation scenarios)

2. **BERT variants** (`prajjwal1/bert-*`) are correctly mapped to their respective configurations in all scripts

3. **Model configurations** are automatically set based on the model name, including:
   - Layer count (`llm_layers`)
   - Hidden dimensions (`llm_dim`)
   - Model comments for tracking

4. **Filename sanitization** is applied automatically for model names with forward slashes

5. **Task-specific heads are required**. Classification is not enabled by
  changing a forecasting loss. The ECG path uses waveform patch embeddings,
  optional RR fusion, pooled backbone states, and a two- or five-class head.

6. **Checkpoint compatibility is task-specific**. Forecasting checkpoints do
  not load into the ECG classifier. Frozen ECG checkpoints store task modules
  and reconstruct the configured pretrained backbone.

7. **Chronos remains forecasting-only** in this repository. The implemented
  ECG classifier uses the Time-LLM backbone family listed above.

## 🔍 Verification

To verify model support, you can test the mappings:

```bash
cd /path/to/fair-teacher-fair-student
python3 -c "
from distillation.scripts.train_teachers import TeacherTrainer
teacher = TeacherTrainer()
print('Supported teacher models:', list(teacher.teacher_models.keys()))
"
```

This ensures consistent model support across all components of the distillation pipeline.