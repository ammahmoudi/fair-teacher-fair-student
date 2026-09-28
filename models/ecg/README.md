# Time-LLM ECG Classifier

This directory contains the task-specific MIT-BIH beat classifier. It extends
the Time-LLM patch/backbone pattern without changing the original BG
forecasting models.

## Architecture

`TimeLLMEcgClassifier` applies:

1. per-channel normalization;
2. beat-window patch embedding;
3. projection into the configured pretrained backbone hidden size;
4. frozen or optionally trainable transformer backbone processing through
   `inputs_embeds`;
5. mean, last-token, or center-token pooling;
6. optional previous/next RR-interval projection and fusion;
7. layer normalization, dropout, and a configurable categorical head.

The head size comes from `num_classes`; it is not fixed to five classes.

## Label Modes

| Mode | Classes | Notes |
| --- | --- | --- |
| `aami5` | N, S, V, F, Q | Primary ECG benchmark |
| `binary_ectopy` | N vs S/V/F | Q excluded before training and evaluation |
| `binary_non_n` | N vs non-N | Includes every non-N AAMI class as positive |

## Backbones

The classifier explicitly constructs BERT, DistilBERT, GPT-2, TinyBERT,
MiniLM, MobileBERT, ALBERT, BERT-tiny, OPT-125M, and LLaMA backbones. The
locked five-seed evidence uses BERT as Teacher and TinyBERT as Student. Other
backbones are engineering support, not validated classification results.

## Training and Distillation

- Standalone wrapper: `llms/time_llm_ecg.py`
- Classification KD wrapper: `distillation/core/ecg_classification_wrapper.py`
- Standalone config generator: `scripts/time_llm/config_generator_mitbih.py`
- KD config generator: `scripts/time_llm/config_generator_mitbih_distillation.py`
- Production pipeline: `scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh`

Standalone training uses cross-entropy. When a weighted sampler is active,
ordinary CE is used; otherwise inverse-frequency class weights may be applied.
Classification KD combines supervised CE with temperature-scaled KL teacher
matching.

## Checkpoints

Backbones are frozen by default. A frozen checkpoint stores only the ECG task
modules and reconstructs the configured pretrained backbone at load time.
Unfrozen runs save the complete model state. Forecasting and ECG checkpoints
are not interchangeable.

## Outputs

Prediction CSVs preserve record ID, beat location, original AAMI class,
selected label mode, RR metadata, subgroup attributes, labels, predictions,
and per-class probabilities. Report utility with accuracy, macro-F1,
weighted-F1, and class recall; report fairness only for subgroup/class cells
that pass support and effective-recall qualification.