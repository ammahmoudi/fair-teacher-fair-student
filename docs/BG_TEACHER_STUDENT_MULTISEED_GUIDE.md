# BG Teacher and Student Five-Seed Guide

This suite fills the two single-seed reference gaps in the BG fairness article:

1. Original BERT Teacher.
2. Independently trained BERT-tiny Student without knowledge distillation.

It does not run KD, T1, O2, T1+O2, O3, T2, K1, K3, or K4. It uses the canonical
seeds `831363`, `809906`, `427368`, `238822`, and `247659` and trains both
reference models for 10 epochs per seed before per-patient OhioT1DM inference.

## Current Checkout

Verify the machine first:

```bash
source venv/bin/activate
bash scripts/pipelines/verify_bg_teacher_student_transfer.sh
```

Inspect the exact five-seed plan without training:

```bash
DRY_RUN=1 bash scripts/pipelines/run_bg_teacher_student_multiseed.sh
```

Start the suite in the background:

```bash
export TOKENIZERS_PARALLELISM=false
nohup env PYTHONUNBUFFERED=1 bash scripts/pipelines/run_bg_teacher_student_multiseed.sh \
  > bg_teacher_student_multiseed.log 2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f bg_teacher_student_multiseed.log
```

## Older Checkout

First extract the current source overlay from
`LLM-TIME-bg-o2-transfer-20260727.zip`, then extract the small reference-suite
add-on archive over the same project root:

```bash
unzip -o LLM-TIME-bg-o2-transfer-20260727.zip -d .
unzip -o LLM-TIME-bg-teacher-student-transfer-20260729.zip -d .
source venv/bin/activate
bash scripts/pipelines/verify_bg_teacher_student_transfer.sh
```

The final aggregate is written to:

```text
distillation_experiments/all_patients_pipeline/teacher_student_multiseed/teacher_student_multiseed_results.csv
```

Rerunning the same command skips finished seed checkpoints and completed
per-patient inference. If training is interrupted before a checkpoint is saved,
that condition restarts from the beginning of that seed.