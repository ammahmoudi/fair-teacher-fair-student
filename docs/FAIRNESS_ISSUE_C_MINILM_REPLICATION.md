# Fairness Issue C: Why MiniLM Improves Fairness (Replicate and Understand)

**Priority:** MEDIUM — investigative; findings can improve all future distillation runs  
**Status:** Open  
**Related files:** `distillation/scripts/distill_students.py`, `distillation/core/distillation_trainer.py`

---

## Finding

MiniLM distillation **improves** fairness over the baseline student, while TinyBERT distillation
**worsens** it.  This is the opposite of Issue A and suggests the KD configuration parameters
matter as much as the architecture.

### Observed Numbers

| Metric | Baseline Student | TinyBERT Distilled | MiniLM Distilled |
|---|---|---|---|
| Age EO Gap change | 0 (reference) | **+0.24%** 🔴 | **−2.1%** 🟢 |
| Gender EO Gap change | 0 (reference) | **+0.37%** 🔴 | **−1.74%** 🟢 |
| Cohort EO Gap change | 0 (reference) | slight increase | **−0.69%** 🟢 |

MiniLM consistently reduces EO Gap across all demographic features while TinyBERT increases it.

---

## Key Configuration Differences

| Parameter | TinyBERT | MiniLM |
|---|---|---|
| Student model | `prajjwal1/bert-tiny` | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` |
| Alpha (α) | 0.3 | **0.5** |
| Beta (β) | 0.3 | 0.5 |
| Learning rate | 5e-4 | **1e-3** |
| Architecture | 2 layers, 128 hidden | 6 layers, 384 hidden |

Three things differ: **α**, **learning rate**, and **student architecture/capacity**.

---

## Hypotheses

**H1 — Higher α reduces bias copying**  
α=0.5 gives more weight to ground truth loss vs teacher loss (compared to α=0.3).  
More emphasis on ground truth → student corrects teacher biases rather than copying them.

**H2 — Higher learning rate enables fairer convergence**  
A higher LR (1e-3 vs 5e-4) may help the model escape biased teacher-influenced local minima
more easily, allowing it to find a solution that is more consistent across groups.

**H3 — Better student architecture**  
MiniLM has 6 layers and 384 hidden dimensions vs TinyBERT's 2 layers / 128 hidden.  
A more expressive student may be able to represent patient-specific patterns more accurately,
reducing the need to rely on group-level shortcuts.

---

## Fix Plan (Ablation Study)

The goal is to **isolate which factor causes MiniLM's fairness improvement**.  Run four
controlled experiments using TinyBERT architecture with varying parameters:

### Experiment 1 — Baseline (already done)
```
Model: prajjwal1/bert-tiny, α=0.3, lr=5e-4
```
This is the existing TinyBERT result (EO Gap increases by +0.24% / +0.37%).

### Experiment 2 — Higher α only (test H1)
```bash
python distillation/scripts/distill_students.py \
    --student_model prajjwal1/bert-tiny \
    --alpha 0.5 \
    --beta 0.5 \
    --lr 5e-4
```
If EO Gap decreases compared to Experiment 1 → H1 confirmed.

### Experiment 3 — Higher LR only (test H2)
```bash
python distillation/scripts/distill_students.py \
    --student_model prajjwal1/bert-tiny \
    --alpha 0.3 \
    --beta 0.3 \
    --lr 1e-3
```
If EO Gap decreases compared to Experiment 1 → H2 confirmed.

### Experiment 4 — Both α and LR (test combined effect)
```bash
python distillation/scripts/distill_students.py \
    --student_model prajjwal1/bert-tiny \
    --alpha 0.5 \
    --beta 0.5 \
    --lr 1e-3
```
This matches MiniLM's config but on TinyBERT architecture.
If EO Gap decreases similarly to MiniLM → H3 (architecture) is not the main factor.
If EO Gap stays high → H3 is confirmed (architecture matters).

### Experiment 5 — MiniLM with lower α (reverse test)
```bash
python distillation/scripts/distill_students.py \
    --student_model nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large \
    --alpha 0.3 \
    --beta 0.3 \
    --lr 5e-4
```
If MiniLM with TinyBERT's config loses its fairness advantage → H3 is less important.

---

## Decision Table (After Ablation)

| Outcome of Exp 2 | Outcome of Exp 4 | Conclusion | Action |
|---|---|---|---|
| EO Gap ↓ | EO Gap ↓ similar to MiniLM | H1 confirmed, architecture doesn't matter | Switch all distillation to α=0.5, β=0.5 |
| EO Gap → same | EO Gap ↓ | H1+H2 together needed | Use both higher α and LR |
| EO Gap → same | EO Gap → same | H3: architecture is the key | Switch to MiniLM for all distillation |

---

## Running the Fairness Analysis Per Experiment

After each experiment:

```bash
python fairness/scripts/apply_advanced_metrics_to_results.py
python fairness/scripts/analyze_training_impact.py
```

Compare EO Gap values in `fairness/analysis_results/` with the baseline.

---

## What to Do With the Findings

### If H1 (higher α) is confirmed:
- Update the default `--alpha` from 0.3 to 0.5 in `distillation/scripts/distill_students.py`
- Update `docs/DISTILLATION_MODEL_PAIRS.md` with the new recommended config
- Re-run all per-patient distillation experiments with α=0.5

### If H3 (architecture) is confirmed:
- Use MiniLM as the primary student for all future experiments
- Document this in `docs/DISTILLATION_MODEL_PAIRS.md`
- Consider removing TinyBERT experiments from the main pipeline

### Combined with Issue A fix:
If Issue A (`EqualizedOddsLoss`) is also implemented, run the same ablation *with the fairness
loss added* to quantify the interaction — does EO loss matter more or less depending on
α value?

---

## Expected Outcome

- Identify the single most important factor driving MiniLM's fairness benefit
- Have a clear recommendation: "use α=0.5" or "use MiniLM architecture" for all distillation
- Reduce TinyBERT EO Gap from +0.24%/+0.37% to negative (fairness improvement) without
  changing the student architecture

---

## Effort Estimate

- Setting up 4 ablation runs: ~2 hours (mostly compute wait time)
- Analysis and interpretation: ~1 hour
- Documentation update: ~30 minutes
