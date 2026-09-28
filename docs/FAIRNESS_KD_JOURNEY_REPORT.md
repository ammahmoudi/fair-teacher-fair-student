# Fairness in Knowledge Distillation for Time-LLM Blood Glucose Forecasting

## Abstract

This report summarizes the complete fairness investigation around knowledge distillation (KD) for Time-LLM blood glucose (BG) forecasting. It is self-contained: it explains the baseline forecasting context, the all-patients extension, the dataset split, the windowed output format, the fairness metrics, the exact problem, every mitigation scenario, formulas, and the final interpretation.

Prior work showed that Time-LLM and Chronos can forecast CGM-based BG values for Type 1 Diabetes Mellitus (T1DM), and that KD can compress a Time-LLM BERT teacher into a smaller TinyBERT student for edge deployment with limited accuracy loss. The follow-up fairness study found a clinical safety problem: standard KD worsened hypoglycemia detection fairness, especially gender Equal Opportunity (EO) Gap. Baseline KD reached EO_raw ≈ 0.217, a critical safety disparity. A broad intervention sweep tested teacher fixes, KD-transfer fixes, student-objective fixes, and output calibration. Most in-training fairness methods failed. The only method that consistently reduced raw and calibrated EO Gap was a learned per-group calibration head (O2), validated across five seeds.

---

## 1. Clinical Forecasting Context

The baseline forecasting study examined LLM-based time-series forecasting for diabetes management. It targets Smart and Connected Health systems where Continuous Glucose Monitoring (CGM) and automated insulin delivery need accurate near-future BG prediction.

Baseline models:

- **Time-LLM**: transforms continuous time-series patches into embeddings usable by frozen language-model backbones such as BERT, GPT-2, or LLaMA.
- **Chronos**: quantizes continuous time series into discrete tokens and forecasts through language-model-style token prediction.

Baseline contributions:

1. Adapt Time-LLM and Chronos to CGM-based BG forecasting.
2. Evaluate zero-shot and fine-tuned models on OhioT1DM and D1NAMO.
3. Show fine-tuned Time-LLM improves RMSE/MAE against strong baselines.
4. Use denoising autoencoder preprocessing for noisy/missing data.
5. Use KD to compress Time-LLM for edge deployment.

This report extends item 5 with fairness: compression is not enough if the student detects hypoglycemia unequally across demographic groups.

---

## 2. Forecasting Task

The task is multistep CGM forecasting. Given a fixed-length history of BG values, predict future BG values.

In the baseline setup:

- CGM sampling interval: 5 minutes.
- Input context: 30 minutes = 6 historical readings.
- Forecast horizons: 30 and 45 minutes.
- Output: a forecast window, not one independent scalar.

Formally:

$$
x_t = \{g_{t-w+1}, \dots, g_t\}, \quad y_t = \{g_{t+1}, \dots, g_{t+h}\}
$$

$$
\hat{y}_t = f_\theta(x_t)
$$

where:

- $g_t$ is BG at time $t$.
- $w$ is input window length.
- $h$ is forecast horizon length.
- $\hat{y}_t$ is a vector of predicted future BG values.

Because windows slide over time, the same future timestamp can appear in multiple output windows. Section 5 explains how metrics handle this.

---

## 3. Dataset and Split

### 3.1 Dataset

The fairness experiments focus on **OhioT1DM**:

- 12 adults with T1DM.
- CGM readings every 5 minutes.
- Around 8 weeks of data.
- Metadata includes gender, age group, pump model, sensor band, and cohort.

D1NAMO appears in the earlier forecasting study as external validation, but the KD fairness experiments in this report use OhioT1DM because fairness analysis needs subgroup metadata and all-patient distillation artifacts.

### 3.2 Prior Work vs All-Patients Pipeline

The earlier forecasting study mainly reports per-patient and cross-patient settings. The fairness work adds an **all-patients pipeline**.

**Per-patient pipeline**:

- Train one model per patient.
- Evaluate on that patient.
- Good for personalization.
- Weak for subgroup fairness because each model sees one patient.

**All-patients pipeline**:

- Combine all 12 OhioT1DM patients into one training dataset.
- Train one shared teacher, one shared student baseline, and one shared distilled student.
- Evaluate by per-patient inference on all 12 patients.
- Enables demographic fairness comparison because one model is tested across all groups.

Generated combined files:

| File | Rows | Meaning |
|---|---:|---|
| `data/ohiot1dm/all_patients_combined/all_patients_training.csv` | 134,790 | training rows from all 12 patients |
| `data/ohiot1dm/all_patients_combined/all_patients_testing.csv` | 31,743 | testing rows from all 12 patients |
| `data/ohiot1dm/all_patients_combined/all_patients_complete.csv` | 166,533 | train + test combined |

The split preserves the existing OhioT1DM train/test separation per patient, then concatenates across patients. It is not a random row split across all data. This avoids mixing test rows into training.

### 3.3 Fairness Calibration Split

Two EO metrics appear in results:

- **EO_raw**: EO Gap using the clinical threshold directly, usually 70 mg/dL for both groups.
- **EO_cal**: EO Gap after fitting group-specific decision thresholds with leakage-free patient-holdout calibration.

Patient-holdout calibration:

1. Split male patients into 2 folds and female patients into 2 folds.
2. For each fold, fit male/female thresholds on calibration patients only.
3. Apply thresholds to disjoint held-out patients.
4. Concatenate held-out predictions across folds.
5. Compute EO Gap on held-out calibrated predictions.

This prevents threshold tuning on the same patient data used for evaluation.

---

## 4. Model and Experiment Configuration

### 4.1 Models

Main fairness grid:

| Role | Model | Meaning |
|---|---|---|
| Teacher | Time-LLM BERT / BERT-base-style backbone | full-size model |
| Raw student / student baseline | Time-LLM TinyBERT (`prajjwal1/bert-tiny`) trained on ground truth only | small model, trained but **not distilled** |
| Distilled student | same TinyBERT student trained with ground truth + teacher predictions | compressed KD model |

Important terminology:

- **Student baseline** in this report means a trained TinyBERT student trained directly on ground-truth BG, without KD.
- It does **not** mean an untrained random student.
- **Baseline KD** means standard distilled student with no fairness intervention.

### 4.2 Pipeline Stages

All-patients KD pipeline:

1. Train teacher on `all_patients_training.csv`.
2. Train student baseline on `all_patients_training.csv` without KD.
3. Train distilled student on same training data using teacher predictions.
4. Run per-patient inference for teacher, student baseline, and distilled student on all 12 patients.
5. Flatten or reconstruct window outputs for fairness metrics.

### 4.3 Training Objective

Standard KD objective:

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{GT} + \beta \mathcal{L}_{T}
$$

where:

$$
\mathcal{L}_{GT} = \frac{1}{N}\sum_n \|\hat{y}^{(n)}_S - y^{(n)}\|_2^2
$$

$$
\mathcal{L}_{T} = \frac{1}{N}\sum_n \|\hat{y}^{(n)}_S - \hat{y}^{(n)}_T\|_2^2
$$

Symbols:

- $\hat{y}_S$: student prediction window.
- $\hat{y}_T$: teacher prediction window.
- $y$: true BG window.
- $\alpha$: weight on ground-truth supervision.
- $\beta$: weight on teacher imitation.

The baseline distillation used $\alpha=0.3$, $\beta=0.3$ from validation search. General all-patients guide defaults may show $\alpha=0.5$, $\beta=0.5$; the fairness roadmap comparison reports the completed run artifacts as listed in Section 10.

---

## 5. Windowed Outputs and Metric Calculation

Time-LLM outputs forecast windows. With sliding windows, one physical timestamp can receive multiple predictions from overlapping forecast windows.

Example:

- Window starting at time 0 predicts times 1..9.
- Window starting at time 1 predicts times 2..10.
- Timestamp 5 may therefore have predictions from several windows.

The fairness framework supports three calculation modes.

### 5.1 Simple Mode

Flatten every predicted window element into one row:

$$
\{(\hat{y}_{r,k}, y_{r,k}, A_r)\}_{r,k}
$$

where:

- $r$ is window row index.
- $k$ is forecast step inside the output window.
- $A_r$ is group label for the patient/window.

Then compute metrics directly on the flattened points. If one timestamp appears in multiple windows, each occurrence counts separately.

This mode was used by `scripts/fairness/compute_fairness_comparison.py` for main EO_raw/EO_cal comparison: it loads `inference_results_reformatted.csv`, iterates all true/pred columns, and flattens them into point rows.

### 5.2 Timeline Reconstruction Mode

For each timestamp, average all overlapping predictions first:

$$
\bar{\hat{g}}_t = \frac{1}{|W_t|}\sum_{(r,k) \in W_t} \hat{y}_{r,k}
$$

where $W_t$ is the set of window outputs covering timestamp $t$.

Then binarize reconstructed values and compute metrics once per timestamp.

### 5.3 Window Majority Mode

First binarize each window value:

$$
\hat{z}_{r,k} = \mathbb{1}[\hat{y}_{r,k} \le 70]
$$

For each timestamp, take a majority vote over overlapping binary predictions:

$$
\hat{z}_t = \mathbb{1}\left[\frac{1}{|W_t|}\sum_{(r,k)\in W_t}\hat{z}_{r,k} > 0.5\right]
$$

Then compute DP, EO, and FVO on timestamp-level binary values.

### 5.4 Which Mode Does This Report Use?

Main comparison table uses flattened point-level/simple mode for EO_raw and patient-holdout EO_cal. The fairness framework also supports timeline reconstruction and window-majority modes for sensitivity analysis. When reporting final paper numbers, mode must be stated next to the metric.

---

## 6. Fairness Metrics: Full Definitions

Let:

- $A$ be a protected/group attribute, e.g., gender.
- $a,b$ be two groups, e.g., Male and Female.
- $Y$ be binary true clinical event.
- $\hat{Y}$ be binary predicted event.
- For hypoglycemia, $Y=1$ means true BG $\le 70$ mg/dL.
- $\hat{Y}=1$ means predicted BG $\le \tau$, where $\tau=70$ for EO_raw or group-specific calibrated threshold for EO_cal.

### 6.1 RMSE and RMSE Ratio

Group RMSE:

$$
\text{RMSE}_g = \sqrt{\frac{1}{N_g}\sum_{n:A_n=g}(y_n-\hat{y}_n)^2}
$$

RMSE Ratio:

$$
\text{RMSE Ratio}=\frac{\max_g \text{RMSE}_g}{\min_g \text{RMSE}_g}
$$

Near 1.0 is fairer.

### 6.2 Demographic Parity Gap

DP Gap measures whether alert rates are equal across groups, regardless of true events:

$$
\text{DP Gap}=\max_{i,j}|P(\hat{Y}=1\mid A=i)-P(\hat{Y}=1\mid A=j)|
$$

It answers: “Does the model issue hypoglycemia alerts at similar rates across groups?”

### 6.3 TPR: True Positive Rate

TPR means True Positive Rate, also called sensitivity or recall:

$$
\text{TPR}_g=\frac{TP_g}{TP_g+FN_g}=P(\hat{Y}=1\mid Y=1,A=g)
$$

For this study:

- $TP_g$: actual hypoglycemia points for group $g$ that were predicted as hypoglycemia.
- $FN_g$: actual hypoglycemia points for group $g$ that were missed by the model.

High TPR means the model catches actual hypoglycemia events.

### 6.4 Equal Opportunity Gap

EO Gap compares TPR across groups:

$$
\text{EO Gap}=\max_{i,j}|\text{TPR}_i-\text{TPR}_j|
$$

For gender:

$$
\text{EO Gap}=|\text{TPR}_{Male}-\text{TPR}_{Female}|
$$

Interpretation:

| EO Gap | Rating |
|---:|---|
| < 0.05 | Excellent |
| 0.05--0.10 | Good |
| 0.10--0.20 | Concerning |
| > 0.20 | Critical safety issue |

### 6.5 EO_raw and EO_cal

**EO_raw**:

$$
\hat{Y}=\mathbb{1}[\hat{y}\le70]
$$

Same clinical threshold for every group.

**EO_cal**:

$$
\hat{Y}=\mathbb{1}[\hat{y}\le\tau_{A}]
$$

where $\tau_A$ is a group-specific threshold fitted on calibration patients and evaluated on held-out patients. EO_cal measures remaining disparity after leakage-free group threshold calibration.

### 6.6 Accuracy and FVO

FVO means Fairness Violation Objective. It measures the largest performance gap between any two groups.

Classification accuracy for group $i$:

$$
\text{Acc}_i = \frac{TP_i+TN_i}{TP_i+TN_i+FP_i+FN_i}
$$

FVO:

$$
\text{FVO}=\max_{i,j}|\text{Acc}_i-\text{Acc}_j|
$$

Meaning of symbols:

- $i$ and $j$ are group indices, e.g., Male/Female, 20--40/40--60/60--80, 530G/630G.
- $\text{Acc}_i$ is the event-classification accuracy for group $i$.
- For classification FVO, BG predictions are first converted to binary clinical events using hypoglycemia or hyperglycemia threshold.
- For regression FVO, the code can use an accuracy proxy: $1-\text{RMSE}_i/\text{global BG range}$.

Low FVO means model reliability is similar across groups.

---

## 7. Exact Fairness Problem Before Fixes

The core problem is not just different RMSE. The safety-critical issue is unequal hypoglycemia detection.

In the all-patients Time-LLM BERT $\rightarrow$ TinyBERT KD setting:

1. A trained TinyBERT student baseline already has nonzero gender EO Gap.
2. Standard KD makes the gender EO Gap worse.
3. The distilled student has EO_raw > 0.20, classified as a critical safety issue.
4. The gap is specifically about TPR: one gender’s actual hypoglycemia events are detected at a different rate than the other gender’s.

Initial diagnosis across features:

| Feature | Trained student baseline EO Gap | Distilled student EO Gap | Change |
|---|---:|---:|---:|
| Age | 0.163 | 0.207 | +27.0% worse |
| Gender | 0.195 | 0.217 | +11.3% worse |
| Pump type | 0.082 | 0.106 | +29.3% worse |

Gender became the main fairness target because it was clinically important, consistently problematic, and linked to a real male/female hypoglycemia prevalence imbalance.

### 7.1 Reference Models Needed for Comparison

All method comparisons should include at least:

| Reference | Why needed |
|---|---|
| Teacher baseline | source model before fairness fix |
| Trained student baseline, no KD | small model without teacher copying |
| Baseline KD, no fairness | direct problem model |
| Fair teacher only / T1 | source-side improvement baseline |
| Proposed fix | actual mitigation |

This report includes these references in the full result table.

---

## 8. Fix Scenario Families

The interventions were organized by where they act:

| Family | Question |
|---|---|
| Teacher-side | Can we make the teacher fair before KD? |
| Transfer-side | Can we change what knowledge is transferred? |
| Student objective | Can we train the student with fairness constraints? |
| Output calibration | Can we correct the final regression output near clinical threshold? |

All percentage changes below use Baseline KD EO_raw = 0.2171 unless stated otherwise.

Percentage change formula:

$$
\%\Delta = 100\times\frac{\text{EO}_{method}-\text{EO}_{baseline}}{\text{EO}_{baseline}}
$$

Negative means fairness improved.

---

## 9. Methods, Scenarios, and Formulas

### 9.1 Standard KD: Baseline Problem Model

Scenario:

- Train BERT teacher on all patients.
- Train TinyBERT student with KD.
- No fairness loss, no sampling fix, no calibration head.

Formula:

$$
\mathcal{L}=\alpha\mathcal{L}_{GT}+\beta\mathcal{L}_T
$$

Result:

| Model | RMSE | EO_raw | EO_cal |
|---|---:|---:|---:|
| Baseline KD | 23.361 | 0.2171 | 0.0766 |

This is the main baseline.

### 9.2 T1: Fair Teacher Retraining

Scenario:

- Retrain teacher with fair sampling / fairness-aware sampling on gender.
- Distill TinyBERT student from this fair teacher.

Conceptual sampling weight:

$$
p(n) \propto \frac{1}{N_{A_n}}
$$

where $N_{A_n}$ is count of samples in group $A_n$.

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change vs Baseline KD |
|---|---:|---:|---:|---:|
| Fair teacher itself | 22.564 | 0.1925 | 0.0478 | -11.3% |
| Distilled from fair teacher | 23.828 | 0.2089 | 0.0579 | -3.8% |

Interpretation: teacher improves, but student only inherits part of the fairness gain.

### 9.3 T2: Per-Group Teachers

Scenario:

- Train separate male and female teachers.
- Route each sample to teacher matching its gender.
- Distill into one student.
- This changes the **teacher signal inside KD training**: the student no longer matches one global teacher, but a group-routed teacher output for each sample.

Formula:

$$
\hat{y}_T^{(n)} = f_{T,A_n}(x^{(n)})
$$

$$
\mathcal{L}=\alpha\|\hat{y}_S-y\|^2+\beta\|\hat{y}_S-\hat{y}_{T,A_n}\|^2
$$

Symbols:

- $n$: training sample index.
- $x^{(n)}$: input CGM history window for sample $n$.
- $y^{(n)}$: ground-truth future BG window for sample $n$.
- $A_n$: group label for sample $n$; here gender, so $A_n \in \{Male,Female\}$.
- $f_{T,A_n}$: teacher model selected by group $A_n$.
- $f_{T,Male}$: teacher trained on male-patient data.
- $f_{T,Female}$: teacher trained on female-patient data.
- $\hat{y}_{T,A_n}$: output of the routed teacher for sample $n$.
- $\hat{y}_S$: student output for the same input.
- $\alpha$: ground-truth loss weight.
- $\beta$: routed-teacher imitation loss weight.

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T2 per-group teachers | 22.956 | 0.2197 | 0.0732 | +1.2% worse |

Interpretation: separate teachers learn separate group base rates and transmit them. This makes the fairness problem worse, not better.

### 9.4 T3: Counterfactual Data Augmentation

Scenario considered:

- Generate synthetic female hypoglycemia windows to reduce male/female hypoglycemia prevalence imbalance.

Not run because physiologically valid CGM generation is a separate research project. Simple interpolation may create impossible glucose traces.

### 9.5 K1: Calibrated Soft Labels

Scenario:

- Estimate per-gender teacher offsets.
- Shift teacher targets before KD.

Formula:

$$
\tilde{y}_T = \hat{y}_T + \delta_{A}
$$

$$
\mathcal{L}=\alpha\|\hat{y}_S-y\|^2+\beta\|\hat{y}_S-\tilde{y}_T\|^2
$$

Symbols:

- $\hat{y}_T$: original teacher forecast window.
- $\delta_A$: learned or estimated additive calibration offset for group $A$.
- $\tilde{y}_T$: calibrated teacher target used as soft label.
- $\hat{y}_S$: student forecast window.
- $y$: ground-truth future BG window.
- $\alpha,\beta$: same KD weights as standard KD.

Effect on KD training: K1 changes the **teacher target** in the KD loss. The student still trains with a normal KD objective, but it imitates $\tilde{y}_T$ instead of raw $\hat{y}_T$.

Results:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| K1 only | 25.677 | 0.2326 | 0.0706 | +7.1% worse |
| T1 + K1 | 22.654 | 0.1981 | 0.0719 | -8.8% |

Interpretation: static soft-label shifting is unstable. On fair teacher it helps somewhat, but not enough.

### 9.6 K2: Group-Conditional KD Temperature

Status: removed.

Reason: temperature is a softmax-logit concept for classification:

$$
p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

This Time-LLM KD model is point-valued regression. It emits BG values, not class logits. There is no meaningful KD temperature to tune.

### 9.7 K3: Feature Alignment

Scenario:

- Capture student hidden states.
- Align male/female representations with scale-invariant CORAL.

Let $H_g$ be hidden features for group $g$. Standardize by pooled per-dim standard deviation, then compute correlation/covariance matrices $C_g$.

Formula:

$$
\mathcal{L}_{K3}=\|C_{Male}-C_{Female}\|_F^2
$$

Training:

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda_{align}\mathcal{L}_{K3}
$$

Symbols:

- $H_g$: hidden-state matrix for samples from group $g$.
- $C_g$: correlation/covariance matrix computed from standardized $H_g$.
- $\|\cdot\|_F$: Frobenius norm.
- $\lambda_{align}$: feature-alignment loss weight.

Effect on KD training: K3 adds an extra representation-level penalty during KD. It does not change teacher outputs or final inference architecture.

Results:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + K3, w=100 | 22.496 | 0.1974 | 0.0417 | -9.1% |
| T1 + K3, w=200 | 22.438 | 0.1919 | 0.0536 | -11.6% |

Interpretation: best RMSE, but EO_raw remains concerning and close to student/fair-teacher range. Alignment term stayed flat near 0.169, so extra weight did not solve the detection gap.

### 9.8 K4: Selective KD Replay

Scenario:

- Upweight KD loss on minority-group female hypoglycemia windows.
- Factor = 4.
- Normalize weights to mean 1.

Formula:

$$
w_n = \begin{cases}
4, & A_n=Female \text{ and } y_n \le 70 \\
1, & \text{otherwise}
\end{cases}
$$

$$
\mathcal{L}=\alpha\mathcal{L}_{GT}+\beta\frac{1}{N}\sum_n w_n\|\hat{y}^{(n)}_S-\hat{y}^{(n)}_T\|^2
$$

Symbols:

- $w_n$: KD-loss multiplier for sample $n$.
- $A_n$: group label for sample $n$.
- $N$: number of training samples in batch or epoch aggregation.
- $\hat{y}^{(n)}_S$, $\hat{y}^{(n)}_T$: student and teacher forecast windows for sample $n$.

Effect on KD training: K4 changes only the **weight of the teacher-imitation term** for selected windows. It does not alter the teacher or model architecture.

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + K4 | 23.413 | 0.2056 | 0.0645 | -5.3% |

Interpretation: loss engaged, but raw EO remained critical.

### 9.9 EqualizedOdds / HypoglycemiaTPR Losses

Scenario:

- Add differentiable penalty for TPR disparity during training.

Soft TPR form:

$$
\widetilde{TPR}_g = \frac{\sum_{n:A_n=g} \mathbb{1}[y_n\le70]s(\hat{y}_n)}{\sum_{n:A_n=g}\mathbb{1}[y_n\le70]+\epsilon}
$$

where $s(\hat{y})$ is a differentiable soft hypoglycemia indicator.

Fairness penalty:

$$
\mathcal{L}_{EO}=|\widetilde{TPR}_{Male}-\widetilde{TPR}_{Female}|
$$

Training:

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda_{EO}\mathcal{L}_{EO}
$$

Results:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| HypoglycemiaTPR v1 | 23.672 | 0.2194 | 0.0629 | +1.1% worse |
| HypoglycemiaTPR v2 | 24.426 | 0.2300 | 0.0764 | +5.9% worse |

Interpretation: direct in-training EO penalties did not reduce thresholded raw EO Gap.

### 9.10 Oversampling

Scenario:

- Oversample fairness-critical windows.
- Test alone and combined with TPR loss.

Sampling idea:

$$
p(n) \propto w_n
$$

Results:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| Oversampling only | 24.918 | 0.2326 | 0.0752 | +7.1% worse |
| Oversampling + TPR loss | 24.141 | 0.2225 | 0.0650 | +2.5% worse |

Interpretation: more exposure to rare/minority windows did not fix event-detection parity.

### 9.11 O1: Projected Dual-Ascent EO Constraint

Scenario:

- Treat EO constraint as constrained optimization.
- Adaptive dual variable penalizes violation.

Constraint:

$$
g(\theta)=|TPR_{Male}-TPR_{Female}|-\epsilon \le 0
$$

Lagrangian-style objective:

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda\max(0,g(\theta))
$$

Dual update:

$$
\lambda \leftarrow \max(0, \lambda + \eta g(\theta))
$$

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O1 | 23.141 | 0.2135 | 0.0656 | -1.7% |

Interpretation: valid optimization mechanism, but raw EO stayed critical.

### 9.12 O2: Learned Per-Group Calibration Head

Scenario:

- Add small group-aware affine output correction to student.
- Train jointly with distillation.
- Save calibration sidecar with student checkpoint.
- Apply the same calibration head automatically at inference.
- This is not post-hoc threshold tuning; it is a learned output layer in the student prediction path.

Formula:

$$
\hat{y}_{final}=W_A\hat{y}_S+b_A
$$

Training:

$$
\mathcal{L}=\alpha\|\hat{y}_{final}-y\|^2+\beta\|\hat{y}_{final}-\hat{y}_T\|^2
$$

Symbols:

- $A$: group label used by the calibration head; in the main run, gender.
- $\hat{y}_S$: raw TinyBERT student forecast window before calibration.
- $W_A$: learned group-specific scale. For window outputs, this can be scalar shared across forecast steps or a vector/matrix matching output shape, depending on implementation.
- $b_A$: learned group-specific bias/intercept.
- $\hat{y}_{final}$: calibrated student forecast used for loss, saved predictions, and inference.
- $y$: ground-truth BG forecast window.
- $\hat{y}_T$: teacher forecast window.
- $\alpha$: ground-truth supervision weight.
- $\beta$: teacher-imitation weight.

How O2 affects KD training:

1. Student backbone produces raw output $\hat{y}_S$.
2. O2 transforms it to $\hat{y}_{final}=W_A\hat{y}_S+b_A$.
3. Both KD loss terms are computed on $\hat{y}_{final}$, not raw $\hat{y}_S$.
4. Gradients flow through the calibration head into the student backbone.
5. The calibration parameters $(W_A,b_A)$ are learned jointly with distillation.
6. At inference, predictions must pass through the saved calibration head; otherwise the O2 fairness gain is lost.

Why this is different from EO_cal:

- **O2** changes model predictions during training and inference.
- **EO_cal** only changes the decision threshold used during evaluation.
- O2 can improve EO_raw because the actual predicted BG values move relative to the fixed 70 mg/dL threshold.
- EO_cal can reduce measured gap after threshold calibration, but it does not change model outputs.

Why O2 works best:

- The fairness problem appears at the final thresholded hypoglycemia decision.
- In-training penalties tried to reshape hidden representations or sample weights, but raw event detection stayed unequal.
- O2 directly adjusts the final regression scale/bias per group, so it can correct systematic under- or over-prediction near the hypoglycemia boundary.
- This matches the diagnosis that the gap is prevalence/threshold driven, not only representation driven.

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O2 | 22.645 | 0.1189 | 0.0286 | -45.2% |

Interpretation: only method with large EO_raw and EO_cal improvement while keeping RMSE competitive. It works because it changes the final numeric prediction relative to the hypoglycemia threshold.

### 9.13 O3: Adversarial Group Erasure

Scenario:

- Add discriminator predicting gender from student hidden states.
- Use gradient reversal so student learns gender-invariant representation.

Discriminator loss:

$$
\mathcal{L}_{adv}=CE(D(h_S), A)
$$

Student objective with gradient reversal:

$$
\mathcal{L}=\mathcal{L}_{KD}-\lambda_{adv}\mathcal{L}_{adv}
$$

Result:

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O3 | 23.048 | 0.2118 | 0.0557 | -2.4% |

The adversarial term stayed near $\ln(2)\approx0.693$, meaning the discriminator was at chance. Gender erasure worked technically.

Interpretation: most informative negative result. If group erasure works but EO Gap remains, the disparity is not mainly representation bias. It is base-rate/outcome-distribution driven.

---

## 10. Full Result Table With References and Percent Change

Baseline for EO_raw change: **Baseline KD EO_raw = 0.2171**.

| Model | RMSE | EO_raw | EO_cal | EO_raw change vs Baseline KD | Raw assessment |
|---|---:|---:|---:|---:|---|
| Teacher baseline (BERT) | 24.162 | 0.2302 | 0.0761 | +6.0% worse | Critical |
| Teacher fair sampling (T1 teacher) | 22.564 | 0.1925 | 0.0478 | -11.3% | Concerning |
| **Trained student baseline, no KD** | **21.989** | **0.1948** | **0.0534** | **-10.3%** | Concerning |
| **Baseline KD, no fairness** | **23.361** | **0.2171** | **0.0766** | **0.0%** | Critical |
| HypoglycemiaTPR loss v1 | 23.672 | 0.2194 | 0.0629 | +1.1% worse | Critical |
| HypoglycemiaTPR loss v2 | 24.426 | 0.2300 | 0.0764 | +5.9% worse | Critical |
| Oversampling only | 24.918 | 0.2326 | 0.0752 | +7.1% worse | Critical |
| Oversampling + HypoglycemiaTPR loss | 24.141 | 0.2225 | 0.0650 | +2.5% worse | Critical |
| Distilled from fair teacher (T1) | 23.828 | 0.2089 | 0.0579 | -3.8% | Critical |
| T1 + O1 constraint | 23.141 | 0.2135 | 0.0656 | -1.7% | Critical |
| T1 + K1 calibrated soft labels | 22.654 | 0.1981 | 0.0719 | -8.8% | Concerning |
| **T1 + O2 calibration head** | **22.645** | **0.1189** | **0.0286** | **-45.2%** | **Moderate / best** |
| T1 + K3 feature alignment, w=100 | 22.496 | 0.1974 | 0.0417 | -9.1% | Concerning |
| T1 + K3 feature alignment, w=200 | 22.438 | 0.1919 | 0.0536 | -11.6% | Concerning |
| T1 + K4 selective KD replay | 23.413 | 0.2056 | 0.0645 | -5.3% | Critical |
| T1 + O3 adversarial erasure | 23.048 | 0.2118 | 0.0557 | -2.4% | Critical |
| T2 per-group teachers | 22.956 | 0.2197 | 0.0732 | +1.2% worse | Critical |
| K1 only | 25.677 | 0.2326 | 0.0706 | +7.1% worse | Critical |

Key comparison:

- Student baseline is trained and has EO_raw 0.1948.
- Baseline KD worsens that to 0.2171.
- O2 improves beyond both: 0.1189.

---

## 11. Multi-Seed Robustness

Repeated over five seeds: 831363, 809906, 427368, 238822, 247659.

| Method | RMSE | EO_raw | EO_cal | EO_raw change vs Baseline KD mean |
|---|---:|---:|---:|---:|
| Baseline KD | 23.528 ± 0.270 | 0.217 ± 0.006 | 0.067 ± 0.008 | 0.0% |
| T1 fair teacher distillation | 22.845 ± 0.331 | 0.201 ± 0.008 | 0.057 ± 0.006 | -7.4% |
| **T1 + O2** | 22.942 ± 0.667 | **0.118 ± 0.033** | **0.045 ± 0.004** | **-45.6%** |

O2 EO_raw per seed:

```text
[0.115, 0.155, 0.153, 0.072, 0.094]
```

O2 beats baseline KD on raw EO Gap in all five seeds. Its improvement size varies, but direction is stable.

---

### 11.1 Cross-Domain ECG Stress Test

The fairness-aware KD framework was also evaluated on MIT-BIH ECG classification
using the same five random seeds and record-disjoint train/validation/test
splits. This is a cross-domain stress test, not an additional BG forecasting
cohort. Utility and fairness magnitudes are therefore compared only within each
task.

| Task | Method | Utility | Fairness | Paired result vs Baseline KD |
|---|---|---:|---:|---|
| BG forecasting | T1+O2 | 22.942 +/- 0.667 RMSE | 0.1179 +/- 0.0325 hypoglycemia EO | lower RMSE; EO -0.0995; 5/5 EO wins |
| ECG AAMI-5 | T1 | 0.4375 +/- 0.0174 macro-F1 | 0.1259 +/- 0.0536 N/V sex EO | macro-F1 -0.0041; EO -0.0191; 3/5 EO wins |
| ECG binary ectopy | O2 | 0.7127 +/- 0.0160 macro-F1 | 0.0239 +/- 0.0211 ectopy sex EO | macro-F1 +0.0127; EO -0.0357; 3/5 EO wins |
| ECG binary ectopy | T1+O2 | 0.7265 +/- 0.0345 macro-F1 | 0.0349 +/- 0.0400 ectopy sex EO | macro-F1 +0.0264; EO -0.0248; 4/5 EO wins |

The binary-ectopy locked test is supportive: T1+O2 improves macro-F1 in all
five seeds and EO in four. However, Baseline KD remains strongest on the
five-seed validation averages, the fairness deltas are not statistically
conclusive at five seeds, and one high-gap Baseline KD seed strongly influences
the mean EO reduction. AAMI-5 is also mixed and has near-zero S recall for the
distilled variants. The defensible conclusion is therefore that output-level
fairness correction is portable enough to merit task-specific evaluation, not
that O2 universally improves every clinical task.

The implementations are conceptually aligned but not identical: BG O2
calibrates continuous glucose outputs, while ECG O2 jointly learns
group-conditional affine transformations of class logits.

---

## 12. Why Most Fixes Failed

The final interpretation is not simply “teacher bias copied into student.” The evidence points to subgroup base-rate imbalance.

Evidence:

1. **T1**: fair teacher improves, but student only partially inherits fairness.
2. **T2**: per-group teachers worsen the gap because they learn group-specific base rates.
3. **K3**: representation alignment improves RMSE but not raw EO enough.
4. **O3**: gender erasure succeeds, but EO Gap stays critical.
5. **O2**: output-level group correction works.

Conclusion:

> The raw hypoglycemia EO Gap is driven mainly by subgroup event prevalence and thresholded regression behavior, not by a removable gender signal in the hidden representation.

This explains why hiding the group, reweighting samples, or aligning hidden states is insufficient. The disparity appears at the final numeric output relative to a clinical threshold; therefore, an output-level correction works best.

---

## 13. Related Finding: Denoised Inference Fairness Collapse

Separate robustness experiments found a preprocessing fairness issue. Chronos was relatively fair on noisy data but unfair on denoised data.

| Feature | Metric | Noisy data | Denoised data | Change |
|---|---|---:|---:|---:|
| Gender | RMSE ratio | ~1.07 | 1.506 | +40% worse |
| Cohort | RMSE ratio | ~1.07 | 1.900 | +78% worse |
| Age | RMSE ratio | ~1.07 | 1.304 | +22% worse |
| Age | EO Gap | 0.157 | 0.623 | +297% worse |

Interpretation: one fixed denoising filter affects patients unequally. It can remove useful glucose variability for some groups while helping others. This supports the broader lesson: global preprocessing or training objectives can improve average accuracy while harming subgroups.

Recommended mitigation:

$$
\text{choose smoothing window per patient so } \frac{\sigma(\text{smoothed})}{\sigma(\text{original})}\approx c
$$

Then rerun subgroup fairness validation.

---

## 14. MiniLM Observation

MiniLM distillation showed a different pattern from TinyBERT. MiniLM slightly improved fairness, while TinyBERT worsened it.

| Metric | Baseline Student | TinyBERT Distilled | MiniLM Distilled |
|---|---:|---:|---:|
| Age EO Gap change | reference | +0.24% | -2.1% |
| Gender EO Gap change | reference | +0.37% | -1.74% |
| Cohort EO Gap change | reference | slight increase | -0.69% |

Possible causes:

- Higher $\alpha$ gives more ground-truth pressure.
- Higher learning rate changes convergence.
- MiniLM has more capacity than TinyBERT.

This is an open ablation thread, not the main validated solution. O2 remains the validated fix for the main all-patients TinyBERT pipeline.

---

## 15. Contributions

1. Shows that Time-LLM KD compression can worsen clinical fairness even when average accuracy remains acceptable.
2. Defines an all-patients fairness evaluation pipeline for OhioT1DM.
3. Evaluates EO_raw and leakage-free EO_cal for hypoglycemia detection.
4. Tests teacher-side, transfer-side, student-objective, and output-calibration interventions.
5. Shows broad negative results for common in-training fairness fixes.
6. Shows adversarial group erasure does not solve EO Gap, proving representation bias is not the main driver.
7. Validates learned per-group calibration head as the only strong fix in this grid.

---

## 16. Limitations

- Main fairness analysis focuses on OhioT1DM and gender EO Gap.
- OhioT1DM has only 12 patients, so subgroup estimates can be noisy.
- MIT-BIH is a different classification domain, not an independent BG forecasting cohort; it cannot resolve the small OhioT1DM patient sample.
- ECG fairness transfer is mixed across AAMI-5 and binary endpoints, and all ECG seeds reuse the same fixed test records.
- O2 uses group labels at inference, which may raise deployment and privacy questions.
- O2 corrects output behavior but does not remove underlying clinical prevalence imbalance.
- Counterfactual augmentation was not run because physiologically valid CGM generation is out of scope.
- MiniLM fairness behavior needs controlled ablation.
- Denoising fairness collapse is identified but not solved here.

---

## 17. Conclusion

Prior work showed that Time-LLM can forecast BG accurately and that KD can compress it for edge deployment. This fairness follow-up shows that compression must be audited with clinical event-level metrics. Standard BERT $\rightarrow$ TinyBERT KD worsened hypoglycemia EO Gap relative to the trained non-distilled student baseline. Most generic fairness interventions did not solve the raw gap.

The decisive evidence is O3: adversarial gender erasure worked, yet EO Gap stayed critical. Therefore, the fairness problem is not mainly caused by gender information in the representation. It is driven by subgroup event prevalence and thresholded regression behavior. O2 works because it calibrates the final output per group, directly changing event detection near the hypoglycemia threshold.

For clinical KD in BG forecasting: always report the trained student baseline, baseline KD, EO_raw, EO_cal, and percent changes. Accuracy alone is insufficient.

---

## References to Project Artifacts

- Private article source: `fairness_article/`
- All-patients guide: `docs/ALL_PATIENTS_TRAINING_GUIDE.md`
- Fairness metrics documentation: `docs/FAIRNESS_METRICS.md`
- Main fairness roadmap and results: `docs/FAIRNESS_SOLUTIONS_ROADMAP.md`
- Distillation EO issue: `docs/FAIRNESS_ISSUE_A_DISTILLATION_EO_GAP.md`
- Denoised inference issue: `docs/FAIRNESS_ISSUE_B_DENOISED_INFERENCE.md`
- MiniLM fairness observation: `docs/FAIRNESS_ISSUE_C_MINILM_REPLICATION.md`
- Metric implementation: `fairness/metrics/advanced_fairness_metrics.py`
- Fairness comparison script: `scripts/fairness/compute_fairness_comparison.py`
- Distillation trainer: `distillation/core/distillation_trainer.py`
- Fairness losses: `fairness/loss_functions/fairness_losses.py`
- Binary-ectopy five-seed report: `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`
- Three-task comparison: `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`
