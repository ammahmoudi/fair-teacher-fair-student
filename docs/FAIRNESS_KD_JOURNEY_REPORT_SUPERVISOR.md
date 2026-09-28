# Fairness in Knowledge Distillation for Time-LLM Blood Glucose Forecasting

## Executive Summary

This report presents a polished supervisor-ready version of the fairness investigation around knowledge distillation (KD) for Time-LLM blood glucose (BG) forecasting in Type 1 Diabetes Mellitus (T1DM). Prior work showed that Time-LLM can forecast CGM-based BG values and that KD can compress a larger Time-LLM BERT teacher into a smaller TinyBERT student for edge deployment.

The fairness follow-up found a clinically important issue: standard KD worsened hypoglycemia detection fairness. The main disparity appeared in gender Equal Opportunity (EO) Gap, which measures whether true hypoglycemia events are detected at similar rates across groups. Baseline KD reached EO_raw = 0.2171, which is above the 0.20 threshold used here for a critical safety concern.

A broad mitigation sweep tested teacher-side fixes, transfer-side fixes, student-objective fixes, and output calibration. Most in-training fairness methods did not sufficiently reduce the raw EO Gap. The strongest and most consistent method was a learned per-group calibration head (O2), which reduced EO_raw to 0.1189 in the main run and reduced the mean EO_raw by about 45% across five seeds while keeping RMSE competitive.

## 1. Study Context

The baseline forecasting study examined LLM-based time-series forecasting for diabetes management. It targets settings where Continuous Glucose Monitoring (CGM) and automated insulin delivery require accurate near-future BG prediction.

The baseline forecasting contributions included adapting Time-LLM and Chronos to CGM forecasting, evaluating models on OhioT1DM and D1NAMO, improving RMSE/MAE over baselines, and using KD to compress Time-LLM for edge deployment.

This report extends the KD part with fairness analysis. Compression is useful only if the compressed model preserves clinically important behavior across demographic groups. In this study, the key clinical event is hypoglycemia detection.

## 2. Forecasting Task

The task is multistep CGM forecasting. Given a fixed-length history of BG values, the model predicts future BG values.

In the baseline setup:

- CGM sampling interval: 5 minutes.
- Input context: 30 minutes, equal to 6 historical readings.
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
- $w$ is the input window length.
- $h$ is the forecast horizon length.
- $\hat{y}_t$ is the predicted future BG window.

For fairness evaluation, hypoglycemia is defined as BG <= 70 mg/dL.

## 3. Dataset and Evaluation Setup

The fairness experiments use OhioT1DM:

- 12 adults with T1DM.
- CGM readings every 5 minutes.
- Around 8 weeks of data.
- Metadata includes gender, age group, pump model, sensor band, and cohort.

The analysis uses an all-patients pipeline. Instead of training one model per patient, all 12 patients are combined into one shared training dataset while preserving each patient's original train/test split. This enables subgroup fairness comparison because one shared model is evaluated across all groups.

Combined files:

| File | Rows | Meaning |
|---|---:|---|
| `data/ohiot1dm/all_patients_combined/all_patients_training.csv` | 134,790 | Training rows from all 12 patients |
| `data/ohiot1dm/all_patients_combined/all_patients_testing.csv` | 31,743 | Testing rows from all 12 patients |
| `data/ohiot1dm/all_patients_combined/all_patients_complete.csv` | 166,533 | Train and test rows combined |

The split is not a random row split across the full dataset. It preserves the existing OhioT1DM train/test separation per patient, then concatenates across patients. This avoids mixing test rows into training.

## 4. Models and KD Objective

Main models:

| Role | Model | Meaning |
|---|---|---|
| Teacher | Time-LLM BERT / BERT-base-style backbone | Full-size source model |
| Student baseline | Time-LLM TinyBERT trained on ground truth only | Small model without KD |
| Distilled student | Same TinyBERT trained with ground truth and teacher predictions | Compressed KD model |

Important terminology:

- Student baseline means a trained TinyBERT model trained directly on ground-truth BG values.
- Baseline KD means the standard distilled TinyBERT student with no fairness intervention.
- The student baseline is required because it shows whether KD itself improves or worsens fairness.

The standard KD objective is:

$$
\mathcal{L}_{KD} = \alpha \mathcal{L}_{GT} + \beta \mathcal{L}_T
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
- $\alpha$: ground-truth supervision weight.
- $\beta$: teacher-imitation weight.

The key question is whether this KD process changes clinical event detection fairness compared with the trained student baseline.

## 5. Windowed Outputs and Metric Calculation

Time-LLM outputs forecast windows. With sliding windows, one physical timestamp can receive multiple predictions from overlapping forecast windows. The main result table uses flattened point-level evaluation.

Flattened evaluation represents each output element as:

$$
\{(\hat{y}_{r,k}, y_{r,k}, A_r)\}_{r,k}
$$

where:

- $r$ is the forecast-window row index.
- $k$ is the forecast step inside the output window.
- $A_r$ is the patient or group label for that row.

Metrics are then computed directly on the flattened prediction points. This is the mode used for the main EO_raw and patient-holdout EO_cal comparisons.

The framework also supports timeline reconstruction by averaging overlapping predictions:

$$
\bar{\hat{g}}_t = \frac{1}{|W_t|}\sum_{(r,k) \in W_t} \hat{y}_{r,k}
$$

where $W_t$ is the set of forecast-window outputs covering timestamp $t$. This report keeps the main comparison in flattened mode for consistency with the completed experiment artifacts.

## 6. Fairness Metrics

Let:

- $A$ be a protected or subgroup attribute, e.g. gender.
- $Y$ be the true binary clinical event.
- $\hat{Y}$ be the predicted binary clinical event.
- For hypoglycemia, $Y=1$ means true BG <= 70 mg/dL.
- For EO_raw, $\hat{Y}=1$ means predicted BG <= 70 mg/dL.

### 6.1 RMSE

Group RMSE is:

$$
\text{RMSE}_g = \sqrt{\frac{1}{N_g}\sum_{n:A_n=g}(y_n-\hat{y}_n)^2}
$$

RMSE measures forecasting accuracy, but it does not directly measure whether clinical events are detected equally across groups.

### 6.2 True Positive Rate

True Positive Rate (TPR), also called sensitivity or recall, is:

$$
\text{TPR}_g = \frac{TP_g}{TP_g+FN_g}=P(\hat{Y}=1 \mid Y=1,A=g)
$$

For hypoglycemia:

- $TP_g$ is the number of true hypoglycemia points in group $g$ that were predicted as hypoglycemia.
- $FN_g$ is the number of true hypoglycemia points in group $g$ that were missed.

High TPR means the model catches true hypoglycemia events.

### 6.3 Equal Opportunity Gap

EO Gap compares TPR across groups:

$$
\text{EO Gap}=\max_{i,j}|\text{TPR}_i-\text{TPR}_j|
$$

For gender:

$$
\text{EO Gap}=|\text{TPR}_{Male}-\text{TPR}_{Female}|
$$

Interpretation used in this report:

| EO Gap | Rating |
|---:|---|
| < 0.05 | Excellent |
| 0.05 to 0.10 | Good |
| 0.10 to 0.20 | Concerning |
| > 0.20 | Critical safety issue |

### 6.4 EO_raw and EO_cal

EO_raw uses the same clinical threshold for every group:

$$
\hat{Y}=\mathbb{1}[\hat{y}\le70]
$$

EO_cal uses leakage-free patient-holdout calibration with group-specific thresholds:

$$
\hat{Y}=\mathbb{1}[\hat{y}\le\tau_A]
$$

where $\tau_A$ is fitted on calibration patients and evaluated on disjoint held-out patients.

EO_raw is the stricter measure because it evaluates the model directly against the clinical threshold without threshold adjustment.

## 7. Main Fairness Problem

The core issue is unequal hypoglycemia detection, not only different RMSE. In the all-patients Time-LLM BERT -> TinyBERT KD setting:

1. The trained TinyBERT student baseline already has nonzero gender EO Gap.
2. Standard KD makes the gender EO Gap worse.
3. The distilled student reaches EO_raw > 0.20, which is a critical safety issue.
4. The gap is specifically about TPR: true hypoglycemia events are detected at different rates across groups.

Initial diagnosis:

| Feature | Student baseline EO Gap | Baseline KD EO Gap | Change |
|---|---:|---:|---:|
| Age | 0.163 | 0.207 | +27.0% worse |
| Gender | 0.195 | 0.217 | +11.3% worse |
| Pump type | 0.082 | 0.106 | +29.3% worse |

Gender became the main fairness target because it was clinically important, consistently problematic, and linked to a real male/female hypoglycemia prevalence imbalance.

## 8. Mitigation Families

The interventions were organized by where they act:

| Family | Question |
|---|---|
| Teacher-side | Can the teacher be made fairer before KD? |
| Transfer-side | Can the teacher signal be adjusted before the student copies it? |
| Student objective | Can fairness constraints or losses improve the student during training? |
| Output calibration | Can final numeric predictions be corrected near the clinical threshold? |

All percentage changes use Baseline KD EO_raw = 0.2171:

$$
\%\Delta = 100\times\frac{\text{EO}_{method}-\text{EO}_{baseline}}{\text{EO}_{baseline}}
$$

Negative values mean fairness improved.

## 9. Method Results and Formulations

### 9.1 Baseline KD

Standard KD uses the normal KD objective with no fairness loss, no sampling fix, and no calibration head.

| Model | RMSE | EO_raw | EO_cal |
|---|---:|---:|---:|
| Baseline KD | 23.361 | 0.2171 | 0.0766 |

This is the main problem model.

### 9.2 T1: Fair Teacher Retraining

T1 retrains the teacher with fairness-aware sampling before distillation. Conceptually, samples are weighted inversely to group frequency:

$$
p(n) \propto \frac{1}{N_{A_n}}
$$

where $N_{A_n}$ is the number of samples in group $A_n$.

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| Fair teacher itself | 22.564 | 0.1925 | 0.0478 | -11.3% |
| Distilled from fair teacher | 23.828 | 0.2089 | 0.0579 | -3.8% |

The teacher improves, but the student inherits only part of the fairness gain.

### 9.3 T2: Per-Group Teachers

T2 trains separate male and female teachers, then routes each training sample to the matching teacher during KD:

$$
\hat{y}_T^{(n)} = f_{T,A_n}(x^{(n)})
$$

$$
\mathcal{L}=\alpha\|\hat{y}_S-y\|^2+\beta\|\hat{y}_S-\hat{y}_{T,A_n}\|^2
$$

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T2 per-group teachers | 22.956 | 0.2197 | 0.0732 | +1.2% worse |

Separate teachers appear to learn and transmit group-specific base rates, making the EO Gap worse.

### 9.4 K1: Calibrated Soft Labels

K1 shifts teacher targets before KD using a group-specific offset:

$$
\tilde{y}_T = \hat{y}_T + \delta_A
$$

$$
\mathcal{L}=\alpha\|\hat{y}_S-y\|^2+\beta\|\hat{y}_S-\tilde{y}_T\|^2
$$

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| K1 only | 25.677 | 0.2326 | 0.0706 | +7.1% worse |
| T1 + K1 | 22.654 | 0.1981 | 0.0719 | -8.8% |

Static soft-label shifting is unstable. It helps somewhat when combined with a fair teacher, but not enough.

### 9.5 K3: Feature Alignment

K3 aligns male and female hidden representations using a CORAL-style covariance penalty:

$$
\mathcal{L}_{K3}=\|C_{Male}-C_{Female}\|_F^2
$$

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda_{align}\mathcal{L}_{K3}
$$

where $C_g$ is the covariance or correlation matrix of standardized hidden features for group $g$.

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + K3, w=100 | 22.496 | 0.1974 | 0.0417 | -9.1% |
| T1 + K3, w=200 | 22.438 | 0.1919 | 0.0536 | -11.6% |

K3 gives the best RMSE, but EO_raw remains in the concerning range and does not approach O2.

### 9.6 K4: Selective KD Replay

K4 upweights the KD loss for fairness-critical windows, especially female hypoglycemia windows:

$$
w_n = \begin{cases}
4, & A_n=Female \text{ and } y_n \le 70 \\
1, & \text{otherwise}
\end{cases}
$$

$$
\mathcal{L}=\alpha\mathcal{L}_{GT}+\beta\frac{1}{N}\sum_n w_n\|\hat{y}^{(n)}_S-\hat{y}^{(n)}_T\|^2
$$

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + K4 | 23.413 | 0.2056 | 0.0645 | -5.3% |

The loss engages, but raw EO remains critical.

### 9.7 Equalized Odds / Hypoglycemia TPR Loss

This method adds a differentiable penalty for TPR disparity:

$$
\widetilde{TPR}_g = \frac{\sum_{n:A_n=g}\mathbb{1}[y_n\le70]s(\hat{y}_n)}{\sum_{n:A_n=g}\mathbb{1}[y_n\le70]+\epsilon}
$$

$$
\mathcal{L}_{EO}=|\widetilde{TPR}_{Male}-\widetilde{TPR}_{Female}|
$$

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda_{EO}\mathcal{L}_{EO}
$$

where $s(\hat{y})$ is a differentiable soft hypoglycemia indicator.

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| HypoglycemiaTPR v1 | 23.672 | 0.2194 | 0.0629 | +1.1% worse |
| HypoglycemiaTPR v2 | 24.426 | 0.2300 | 0.0764 | +5.9% worse |

Direct in-training EO penalties did not reduce thresholded raw EO Gap.

### 9.8 O1: Projected Dual-Ascent EO Constraint

O1 treats EO as a constrained optimization problem:

$$
g(\theta)=|TPR_{Male}-TPR_{Female}|-\epsilon \le 0
$$

$$
\mathcal{L}=\mathcal{L}_{KD}+\lambda\max(0,g(\theta))
$$

$$
\lambda \leftarrow \max(0, \lambda + \eta g(\theta))
$$

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O1 | 23.141 | 0.2135 | 0.0656 | -1.7% |

The optimization mechanism is valid, but raw EO stays critical.

### 9.9 O2: Learned Per-Group Calibration Head

O2 adds a small group-aware affine output correction to the student. It is trained jointly with distillation and applied during inference:

$$
\hat{y}_{final}=W_A\hat{y}_S+b_A
$$

Training uses the calibrated output in both loss terms:

$$
\mathcal{L}=\alpha\|\hat{y}_{final}-y\|^2+\beta\|\hat{y}_{final}-\hat{y}_T\|^2
$$

where:

- $A$ is the group label used by the calibration head.
- $W_A$ is the learned group-specific scale.
- $b_A$ is the learned group-specific bias.
- $\hat{y}_{final}$ is the prediction used for training, saving, inference, and fairness evaluation.

O2 is different from EO_cal. EO_cal changes only the decision threshold during evaluation. O2 changes the actual predicted BG values before the clinical threshold is applied.

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O2 | 22.645 | 0.1189 | 0.0286 | -45.2% |

O2 is the strongest method because the fairness problem appears at the final thresholded regression output. It directly corrects systematic prediction behavior near the hypoglycemia boundary.

### 9.10 O3: Adversarial Group Erasure

O3 adds a discriminator that tries to predict gender from student hidden states. Gradient reversal trains the student to make hidden states less predictive of gender:

$$
\mathcal{L}_{adv}=CE(D(h_S), A)
$$

$$
\mathcal{L}=\mathcal{L}_{KD}-\lambda_{adv}\mathcal{L}_{adv}
$$

| Model | RMSE | EO_raw | EO_cal | EO_raw change |
|---|---:|---:|---:|---:|
| T1 + O3 | 23.048 | 0.2118 | 0.0557 | -2.4% |

The adversarial term stayed near ln(2), meaning the discriminator was close to chance. Gender erasure worked technically, but EO Gap remained critical. This is strong evidence that the disparity is not mainly caused by removable gender information in the hidden representation.

## 10. Full Result Table

Baseline for percentage change: Baseline KD EO_raw = 0.2171.

| Model | RMSE | EO_raw | EO_cal | EO_raw change vs Baseline KD | Assessment |
|---|---:|---:|---:|---:|---|
| Teacher baseline (BERT) | 24.162 | 0.2302 | 0.0761 | +6.0% worse | Critical |
| Teacher fair sampling (T1 teacher) | 22.564 | 0.1925 | 0.0478 | -11.3% | Concerning |
| Student baseline, no KD | 21.989 | 0.1948 | 0.0534 | -10.3% | Concerning |
| Baseline KD, no fairness | 23.361 | 0.2171 | 0.0766 | 0.0% | Critical |
| HypoglycemiaTPR loss v1 | 23.672 | 0.2194 | 0.0629 | +1.1% worse | Critical |
| HypoglycemiaTPR loss v2 | 24.426 | 0.2300 | 0.0764 | +5.9% worse | Critical |
| Oversampling only | 24.918 | 0.2326 | 0.0752 | +7.1% worse | Critical |
| Oversampling + HypoglycemiaTPR loss | 24.141 | 0.2225 | 0.0650 | +2.5% worse | Critical |
| Distilled from fair teacher (T1) | 23.828 | 0.2089 | 0.0579 | -3.8% | Critical |
| T1 + O1 constraint | 23.141 | 0.2135 | 0.0656 | -1.7% | Critical |
| T1 + K1 calibrated soft labels | 22.654 | 0.1981 | 0.0719 | -8.8% | Concerning |
| T1 + O2 calibration head | 22.645 | 0.1189 | 0.0286 | -45.2% | Best |
| T1 + K3 feature alignment, w=100 | 22.496 | 0.1974 | 0.0417 | -9.1% | Concerning |
| T1 + K3 feature alignment, w=200 | 22.438 | 0.1919 | 0.0536 | -11.6% | Concerning |
| T1 + K4 selective KD replay | 23.413 | 0.2056 | 0.0645 | -5.3% | Critical |
| T1 + O3 adversarial erasure | 23.048 | 0.2118 | 0.0557 | -2.4% | Critical |
| T2 per-group teachers | 22.956 | 0.2197 | 0.0732 | +1.2% worse | Critical |
| K1 only | 25.677 | 0.2326 | 0.0706 | +7.1% worse | Critical |

Key comparison:

- Student baseline: EO_raw = 0.1948.
- Baseline KD: EO_raw = 0.2171.
- O2 calibration head: EO_raw = 0.1189.

Therefore, baseline KD worsens fairness relative to the trained non-distilled student, while O2 improves beyond both.

## 11. Multi-Seed Robustness

The strongest methods were repeated over five seeds: 831363, 809906, 427368, 238822, and 247659.

| Method | RMSE | EO_raw | EO_cal | EO_raw change vs Baseline KD mean |
|---|---:|---:|---:|---:|
| Baseline KD | 23.528 +/- 0.270 | 0.217 +/- 0.006 | 0.067 +/- 0.008 | 0.0% |
| T1 fair teacher distillation | 22.845 +/- 0.331 | 0.201 +/- 0.008 | 0.057 +/- 0.006 | -7.4% |
| T1 + O2 | 22.942 +/- 0.667 | 0.118 +/- 0.033 | 0.045 +/- 0.004 | -45.6% |

O2 EO_raw by seed:

```text
[0.115, 0.155, 0.153, 0.072, 0.094]
```

O2 beats baseline KD on EO_raw in all five seeds. The improvement size varies, but the direction is stable.

### 11.1 Cross-Domain ECG Stress Test

MIT-BIH ECG provides a cross-domain stress test of the fairness-aware KD
framework. It is not an additional BG forecasting cohort, and absolute utility
or EO values are not compared across tasks.

| Task | Best reportable direction | Utility delta vs KD | EO delta vs KD | EO wins |
|---|---|---:|---:|---:|
| BG forecasting | T1+O2 | RMSE -0.5864 | -0.0995 | 5/5 |
| ECG AAMI-5 | T1 | macro-F1 -0.0041 | -0.0191 | 3/5 |
| ECG binary ectopy | O2 | macro-F1 +0.0127 | -0.0357 | 3/5 |
| ECG binary ectopy | T1+O2 | macro-F1 +0.0264 | -0.0248 | 4/5 |

The binary locked test supports the portability of output-level correction:
T1+O2 improves macro-F1 in 5/5 seeds and ectopy EO in 4/5. This is supportive,
not confirmatory, because Baseline KD remains strongest on the five-seed
validation averages, the fairness deltas are not statistically conclusive, and
one high-gap Baseline KD seed influences the mean improvement. AAMI-5 remains
mixed, with persistent S-class failure. BG O2 calibrates continuous forecasts;
ECG O2 is a jointly trained group-conditional affine logit head, so this is a
conceptual transfer rather than an identical estimator.

## 12. Interpretation

The result is not simply that the teacher contains bias and the student copies it. The evidence points to subgroup event prevalence and thresholded regression behavior:

1. T1 improves the teacher, but the student inherits only part of the gain.
2. T2 worsens the gap because per-group teachers learn group-specific base rates.
3. K3 improves RMSE but does not reduce EO_raw enough.
4. O3 removes gender information from hidden states, but EO Gap remains critical.
5. O2 works because it changes final predictions relative to the hypoglycemia threshold.

The key interpretation is:

> The raw hypoglycemia EO Gap is driven mainly by subgroup event prevalence and thresholded regression behavior, not only by removable gender information in hidden representations.

This explains why hiding the group, reweighting samples, or aligning hidden states is insufficient. The disparity appears at the final numeric output relative to a clinical threshold; therefore, an output-level correction works best.

## 13. Contributions

1. Shows that Time-LLM KD compression can worsen clinical fairness even when average forecasting accuracy remains acceptable.
2. Defines an all-patients fairness evaluation pipeline for OhioT1DM.
3. Evaluates EO_raw and leakage-free EO_cal for hypoglycemia detection.
4. Tests teacher-side, transfer-side, student-objective, and output-calibration interventions.
5. Shows that common in-training fairness fixes do not reliably solve the raw EO Gap.
6. Shows that adversarial group erasure does not solve EO Gap, so representation bias is not the main driver.
7. Validates learned per-group calibration as the strongest fix in this grid.

## 14. Limitations

- The main fairness analysis focuses on OhioT1DM and gender EO Gap.
- OhioT1DM has only 12 patients, so subgroup estimates can be noisy.
- MIT-BIH is a different classification task, not an independent BG cohort, and cannot resolve the limited OhioT1DM patient sample.
- ECG transfer is endpoint- and seed-dependent; all ECG seeds evaluate the same fixed test records.
- O2 uses group labels at inference, which may raise deployment and privacy questions.
- O2 corrects output behavior but does not remove underlying clinical prevalence imbalance.
- Counterfactual augmentation was not run because physiologically valid CGM generation is outside the scope of this study.
- Further validation on larger datasets is needed before clinical deployment.

## 15. Conclusion

Prior work showed that Time-LLM can forecast BG accurately and that KD can compress it for edge deployment. This fairness follow-up shows that compression must also be audited with clinical event-level metrics.

Standard BERT -> TinyBERT KD worsened hypoglycemia EO Gap relative to the trained non-distilled student baseline. Most generic fairness interventions did not solve the raw gap. The strongest solution was O2, a learned per-group calibration head that directly adjusts the final predicted BG values before hypoglycemia classification.

For clinical KD in BG forecasting, the recommended reporting set is: teacher baseline, trained student baseline, baseline KD, EO_raw, EO_cal, and percent change. Accuracy alone is insufficient.

## Source Artifacts

- BG five-seed results: `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/multiseed_robustness_results.csv`
- AAMI-5 five-seed analysis: `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/MULTISEED_ANALYSIS.md`
- Binary-ectopy five-seed analysis: `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`
- Canonical three-task comparison: `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`
- Binary protocol and split: `experiments/mitbih_binary_ectopy_five_seed/protocol_manifest.json` and `experiments/mitbih_binary_ectopy_five_seed/data_split_manifest.json`
