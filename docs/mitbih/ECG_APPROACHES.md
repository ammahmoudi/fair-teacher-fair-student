# MIT-BIH ECG Modeling Approaches

## Goal

Document the two practical ways to use the MIT-BIH Arrhythmia dataset in this repository.

The two approaches are:

1. **Approach 1 — ECG waveform forecasting**
2. **Approach 2 — Beat-centered ECG classification**

Approach 2 is the better choice for clinically meaningful fairness evaluation.
Approach 1 is still useful because it matches the current Time-LLM forecasting pipeline more directly.

## Implementation Update — 2026-07-15

Approach 2 was implemented with AAMI-5 as the primary endpoint and binary
ectopy (N versus S/V/F, Q excluded) as a locked secondary endpoint. Both use
record-disjoint splits and five-seed teacher/student/KD comparisons. The
binary result includes previous/next RR features and is summarized in
`experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`.
It supports task-specific fairness-aware KD evaluation but is not an external
BG forecasting replication.

---

## 1. Quick Summary

| Approach | Raw sample unit | Input window | Output window / target | Model type | Closest to the baseline forecasting setup? | Fairness is checked on | Main problems |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. ECG waveform forecasting | continuous ECG timeline | past ECG samples, e.g. 256 points | future ECG samples, e.g. next 32 / 64 / 128 points | forecasting | **Yes** | predictive system that forecasts future ECG waveform | strong pipeline reuse, but weak direct clinical meaning for arrhythmia fairness |
| 2. Beat-centered ECG classification | annotated heartbeat | fixed beat-centered segment, e.g. 256 samples around one beat | AAMI-5 class, or locked N-vs-S/V/F secondary label | classification | **No, not directly** | class-wise or ectopy-recall subgroup gaps | clinically meaningful fairness, but not the same task family as Time-LLM forecasting |

### Quick interpretation

- **Approach 1** is the closest match to what was done in the baseline forecasting framework: predict future values first, then run fairness checks on that predictive system.
- **Approach 2** is not the same modeling setup. It is a direct classification task built from ECG beat windows.
- So if the question is "what is most like OhioT1DM + Time-LLM?", the answer is **Approach 1**.
- If the question is "what is the better MIT-BIH fairness study?", the answer is **Approach 2**.

---

## 2. How MIT-BIH ECG Data Looks

Each record has:

- a continuous ECG waveform
- usually one or two leads
- sampling rate `360 Hz`
- beat annotations in the `.atr` file
- metadata in the `.hea` file

Example header:

```text
100 2 360 650000
100.dat 212 200 11 1024 995 -22131 0 MLII
100.dat 212 200 11 1024 1011 20052 0 V5
# 69 M 1085 1629 x1
# Aldomet, Inderal
```

This means:

- record ID = `100`
- number of channels = `2`
- sampling rate = `360 Hz`
- total length is about `650000` samples
- lead names = `MLII`, `V5`
- age = `69`
- sex = `M`

Conceptually, the raw signal looks like this:

```text
sample_index, lead_1_value, lead_2_value
0,           x1,           y1
1,           x2,           y2
2,           x3,           y3
...
```

And the annotations look like this conceptually:

```text
record_id, sample_index, symbol
100,       370,          N
100,       662,          V
100,       941,          N
...
```

---

## 3. How Current Time-LLM Input Looks

The current forecasting pipeline expects time-series rows such as:

```csv
item_id,timestamp,_value
540,2020-01-01 00:00:00,145.0
540,2020-01-01 00:05:00,147.0
540,2020-01-01 00:10:00,142.0
```

### Core idea

- one time series per `item_id`
- one scalar target value per timestamp
- model input = past window
- model output = future window

That matches OhioT1DM naturally:

- past glucose values → future glucose values

### Important framing relative to the baseline forecasting framework

In the original baseline forecasting setup, the pipeline was:

1. input past BG values
2. predict future BG values
3. derive fairness metrics from the predictive system outputs

So fairness was **not** checked on a direct classifier. It was checked on a forecasting system.

That means:

- **Approach 1** matches the original framing much better
- **Approach 2** changes the task family from forecasting to classification

This is the key reason to keep both approaches documented.

---

## 4. Approach 1 — ECG Waveform Forecasting

### 4.1 Idea

Convert the ECG waveform into the same row-based scalar time-series format used by the current Time-LLM pipeline.

Then train the model to do:

- past ECG samples → future ECG samples

This is the closest analogue to the current OhioT1DM setup.

This is also the closest match to the original fairness workflow:

- train a predictive system on past values
- predict future values
- evaluate subgroup fairness on the outputs of that predictive system

---

### 4.2 Data conversion

Choose one ECG lead, preferably:

1. `MLII` if available
2. otherwise the first lead in the record

Then convert the signal to rows:

```csv
item_id,timestamp,_value
100,0.000000,0.15
100,0.002778,0.17
100,0.005556,0.14
100,0.008333,0.11
```

Where:

- `item_id` = record ID
- `timestamp` = `sample_index / 360`
- `_value` = ECG amplitude from the selected lead

So a record becomes a single continuous scalar time series.

---

### 4.3 Transformation steps

#### Approach 1 input source

- raw `.dat` waveform
- `.hea` metadata

#### Approach 1 steps

1. read waveform
2. choose one lead
3. create timestamps from sample index
4. normalize signal
5. save as `item_id,timestamp,_value`

#### Approach 1 recommended normalization

- per-record robust normalization
- median centering
- MAD scaling
- fallback to mean/std if needed

#### Optional output schema

```csv
item_id,timestamp,_value,sex,age,record_id
100,0.000000,0.15,M,69,100
100,0.002778,0.17,M,69,100
```

The current Time-LLM loader may only need:

```csv
item_id,timestamp,_value
```

Group metadata can be joined later for fairness analysis.

---

### 4.4 Windowing in forecasting mode

Time-LLM would build windows like:

- input window = past ECG points
- prediction window = future ECG points

Example ECG forecasting configuration:

| Parameter | Example value |
| --- | --- |
| sequence length | 256 |
| context length | 256 |
| prediction length | 64 |
| stride | 1 or dataset default |

Possible forecasting settings:

| Setting | Input samples | Predicted samples | Notes |
| --- | ---: | ---: | --- |
| short horizon | 256 | 32 | easier forecast |
| medium horizon | 256 | 64 | good first baseline |
| longer horizon | 256 | 128 | harder, more unstable |

Because ECG is sampled at `360 Hz`:

- `32` samples ≈ `0.089 s`
- `64` samples ≈ `0.178 s`
- `128` samples ≈ `0.356 s`

---

### 4.5 What fairness means in Approach 1

This approach measures:

- fairness of **ECG waveform forecasting**

It does **not** directly measure:

- fairness of arrhythmia detection

#### Easy fairness metrics in Approach 1

- RMSE by group
- MAE by group
- fairness ratio by group
- group-wise forecast error plots

#### Harder fairness metrics in Approach 1

TPR and EO Gap are not automatic here.

To define TPR, we would need to convert future windows into event labels.

Possible event definition:

- a future window is positive if it contains at least one abnormal annotated beat

Then we would need a rule for mapping predicted future ECG waveform to predicted event.
That requires a second stage such as:

- abnormality detector on predicted waveform
- surrogate threshold on waveform morphology
- beat detector + classifier on predicted waveform

That makes Approach 1 less clean for clinical fairness.

#### Threshold policy in Approach 1

There is **no single direct clinical amplitude threshold** on raw forecasted ECG waveform that behaves like the BG threshold `< 70`.

So thresholding in Approach 1 must be done on a **derived event definition**, not directly on voltage.

Recommended threshold policies for Approach 1:

1. **Forecasting-only policy**
   - no event threshold
   - fairness is evaluated with RMSE, MAE, fairness ratio
   - this is the cleanest policy if the goal is only waveform forecasting fairness

2. **Window event threshold: at least 1 abnormal beat**
   - true future window is positive if it contains `>= 1` abnormal beat
   - predicted future window is positive if a downstream detector/classifier finds `>= 1` abnormal beat in the forecasted waveform
   - this is the simplest clinically meaningful event policy for forecasting mode

3. **Window event threshold: at least k abnormal beats**
   - future window is positive if it contains `>= 2` abnormal beats, or another fixed `k`
   - useful when single-beat positivity is too noisy

4. **Abnormal burden threshold**
   - future window is positive if abnormal beat fraction is above a threshold such as `>= 20%`
   - more stable in dense abnormal windows, but harder to explain

Main problem with all event thresholds in Approach 1:

- they require a **second-stage event detector** on top of the waveform forecaster
- fairness is then measured on a derived system, not directly on the forecasting output

---

### 4.6 Pros of Approach 1

- closest reuse of current Time-LLM infrastructure
- closest analogue to OhioT1DM forecasting
- little conceptual change in windowing pipeline
- good for proving the framework can run on another time series domain

---

### 4.7 Cons of Approach 1

- fairness target is waveform forecasting, not arrhythmia detection
- TPR / EO Gap need an extra event-definition layer
- clinical interpretation is weaker
- predicted waveform quality may not map cleanly to beat abnormality detection

---

### 4.8 When to use Approach 1

Use it when the goal is:

- quick reuse of Time-LLM forecasting pipeline
- direct comparison with OhioT1DM-style setup
- fairness of continuous ECG forecasting
- low engineering overhead first experiment

---

## 5. Approach 2 — Beat-Centered ECG Classification

### 5.1 Idea

Use MIT-BIH as it is naturally intended:

- extract a fixed ECG window around each annotated beat
- predict whether the beat is normal or abnormal

This makes the task:

- beat-centered classification

This is the better fit for meaningful fairness analysis on MIT-BIH.

But this is an important change in task definition:

- it is **not** native Time-LLM forecasting anymore
- it is a **classification** pipeline
- the fairness check is on a classifier, not on a forecaster

So Approach 2 is better clinically, but less faithful to the original baseline forecasting setup.

---

### 5.2 Data conversion

For each annotated beat:

1. take the beat sample index from `.atr`
2. extract a fixed-length ECG window around it
3. assign label from annotation symbol
4. attach metadata group label from the record

Example conceptual sample:

```text
sample_id: 100_00023
record_id: 100
sex: M
age_group: 70+
window: [x0, x1, x2, ..., x255]
label: 1
symbol: V
```

---

### 5.3 Recommended window sizes

Because this is beat-centered morphology classification, use symmetric windows.

Primary default:

- **256 samples total**
- `128` before the beat
- `128` after the beat

Window sweep:

| Setting | Samples | Seconds | Meaning |
| --- | ---: | ---: | --- |
| small | 180 | 0.50 s | compact morphology |
| default | 256 | 0.71 s | recommended first baseline |
| medium | 300 | 0.83 s | more context |
| large | 360 | 1.00 s | widest local context |

---

### 5.4 Transformation steps

#### Approach 2 input source

- `.dat` waveform
- `.hea` metadata
- `.atr` beat annotations

#### Approach 2 steps

1. read waveform
2. choose one lead
3. normalize per record
4. read beat annotations
5. map symbols to class labels
6. extract beat-centered fixed window
7. pad if needed near boundaries
8. store label and subgroup metadata

#### Approach 2 output representation

This can be stored as:

##### Option A — wide sample table

```text
sample_id,record_id,sex,age_group,label,x0,x1,...,x255
```

##### Option B — array dataset

```text
X shape = [num_beats, window_size, 1]
y shape = [num_beats]
group labels shape = [num_beats]
```

Option B is the cleaner implementation.

---

### 5.5 What fairness means in Approach 2

This approach measures:

- fairness of **abnormal beat detection**

This is clinically meaningful.

#### Positive event definition

- positive = abnormal beat

#### Metrics

- TPR by group
- EO Gap by group
- DP Gap by group
- FVO by group
- F1 / AUROC by group

#### Why this is better

TPR is direct here:

```text
TPR = correctly detected abnormal beats / all true abnormal beats
```

No extra conversion layer is needed.

#### Threshold policy in Approach 2

Approach 2 is a classifier, so thresholding is direct.

Recommended threshold policies for Approach 2:

1. **Default probability threshold**
   - predict abnormal if `p(abnormal) >= 0.50`

2. **Validation-optimized threshold**
   - choose threshold on validation set to maximize F1
   - use this as the main operating threshold if class imbalance is strong

3. **Safety-oriented threshold**
   - choose threshold to achieve target recall, for example recall `>= 0.90`
   - useful if the supervisor wants a safety-first operating point

This makes TPR, EO Gap, and DP Gap much easier to define and defend.

---

### 5.6 Pros of Approach 2

- clinically meaningful MIT-BIH task
- direct abnormal-vs-normal interpretation
- TPR and EO Gap are natural
- easier fairness claims
- aligns with beat annotations directly

---

### 5.7 Cons of Approach 2

- requires a classification dataset path, not the current forecasting CSV path
- less direct reuse of current Time-LLM input format
- needs a new dataset loader and possibly a new model head

---

### 5.8 When to use Approach 2

Use it when the goal is:

- meaningful fairness testing on MIT-BIH
- abnormal beat detection fairness
- subgroup TPR / EO analysis
- clinically interpretable results

---

## 6. Side-by-Side Comparison

| Aspect | Approach 1: Forecasting | Approach 2: Beat Classification |
| --- | --- | --- |
| raw input | continuous ECG signal | continuous ECG + beat annotations |
| converted input | `item_id,timestamp,_value` rows | fixed beat-centered windows |
| target | future ECG waveform | normal / abnormal beat label |
| input window meaning | past signal context | local morphology around one beat |
| output meaning | future signal continuation | direct clinical beat label |
| easiest reuse of current code | yes | partial |
| closest to OhioT1DM | yes | no |
| same fairness pattern as the baseline forecasting framework | yes: forecast first, fairness second | no: direct classification fairness |
| input window | e.g. past 256 samples | e.g. centered 256-sample beat window |
| output window | e.g. next 64 samples | no output window, single class label |
| direct TPR meaning | weak / indirect | strong / direct |
| direct EO Gap meaning | weak / indirect | strong / direct |
| threshold policy | derived event threshold after forecasting, or no event threshold at all | direct class-probability threshold |
| best clinical fairness interpretation | no | yes |
| recommended for first fairness paper result | maybe | yes |

### Practical reading of the comparison

- If you want **the same logic as the baseline forecasting framework**, use **Approach 1**.
- If you want **the best MIT-BIH fairness story**, use **Approach 2**.
- If you want both, use Approach 1 as the pipeline-transfer baseline and Approach 2 as the main ECG fairness result.

### Threshold comparison in one sentence

- **Approach 1**: threshold must be built on a derived event after waveform forecasting.
- **Approach 2**: threshold is directly on abnormal-beat probability.

---

## 7. Relation to OhioT1DM

### 7.1 Why Approach 1 feels similar

OhioT1DM does:

- past glucose → future glucose

Approach 1 does:

- past ECG → future ECG

So the structure is the same:

- scalar time series
- sliding windows
- forecasting target
- group-wise error fairness

That is why Approach 1 is useful.

---

### 7.2 Why Approach 2 is still better for MIT-BIH

OhioT1DM forecasting naturally supports event thresholds like:

- predicted glucose < 70

That gives direct hypo detection fairness.

MIT-BIH waveform forecasting does not naturally produce:

- abnormal beat label

Approach 2 fixes that by making the target itself the clinical event of interest.

### 7.3 Direct answer to the Time-LLM question

Time-LLM is a **forecasting** model family in this repository.

So:

- **Approach 1** fits Time-LLM naturally
- **Approach 2** does **not** fit Time-LLM in the current form naturally

Approach 2 would require one of these changes:

1. use Time-LLM only as a feature extractor / encoder and add a classification head
2. build a separate ECG classifier outside the standard Time-LLM forecasting path
3. recast ECG windows into another sequence-learning setup that is no longer the same as the current BG forecasting pipeline

So when we say Approach 2 is better, we mean:

- better for the **dataset’s clinical task**
- not better as a **direct reuse of Time-LLM forecasting code**

---

## 8. Recommended project strategy

Use both approaches, but in this order.

### Stage A — main MIT-BIH fairness result

Start with **Approach 2**.

Reason:

- strongest fairness interpretation
- direct TPR and EO Gap
- best use of MIT-BIH annotations

But note:

- this is a classification fairness result, not a strict forecasting-pipeline fairness result

### Stage B — pipeline-transfer experiment

Then optionally run **Approach 1**.

Reason:

- shows Time-LLM forecasting pipeline can also operate on ECG
- provides a direct analogue to OhioT1DM
- demonstrates broader framework reuse

This is the experiment that answers:

- can the existing forecasting-plus-fairness pipeline transfer from BG to ECG?

---

## 9. Final recommendation

### Recommended main approach

#### Approach 2 — Beat-centered ECG classification

Use this for:

- primary fairness report
- sex and age fairness analysis
- EO Gap / DP Gap / FVO reporting

This is the best **clinical MIT-BIH fairness** result.

### Recommended supporting approach

#### Approach 1 — ECG waveform forecasting

Use this for:

- infrastructure transfer demo
- comparison to OhioT1DM-style forecasting
- supplementary experiment if needed

This is the best **forecasting-style pipeline transfer** result.

---

## 10. Practical rule

If the question is:

### Can we feed ECG into current Time-LLM format?

Yes. Use Approach 1.

### Can we make a meaningful MIT-BIH fairness claim?

Yes. Better with Approach 2.

### Which one should be primary?

Approach 2.

### Should we still keep Approach 1 documented?

Yes, because it may still be useful for baseline reuse and repo consistency.
