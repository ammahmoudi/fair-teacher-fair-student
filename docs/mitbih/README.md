# MIT-BIH ECG Documentation

This folder contains the design, implementation, fairness, tuning, and final
reporting guidance for the Time-LLM MIT-BIH classification extension.

## Start Here

- `ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`: completed protocol, final
  binary-ectopy results, limitations, and source artifacts.
- `FAIRNESS_PLAN.md`: fairness definitions, support gates, and evaluation plan.
- `TIMELLM_CLASSIFICATION_VARIANT.md`: classifier architecture and label-space
  decisions.
- `TIMELLM_DISTILLATION_ROADMAP.md`: ECG teacher/student/KD design and completed
  execution status.
- `TIMELLM_IMPLEMENTATION_PLAN.md`: implementation boundaries and module map.
- `ECG_APPROACHES.md`: comparison of waveform forecasting and beat-centered
  classification.

## Final Result Artifacts

- AAMI-5 analysis:
  `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/MULTISEED_ANALYSIS.md`
- Binary-ectopy analysis:
  `experiments/mitbih_binary_ectopy_five_seed/MULTISEED_ANALYSIS.md`
- Canonical three-task comparison:
  `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`

AAMI-5 is the primary ECG benchmark. Binary ectopy (N versus S/V/F, Q
excluded) is a reportable secondary stress test. Locked test results must not
be used for post-hoc method selection or retuning.
