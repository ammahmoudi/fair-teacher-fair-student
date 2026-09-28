# Fair Teacher, Fair Student

Standalone implementation of fairness-aware knowledge distillation for
clinical time-series models. The repository contains the complete training,
evaluation, and analysis components required to run the experiments.

The implementation studies two complementary interventions:

- **Fair Teacher (T1 / EBTD):** modifies teacher training exposure for rare,
  safety-critical events.
- **Fair Student (O2 / GCOA):** calibrates group-conditional student outputs
  during knowledge distillation.

The primary task is blood-glucose forecasting on OhioT1DM. A secondary
MIT-BIH ECG classification task evaluates whether the fairness behavior
transfers across clinical time-series settings.

## Repository layout

| Path | Purpose |
| --- | --- |
| `fairness/` | Fairness losses, metrics, analyzers, calibration, and sampling |
| `distillation/` | Teacher/student training and task-specific KD wrappers |
| `scripts/pipelines/` | End-to-end BG and ECG experiment entry points |
| `scripts/fairness/` | Multi-seed aggregation and fairness comparisons |
| `scripts/clinical_metrics/` | Clinical figure and metric generation |
| `models/`, `llms/` | Time-LLM teacher/student and ECG model implementations |
| `data_processing/` | BG and ECG dataset loaders and preprocessing |
| `efficiency_toolkit/` | Parameter, memory, and latency reporting |
| `docs/` | Experiment protocols and implementation notes |
| `fairness_article/` | Private manuscript submodule for authorized collaborators |

Datasets, checkpoints, generated results, and manuscript sources are not
stored in the public code history.

## Setup

```bash
git clone https://github.com/ammahmoudi/fair-teacher-fair-student.git
cd fair-teacher-fair-student

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The data directories are intentionally ignored. Follow the dataset-specific
guides in `docs/` and keep raw or derived clinical data outside version
control.

## Main experiment entry points

Blood-glucose teacher/student multi-seed evaluation:

```bash
bash scripts/pipelines/run_bg_teacher_student_multiseed.sh --help
```

Fair-teacher training:

```bash
bash scripts/pipelines/run_fair_teacher_experiment.sh --help
```

MIT-BIH fairness-aware distillation:

```bash
bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh --help
```

Locked binary-ectopy five-seed protocol:

```bash
bash scripts/pipelines/run_mitbih_binary_ectopy_five_seed.sh --help
```

Start with these guides:

- `docs/BG_TEACHER_STUDENT_MULTISEED_GUIDE.md`
- `docs/BG_O2_TRANSFER_GUIDE.md`
- `docs/FAIRNESS_METRICS.md`
- `docs/mitbih/README.md`
- `docs/mitbih/FAIRNESS_PLAN.md`

## Private manuscript

The paper is maintained in the private `fairness_article` GitHub repository
and linked here as a submodule. A normal clone of this code repository does
not download the manuscript. Authorized collaborators can initialize it with:

```bash
git submodule update --init fairness_article
```

GitHub access to the private manuscript repository is required.

## Acknowledgements

This work builds on Time-LLM and Chronos. Consult the corresponding source
files and documentation for their citations and license terms.

## License

MIT. See `LICENSE`.
