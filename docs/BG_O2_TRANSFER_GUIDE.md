# BG O2 Transfer Guide

This transfer overlay updates an older LLM-TIME checkout so it can run the
five-seed, standalone BG O2 calibration-head ablation. It includes the O2
training/inference code, the runner, its aggregator, and the canonical 537 MB
baseline BERT teacher checkpoint.

The overlay intentionally does not include data, prior experiment results,
`.git`, or `venv`. Keep the existing OhioT1DM data in the destination checkout.

## Destination Setup

From the root of the older LLM-TIME checkout, extract the archive over the
existing project:

```bash
unzip -o LLM-TIME-bg-o2-transfer-20260727.zip -d .
```

If `unzip` is unavailable, use Python instead:

```bash
python3 -m zipfile -e LLM-TIME-bg-o2-transfer-20260727.zip .
```

Create or refresh the Python environment only when the existing one does not
already provide the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the data, bundled teacher checkpoint, Python packages, and CUDA before
starting the experiment:

```bash
source venv/bin/activate
bash scripts/pipelines/verify_bg_o2_transfer.sh
```

## Run

First inspect the five-seed plan without training:

```bash
source venv/bin/activate
DRY_RUN=1 bash scripts/pipelines/run_bg_o2_only_multiseed.sh
```

Start the actual resumable five-seed suite in the background:

```bash
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
nohup env PYTHONUNBUFFERED=1 bash scripts/pipelines/run_bg_o2_only_multiseed.sh \
  > bg_o2_only_multiseed.log 2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f bg_o2_only_multiseed.log
```

The runner uses the five canonical seeds `831363`, `809906`, `427368`,
`238822`, and `247659`. It trains only standalone O2 from the baseline BERT
teacher; it does not rerun T1, T1+O2, O3, T2, K1, K3, or K4.

If the machine reboots or the job stops, rerun the same `nohup` command. The
runner skips finished seed checkpoints and completed per-patient inference, but
an interrupted training epoch restarts that seed's training stage.

The final aggregate is written to:

```text
distillation_experiments/all_patients_pipeline/o2_only_multiseed/o2_only_multiseed_results.csv
```