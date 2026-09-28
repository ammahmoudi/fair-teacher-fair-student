# Accelerate Checkpoint Hotfix

## Symptom

Training completes all epochs, then fails while saving `checkpoint.pth` with:

```text
deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist
```

The failure occurs because Accelerate attempts to import DeepSpeed inside
`unwrap_model()` even though this is a one-process GPU run. The model itself is
already directly serializable in that mode.

## Apply on Sara's Machine

From the project that produced the failure:

```bash
cd ~/fair-teacher-fair-student_20260620_103613
unzip -o /path/to/LLM-TIME-accelerate-checkpoint-hotfix-20260729.zip -d .
source venv/bin/activate
python -m py_compile utils/time_llm_utils.py llms/time_llm.py llms/student_llm.py
```

No CUDA toolkit installation, `CUDA_HOME` setting, or DeepSpeed rebuild is
needed for this single-GPU suite.

## Resume the Reference Suite

The failed Teacher seed `831363` completed its epoch loop but did not save its
checkpoint, so rerun the same launcher after applying the hotfix:

```bash
export TOKENIZERS_PARALLELISM=false
nohup env PYTHONUNBUFFERED=1 bash scripts/pipelines/run_bg_teacher_student_multiseed.sh \
  > bg_teacher_student_multiseed.log 2>&1 &
echo $!
```

The runner will retrain that incomplete Teacher seed once, then continue with
its per-patient inference and the remaining Teacher/Student seeds. Completed
checkpoints and inference trees are skipped automatically.