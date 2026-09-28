# grid_search_distillation.py
import itertools
import os
import torch
import torch.optim as optim
from distillation_trainer import DistillationTrainer
from utils.time_llm_utils import EarlyStopping, adjust_learning_rate
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs

def run_distillation_experiment(
    teacher,
    student_fn,
    dataloader,
    device,
    accelerator,
    alpha,
    beta,
    train_epochs,
    output_dir,
    llm_settings
):
    student = student_fn()

    optimizer = optim.Adam(student.parameters(), lr=llm_settings["learning_rate"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
        epochs=train_epochs,
        max_lr=llm_settings["learning_rate"],
    )

    early_stopping = EarlyStopping(accelerator=accelerator, patience=llm_settings["patience"], verbose=True, save_mode=False)

    dataloader, student, optimizer, scheduler = accelerator.prepare(
        dataloader, student, optimizer, scheduler
    )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        accelerator=accelerator,
        scheduler=scheduler,
        early_stopping=early_stopping,
        alpha=alpha,
        beta=beta,
        train_epochs=train_epochs,
    )

    loss_history = trainer.train()

    # Save loss history for analysis
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"loss_alpha{alpha}_beta{beta}.txt")
    with open(log_path, "w") as f:
        for epoch, loss in enumerate(loss_history):
            f.write(f"Epoch {epoch+1}, Loss: {loss:.7f}\n")

def grid_search_distillation(
    teacher,
    student_fn,
    dataloader,
    device,
    alphas,
    betas,
    train_epochs,
    output_dir,
    llm_settings
):
    grid = list(itertools.product(alphas, betas))

    print(f"Running grid search with {len(grid)} combinations...")

    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./configs/ds_config_zero2.json")
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    teacher.to(accelerator.device)

    for alpha, beta in grid:
        print(f"\n--- alpha={alpha}, beta={beta} ---")

        run_distillation_experiment(
            teacher,
            student_fn,
            dataloader,
            accelerator.device,
            accelerator,
            alpha,
            beta,
            train_epochs,
            output_dir,
            llm_settings
        )

if __name__ == "__main__":
    import gin
    import random
    import numpy as np
    import torch
    import os
    import logging

    from models.time_llm import Model as TeacherModel
    from models.time_llm import Model as StudentModel
    from data_processing.data_loader import TimeLLMDataHandler
    from utils.tools import smart_load_state_dict


    @gin.configurable
    def run(log_dir="./logs", llm_settings=None, data_settings=None):

        # Set random seed
        if "seed" in llm_settings:
            seed = llm_settings["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        # Set dtype globally
        if llm_settings["torch_dtype"] == "bfloat16":
            torch.set_default_dtype(torch.bfloat16)
        elif llm_settings["torch_dtype"] == "float32":
            torch.set_default_dtype(torch.float32)
        elif llm_settings["torch_dtype"] == "float16":
            torch.set_default_dtype(torch.float16)


        os.makedirs(log_dir, exist_ok=True)

        # Teacher
        teacher = TeacherModel(llm_settings)
        teacher_ckpt_path = llm_settings.get("teacher_checkpoint_path", None)
        if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
            smart_load_state_dict(teacher, teacher_ckpt_path, device="cpu")
            print(f"✅ Loaded fine-tuned teacher model from: {teacher_ckpt_path}")
        else:
            print("⚠️ No teacher checkpoint provided or file not found. Using fresh teacher model.")

        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()

        # Student factory
        def student_fn():
            student_cfg = llm_settings.copy()
            student_cfg["llm_model"] = llm_settings.get("student_model", "DistilBERT")
            student_cfg["llm_layers"] = llm_settings.get("student_layers", 6)
            student_cfg["llm_dim"] = llm_settings.get("student_dim", 768)
            student_cfg["d_ff"] = llm_settings.get("student_d_ff", 32)
            student = TeacherModel(student_cfg)
            return student

        # Data loader
        data_loader = TimeLLMDataHandler(settings={
            "input_features": data_settings["input_features"],
            "labels": data_settings["labels"],
            "preprocessing_method": data_settings["preprocessing_method"],
            "preprocess_input_features": data_settings["preprocess_input_features"],
            "preprocess_label": data_settings["preprocess_label"],
            "frequency": data_settings["frequency"],
            "num_workers": llm_settings["num_workers"],
            "sequence_length": llm_settings["sequence_length"],
            "context_length": llm_settings["context_length"],
            "prediction_length": llm_settings["prediction_length"],
            "percent": data_settings["percent"],
            "val_split": data_settings["val_split"],
        })

        _, train_loader = data_loader.load_from_csv(
            data_settings["path_to_train_data"],
            batch_size=llm_settings["train_batch_size"],
            split="train"
        )

        # Grid params
        alphas = [ 0.5]
        betas = [0.7]
        # No KL weights or temperatures needed for simplified distillation

        grid_search_distillation(
            teacher,
            student_fn,
            train_loader,
            device="cuda",
            alphas=alphas,
            betas=betas,
            train_epochs=20,
            # train_epochs=llm_settings["train_epochs"],
            output_dir=os.path.join(log_dir, "distillation_grid_logs"),
            llm_settings=llm_settings
        )

    from absl import app, flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("config_path", "configs/dist/config_distil_584_train.gin", "Path to gin config file.")

    def main(argv):
        
        print(f"✅ Loaded config: {FLAGS.config_path}")
        gin.parse_config_file(FLAGS.config_path)
        run()

    app.run(main)