import logging
import os
import torch
import time
import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim, nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils.result_saver import save_results_and_generate_plots
from utils.time_llm_utils import EarlyStopping, adjust_learning_rate, model_for_checkpoint, vali
from utils.file_utils import load_txt_content
from utils.timefeatures import decode_manual_time_features, decode_time_features
from models.layers.student_model import Model as StudentModel
from llms.ts_llm import TimeSeriesLLM


class StudentLLM(TimeSeriesLLM):
    def __init__(self, settings, data_settings, log_dir="./logs", name="student_llm"):
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing StudentLLM model...")

        super(StudentLLM, self).__init__(name=name)

        self._llm_settings = settings
        self._data_settings = data_settings
        self._log_dir = log_dir

        # Load prompt content
        self._llm_settings["content"] = load_txt_content(self._data_settings["prompt_path"])

        # Model and accelerator setup
        self.llm_model = StudentModel(settings).float()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./configs/ds_config_zero2.json")
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

        self.fitted_scaler = None
        self._setup_external_loggers()

    def _setup_external_loggers(self):
        loggers_to_propagate = ["accelerate", "deepspeed", "transformers", "torch.distributed", "logging"]
        for logger_name in loggers_to_propagate:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.propagate = True
            for handler in logging.getLogger().handlers:
                logger.addHandler(handler)

    def load_model(self, checkpoint_path, is_inference=True):
        self.logger.info(f"Loading model from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.accelerator.device)
            self.llm_model.load_state_dict(state_dict)
            self.llm_model.to(self.accelerator.device).to(self._llm_settings["torch_dtype"])
            self.logger.info("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def train(self, train_data, train_loader, val_loader=None):
        if hasattr(train_data, "scaler"):
            self.fitted_scaler = train_data.scaler

        self.logger.info("Starting Training...")
        train_steps = len(train_loader)

        model_optim = optim.Adam(self.llm_model.parameters(), lr=self._llm_settings["learning_rate"])
        scheduler = self._initialize_scheduler(model_optim, train_steps)
        checkpoint_dir = os.path.join(self._log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self._llm_settings["patience"], verbose=True, save_mode=False)

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, val_loader, self.llm_model, model_optim, scheduler = self.accelerator.prepare(train_loader, val_loader, self.llm_model, model_optim, scheduler)

        train_loss_l, val_loss_l = [], []

        for epoch in range(self._llm_settings["train_epochs"]):
            train_loss = self._run_epoch(train_loader, criterion, model_optim, scheduler, is_training=True)
            train_loss_l.append(train_loss)

            if val_loader and len(val_loader) > 0:
                val_loss, val_mae_loss = vali(self._llm_settings, self.accelerator, self.llm_model, val_loader, criterion, mae_metric)
                val_loss_l.append(val_loss)
                self.logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}")
                early_stopping(val_loss, self.llm_model, checkpoint_dir)

                if early_stopping.early_stop:
                    self.logger.info("Early stopping triggered.")
                    break

            self._adjust_scheduler(scheduler, epoch, model_optim)

        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.logger.info(f"Saving model checkpoint at {final_checkpoint_path}.")
        model_to_save = model_for_checkpoint(self.accelerator, self.llm_model)
        torch.save(model_to_save.state_dict(), final_checkpoint_path)

        return final_checkpoint_path, train_loss_l, val_loss_l

    def predict(self, test_loader, output_dir):
        """
        Predict the output for the test data, save predictions to files,
        and generate plots for analysis.

        :param test_loader: DataLoader for test data.
        :param output_dir: Directory to save the predictions, formatted results, and plots.
        :return: Tuple of (predictions, targets) as numpy arrays for evaluation.
        """
        self.llm_model.eval()
        (
            predictions,
            targets,
            inputs,
            input_timestamps,
            output_timestamps,
            attention_maps,
        ) = ([], [], [], [], [], [])

        if len(test_loader) == 0:
            self.logger.info("Warning: The test loader is empty. No data to predict.")
            return None, None

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.accelerator.device)
                batch_y = batch_y.float().to(self.accelerator.device)

                # Prepare decoder input
                dec_inp = (
                    torch.zeros_like(
                        batch_y[:, -self._llm_settings["prediction_length"] :, :]
                    )
                    .float()
                    .to(self.accelerator.device)
                )
                dec_inp = torch.cat(
                    [batch_y[:, : self._llm_settings["context_length"], :], dec_inp],
                    dim=1,
                )

                outputs = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self._llm_settings["features"] == "MS" else 0
                outputs = outputs[:, -self._llm_settings["prediction_length"] :, f_dim:]

                predictions.append(outputs.cpu().numpy())
                # print(outputs.cpu().numpy().shape)
                targets.append(
                    batch_y[:, -self._llm_settings["prediction_length"] :, f_dim:]
                    .cpu()
                    .numpy()
                )
                inputs.append(batch_x.cpu().numpy())
                input_timestamps.append(batch_x_mark.cpu().numpy())
                output_timestamps.append(batch_y_mark.cpu().numpy())

        if not predictions:
            logging.error(
                "No predictions were made. Ensure the test loader contains data."
            )
            return None, None

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        input_timestamps = np.concatenate(input_timestamps, axis=0)
        output_timestamps = np.concatenate(output_timestamps, axis=0)

        # Decode x timestamps
        decoded_x_timestamps = []
        if self._llm_settings["timeenc"] == 0:
            for batch in input_timestamps:
                decoded_x_timestamps.extend(
                    decode_manual_time_features(
                        batch, freq=self._data_settings["frequency"]
                    )
                )
        else:
            decoded_x_timestamps = decode_time_features(
                input_timestamps.T, freq=self._data_settings["frequency"]
            )

        # Decode y timestamps
        decoded_y_timestamps = []
        if self._llm_settings["timeenc"] == 0:
            for batch in output_timestamps:
                decoded_y_timestamps.extend(
                    decode_manual_time_features(
                        batch, freq=self._data_settings["frequency"]
                    )
                )
        else:
            decoded_y_timestamps = decode_time_features(
                output_timestamps.T, freq=self._data_settings["frequency"]
            )

        grouped_x_timestamps = [
            decoded_x_timestamps[
                i
                * self._llm_settings["context_length"] : (i + 1)
                * self._llm_settings["context_length"]
            ]
            for i in range(len(inputs))
        ]
        grouped_y_timestamps = [
            decoded_y_timestamps[
                i
                * self._llm_settings["prediction_length"] : (i + 1)
                * self._llm_settings["prediction_length"]
            ]
            for i in range(len(targets))
        ]

        grouped_x_timestamps = [
            ", ".join(
                ts.strftime("%d-%m-%Y %H:%M:%S") for ts in group if ts is not None
            )
            for group in grouped_x_timestamps
        ]
        grouped_y_timestamps = [
            ", ".join(
                ts.strftime("%d-%m-%Y %H:%M:%S") for ts in group if ts is not None
            )
            for group in grouped_y_timestamps
        ]

        save_results_and_generate_plots(
            output_dir,
            predictions,
            targets,
            inputs,
            grouped_x_timestamps,
            grouped_y_timestamps,
        )

        return predictions, targets

    def evaluate(self, predictions, targets, metrics=["rmse", "mae", "mape"]):
        return super().evaluate(predictions, targets, metrics)

    def _initialize_scheduler(self, optimizer, train_steps):
        if self._llm_settings["lradj"] == "COS":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self._llm_settings["pct_start"],
            epochs=self._llm_settings["train_epochs"],
            max_lr=self._llm_settings["learning_rate"]
        )

    def _adjust_scheduler(self, scheduler, epoch, optimizer):
        if self._llm_settings["lradj"] != "TST":
            if self._llm_settings["lradj"] == "COS":
                scheduler.step()
                self.logger.info(f"Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.10f}")
            else:
                if epoch == 0:
                    self._llm_settings["learning_rate"] = optimizer.param_groups[0]["lr"]
                adjust_learning_rate(self.accelerator, optimizer, scheduler, epoch + 1, self._llm_settings, printout=True)
        else:
            self.logger.info(f"Epoch {epoch+1} | LR (TST): {scheduler.get_last_lr()[0]:.10f}")

    def _run_epoch(self, loader, criterion, optimizer, scheduler, is_training=True):
        self.llm_model.train() if is_training else self.llm_model.eval()
        epoch_loss = []
        iter_count = 0
        time_now = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(loader)):
            batch_x = batch_x.float().to(self.accelerator.device)
            batch_y = batch_y.float().to(self.accelerator.device)
            batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
            batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -self._llm_settings["prediction_length"]:, :]).float().to(self.accelerator.device)
            dec_inp = torch.cat([batch_y[:, :self._llm_settings["context_length"], :], dec_inp], dim=1)

            if is_training:
                optimizer.zero_grad()

            outputs = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self._llm_settings["features"] == "MS" else 0
            outputs = outputs[:, -self._llm_settings["prediction_length"]:, f_dim:]
            batch_y = batch_y[:, -self._llm_settings["prediction_length"]:, f_dim:]
            loss = criterion(outputs, batch_y)

            if is_training:
                self.accelerator.backward(loss)
                optimizer.step()
                if self._llm_settings["lradj"] == "TST":
                    adjust_learning_rate(self.accelerator, optimizer, scheduler, iter_count + 1, self._llm_settings, printout=False)

            epoch_loss.append(loss.item())
            iter_count += 1

        return np.mean(epoch_loss)
