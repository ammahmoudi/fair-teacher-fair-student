import io
import logging
import os
import random
from typing import Optional
import numpy as np
import yaml
import torch
from llms.ts_llm import TimeSeriesLLM

from models.chronos.src.chronos.chronos import ChronosPipeline
from utils.result_saver import generate_run_title, save_results_and_generate_plots
from peft import PeftModel


class ChronosLLM(TimeSeriesLLM):
    def __init__(
        self,
        settings={
            "model": "amazon/chronos-t5-base",
            "device_map": "cuda",
            "torch_dtype": torch.float32,
        },
        data_settings={},
        log_dir="./logs",
        name="chronos_llm"
        """
        Initialize the Chronos LLM model.

        Args:
            settings (dict): Settings for the model, including model path, device map, and torch dtype.
            log_dir (str): Directory for logging outputs.
            name (str): Name of the LLM instance.
        """,
    ):

        super(ChronosLLM, self).__init__(name=name)
        self._llm_settings = settings
        self._data_settings = data_settings
        self._log_dir = log_dir

        # Get the existing root logger from setup_logging
        self.logger = logging.getLogger()
        self.logger.info("Initializing Chronos LLM model...")

        # Capture logs from ChronosPipeline
        chronos_logger = logging.getLogger("chronos")
        chronos_logger.setLevel(logging.DEBUG)
        chronos_logger.propagate = True  # Ensure it sends logs to the root logger

        # Load the Chronos pipeline with logging
        try:
            self.llm_model = ChronosPipeline.from_pretrained(
                pretrained_model_name_or_path=settings["model"],
                device_map=settings["device_map"],
                torch_dtype=settings["torch_dtype"],
                # cache_dir=self._log_dir,
            )
            self.logger.info("Chronos LLM model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Chronos LLM model: {e}")
            raise

    def load_model(self, ckpt_path=None, lora_path: Optional[str] = None):
        """
        Load a model checkpoint with optional LoRA adapters.

        Parameters:
        -----------
        ckpt_path : str
            Path to the base model checkpoint directory.
        lora_path : Optional[str], default=None
            Optional path to a LoRA checkpoint. If None, will check inside `ckpt_path`.
        """
        if ckpt_path:
            self.logger.info(f"Loading model checkpoint from {ckpt_path}")
            

            # If lora_path is not provided, check inside ckpt_path for a LoRA checkpoint
            if lora_path is None:
                default_lora_path = ckpt_path.replace("checkpoint-final", "lora-final")
                lora_path = default_lora_path if os.path.exists(default_lora_path) else None

            # Load the model with optional LoRA support
            self.llm_model = ChronosPipeline.from_pretrained(
                pretrained_model_name_or_path=ckpt_path,
                device_map=self._llm_settings["device_map"],
                torch_dtype=self._llm_settings["torch_dtype"],
                lora_path=lora_path,
            )

            if lora_path:
                self.logger.info(f"LoRA adapters loaded from {lora_path}")
            else:
                self.logger.info("No LoRA adapters found, using checkpoint model.")
        elif lora_path:
            self.llm_model = ChronosPipeline.from_pretrained(
                pretrained_model_name_or_path=self._llm_settings["model"],
                device_map=self._llm_settings["device_map"],
                torch_dtype=self._llm_settings["torch_dtype"],
                lora_path=lora_path,
                
            )
            if lora_path:
                self.logger.info(f"LoRA adapters loaded from {lora_path} with base model {self._llm_settings['model']}")
            else:
                self.logger.info("No LoRA adapters found, using base model.")
            

    def predict(
        self,
        test_data,
        data_loader,
        settings={
            "prediction_length": 64,
            "num_samples": 100,
            "batch_size": None,
            "auto_split": False,
        },
        benchmark_efficiency=True,
    ):
        """
        Predict method to generate predictions and save inputs, targets, and predictions.

        Args:
            test_data: DataFrame containing both input features and targets.
            data_loader: Data loader of test data
            settings: Dictionary of settings for prediction.
            benchmark_efficiency: Whether to run efficiency benchmarking during prediction.

        Returns:
            llm_prediction: Numpy array of model predictions.
            efficiency_metrics: EfficiencyMetrics object if benchmarking is enabled.
        """

        logging.info("Starting prediction process...")

        # Split test data into input features and labels
        test_input_features, test_labels = (
            data_loader.split_dataframe_input_features_vs_labels(test_data)
        )
        logging.debug(
            f"Split test data into input features (shape: {test_input_features.shape}) and labels (shape: {test_labels.shape})"
        )

        # Batch the data into smaller sequences if batch size is specified
        if settings["batch_size"] is not None:
            test_input_batches = [
                test_input_features[i : i + settings["batch_size"]]
                for i in range(0, len(test_input_features), settings["batch_size"])
            ]
            test_label_batches = [
                test_labels[i : i + settings["batch_size"]]
                for i in range(0, len(test_labels), settings["batch_size"])
            ]
            logging.info(f"Data is batched with batch size: {settings['batch_size']}")
        else:
            test_input_batches = [test_input_features]
            test_label_batches = [test_labels]
            logging.info("Processing data without batching.")

        prediction_results = []
        input_results = []
        target_results = []

        # Initialize efficiency benchmarking if requested
        if benchmark_efficiency and len(test_input_batches) > 0:
            # Create sample input for benchmarking
            sample_batch = test_input_batches[0].values[:1]  # Single sample
            sample_input = torch.tensor(sample_batch)
            
            # Efficiency metrics are now calculated automatically in main.py
            logging.info("Efficiency metrics will be calculated automatically")

        for batch_idx, (input_batch, target_batch) in enumerate(
            zip(test_input_batches, test_label_batches)
        ):
            batch_inputs = input_batch.values
            batch_targets = target_batch.values

            input_results.append(batch_inputs)
            target_results.append(batch_targets)

            logging.debug(
                f"Processing batch {batch_idx+1}/{len(test_input_batches)} with input shape {batch_inputs.shape}"
            )

            if settings["prediction_length"] > 64 and settings["auto_split"]:
                logging.info("Using auto-split mode for long sequence prediction.")
                split_results = []
                predictions_remaining = settings["prediction_length"]
                num_splits = settings["prediction_length"] // 64
                num_splits = (
                    num_splits
                    if settings["prediction_length"] % 64 == 0
                    else num_splits + 1
                )

                for split_idx in range(num_splits):
                    prediction_length = min(64, predictions_remaining)

                    logging.debug(
                        f"Split {split_idx+1}/{num_splits}: Predicting {prediction_length} time steps."
                    )

                    current_llm_prediction = self.llm_model.predict(
                        context=torch.tensor(batch_inputs),
                        prediction_length=prediction_length,
                        num_samples=settings["num_samples"],
                        limit_prediction_length=True,
                    )

                    if settings["num_samples"] > 1:
                        current_llm_prediction = torch.mean(
                            current_llm_prediction, dim=1
                        )

                    split_results.append(torch.squeeze(current_llm_prediction))

                    if predictions_remaining > 64:
                        batch_inputs[:, : (batch_inputs.shape[1] - 64)] = batch_inputs[
                            :, 64:
                        ]
                        batch_inputs[:, :64] = current_llm_prediction
                        predictions_remaining -= 64
                        logging.debug(
                            f"Sliding window updated: {predictions_remaining} steps remaining."
                        )

                current_llm_prediction = torch.cat(split_results, dim=1)

            else:
                logging.info("Predicting in a single pass without auto-split.")
                current_llm_prediction = self.llm_model.predict(
                    context=torch.tensor(batch_inputs),
                    prediction_length=settings["prediction_length"],
                    num_samples=settings["num_samples"],
                    limit_prediction_length=False,
                )

                if settings["num_samples"] > 1:
                    current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                current_llm_prediction = torch.squeeze(current_llm_prediction)

            prediction_results.append(current_llm_prediction)
            logging.debug(f"Batch {batch_idx+1} prediction completed.")

        # Combine results
        llm_prediction = torch.cat(prediction_results, dim=0).cpu().detach().numpy()
        inputs = np.concatenate(input_results, axis=0)
        targets = np.concatenate(target_results, axis=0)

        logging.info(
            f"Prediction process completed. Shape of predictions: {llm_prediction.shape}"
        )

        # Save the results
        short_name = generate_run_title(self._data_settings, self._llm_settings)
        save_results_and_generate_plots(
            self._log_dir,
            predictions=llm_prediction,
            targets=targets,
            inputs=inputs,
            name=short_name,
        )

        logging.info("Results saved successfully.")
        
        return llm_prediction, targets

    def train(
        self,
        train_data_path,
        chronos_dir,
        settings={
            "max_steps": 10000,
            "random_init": True,
            "save_steps": 1000,
            "batch_size": 32,
            "prediction_length": 64,
            "context_length": 64,
            "min_past": 64,
            "ntokens": 4096,
            "tokenizer_kwargs": "{'low_limit': 35,'high_limit': 500}",
            "use_peft": False,  # New: Enable PEFT training
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
    ):
        # Load YAML config
        with open(
            "{}/chronos/scripts/training/configs/{}.yaml".format(
                chronos_dir, self._llm_settings["model"].split("/")[-1]
            ),
            "r",
        ) as stream:
            data_loaded = yaml.safe_load(stream)

        # Update config for training
        data_loaded["model_id"] = self._llm_settings["model"]
        data_loaded["training_data_paths"] = [train_data_path]
        data_loaded["probability"] = [1.0]
        data_loaded["prediction_length"] = settings["prediction_length"]
        data_loaded["context_length"] = settings["context_length"]
        data_loaded["min_past"] = settings["min_past"]
        data_loaded["tokenizer_kwargs"] = settings["tokenizer_kwargs"]
        data_loaded["ntokens"] = settings["ntokens"]
        data_loaded["random_init"] = settings["random_init"]
        data_loaded["per_device_train_batch_size"] = settings["batch_size"]
        data_loaded["max_steps"] = settings["max_steps"]
        data_loaded["save_steps"] = settings["save_steps"]
        data_loaded["learning_rate"] = settings["learning_rate"]
        data_loaded["log_steps"] = settings["log_steps"]
        
        # 🆕 Add GPU-specific settings
        if torch.cuda.is_available():
            data_loaded["fp16"] = False  # Use FP32 as specified in settings
            data_loaded["bf16"] = False  # Disable BF16 for compatibility
            data_loaded["dataloader_pin_memory"] = True
            data_loaded["dataloader_num_workers"] = 4  # Optimize for GPU training
            data_loaded["gradient_checkpointing"] = True  # Save memory
            # Force GPU usage in HuggingFace Trainer  
            data_loaded["no_cuda"] = False  # Explicitly enable CUDA
            data_loaded["use_cpu"] = False  # Don't use CPU
            # Increase batch size for GPU training
            data_loaded["per_device_train_batch_size"] = max(8, settings["batch_size"])
            # Enable mixed precision for GPU (but keep FP32 for compatibility)
            data_loaded["fp16"] = False
            data_loaded["bf16"] = False
            self.logger.info("🔧 Configured training for GPU with memory optimizations")
        else:
            data_loaded["dataloader_num_workers"] = 1  # Conservative for CPU
            data_loaded["gradient_checkpointing"] = False
            data_loaded["no_cuda"] = True  # Disable CUDA
            self.logger.info("⚠️  Configured training for CPU")

        # Enable PEFT fine-tuning
        if settings["use_peft"]:
            data_loaded["use_peft"] = settings["use_peft"]  # New setting for LoRA
            data_loaded["lora_r"] = settings["lora_r"]
            data_loaded["lora_alpha"] = settings["lora_alpha"]
            data_loaded["lora_dropout"] = settings["lora_dropout"]

        if self._llm_settings["seed"]:
            data_loaded["seed"] = self._llm_settings["seed"]

        output_dir = self._log_dir + "/{}".format(
            self._llm_settings["model"].split("/")[-1]
        )
        data_loaded["output_dir"] = output_dir
        # Save updated config
        custom_config_path = (
            "{}/chronos/scripts/training/configs/{}_custom.yaml".format(
                chronos_dir, self._llm_settings["model"].split("/")[-1]
            )
        )
        with io.open(custom_config_path, "w", encoding="utf8") as outfile:
            yaml.dump(data_loaded, outfile)

        # Train the model with proper GPU configuration
        if torch.cuda.is_available():
            # Use direct GPU training with explicit environment variables
            env_vars = "CUDA_VISIBLE_DEVICES=0 TORCH_USE_CUDA_DSA=1"
            training_command = f"{env_vars} python {chronos_dir}/chronos/scripts/training/train.py --config {custom_config_path}"
            self.logger.info(f"🚀 Training with GPU: {training_command}")
            
            # Set environment variables in current process too
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
        else:
            # Fallback to CPU training
            training_command = f"python {chronos_dir}/chronos/scripts/training/train.py --config {custom_config_path}"
            self.logger.info(f"⚠️  Training with CPU (no GPU available): {training_command}")
        
        os.system(training_command)

        # Load the trained model with LoRA
        if settings["use_peft"]:
            output_dir = "{}/run-0/lora-final".format(self._log_dir)
        else:
            output_dir = "{}/run-0/checkpoint-final".format(self._log_dir)

        # lora_ckpt_path = os.path.join(output_dir, "lora_checkpoint")

        # if os.path.exists(lora_ckpt_path):
        #     self.logger.info(
        #         f"LoRA checkpoint found at {lora_ckpt_path}, loading model with adapters."
        #     )
        #     self.llm_model = PeftModel.from_pretrained(self.llm_model, lora_ckpt_path)
        # else:
        #     self.logger.info(
        #         f"No LoRA checkpoint found, loading base model from {output_dir}."
        #     )
        #     self.load_model(output_dir)
        self.logger.info(output_dir)

        return output_dir
