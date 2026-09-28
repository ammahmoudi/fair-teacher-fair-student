"""
Fair Teacher–Fair Student: An LLM-Based Framework for Blood Glucose Prediction in Type 1 Diabetes

Main training and inference script for blood glucose prediction using Large Language Models.
Supports multiple LLM architectures (Time-LLM, Chronos) with knowledge distillation capabilities.

Authors: Amirhossein Mahmoudi, Ghazal Farahani, Peter Domanski, Bahar Farahani, 
         Farshad Firouzi, and Krishnendu Chakrabarty

Usage:
    python main.py --config_path=./configs/config.gin
"""

import os
import random
import gin
import json
import numpy as np
import torch
import datetime
from data_processing.data_loader import ChronosDataHandler, EcgTimeLLMDataHandler, TimeLLMDataHandler
from absl import app, flags
from llms.chronos import ChronosLLM
from llms.student_llm import StudentLLM
from llms.time_llm_ecg import TimeLLMECGClassifier
from llms.time_llm import TimeLLM
from distillation.core.ecg_classification_wrapper import ECGClassificationDistillationWrapper
from distillation.core.distillation_wrapper import DistillationWrapper
from utils.logger import setup_logging
import logging
import pickle
import shutil
from efficiency_toolkit.core.efficiency_calculator import auto_calculate_efficiency
from efficiency_toolkit.core.real_time_profiler import RealTimeProfiler
from efficiency_toolkit.analysis.combine_reports import combine_performance_reports


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_path", "./configs/config_time_llm.gin", "Path to config file."
)
flags.DEFINE_string(
    "log_level", "DEBUG", "Logging level: DEBUG, INFO, WARNING, ERROR, or FATAL."
)
flags.DEFINE_bool(
    "remove_checkpoints",
    False,
    "Whether to remove checkpoint files/folders after training+inference.",
)

import torch.distributed as dist

if dist.is_initialized() and dist.get_rank() != 0:
    logging.getLogger().setLevel(
        logging.WARNING
    )  # Suppress logging for non-main processes


@gin.configurable
def run(
    data_settings={
        "path_to_train_data": "path/to/train/data",
        "path_to_test_data": "path/to/test/data",
        "input_features": ["feature1", "feature2"],
        "label": ["label1"],
        "preprocessing_method": "min_max",
        "preprocess_input_features": False,
        "preprocess_label": False,
        "percent": 100,
    },
    llm_settings={
        "mode": "inference",
        "method": "chronos",
        "model": "amazon/chronos-t5-base",
        "torch_dtype": "float32",
        "ntokens": 4096,
        "tokenizer_kwargs": "{'low_limit': 35,'high_limit': 500}",
        "prediction_length": 64,
        "num_samples": 100,
        "context_length": 64,
        "min_past": 64,
        "prediction_batch_size": 64,
        "prediction_use_auto_split": False,
        "max_train_steps": 10000,
        "train_batch_size": 32,
        "random_init": False,
        "save_steps": 1000,
        "eval_metrics": ["rmse", "mae", "mape"],
        "restore_from_checkpoint": False,
        "restore_checkpoint_path": "path/to/checkpoint",
    },
    log_dir="./logs",
    chronos_dir=".",
):
    # logging files
    # Parse the log level from the command-line arguments
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "FATAL": logging.FATAL,
    }
    log_level = log_level_map.get(FLAGS.log_level.upper(), logging.INFO)

    # Set up logging
    log_dir = setup_logging(log_dir, log_level=log_level)
    logging.info("Logging initialized at directory: {}".format(log_dir))
    
    # Set default values for missing keys in llm_settings (for inference-only configs)
    if "train_batch_size" not in llm_settings:
        llm_settings["train_batch_size"] = llm_settings.get("prediction_batch_size", 32)
    
    # Set fixed random seed for reproducibility
    if "seed" in llm_settings:
        fix_seed = llm_settings["seed"]
        logging.info(f"using fix seed:{fix_seed}")
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    # convert torch_dtype in llm_settings
    if llm_settings["torch_dtype"] == "float32":
        llm_settings["torch_dtype"] = torch.float32
    elif llm_settings["torch_dtype"] == "bfloat16":
        llm_settings["torch_dtype"] = torch.bfloat16
    elif llm_settings["torch_dtype"] == "float16":
        llm_settings["torch_dtype"] = torch.float16
    else:
        logging.info(
            "Torch data type {} not supported.".format(llm_settings["torch_dtype"])
        )
        raise NotImplementedError
    # Set globally
    logging.info(f"setting torch dtype to {llm_settings['torch_dtype']}")
    torch.set_default_dtype(llm_settings["torch_dtype"])

    if llm_settings["method"] == "chronos":

        # load data
        data_loader = ChronosDataHandler(
            settings={
                "input_features": data_settings["input_features"],
                "labels": data_settings["labels"],
                "preprocessing_method": data_settings["preprocessing_method"],
                "preprocess_input_features": data_settings["preprocess_input_features"],
                "preprocess_label": data_settings["preprocess_label"],
                "percent": data_settings["percent"],
                "path_to_train_data": data_settings["path_to_train_data"],
            }
        )
        logging.info("Running Chronos model.")
        llm = ChronosLLM(
            settings={
                "model": llm_settings["model"],
                "device_map": "cuda",
                "torch_dtype": llm_settings["torch_dtype"],
                "mode": llm_settings["mode"],
                "method": llm_settings["method"],
                "llm_model": llm_settings["model"],
                "seed": llm_settings["seed"],
            },
            data_settings=data_loader._settings,
            log_dir=log_dir,
        )
        
        # 🚀 AUTOMATIC EFFICIENCY CALCULATION
        logging.info("🔍 Calculating efficiency metrics for Chronos...")
        try:
            efficiency_results = auto_calculate_efficiency(
                model=llm,
                model_name="chronos",
                config={"llm_settings": llm_settings, "data_settings": data_settings},
                log_dir=log_dir
            )
            logging.info("✅ Efficiency metrics calculated and saved successfully!")
        except Exception as e:
            logging.warning(f"⚠️ Could not calculate efficiency metrics: {e}")
        
        if 'use_peft' not in llm_settings: llm_settings['use_peft']=False
        if 'restore_from_checkpoint' not in llm_settings: llm_settings['restore_from_checkpoint']=False

        if llm_settings["restore_from_checkpoint"]:
            if llm_settings["use_peft"]:
                llm.load_model(
                    llm_settings["restore_checkpoint_path"], llm_settings["lora_path"]
                )
                logging.info("loaded from checkpoint and lora")
            else:
                llm.load_model(llm_settings["restore_checkpoint_path"])
                logging.info("loaded from checkpoint")
        elif llm_settings["use_peft"] and llm_settings['mode']=='inference':
            llm.load_model(None, llm_settings["lora_path"])
            logging.info("loaded from base model and lora")

        if llm_settings["mode"] == "inference":
            # check if train and test data are csv files
            # if data_settings["path_to_train_data"].endswith(".csv"):
            #     logging.debug(
            #         "Loading train data from {}.".format(
            #             data_settings["path_to_train_data"]
            #         )
            #     )
            #     train_data_path = data_loader.convert_csv_to_arrow(
            #         data_settings["path_to_train_data"], split="train"
            #     )
            # data_settings["path_to_train_data"] = train_data_path
            if data_settings["path_to_test_data"].endswith(".csv"):
                logging.debug(
                    "Loading test data from {}.".format(
                        data_settings["path_to_test_data"]
                    )
                )
                test_data = data_loader.load_from_csv(
                    data_settings["path_to_test_data"], split="test"
                )

            # Create real-time profiler for capturing actual inference metrics
            profiler = RealTimeProfiler(log_dir)
            profiler.start_monitoring()
            
            logging.info("📊 Starting real inference performance measurement...")
            
            # prediction with real-time performance monitoring
            llm_prediction, targets = profiler.measure_inference_call(
                llm.predict,
                test_data=test_data,
                data_loader=data_loader,
                settings={
                    "prediction_length": llm_settings["prediction_length"],
                    "num_samples": llm_settings["num_samples"],
                    "batch_size": llm_settings["prediction_batch_size"],
                    "auto_split": llm_settings["prediction_use_auto_split"],
                },
            )
            
            profiler.stop_monitoring()
            
            # Generate and save real performance report
            if hasattr(llm, 'model'):
                model_obj = llm.model
            elif hasattr(llm, 'chronos_pipeline') and hasattr(llm.chronos_pipeline, 'model'):
                model_obj = llm.chronos_pipeline.model
            else:
                model_obj = None
            
            if model_obj:
                model_size_mb = sum(p.numel() * p.element_size() for p in model_obj.parameters()) / (1024**2)
                model_params = sum(p.numel() for p in model_obj.parameters())
            else:
                model_size_mb = 0
                model_params = 0
            
            real_report = profiler.get_comprehensive_report(
                model_name=llm_settings["method"],
                model_size_mb=model_size_mb,
                model_params=model_params
            )
            
            real_report_path = profiler.save_report(real_report)
            logging.info(f"📈 Real performance report saved to: {real_report_path}")
            logging.info(f"✅ Captured {len(profiler.inference_times)} real inference measurements")

            if llm_prediction is not None:
                # Evaluate metrics
                metric_results = llm.evaluate(
                    llm_prediction, targets, llm_settings["eval_metrics"]
                )
                logging.info(f"Metric results: {metric_results}")

        elif llm_settings["mode"] == "training":
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos".format(chronos_dir)):
                root_dir = os.getcwd()
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.chdir("cd {}".format(chronos_dir))
                os.system(
                    "git clone https://github.com/amazon-science/chronos-forecasting.git ./chronos"
                )
                os.chdir(root_dir)

            if data_settings["path_to_train_data"].endswith(".csv"):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings["path_to_train_data"], split="train"
                )
                data_settings["path_to_train_data"] = train_data_path
            elif data_settings["path_to_train_data"].endswith(".arrow"):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            # Create real-time profiler for Chronos training
            chronos_training_profiler = RealTimeProfiler(log_dir)
            chronos_training_profiler.start_monitoring()
            
            logging.info("📊 Starting real Chronos training performance measurement...")

            chronos_training_profiler.measure_inference_call(
                llm.train,
                data_settings["path_to_train_data"],
                chronos_dir,
                settings={
                    "max_steps": llm_settings["max_train_steps"],
                    "random_init": llm_settings["random_init"],
                    "save_steps": llm_settings["save_steps"],
                    "batch_size": llm_settings["train_batch_size"],
                    "prediction_length": llm_settings["prediction_length"],
                    "context_length": llm_settings["context_length"],
                    "min_past": llm_settings["min_past"],
                    "ntokens": llm_settings["ntokens"],
                    "tokenizer_kwargs": llm_settings["tokenizer_kwargs"],
                    "log_steps": llm_settings["log_steps"],
                    "learning_rate": llm_settings["learning_rate"],
                    "use_peft": llm_settings["use_peft"],
                    "lora_r": llm_settings["lora_r"],
                    "lora_alpha": llm_settings["lora_alpha"],
                    "lora_dropout": llm_settings["lora_dropout"],
                },
            )
            
            chronos_training_profiler.stop_monitoring()
            
            # Generate Chronos training performance report
            chronos_training_report = chronos_training_profiler.get_comprehensive_report(
                model_name="chronos_training",
                model_size_mb=0,  # Chronos model size estimation
                model_params=0
            )
            
            chronos_training_report_path = chronos_training_profiler.save_report(chronos_training_report)
            logging.info(f"📈 Chronos training performance report saved to: {chronos_training_report_path}")
            logging.info(f"✅ Captured Chronos training performance with {len(chronos_training_profiler.inference_times)} measurements")

        elif llm_settings["mode"] == "training+inference":
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos".format(chronos_dir)):
                root_dir = os.getcwd()
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.chdir(chronos_dir)
                os.system(
                    "git clone https://github.com/amazon-science/chronos-forecasting.git ./chronos"
                )
                os.chdir(root_dir)

            if data_settings["path_to_train_data"].endswith(".csv"):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings["path_to_train_data"], split="train"
                )
                data_settings["path_to_train_data"] = train_data_path
            elif data_settings["path_to_train_data"].endswith(".arrow"):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            # Create real-time profiler for Chronos training+inference mode
            chronos_train_inf_profiler = RealTimeProfiler(log_dir)
            chronos_train_inf_profiler.start_monitoring()
            
            logging.info("📊 Starting real Chronos training performance measurement (training+inference mode)...")

            # training with performance monitoring
            llm_ckpt_path = chronos_train_inf_profiler.measure_inference_call(
                llm.train,
                data_settings["path_to_train_data"],
                chronos_dir,
                settings={
                    "max_steps": llm_settings["max_train_steps"],
                    "random_init": llm_settings["random_init"],
                    "save_steps": llm_settings["save_steps"],
                    "batch_size": llm_settings["train_batch_size"],
                    "prediction_length": llm_settings["prediction_length"],
                    "context_length": llm_settings["context_length"],
                    "min_past": llm_settings["min_past"],
                    "ntokens": llm_settings["ntokens"],
                    "tokenizer_kwargs": llm_settings["tokenizer_kwargs"],
                },
            )
            
            chronos_train_inf_profiler.stop_monitoring()
            
            # Generate Chronos training+inference performance report
            chronos_train_inf_report = chronos_train_inf_profiler.get_comprehensive_report(
                model_name="chronos_training_inference",
                model_size_mb=0,  # Chronos model size estimation
                model_params=0
            )
            
            chronos_train_inf_report_path = chronos_train_inf_profiler.save_report(chronos_train_inf_report)
            logging.info(f"📈 Chronos training+inference performance report saved to: {chronos_train_inf_report_path}")
            logging.info(f"✅ Captured Chronos training performance with {len(chronos_train_inf_profiler.inference_times)} measurements")
            # load checkpoint
            llm.load_model(llm_ckpt_path)
            # inference and evaluation
            if data_settings["path_to_test_data"].endswith(".csv"):
                test_data = data_loader.load_from_csv(
                    data_settings["path_to_test_data"], split="test"
                )

            # Create real-time profiler for Chronos inference
            profiler_chronos = RealTimeProfiler(log_dir)
            profiler_chronos.start_monitoring()
            
            logging.info("📊 Starting Chronos real inference performance measurement...")

            llm_prediction, targets = profiler_chronos.measure_inference_call(
                llm.predict,
                test_data=test_data,
                data_loader=data_loader,
                settings={
                    "prediction_length": llm_settings["prediction_length"],
                    "num_samples": llm_settings["num_samples"],
                    "batch_size": llm_settings["prediction_batch_size"],
                    "auto_split": llm_settings["prediction_use_auto_split"],
                },
            )
            
            profiler_chronos.stop_monitoring()
            
            # Generate Chronos real performance report
            model_size_mb = 0  # Chronos model size will be calculated differently
            model_params = 0   # Will be updated if accessible
            
            try:
                if hasattr(llm, 'model') and hasattr(llm.model, 'parameters'):
                    model_size_mb = sum(p.numel() * p.element_size() for p in llm.model.parameters()) / (1024**2)
                    model_params = sum(p.numel() for p in llm.model.parameters())
            except:
                pass  # Use defaults if model parameters not accessible
            
            chronos_real_report = profiler_chronos.get_comprehensive_report(
                model_name="chronos",
                model_size_mb=model_size_mb,
                model_params=model_params
            )
            
            chronos_report_path = profiler_chronos.save_report(chronos_real_report)
            logging.info(f"📈 Chronos real performance report saved to: {chronos_report_path}")
            logging.info(f"✅ Captured {len(profiler_chronos.inference_times)} real Chronos inference measurements")
            if llm_prediction is not None:
                # Evaluate metrics
                metric_results = llm.evaluate(
                    llm_prediction, targets, llm_settings["eval_metrics"]
                )
                logging.info(f"Metric results: {metric_results}")
            # Check if we need to remove checkpoints
            if FLAGS.remove_checkpoints=="True":
                if os.path.exists(llm_ckpt_path):
                    if os.path.isdir(llm_ckpt_path):
                        shutil.rmtree(llm_ckpt_path)  # Delete the entire folder
                        logging.info(
                            f"Successfully deleted checkpoint folder: {llm_ckpt_path}"
                        )
                    elif os.path.isfile(llm_ckpt_path):
                        os.remove(llm_ckpt_path)  # Delete the file
                        logging.info(
                            f"Successfully deleted checkpoint file: {llm_ckpt_path}"
                        )
                else:
                    logging.warning(
                        f"Checkpoint path {llm_ckpt_path} does not exist, nothing to remove."
                    )

    elif llm_settings["method"] == "time_llm":

        # Initialize TimeLLM DataLoader
        data_loader = TimeLLMDataHandler(
            settings={
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
            }
        )

        # Load datasets (skip train/val if in inference mode and path is empty)
        if llm_settings["mode"] == "inference" and not data_settings.get("path_to_train_data"):
            # Inference-only mode: skip train/val data
            train_data, train_loader = None, None
            val_data, val_loader = None, None
        else:
            # Load train and val data
            train_data, train_loader = data_loader.load_from_csv(
                data_settings["path_to_train_data"],
                batch_size=llm_settings["train_batch_size"],
                split="train",
                fair_sampling=data_settings.get("fair_teacher_sampling", False),
                fair_feature=data_settings.get("fair_teacher_feature", "gender"),
            )
            val_data, val_loader = data_loader.load_from_csv(
                data_settings["path_to_train_data"],
                batch_size=llm_settings["train_batch_size"],
                split="val",
            )
        
        # Always load test data
        test_data, test_loader = data_loader.load_from_csv(
            data_settings["path_to_test_data"],
            batch_size=llm_settings["prediction_batch_size"],
            split="test",
        )

        # Initialize TimeLLM
        llm = TimeLLM(
            name=llm_settings["model_comment"],
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir,
        )
        
        # 🚀 AUTOMATIC EFFICIENCY CALCULATION
        logging.info("🔍 Calculating efficiency metrics for TimeLLM...")
        try:
            efficiency_results = auto_calculate_efficiency(
                model=llm,
                model_name="time_llm",
                config={"llm_settings": llm_settings, "data_settings": data_settings},
                log_dir=log_dir
            )
            logging.info("✅ Efficiency metrics calculated and saved successfully!")
        except Exception as e:
            logging.warning(f"⚠️ Could not calculate efficiency metrics: {e}")
        
        # Handle modes
        if llm_settings["mode"] == "inference":
            if llm_settings["restore_from_checkpoint"]:
                llm.load_model(
                    llm_settings["restore_checkpoint_path"], is_inference=True
                )

            if len(test_loader) == 0:
                logging.warning("Test loader is empty. No predictions will be made.")
            else:
                # Create real-time profiler for capturing actual inference metrics
                profiler = RealTimeProfiler(log_dir)
                profiler.start_monitoring()
                
                logging.info("📊 Starting real inference performance measurement...")
                
                # Save prediction results with real-time monitoring
                llm_prediction, targets, _ = profiler.measure_inference_call(
                    llm.predict,
                    test_loader, 
                    output_dir=log_dir
                )
                
                profiler.stop_monitoring()
                
                # Generate and save real performance report
                try:
                    model_size_mb = sum(p.numel() * p.element_size() for p in llm.llm_model.parameters()) / (1024**2)
                    model_params = sum(p.numel() for p in llm.llm_model.parameters())
                except:
                    model_size_mb = 0
                    model_params = 0
                
                real_report = profiler.get_comprehensive_report(
                    model_name=llm_settings["method"],
                    model_size_mb=model_size_mb,
                    model_params=model_params
                )
                
                real_report_path = profiler.save_report(real_report)
                logging.info(f"📈 Real performance report saved to: {real_report_path}")
                logging.info(f"✅ Captured {len(profiler.inference_times)} real inference measurements")

                if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(
                        llm_prediction, targets, llm_settings["eval_metrics"]
                    )
                    logging.info(f"Metric results: {metric_results}")

        elif llm_settings["mode"] == "training":
            # Create real-time profiler for training performance monitoring
            training_profiler = RealTimeProfiler(log_dir)
            training_profiler.start_monitoring()
            
            logging.info("📊 Starting real training performance measurement...")
            
            final_checkpoint_path, train_loss, val_loss = training_profiler.measure_inference_call(
                llm.train,
                train_data=train_data,
                train_loader=train_loader,
                val_loader=(
                    val_loader if len(val_loader) > 0 else None
                ),  # Pass None if val_loader is empty
            )
            
            training_profiler.stop_monitoring()
            
            # Generate training performance report
            try:
                model_size_mb = sum(p.numel() * p.element_size() for p in llm.llm_model.parameters()) / (1024**2)
                model_params = sum(p.numel() for p in llm.llm_model.parameters())
            except:
                model_size_mb = 0
                model_params = 0
            
            training_report = training_profiler.get_comprehensive_report(
                model_name=f"{llm_settings['method']}_training",
                model_size_mb=model_size_mb,
                model_params=model_params
            )
            
            training_report_path = training_profiler.save_report(training_report)
            logging.info(f"📈 Training performance report saved to: {training_report_path}")
            logging.info(f"✅ Captured training performance with {len(training_profiler.inference_times)} measurements")
            with open(log_dir + "/loss.pkl", "wb") as f:
                pickle.dump((train_loss, val_loss), f)

        elif llm_settings["mode"] == "training+inference":
            # Create real-time profiler for training+inference mode
            training_inference_profiler = RealTimeProfiler(log_dir)
            training_inference_profiler.start_monitoring()
            
            logging.info("📊 Starting real training performance measurement (training+inference mode)...")
            
            # Train the model and retrieve checkpoint path with performance monitoring
            final_checkpoint_path, train_loss, val_loss = training_inference_profiler.measure_inference_call(
                llm.train,
                train_data=train_data,
                train_loader=train_loader,
                val_loader=(
                    val_loader if len(val_loader) > 0 else None
                ),  # Pass None if val_loader is empty
            )
            
            training_inference_profiler.stop_monitoring()
            
            # Generate training performance report for training+inference mode
            try:
                model_size_mb = sum(p.numel() * p.element_size() for p in llm.llm_model.parameters()) / (1024**2)
                model_params = sum(p.numel() for p in llm.llm_model.parameters())
            except:
                model_size_mb = 0
                model_params = 0
            
            training_inference_report = training_inference_profiler.get_comprehensive_report(
                model_name=f"{llm_settings['method']}_training_inference",
                model_size_mb=model_size_mb,
                model_params=model_params
            )
            
            training_inference_report_path = training_inference_profiler.save_report(training_inference_report)
            logging.info(f"📈 Training+Inference performance report saved to: {training_inference_report_path}")
            logging.info(f"✅ Captured training performance with {len(training_inference_profiler.inference_times)} measurements")
            
            print(final_checkpoint_path)
            with open(log_dir + "/loss.pkl", "wb") as f:
                pickle.dump((train_loss, val_loss), f)

            # Load the saved checkpoint for inference
            llm.load_model(final_checkpoint_path)

            if len(test_loader) == 0:
                logging.warning("Test loader is empty. No predictions will be made.")
            else:
                # Create real-time profiler for training mode inference
                profiler_training = RealTimeProfiler(log_dir)
                profiler_training.start_monitoring()
                
                logging.info("📊 Starting real inference performance measurement (training mode)...")
                
                # Handle efficiency metrics return value (now calculated in main.py)
                llm_prediction, targets, _ = profiler_training.measure_inference_call(
                    llm.predict,
                    test_loader, 
                    output_dir=log_dir
                )
                
                profiler_training.stop_monitoring()
                
                # Generate real performance report for training mode
                try:
                    model_size_mb = sum(p.numel() * p.element_size() for p in llm.llm_model.parameters()) / (1024**2)
                    model_params = sum(p.numel() for p in llm.llm_model.parameters())
                except:
                    model_size_mb = 0
                    model_params = 0
                
                training_real_report = profiler_training.get_comprehensive_report(
                    model_name=f"{llm_settings['method']}_training_mode",
                    model_size_mb=model_size_mb,
                    model_params=model_params
                )
                
                training_report_path = profiler_training.save_report(training_real_report)
                logging.info(f"📈 Training mode real performance report saved to: {training_report_path}")
                logging.info(f"✅ Captured {len(profiler_training.inference_times)} real inference measurements")
                if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(
                        llm_prediction, targets, llm_settings["eval_metrics"]
                    )
                    logging.info(f"Metric results: {metric_results}")

            # Check if we need to remove checkpoints
            if FLAGS.remove_checkpoints=="True":
                if os.path.exists(final_checkpoint_path):
                    if os.path.isdir(final_checkpoint_path):
                        shutil.rmtree(final_checkpoint_path)  # Delete the entire folder
                        logging.info(
                            f"Successfully deleted checkpoint folder: {final_checkpoint_path}"
                        )
                    elif os.path.isfile(final_checkpoint_path):
                        os.remove(final_checkpoint_path)  # Delete the file
                        logging.info(
                            f"Successfully deleted checkpoint file: {final_checkpoint_path}"
                        )
                else:
                    logging.warning(
                        f"Checkpoint path {final_checkpoint_path} does not exist, nothing to remove."
                    )

        else:
            logging.error(f"Unsupported mode: {llm_settings['mode']}")
            raise NotImplementedError
    elif llm_settings["method"] == "time_llm_ecg_classifier":
        data_loader = EcgTimeLLMDataHandler(
            settings={
                "dataset_dir": data_settings["dataset_dir"],
                "window_size": llm_settings.get("sequence_length", 256),
                "batch_size": llm_settings.get("train_batch_size", 64),
                "num_workers": llm_settings.get("num_workers", 0),
                "include_duplicate_202": data_settings.get("include_duplicate_202", False),
                "beat_index_csv": data_settings.get("beat_index_csv"),
                "metadata_csv": data_settings.get("metadata_csv"),
                "label_mode": data_settings.get("label_mode", "aami5"),
                "sampling_rate_hz": data_settings.get("sampling_rate_hz", 360.0),
                "rr_clip_seconds": data_settings.get("rr_clip_seconds", 3.0),
            }
        )

        train_data, train_loader = data_loader.load_from_index(
            split="train",
            batch_size=llm_settings.get("train_batch_size", 64),
            fair_sampling=data_settings.get("fair_teacher_sampling", False),
            fair_feature=data_settings.get("fair_teacher_feature", "sex"),
            fair_max_oversample=data_settings.get("fair_teacher_max_oversample", 4.0),
            class_balanced_sampling=data_settings.get("class_balanced_sampling", False),
            class_balanced_max_oversample=data_settings.get("class_balanced_max_oversample", 50.0),
        )
        val_data, val_loader = data_loader.load_from_index(
            split="val", batch_size=llm_settings.get("train_batch_size", 64), shuffle=False
        )
        test_data, test_loader = data_loader.load_from_index(
            split="test", batch_size=llm_settings.get("prediction_batch_size", 64), shuffle=False
        )

        llm = TimeLLMECGClassifier(
            name=llm_settings.get("model_comment", "time_llm_ecg_classifier"),
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir,
        )

        # 🚀 AUTOMATIC EFFICIENCY CALCULATION
        logging.info("🔍 Calculating efficiency metrics for ECG classifier...")
        try:
            efficiency_results = auto_calculate_efficiency(
                model=llm,
                model_name="time_llm_ecg_classifier",
                config={"llm_settings": llm_settings, "data_settings": data_settings},
                log_dir=log_dir,
            )
            logging.info("✅ Efficiency metrics calculated and saved successfully!")
        except Exception as e:
            logging.warning(f"⚠️ Could not calculate efficiency metrics: {e}")

        if llm_settings["mode"] == "inference":
            if llm_settings.get("restore_from_checkpoint", False):
                llm.load_model(llm_settings["restore_checkpoint_path"])
            preds, targets, pred_df = llm.predict(test_loader, output_dir=log_dir)
            if preds is not None:
                metric_results = llm.evaluate(preds, targets, llm_settings.get("eval_metrics", ["accuracy", "macro_f1", "weighted_f1"]))
                logging.info(f"ECG classifier metrics: {metric_results}")
                with open(os.path.join(log_dir, "classification_report.json"), "w") as f:
                    json.dump(llm.classification_report_dict(targets, preds), f, indent=2)

        elif llm_settings["mode"] in ["training", "training+inference"]:
            checkpoint_path, train_loss, val_loss = llm.train(train_data, train_loader, val_loader)
            with open(log_dir + "/loss.pkl", "wb") as f:
                pickle.dump((train_loss, val_loss), f)

            llm.load_model(checkpoint_path)
            if val_loader is not None and len(val_loader) > 0:
                llm.predict(val_loader, output_dir=log_dir, filename="val_predictions.csv")

            if llm_settings["mode"] == "training+inference":
                preds, targets, pred_df = llm.predict(test_loader, output_dir=log_dir)
                if preds is not None:
                    metric_results = llm.evaluate(preds, targets, llm_settings.get("eval_metrics", ["accuracy", "macro_f1", "weighted_f1"]))
                    logging.info(f"ECG classifier metrics: {metric_results}")
                    with open(os.path.join(log_dir, "classification_report.json"), "w") as f:
                        json.dump(llm.classification_report_dict(targets, preds), f, indent=2)
        else:
            logging.error(f"Unsupported mode for ECG classifier: {llm_settings['mode']}")
            raise NotImplementedError

    elif llm_settings["method"] == "distillation_ecg_classifier":
        data_loader = EcgTimeLLMDataHandler(
            settings={
                "dataset_dir": data_settings["dataset_dir"],
                "window_size": llm_settings.get("sequence_length", 256),
                "batch_size": llm_settings.get("train_batch_size", 64),
                "num_workers": llm_settings.get("num_workers", 0),
                "include_duplicate_202": data_settings.get("include_duplicate_202", False),
                "beat_index_csv": data_settings.get("beat_index_csv"),
                "metadata_csv": data_settings.get("metadata_csv"),
                "label_mode": data_settings.get("label_mode", "aami5"),
                "sampling_rate_hz": data_settings.get("sampling_rate_hz", 360.0),
                "rr_clip_seconds": data_settings.get("rr_clip_seconds", 3.0),
            }
        )

        train_data, train_loader = data_loader.load_from_index(
            split="train",
            batch_size=llm_settings.get("train_batch_size", 64),
            fair_sampling=data_settings.get("fair_teacher_sampling", False),
            fair_feature=data_settings.get("fair_teacher_feature", "sex"),
            fair_max_oversample=data_settings.get("fair_teacher_max_oversample", 4.0),
            class_balanced_sampling=data_settings.get("class_balanced_sampling", False),
            class_balanced_max_oversample=data_settings.get("class_balanced_max_oversample", 50.0),
        )
        val_data, val_loader = data_loader.load_from_index(
            split="val", batch_size=llm_settings.get("train_batch_size", 64), shuffle=False
        )
        test_data, test_loader = data_loader.load_from_index(
            split="test", batch_size=llm_settings.get("prediction_batch_size", 64), shuffle=False
        )

        distillation_driver = ECGClassificationDistillationWrapper(
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir,
            teacher_checkpoint_path=llm_settings["teacher_checkpoint_path"],
        )

        # 🚀 AUTOMATIC EFFICIENCY CALCULATION (for the distilled student)
        logging.info("🔍 Calculating efficiency metrics for distilled ECG student...")
        try:
            class _EcgStudentEfficiencyAdapter:
                """Lightweight adapter so auto_calculate_efficiency's generic
                `model.llm_model` convention works uniformly for the bare
                student TimeLLMEcgClassifier (matches the wrapper convention
                used by time_llm_ecg_classifier's TimeLLMECGClassifier)."""

                def __init__(self, classifier):
                    self.llm_model = classifier

            efficiency_results = auto_calculate_efficiency(
                model=_EcgStudentEfficiencyAdapter(distillation_driver.student),
                model_name="distillation_ecg_classifier",
                config={"llm_settings": llm_settings, "data_settings": data_settings},
                log_dir=log_dir,
            )
            logging.info("✅ Efficiency metrics calculated and saved successfully!")
        except Exception as e:
            logging.warning(f"⚠️ Could not calculate efficiency metrics: {e}")

        if llm_settings["mode"] == "training+inference":
            checkpoint_path, train_loss, val_loss = distillation_driver.distill_knowledge(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=llm_settings.get("train_epochs", 5),
            )
            with open(log_dir + "/loss.pkl", "wb") as f:
                pickle.dump((train_loss, val_loss), f)

            if val_loader is not None and len(val_loader) > 0:
                distillation_driver.predict(val_loader, output_dir=log_dir, filename="val_predictions.csv")

            preds, targets, _pred_df = distillation_driver.predict(test_loader, output_dir=log_dir)
            metric_results = distillation_driver.evaluate(
                preds,
                targets,
                llm_settings.get("eval_metrics", ["accuracy", "macro_f1", "weighted_f1"]),
            )
            logging.info(f"ECG distilled classifier metrics: {metric_results}")
            with open(os.path.join(log_dir, "classification_report.json"), "w") as f:
                json.dump(distillation_driver.classification_report_dict(targets, preds), f, indent=2)

        elif llm_settings["mode"] == "inference":
            distillation_driver.student.load_checkpoint_state_dict(
                torch.load(llm_settings["restore_checkpoint_path"], map_location=distillation_driver.device, weights_only=True)
            )
            if distillation_driver.student_calibration_enabled:
                calibration_dir = os.path.dirname(os.path.dirname(llm_settings["restore_checkpoint_path"]))
                distillation_driver.load_student_calibration(
                    os.path.join(calibration_dir, "student_calibration_head.json")
                )
            preds, targets, _pred_df = distillation_driver.predict(test_loader, output_dir=log_dir)
            metric_results = distillation_driver.evaluate(
                preds,
                targets,
                llm_settings.get("eval_metrics", ["accuracy", "macro_f1", "weighted_f1"]),
            )
            logging.info(f"ECG distilled classifier metrics: {metric_results}")
            with open(os.path.join(log_dir, "classification_report.json"), "w") as f:
                json.dump(distillation_driver.classification_report_dict(targets, preds), f, indent=2)
        else:
            logging.error(f"Unsupported mode for ECG classification distillation: {llm_settings['mode']}")
            raise NotImplementedError

    elif llm_settings["method"] == "student_llm":
        # Initialize TimeLLM-compatible data loader
        data_loader = TimeLLMDataHandler(
            settings={
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
            }
        )

                # Load datasets
        train_data, train_loader = data_loader.load_from_csv(
            data_settings["path_to_train_data"],
            batch_size=llm_settings["train_batch_size"],
            split="train",
        )
        val_data, val_loader = data_loader.load_from_csv(
            data_settings["path_to_train_data"],
            batch_size=llm_settings["train_batch_size"],
            split="val",
        )
        test_data, test_loader = data_loader.load_from_csv(
            data_settings["path_to_test_data"],
            batch_size=llm_settings["prediction_batch_size"],
            split="test",
        )

        # Initialize student model wrapper
        llm = StudentLLM(
            name=llm_settings["model_comment"],
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir,
        )

        if llm_settings["restore_from_checkpoint"]:
            llm.load_model(llm_settings["restore_checkpoint_path"], is_inference=True)

        if len(test_loader) == 0:
            logging.warning("Test loader is empty. No predictions will be made.")
        else:
            predictions, targets, _ = llm.predict(test_loader, output_dir=log_dir)
            if predictions is not None:
                metric_results = llm.evaluate(predictions, targets, llm_settings["eval_metrics"])
                logging.info(f"Student Model Evaluation Metrics: {metric_results}")

    elif llm_settings["method"] == "distillation":
        logging.info("Starting Knowledge Distillation...")
        
        # Initialize distillation wrapper with proper trainer
        distillation_driver = DistillationWrapper(
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir,
            teacher_checkpoint_path=llm_settings["teacher_checkpoint_path"]
        )
        
        # Initialize TimeLLM-compatible data loader
        data_loader = TimeLLMDataHandler(
            settings={
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
            }
        )

        # Load datasets
        train_data, train_loader = data_loader.load_from_csv(
            data_settings["path_to_train_data"],
            batch_size=llm_settings["train_batch_size"],
            split="train",
        )
        val_data, val_loader = data_loader.load_from_csv(
            data_settings["path_to_train_data"],
            batch_size=llm_settings["train_batch_size"],
            split="val",
        )
        test_data, test_loader = data_loader.load_from_csv(
            data_settings["path_to_test_data"],
            batch_size=llm_settings["prediction_batch_size"],
            split="test",
        )
        
        if llm_settings["mode"] == "training+inference":
            # Perform distillation training
            final_checkpoint_path, train_loss, val_loss = distillation_driver.distill_knowledge(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=llm_settings["train_epochs"]
            )
            logging.info(f"✅ Knowledge distillation completed. Final checkpoint: {final_checkpoint_path}")
            
            # Run inference on test set
            if len(test_loader) > 0:
                predictions, targets, _ = distillation_driver.predict(test_loader, output_dir=log_dir)
                if predictions is not None:
                    metric_results = distillation_driver.evaluate(predictions, targets, llm_settings["eval_metrics"])
                    logging.info(f"Distilled Model Evaluation Metrics: {metric_results}")
            else:
                logging.warning("Test loader is empty. No predictions will be made.")
                
        elif llm_settings["mode"] == "inference":
            # Only run inference
            if len(test_loader) > 0:
                predictions, targets, _ = distillation_driver.predict(test_loader, output_dir=log_dir)
                if predictions is not None:
                    metric_results = distillation_driver.evaluate(predictions, targets, llm_settings["eval_metrics"])
                    logging.info(f"Distilled Model Evaluation Metrics: {metric_results}")
            else:
                logging.warning("Test loader is empty. No predictions will be made.")
        else:
            logging.error(f"Unsupported mode for distillation: {llm_settings['mode']}")
            raise NotImplementedError

    else:
        logging.error("Method {} not supported.".format(llm_settings["method"]))
        raise NotImplementedError

    # Generate comprehensive performance report combining all measurements
    try:
        model_name = llm_settings.get("method", "unknown_model")
        comprehensive_report_path = combine_performance_reports(log_dir, model_name)
        if comprehensive_report_path:
            logging.info(f"✅ Comprehensive performance report saved: {comprehensive_report_path}")
        else:
            logging.info("ℹ️ No performance reports found to combine")
    except Exception as e:
        logging.warning(f"Could not generate comprehensive performance report: {e}")

    # Save Gin's operative config to a file
    txt_file = open(log_dir + "/gin_config.txt", "w+")
    txt_file.write(gin.operative_config_str())
    txt_file.close()


def main(args):
    # parse config file
    gin.parse_config_file(FLAGS.config_path)
    run()


if __name__ == "__main__":
    # pass config path to main function
    app.run(main)
