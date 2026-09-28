import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim.adam import Adam
from torch.utils.data import WeightedRandomSampler

from llms.ts_llm import TimeSeriesLLM
from models.ecg.time_llm_classifier import TimeLLMEcgClassifier


class TimeLLMECGClassifier(TimeSeriesLLM):
    """Time-LLM-inspired ECG classification wrapper.

    This wrapper mirrors the repository's existing model-wrapper pattern while
    staying separate from the BG forecasting path.
    """

    def __init__(self, settings, data_settings, log_dir="./logs", name="time_llm_ecg_classifier"):
        super().__init__(name=name)
        self._llm_settings = settings
        self._data_settings = data_settings
        self._log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.llm_model = TimeLLMEcgClassifier(settings).float().to(self.device)
        total_parameters = sum(param.numel() for param in self.llm_model.parameters())
        trainable_parameters = sum(param.numel() for param in self.llm_model.parameters() if param.requires_grad)
        self.logger.info(
            "ECG model parameters: total=%d, trainable=%d, freeze_llm=%s",
            total_parameters,
            trainable_parameters,
            self.llm_model.freeze_llm,
        )

    def load_model(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.llm_model.load_checkpoint_state_dict(state_dict)
        self.llm_model.to(self.device)
        self.logger.info(f"Loaded ECG classifier checkpoint from {checkpoint_path}")

    def train(self, train_data, train_loader, val_loader=None):
        os.makedirs(os.path.join(self._log_dir, "checkpoints"), exist_ok=True)
        task_learning_rate = float(self._llm_settings.get("learning_rate", 1e-4))
        backbone_learning_rate = float(self._llm_settings.get("backbone_learning_rate", 1e-5))
        task_parameters = []
        backbone_parameters = []
        for name, param in self.llm_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("llm_model."):
                backbone_parameters.append(param)
            else:
                task_parameters.append(param)
        parameter_groups = [{"params": task_parameters, "lr": task_learning_rate}]
        if backbone_parameters:
            parameter_groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})
        optimizer = Adam(parameter_groups)
        self.logger.info(
            "ECG optimizer: task_params=%d at %.2e, backbone_params=%d at %.2e",
            sum(param.numel() for param in task_parameters),
            task_learning_rate,
            sum(param.numel() for param in backbone_parameters),
            backbone_learning_rate,
        )
        uses_weighted_sampler = isinstance(train_loader.sampler, WeightedRandomSampler)
        class_weights = None if uses_weighted_sampler else self._compute_class_weights(train_data)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.logger.info(
            "ECG imbalance objective: weighted_sampler=%s, class_weighted_ce=%s",
            uses_weighted_sampler,
            class_weights is not None,
        )

        best_val_loss = float("inf")
        best_path = os.path.join(self._log_dir, "checkpoints", "checkpoint_best.pth")
        last_path = os.path.join(self._log_dir, "checkpoints", "checkpoint_last.pth")
        train_history: List[Dict[str, float]] = []

        epochs = int(self._llm_settings.get("train_epochs", 5))
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer, criterion, train=True)
            val_loss = self._run_epoch(val_loader, optimizer, criterion, train=False) if val_loader is not None else 0.0
            train_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
            if val_loader is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.llm_model.checkpoint_state_dict(), best_path)

        torch.save(self.llm_model.checkpoint_state_dict(), last_path)
        if val_loader is None:
            best_path = last_path
        with open(os.path.join(self._log_dir, "train_history.json"), "w") as f:
            json.dump(train_history, f, indent=2)
        return best_path, [row["train_loss"] for row in train_history], [row["val_loss"] for row in train_history]

    def predict(self, test_loader, output_dir=None, filename: str = "test_predictions.csv"):
        self.llm_model.eval()
        rows: List[Dict[str, object]] = []
        all_true: List[int] = []
        all_pred: List[int] = []
        all_logits: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y, meta in test_loader:
                logits = self.llm_model(batch_x.to(self.device), metadata=meta)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                true = batch_y.numpy()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_true.extend(true.tolist())
                all_pred.extend(preds.tolist())
                all_logits.append(probs)
                for i in range(len(preds)):
                    row = {
                        "record_id": meta["record_id"][i],
                        "beat_sample_index": int(meta["beat_sample_index"][i]),
                        "raw_symbol": meta["raw_symbol"][i],
                        "aami_class": meta["aami_class"][i],
                        "original_class_id": int(meta["original_class_id"][i]),
                        "label_mode": meta["label_mode"][i],
                        "rr_prev_seconds": float(meta["rr_prev_seconds"][i]),
                        "rr_next_seconds": float(meta["rr_next_seconds"][i]),
                        "sex": meta["sex"][i],
                        "age_group": meta["age_group"][i],
                        "paced_group": meta["paced_group"][i],
                        "difficulty_group": meta["difficulty_group"][i],
                        "y_true": int(true[i]),
                        "y_pred": int(preds[i]),
                    }
                    for class_idx in range(probs.shape[1]):
                        row[f"prob_{class_idx}"] = float(probs[i, class_idx])
                    rows.append(row)

        predictions_df = pd.DataFrame(rows)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            predictions_df.to_csv(os.path.join(output_dir, filename), index=False)
        predictions = np.asarray(all_pred)
        targets = np.asarray(all_true)
        return predictions, targets, predictions_df

    def evaluate(self, llm_prediction, ground_truth_data, metrics=None):
        if metrics is None:
            metrics = ["accuracy", "macro_f1", "weighted_f1"]
        results: Dict[str, float] = {}
        y_pred = np.asarray(llm_prediction)
        y_true = np.asarray(ground_truth_data)
        if "accuracy" in metrics:
            results["accuracy"] = float(accuracy_score(y_true, y_pred))
        if "macro_f1" in metrics:
            results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if "weighted_f1" in metrics:
            results["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        return results

    def classification_report_dict(self, y_true, y_pred):
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    def _compute_class_weights(self, dataset) -> torch.Tensor:
        underlying = getattr(dataset, "_dataset", dataset)
        labels = [sample.class_id for sample in underlying.samples]
        counts = np.bincount(labels, minlength=self._llm_settings.get("num_classes", 5))
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _run_epoch(self, loader, optimizer, criterion, train: bool) -> float:
        if loader is None:
            return 0.0
        self.llm_model.train(train)
        losses = []
        for batch_x, batch_y, meta in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.llm_model(batch_x, metadata=meta)
            loss = criterion(logits, batch_y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0
