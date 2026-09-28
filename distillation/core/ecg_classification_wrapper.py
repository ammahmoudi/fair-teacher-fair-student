"""Classification distillation wrapper for MIT-BIH Time-LLM ECG models.

Fairness-mitigation transfer from BG (blood glucose): this wrapper ports every
fix from `distillation/core/distillation_wrapper.py` / `distillation_trainer.py`
to ECG AAMI-5 and binary beat classification. T1 (fair teacher sampling) lives upstream
in `EcgTimeLLMDataHandler.load_from_index` / `fairness/utils/ecg_sampling.py`.
The remaining fixes are adapted here as follows (BG -> ECG):

- O2 (student calibration): BG learns a per-group affine scale+bias on
    continuous glucose outputs during training. Here the same jointly trained
    head uses per-class scale+bias vectors on classification logits; a scalar
    bias shared by all classes would cancel under softmax and have no effect.
- K1 (calibrated soft labels): BG shifts the teacher's continuous output by a
  scalar per-group mg/dL offset. A uniform shift to classification logits is a
  no-op after softmax, so here the offset is a per-class vector added to the
  teacher logits per group (see `_apply_teacher_calibration`).
- O1 (fairness constraint / projected dual ascent): BG constrains a soft
  hypoglycemia-TPR gap between two groups. Here the gap is a soft per-group
  recall on a configurable set of "target" AAMI classes, generalized to any
  number of groups via max-pairwise gap (see `_compute_soft_recall_gap`).
- K3 (feature alignment) / O3 (adversarial group erasure): unchanged in
  spirit — both reuse `FeatureAlignmentLoss` / `GroupAdversary` from
  `fairness/loss_functions/fairness_losses.py` on the student LLM's pooled
  hidden state, captured via the same forward-hook pattern BG uses.
- K4 (selective KD replay): BG upweights minority-group hypoglycemic
  *windows*. Here it upweights minority-group *samples whose true label is in
  a configurable set of clinically important classes*.
- T2 (multi-teacher): unchanged in spirit — routes each sample to a
  group-specialized second teacher checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim.adam import Adam
from torch.utils.data import WeightedRandomSampler

from data_processing.ecg.label_map import AAMI_CLASS_TO_ID
from fairness.loss_functions.fairness_losses import FeatureAlignmentLoss, GroupAdversary
from fairness.utils.ecg_sampling import build_group_labels_from_samples
from models.ecg.time_llm_classifier import TimeLLMEcgClassifier


class ECGClassificationDistillationWrapper:
    """Teacher-student classification distillation for MIT-BIH ECG."""

    def __init__(self, settings, data_settings, log_dir, teacher_checkpoint_path):
        self.settings = settings
        self.data_settings = data_settings
        self.log_dir = log_dir
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = float(settings.get("distillation_alpha", 0.5))
        self.beta = float(settings.get("distillation_beta", 0.5))
        self.temperature = float(settings.get("distillation_temperature", 2.0))
        self.logger = logging.getLogger(__name__)
        self.checkpoint_selection = settings.get("checkpoint_selection", "loss")
        self.checkpoint_selection_feature = settings.get("checkpoint_selection_feature", "sex")
        self.checkpoint_selection_fairness_classes = [
            int(class_id) for class_id in settings.get("checkpoint_selection_fairness_classes", [0, 2])
        ]
        self.checkpoint_selection_s_class = int(settings.get("checkpoint_selection_s_class", 1))
        self.checkpoint_selection_min_s_recall = float(
            settings.get("checkpoint_selection_min_s_recall", 0.05)
        )
        self.checkpoint_selection_macro_f1_tolerance = float(
            settings.get("checkpoint_selection_macro_f1_tolerance", 0.01)
        )
        self.checkpoint_selection_min_group_class_support = int(
            settings.get("checkpoint_selection_min_group_class_support", 20)
        )
        self.checkpoint_selection_min_best_group_recall = float(
            settings.get("checkpoint_selection_min_best_group_recall", 0.05)
        )
        self.checkpoint_selection_require_s_recall = bool(
            settings.get("checkpoint_selection_require_s_recall", False)
        )
        if self.checkpoint_selection not in {"loss", "utility_fairness"}:
            raise ValueError(
                "checkpoint_selection must be 'loss' or 'utility_fairness', "
                f"got {self.checkpoint_selection!r}"
            )

        # O2: jointly learned per-group affine calibration on student logits,
        # matching the BG trainer's learned calibration-head lifecycle.
        self.student_calibration_enabled = bool(settings.get("student_calibration_enabled", False))
        self.student_calibration_feature = settings.get("student_calibration_feature", "sex")
        self.student_calibration_learning_rate = float(
            settings.get("student_calibration_learning_rate", settings.get("learning_rate", 1e-4))
        )
        self.student_calibration_scale_regularization = float(
            settings.get("student_calibration_scale_regularization", 0.0)
        )
        self.student_calibration_bias_regularization = float(
            settings.get("student_calibration_bias_regularization", 0.0)
        )
        self.student_calibration_scale: nn.Parameter | None = None
        self.student_calibration_bias: nn.Parameter | None = None
        self._calibration_metadata: Dict[str, object] | None = None

        # K1: calibrated soft labels (group-conditional teacher logit offsets)
        self.teacher_calibration_enabled = bool(settings.get("teacher_calibration_enabled", False))
        self.teacher_calibration_feature = settings.get("teacher_calibration_feature", None)
        self.teacher_calibration_offsets: Dict[str, List[float]] = settings.get("teacher_calibration_offsets", {}) or {}

        # O1: constrained fairness optimization via projected dual ascent
        self.fairness_constraint_enabled = bool(settings.get("fairness_constraint_enabled", False))
        self.fairness_constraint_feature = settings.get("fairness_constraint_feature", None)
        self.fairness_constraint_target_classes = settings.get("fairness_constraint_target_classes", [1, 2, 3, 4])
        self.fairness_constraint_epsilon = float(settings.get("fairness_constraint_epsilon", 0.05))
        self.fairness_dual_lr = float(settings.get("fairness_dual_lr", 0.01))
        self.fairness_dual_lambda = float(max(0.0, settings.get("fairness_dual_init", 1.0)))

        # K3: fairness-aware feature alignment (group-invariant student hidden states)
        self.feature_alignment_enabled = bool(settings.get("feature_alignment_enabled", False))
        self.feature_alignment_feature = settings.get("feature_alignment_feature", None)
        self.feature_alignment_weight = float(settings.get("feature_alignment_weight", 0.0))
        self.feature_alignment_loss_fn = None
        if self.feature_alignment_enabled and self.feature_alignment_weight > 0:
            self.feature_alignment_loss_fn = FeatureAlignmentLoss(
                divergence=settings.get("feature_alignment_divergence", "coral")
            )

        # K4: selective KD replay (upweight minority-group samples on target classes)
        self.kd_replay_enabled = bool(settings.get("kd_replay_enabled", False))
        self.kd_replay_feature = settings.get("kd_replay_feature", None)
        self.kd_replay_minority_group = settings.get("kd_replay_minority_group", None)
        self.kd_replay_target_classes = settings.get("kd_replay_target_classes", [1, 2, 3, 4])
        self.kd_replay_factor = float(settings.get("kd_replay_factor", 4.0))

        # O3: adversarial group erasure (gradient-reversal discriminator on hidden states)
        self.adv_erasure_enabled = bool(settings.get("adv_erasure_enabled", False))
        self.adv_erasure_feature = settings.get("adv_erasure_feature", None)
        self.adv_erasure_lambda = float(settings.get("adv_erasure_lambda", 1.0))
        self.adv_erasure_head = None

        # T2: per-group teachers (multi-teacher KD). The primary teacher serves
        # all samples except those in `multi_teacher_group0`, which are served
        # by the second (group-specialized) teacher.
        self.multi_teacher_feature = settings.get("multi_teacher_feature", None)
        self.multi_teacher_group0 = settings.get("multi_teacher_group0", None)
        self.second_teacher_checkpoint_path = settings.get("second_teacher_checkpoint_path", None)
        self.multi_teacher_enabled = bool(settings.get("multi_teacher_enabled", False)) and bool(
            self.second_teacher_checkpoint_path
        )

        self._group_vocabs: Dict[str, Dict[str, int]] = {}
        self._captured_hidden = None
        self._hook_handle = None

        self.teacher = self._build_model(is_student=False)
        self.student = self._build_model(is_student=True)
        self._load_teacher()

        self.second_teacher = None
        if self.multi_teacher_enabled:
            self.second_teacher = self._build_model(is_student=False)
            self._load_second_teacher()

        if self.feature_alignment_loss_fn is not None or (self.adv_erasure_enabled and self.adv_erasure_lambda > 0):
            self._register_hidden_hook()

        if self.teacher_calibration_enabled:
            self.logger.info(f"🧪 K1 ECG calibrated soft labels: feature={self.teacher_calibration_feature}")
        if self.fairness_constraint_enabled:
            self.logger.info(
                f"📏 O1 ECG fairness constraint: eps={self.fairness_constraint_epsilon}, "
                f"feature={self.fairness_constraint_feature}, target_classes={self.fairness_constraint_target_classes}"
            )
        if self.feature_alignment_loss_fn is not None:
            self.logger.info(
                f"🧬 K3 ECG feature alignment: weight={self.feature_alignment_weight}, feature={self.feature_alignment_feature}"
            )
        if self.kd_replay_enabled:
            self.logger.info(
                f"🔁 K4 ECG selective KD replay: factor={self.kd_replay_factor}, "
                f"minority_group={self.kd_replay_minority_group}, feature={self.kd_replay_feature}"
            )
        if self.adv_erasure_enabled and self.adv_erasure_lambda > 0:
            self.logger.info(f"🛡️ O3 ECG adversarial group erasure: lambda={self.adv_erasure_lambda}, feature={self.adv_erasure_feature}")
        if self.multi_teacher_enabled:
            self.logger.info(f"👥 T2 ECG per-group teachers: feature={self.multi_teacher_feature}, group0_teacher={self.second_teacher_checkpoint_path}")
        if self.student_calibration_enabled:
            self.logger.info(
                "🎯 O2 ECG calibration: feature=%s, lr=%.2e, scale_reg=%.2e, bias_reg=%.2e",
                self.student_calibration_feature,
                self.student_calibration_learning_rate,
                self.student_calibration_scale_regularization,
                self.student_calibration_bias_regularization,
            )
        if self.checkpoint_selection == "utility_fairness":
            self.logger.info(
                "ECG checkpoint selection: utility_fairness, feature=%s, classes=%s, "
                "min_s_recall=%.3f, macro_f1_tolerance=%.3f, min_support=%d, require_s=%s",
                self.checkpoint_selection_feature,
                self.checkpoint_selection_fairness_classes,
                self.checkpoint_selection_min_s_recall,
                self.checkpoint_selection_macro_f1_tolerance,
                self.checkpoint_selection_min_group_class_support,
                self.checkpoint_selection_require_s_recall,
            )

    def _teacher_model_name(self) -> str:
        return self.settings.get("teacher_model", self.settings.get("llm_model", "BERT"))

    def _teacher_model_config(self) -> Dict:
        teacher_model = self._teacher_model_name()
        teacher_defaults = {
            "BERT": {"llm_model": "BERT", "llm_layers": 12, "llm_dim": 768},
            "DistilBERT": {"llm_model": "DistilBERT", "llm_layers": 6, "llm_dim": 768},
            "TinyBERT": {"llm_model": "TinyBERT", "llm_layers": 4, "llm_dim": 312},
            "BERT-tiny": {"llm_model": "BERT-tiny", "llm_layers": 2, "llm_dim": 128},
            "MiniLM": {"llm_model": "MiniLM", "llm_layers": 6, "llm_dim": 384},
            "GPT2": {"llm_model": "GPT2", "llm_layers": 12, "llm_dim": 768},
        }
        config = teacher_defaults.get(teacher_model, teacher_defaults["BERT"]).copy()
        config.update(
            {
                "task_name": "ecg_classification",
                "sequence_length": int(self.settings.get("sequence_length", 256)),
                "enc_in": 1,
                "d_model": int(self.settings.get("d_model", 64)),
                "d_ff": int(self.settings.get("d_ff", 128)),
                "dropout": float(self.settings.get("dropout", 0.1)),
                "patch_len": int(self.settings.get("patch_len", 16)),
                "stride": int(self.settings.get("stride", 8)),
                "num_classes": int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID))),
                "pooling": self.settings.get("pooling", "mean"),
                "use_rr_features": bool(self.settings.get("teacher_use_rr_features", False)),
                "rr_fusion_weight": float(self.settings.get("rr_fusion_weight", 1.0)),
                "freeze_llm": bool(self.settings.get("freeze_llm", True)),
            }
        )
        return config

    def _student_model_config(self) -> Dict:
        return {
            "task_name": "ecg_classification",
            "sequence_length": int(self.settings.get("sequence_length", 256)),
            "enc_in": 1,
            "d_model": int(self.settings.get("d_model", 64)),
            "d_ff": int(self.settings.get("d_ff", 128)),
            "dropout": float(self.settings.get("dropout", 0.1)),
            "patch_len": int(self.settings.get("patch_len", 16)),
            "stride": int(self.settings.get("stride", 8)),
            "num_classes": int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID))),
            "pooling": self.settings.get("pooling", "mean"),
            "use_rr_features": bool(self.settings.get("use_rr_features", False)),
            "rr_fusion_weight": float(self.settings.get("rr_fusion_weight", 1.0)),
            "llm_model": self.settings.get("llm_model", "TinyBERT"),
            "llm_layers": int(self.settings.get("llm_layers", 4)),
            "llm_dim": int(self.settings.get("llm_dim", 312)),
            "freeze_llm": bool(self.settings.get("freeze_llm", True)),
        }

    def _build_model(self, is_student: bool) -> TimeLLMEcgClassifier:
        config = self._student_model_config() if is_student else self._teacher_model_config()
        return TimeLLMEcgClassifier(config).float().to(self.device)

    def _load_teacher(self):
        state_dict = torch.load(self.teacher_checkpoint_path, map_location=self.device, weights_only=True)
        self.teacher.load_checkpoint_state_dict(state_dict)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.logger.info(f"Loaded ECG teacher checkpoint from {self.teacher_checkpoint_path}")

    def _load_second_teacher(self):
        """T2: load the group-0 specialized teacher checkpoint."""
        state_dict = torch.load(self.second_teacher_checkpoint_path, map_location=self.device, weights_only=True)
        self.second_teacher.load_checkpoint_state_dict(state_dict)
        self.second_teacher.eval()
        for param in self.second_teacher.parameters():
            param.requires_grad = False
        self.logger.info(f"Loaded ECG second (group-0) teacher checkpoint from {self.second_teacher_checkpoint_path}")

    # ------------------------------------------------------------------
    # Group-label helpers (shared by O1, K3, O3)
    # ------------------------------------------------------------------
    def _group_vocab(self, feature: str, dataset=None) -> Dict[str, int]:
        if feature in self._group_vocabs:
            return self._group_vocabs[feature]
        if dataset is None:
            raise ValueError(f"Group vocabulary for '{feature}' not yet built and no dataset provided.")
        underlying = getattr(dataset, "_dataset", dataset)
        values = build_group_labels_from_samples(underlying, feature)
        unique = sorted(set(str(v) for v in values))
        vocab = {value: idx for idx, value in enumerate(unique)}
        self._group_vocabs[feature] = vocab
        return vocab

    def _encode_batch_groups(self, meta_values, feature: str) -> torch.Tensor:
        vocab = self._group_vocab(feature)
        idx = [vocab.get(str(v), 0) for v in meta_values]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def _prepare_group_vocabs(self, dataset) -> None:
        features = set()
        if self.student_calibration_enabled and self.student_calibration_feature:
            features.add(self.student_calibration_feature)
        if self.fairness_constraint_enabled and self.fairness_constraint_feature:
            features.add(self.fairness_constraint_feature)
        if self.feature_alignment_loss_fn is not None and self.feature_alignment_feature:
            features.add(self.feature_alignment_feature)
        if self.adv_erasure_enabled and self.adv_erasure_lambda > 0 and self.adv_erasure_feature:
            features.add(self.adv_erasure_feature)
        for feature in features:
            self._group_vocab(feature, dataset)

    def _initialize_student_calibration(self) -> None:
        if not self.student_calibration_enabled:
            return
        vocab = self._group_vocab(self.student_calibration_feature)
        num_groups = len(vocab)
        num_classes = int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID)))
        self.student_calibration_scale = nn.Parameter(
            torch.ones((num_groups, num_classes), dtype=torch.float32, device=self.device)
        )
        self.student_calibration_bias = nn.Parameter(
            torch.zeros((num_groups, num_classes), dtype=torch.float32, device=self.device)
        )

    def _apply_student_calibration(self, logits: torch.Tensor, meta) -> torch.Tensor:
        if (
            not self.student_calibration_enabled
            or self.student_calibration_scale is None
            or self.student_calibration_bias is None
        ):
            return logits
        groups = meta.get(self.student_calibration_feature)
        if groups is None:
            return logits
        group_idx = self._encode_batch_groups(groups, self.student_calibration_feature)
        scales = self.student_calibration_scale[group_idx].to(dtype=logits.dtype)
        biases = self.student_calibration_bias[group_idx].to(dtype=logits.dtype)
        return logits * scales + biases

    def _export_student_calibration_metadata(self) -> Dict[str, object] | None:
        if (
            not self.student_calibration_enabled
            or self.student_calibration_scale is None
            or self.student_calibration_bias is None
        ):
            return None
        vocab = self._group_vocab(self.student_calibration_feature)
        groups = [group for group, _ in sorted(vocab.items(), key=lambda item: item[1])]
        return {
            "feature": self.student_calibration_feature,
            "groups": groups,
            "group_scales": self.student_calibration_scale.detach().cpu().tolist(),
            "group_biases": self.student_calibration_bias.detach().cpu().tolist(),
            "num_classes": int(self.student_calibration_bias.shape[1]),
            "training_mode": "joint",
            "learning_rate": self.student_calibration_learning_rate,
            "scale_regularization": self.student_calibration_scale_regularization,
            "bias_regularization": self.student_calibration_bias_regularization,
        }

    def _student_calibration_regularization_loss(self) -> torch.Tensor:
        if (
            not self.student_calibration_enabled
            or self.student_calibration_scale is None
            or self.student_calibration_bias is None
        ):
            return torch.tensor(0.0, device=self.device)
        scale_penalty = torch.mean(torch.square(self.student_calibration_scale - 1.0))
        bias_penalty = torch.mean(torch.square(self.student_calibration_bias))
        return (
            self.student_calibration_scale_regularization * scale_penalty
            + self.student_calibration_bias_regularization * bias_penalty
        )

    def _validation_selection_metrics(self, loader) -> Dict[str, object]:
        self.student.eval()
        all_true: List[int] = []
        all_pred: List[int] = []
        all_groups: List[str] = []
        with torch.no_grad():
            for batch_x, batch_y, meta in loader:
                logits = self.student(batch_x.to(self.device), metadata=meta)
                logits = self._apply_student_calibration(logits, meta)
                all_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
                all_true.extend(batch_y.tolist())
                all_groups.extend(
                    str(group) for group in meta[self.checkpoint_selection_feature]
                )

        y_true = np.asarray(all_true, dtype=int)
        y_pred = np.asarray(all_pred, dtype=int)
        groups = np.asarray(all_groups, dtype=object)
        num_classes = int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID)))
        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        )
        s_mask = y_true == self.checkpoint_selection_s_class
        s_recall = float(np.mean(y_pred[s_mask] == self.checkpoint_selection_s_class)) if s_mask.any() else 0.0

        class_eo: Dict[str, float] = {}
        class_support: Dict[str, Dict[str, int]] = {}
        unique_groups = sorted(set(all_groups))
        for class_id in self.checkpoint_selection_fairness_classes:
            recalls = []
            support_by_group = {}
            for group in unique_groups:
                group_class_mask = (groups == group) & (y_true == class_id)
                support = int(group_class_mask.sum())
                support_by_group[group] = support
                if support > 0:
                    recalls.append(float(np.mean(y_pred[group_class_mask] == class_id)))
            class_support[str(class_id)] = support_by_group
            if (
                len(unique_groups) >= 2
                and len(recalls) == len(unique_groups)
                and all(
                    support >= self.checkpoint_selection_min_group_class_support
                    for support in support_by_group.values()
                )
                and max(recalls) >= self.checkpoint_selection_min_best_group_recall
            ):
                class_eo[str(class_id)] = float(max(recalls) - min(recalls))

        fairness_eo = (
            float(np.mean(list(class_eo.values())))
            if len(class_eo) == len(self.checkpoint_selection_fairness_classes)
            and class_eo
            else None
        )
        return {
            "macro_f1": macro_f1,
            "s_recall": s_recall,
            "fairness_eo": fairness_eo,
            "class_eo": class_eo,
            "class_support": class_support,
            "groups": unique_groups,
        }

    def _select_validation_candidate(self, candidates: List[Dict[str, object]]) -> Dict[str, object]:
        if not candidates:
            raise ValueError("No validation candidates were collected")
        best_macro_f1 = max(float(candidate["macro_f1"]) for candidate in candidates)
        utility_candidates = [
            candidate
            for candidate in candidates
            if float(candidate["macro_f1"])
            >= best_macro_f1 - self.checkpoint_selection_macro_f1_tolerance
        ]
        s_candidates = [
            candidate
            for candidate in utility_candidates
            if float(candidate["s_recall"]) >= self.checkpoint_selection_min_s_recall
        ]
        if self.checkpoint_selection_require_s_recall and not s_candidates:
            raise RuntimeError(
                "No validation checkpoint passed the required S-recall gate: "
                f"minimum={self.checkpoint_selection_min_s_recall:.3f}"
            )
        eligible = s_candidates or utility_candidates
        return min(
            eligible,
            key=lambda candidate: (
                float(candidate["fairness_eo"])
                if candidate["fairness_eo"] is not None
                else float("inf"),
                -float(candidate["s_recall"]),
                -float(candidate["macro_f1"]),
                float(candidate["val_loss"]),
            ),
        )

    def _restore_student_calibration_metadata(self, metadata: Dict[str, object]) -> None:
        self.student_calibration_feature = str(metadata.get("feature", self.student_calibration_feature))
        groups = [str(group) for group in metadata["groups"]]
        self._group_vocabs[self.student_calibration_feature] = {
            group: index for index, group in enumerate(groups)
        }
        biases = torch.tensor(metadata["group_biases"], dtype=torch.float32, device=self.device)
        scales_value = metadata.get("group_scales")
        scales = (
            torch.tensor(scales_value, dtype=torch.float32, device=self.device)
            if scales_value is not None
            else torch.ones_like(biases)
        )
        self.student_calibration_scale = nn.Parameter(scales)
        self.student_calibration_bias = nn.Parameter(biases)
        self._calibration_metadata = metadata

    # ------------------------------------------------------------------
    # K3/O3: shared forward hook capturing the student LLM's hidden state
    # ------------------------------------------------------------------
    def _register_hidden_hook(self) -> None:
        llm_model = getattr(self.student, "llm_model", None)
        if llm_model is None:
            self.logger.warning(
                "K3/O3 enabled but student has no .llm_model attribute; hidden-state losses will be skipped."
            )
            self.feature_alignment_loss_fn = None
            self.adv_erasure_enabled = False
            return

        def _hook(module, inputs, output):
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, tuple):
                hidden = output[0]
            self._captured_hidden = hidden

        self._hook_handle = llm_model.register_forward_hook(_hook)

    def _pooled_hidden(self):
        if self._captured_hidden is None:
            return None
        return self._captured_hidden.mean(dim=1)

    # ------------------------------------------------------------------
    # K1: calibrated soft labels (per-group teacher logit offsets)
    # ------------------------------------------------------------------
    def _apply_teacher_calibration(self, teacher_logits, meta):
        if (
            not self.teacher_calibration_enabled
            or not self.teacher_calibration_offsets
            or self.teacher_calibration_feature is None
        ):
            return teacher_logits
        groups = meta.get(self.teacher_calibration_feature)
        if groups is None:
            return teacher_logits
        num_classes = teacher_logits.shape[1]
        offsets = []
        for g in groups:
            offset = self.teacher_calibration_offsets.get(str(g))
            if offset is None or len(offset) != num_classes:
                offset = [0.0] * num_classes
            offsets.append(offset)
        offsets_t = torch.tensor(offsets, dtype=teacher_logits.dtype, device=teacher_logits.device)
        return teacher_logits + offsets_t

    # ------------------------------------------------------------------
    # O1: soft per-group recall gap + projected dual ascent
    # ------------------------------------------------------------------
    def _compute_soft_recall_gap(self, student_logits, batch_y, group_idx) -> torch.Tensor:
        target_mask = torch.zeros_like(batch_y, dtype=torch.bool)
        for class_id in self.fairness_constraint_target_classes:
            target_mask |= (batch_y == class_id)
        if target_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        probs = torch.softmax(student_logits, dim=1)
        correct_prob = probs.gather(1, batch_y.view(-1, 1)).squeeze(1)

        unique_groups = torch.unique(group_idx[target_mask])
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=self.device)

        recalls = []
        for group in unique_groups:
            mask = target_mask & (group_idx == group)
            if mask.sum() == 0:
                continue
            recalls.append(correct_prob[mask].mean())
        if len(recalls) < 2:
            return torch.tensor(0.0, device=self.device)
        recalls_t = torch.stack(recalls)
        return recalls_t.max() - recalls_t.min()

    # ------------------------------------------------------------------
    # K4: selective KD replay weights
    # ------------------------------------------------------------------
    def _kd_replay_weights(self, batch_y, meta) -> torch.Tensor:
        batch_size = batch_y.shape[0]
        if self.kd_replay_feature is None or self.kd_replay_minority_group is None:
            return torch.ones(batch_size, device=self.device)
        groups = meta.get(self.kd_replay_feature)
        if groups is None:
            return torch.ones(batch_size, device=self.device)
        target_mask = torch.zeros_like(batch_y, dtype=torch.bool)
        for class_id in self.kd_replay_target_classes:
            target_mask |= (batch_y == class_id)
        is_minority = torch.tensor(
            [str(g) == str(self.kd_replay_minority_group) for g in groups],
            dtype=torch.bool,
            device=self.device,
        )
        upweight = (is_minority & target_mask).float()
        weights = 1.0 + (self.kd_replay_factor - 1.0) * upweight
        return weights / weights.mean().clamp_min(1e-8)

    def distill_knowledge(self, train_loader, val_loader=None, epochs=None):
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        self._prepare_group_vocabs(train_loader.dataset)
        self._initialize_student_calibration()
        task_learning_rate = float(self.settings.get("learning_rate", 1e-4))
        backbone_learning_rate = float(self.settings.get("backbone_learning_rate", 1e-5))
        task_parameters = []
        backbone_parameters = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("llm_model."):
                backbone_parameters.append(param)
            else:
                task_parameters.append(param)
        parameter_groups = [{"params": task_parameters, "lr": task_learning_rate}]
        if backbone_parameters:
            parameter_groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})
        if self.student_calibration_enabled:
            parameter_groups.append(
                {
                    "params": [self.student_calibration_scale, self.student_calibration_bias],
                    "lr": self.student_calibration_learning_rate,
                }
            )
        optimizer = Adam(parameter_groups)
        self.logger.info(
            "ECG KD optimizer: task_params=%d at %.2e, backbone_params=%d at %.2e, freeze_llm=%s",
            sum(param.numel() for param in task_parameters),
            task_learning_rate,
            sum(param.numel() for param in backbone_parameters),
            backbone_learning_rate,
            self.student.freeze_llm,
        )
        uses_weighted_sampler = isinstance(train_loader.sampler, WeightedRandomSampler)
        class_weights = None if uses_weighted_sampler else self._compute_class_weights(train_loader.dataset)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.logger.info(
            "ECG KD imbalance objective: weighted_sampler=%s, class_weighted_ce=%s",
            uses_weighted_sampler,
            class_weights is not None,
        )
        epochs = int(epochs or self.settings.get("train_epochs", 5))

        # Lazily construct the O3 adversary now that its required num_groups
        # and the optimizer to attach it to are known.
        if self.adv_erasure_enabled and self.adv_erasure_lambda > 0 and self.adv_erasure_head is None:
            hidden_size = int(getattr(self.student.llm_model.config, "hidden_size", self.settings.get("llm_dim", 312)))
            num_groups = 2
            if self.adv_erasure_feature:
                num_groups = max(2, len(self._group_vocab(self.adv_erasure_feature)))
            self.adv_erasure_head = GroupAdversary(feature_dim=hidden_size, num_groups=num_groups).to(self.device)
            optimizer.add_param_group({"params": self.adv_erasure_head.parameters()})

        best_val = float("inf")
        best_path = os.path.join(self.log_dir, "checkpoints", "checkpoint_best.pth")
        last_path = os.path.join(self.log_dir, "checkpoints", "checkpoint_last.pth")
        train_history: List[Dict[str, float]] = []
        best_calibration_metadata = None
        validation_candidates: List[Dict[str, object]] = []

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer, ce_loss, train=True)
            val_loss = self._run_epoch(val_loader, optimizer, ce_loss, train=False) if val_loader is not None else 0.0
            history_row = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            if val_loader is not None and self.checkpoint_selection == "utility_fairness":
                selection_metrics = self._validation_selection_metrics(val_loader)
                history_row.update(selection_metrics)
                candidate_path = os.path.join(
                    self.log_dir,
                    "checkpoints",
                    f"checkpoint_candidate_epoch_{epoch + 1}.pth",
                )
                torch.save(self.student.checkpoint_state_dict(), candidate_path)
                validation_candidates.append(
                    {
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        **selection_metrics,
                        "student_state_path": candidate_path,
                        "calibration_metadata": self._export_student_calibration_metadata(),
                    }
                )
                self.logger.info(
                    "ECG validation candidate epoch=%d macro_f1=%.4f s_recall=%.4f fairness_eo=%s",
                    epoch + 1,
                    selection_metrics["macro_f1"],
                    selection_metrics["s_recall"],
                    f"{selection_metrics['fairness_eo']:.4f}"
                    if selection_metrics["fairness_eo"] is not None
                    else "unavailable",
                )
            train_history.append(history_row)
            self.logger.info(
                f"ECG KD Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
                f"| O1 lambda={self.fairness_dual_lambda:.5f}"
            )
            if val_loader is not None and val_loss < best_val:
                best_val = val_loss
                torch.save(self.student.checkpoint_state_dict(), best_path)
                best_calibration_metadata = self._export_student_calibration_metadata()

        if validation_candidates:
            try:
                selected = self._select_validation_candidate(validation_candidates)
            except Exception:
                for candidate in validation_candidates:
                    os.remove(candidate["student_state_path"])
                raise
            selected_state = torch.load(
                selected["student_state_path"],
                map_location="cpu",
                weights_only=True,
            )
            torch.save(selected_state, best_path)
            best_calibration_metadata = selected["calibration_metadata"]
            selection_report = {
                "selection_mode": self.checkpoint_selection,
                "selected_epoch": selected["epoch"],
                "selected_metrics": {
                    key: value
                    for key, value in selected.items()
                    if key not in {"student_state_path", "calibration_metadata"}
                },
                "candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in {"student_state_path", "calibration_metadata"}
                    }
                    for candidate in validation_candidates
                ],
            }
            with open(os.path.join(self.log_dir, "checkpoint_selection.json"), "w") as f:
                json.dump(selection_report, f, indent=2)
            self.logger.info(
                "Selected ECG checkpoint epoch=%d by utility/fairness validation rule",
                selected["epoch"],
            )
            for candidate in validation_candidates:
                os.remove(candidate["student_state_path"])

        torch.save(self.student.checkpoint_state_dict(), last_path)
        if val_loader is None:
            best_path = last_path
        with open(os.path.join(self.log_dir, "distillation_history.json"), "w") as f:
            json.dump(train_history, f, indent=2)

        # K3/O3 no longer need the hidden-state hook once training is done.
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

        best_state = torch.load(best_path, map_location=self.device, weights_only=True)
        self.student.load_checkpoint_state_dict(best_state)
        self.logger.info("Loaded best-validation ECG student checkpoint from %s", best_path)

        if best_calibration_metadata is not None:
            self._restore_student_calibration_metadata(best_calibration_metadata)
            calibration_path = os.path.join(self.log_dir, "student_calibration_head.json")
            with open(calibration_path, "w") as f:
                json.dump(best_calibration_metadata, f, indent=2)
            self.logger.info("🎯 O2 ECG student calibration head saved to %s", calibration_path)

        return best_path, [row["train_loss"] for row in train_history], [row["val_loss"] for row in train_history]

    def load_student_calibration(self, calibration_path: str | None = None) -> None:
        """Load a jointly trained O2 calibration head from disk, if present."""
        calibration_path = calibration_path or os.path.join(self.log_dir, "student_calibration_head.json")
        if not os.path.exists(calibration_path):
            self._calibration_metadata = None
            return
        with open(calibration_path, "r") as f:
            metadata = json.load(f)
        self._restore_student_calibration_metadata(metadata)
        self.logger.info(f"Loaded O2 ECG student calibration head from {calibration_path}")

    def predict(self, test_loader, output_dir=None, filename: str = "test_predictions.csv"):
        # Mirrors the BG DistillationWrapper: auto-load the learned sidecar for
        # standalone inference instances.
        if self.student_calibration_enabled and self.student_calibration_scale is None:
            self.load_student_calibration()

        self.student.eval()
        rows: List[Dict[str, object]] = []
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for batch_x, batch_y, meta in test_loader:
                logits = self.student(batch_x.to(self.device), metadata=meta)
                logits = self._apply_student_calibration(logits, meta)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                true = batch_y.numpy()
                all_true.extend(true.tolist())
                all_pred.extend(preds.tolist())
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
        return np.asarray(all_pred), np.asarray(all_true), predictions_df

    def evaluate(self, predictions, targets, metrics=None):
        if metrics is None:
            metrics = ["accuracy", "macro_f1", "weighted_f1"]
        y_pred = np.asarray(predictions)
        y_true = np.asarray(targets)
        results: Dict[str, float] = {}
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
        counts = np.bincount(
            labels,
            minlength=int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID))),
        )
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _run_epoch(self, loader, optimizer, ce_loss, train: bool) -> float:
        if loader is None:
            return 0.0
        self.student.train(train)
        losses: List[float] = []
        for batch_x, batch_y, meta in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(batch_x, metadata=meta)
                # T2: route each sample to its group-specialized teacher.
                if train and self.multi_teacher_enabled and self.multi_teacher_feature is not None:
                    second_logits = self.second_teacher(batch_x, metadata=meta)
                    groups = meta.get(self.multi_teacher_feature, [])
                    is_g0 = torch.tensor(
                        [str(g) == str(self.multi_teacher_group0) for g in groups],
                        dtype=torch.bool,
                        device=self.device,
                    ).view(-1, 1)
                    teacher_logits = torch.where(is_g0, second_logits, teacher_logits)
                # K1: calibrated soft labels (group-conditional teacher logit offsets)
                if train:
                    teacher_logits = self._apply_teacher_calibration(teacher_logits, meta)

            student_logits = self.student(batch_x, metadata=meta)
            student_logits = self._apply_student_calibration(student_logits, meta)

            supervised = ce_loss(student_logits, batch_y)

            # K4: selective KD replay upweights minority-group samples whose
            # true label is a "target" class; disabled it reduces to plain KD.
            kd_weights = (
                self._kd_replay_weights(batch_y, meta)
                if (train and self.kd_replay_enabled)
                else torch.ones(batch_y.shape[0], device=self.device)
            )
            per_sample_kd = F.kl_div(
                torch.log_softmax(student_logits / self.temperature, dim=1),
                torch.softmax(teacher_logits / self.temperature, dim=1),
                reduction="none",
            ).sum(dim=1) * (self.temperature ** 2)
            kd = (kd_weights * per_sample_kd).mean()

            loss = self.alpha * supervised + self.beta * kd
            if train and self.student_calibration_enabled:
                loss = loss + self._student_calibration_regularization_loss()

            loss_align = torch.tensor(0.0, device=self.device)
            loss_adv = torch.tensor(0.0, device=self.device)
            o1_gap = torch.tensor(0.0, device=self.device)
            o1_viol = torch.tensor(0.0, device=self.device)

            if train:
                # O1: fairness constraint via projected dual ascent
                if self.fairness_constraint_enabled and self.fairness_constraint_feature is not None:
                    group_idx = self._encode_batch_groups(meta[self.fairness_constraint_feature], self.fairness_constraint_feature)
                    o1_gap = self._compute_soft_recall_gap(student_logits, batch_y, group_idx)
                    o1_viol = torch.relu(o1_gap - self.fairness_constraint_epsilon)
                    loss = loss + self.fairness_dual_lambda * o1_viol

                # K3: fairness-aware feature alignment on student hidden states
                if self.feature_alignment_loss_fn is not None and self.feature_alignment_feature is not None:
                    pooled = self._pooled_hidden()
                    if pooled is not None:
                        group_idx = self._encode_batch_groups(meta[self.feature_alignment_feature], self.feature_alignment_feature)
                        loss_align = self.feature_alignment_loss_fn(pooled, group_idx)
                        loss = loss + self.feature_alignment_weight * loss_align

                # O3: adversarial group erasure
                if self.adv_erasure_head is not None and self.adv_erasure_feature is not None:
                    pooled = self._pooled_hidden()
                    if pooled is not None:
                        group_idx = self._encode_batch_groups(meta[self.adv_erasure_feature], self.adv_erasure_feature)
                        if len(torch.unique(group_idx)) >= 2:
                            loss_adv = self.adv_erasure_head(pooled, group_idx, lambda_=self.adv_erasure_lambda)
                            loss = loss + loss_adv

            # Clear the capture so a missed hook on the next step can't reuse stale features.
            self._captured_hidden = None

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.fairness_constraint_enabled and self.fairness_constraint_feature is not None:
                    raw_violation = float((o1_gap - self.fairness_constraint_epsilon).detach().item())
                    self.fairness_dual_lambda = max(
                        0.0, self.fairness_dual_lambda + self.fairness_dual_lr * raw_violation
                    )
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0
