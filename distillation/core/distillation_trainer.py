import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

# Allow importing fairness losses from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fairness.loss_functions.fairness_losses import (
    EqualizedOddsLoss,
    HypoglycemiaTPREqualityLoss,
    FeatureAlignmentLoss,
    GroupAdversary,
)


class DistillationTrainer:
    def __init__(
        self,
        teacher,
        student,
        dataloader,
        optimizer,
        device,
        accelerator=None,
        scheduler=None,
        early_stopping=None,
        alpha=0.5,
        beta=0.5,
        train_epochs=10,
        logger=None,
        fairness_weight: float = 0.0,
        target_threshold: float = 70.0,
        pred_threshold: float = 70.0,
        teacher_calibration_enabled: bool = False,
        teacher_calibration_feature: str = None,
        teacher_calibration_group0_offset: float = 0.0,
        teacher_calibration_group1_offset: float = 0.0,
        student_calibration_enabled: bool = False,
        student_calibration_feature: str = None,
        fairness_constraint_enabled: bool = False,
        fairness_constraint_epsilon: float = 0.05,
        fairness_dual_lr: float = 0.01,
        fairness_dual_init: float = 1.0,
        feature_alignment_enabled: bool = False,
        feature_alignment_feature: str = None,
        feature_alignment_weight: float = 0.0,
        feature_alignment_divergence: str = "coral",
        feature_alignment_layer: int = -1,
        kd_replay_enabled: bool = False,
        kd_replay_feature: str = None,
        kd_replay_minority_group: int = 0,
        kd_replay_factor: float = 4.0,
        adv_erasure_enabled: bool = False,
        adv_erasure_feature: str = None,
        adv_erasure_lambda: float = 1.0,
        adv_erasure_layer: int = -1,
        second_teacher=None,
        multi_teacher_group0=0,
    ):
        self.teacher = teacher
        self.student = student
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.alpha = alpha
        self.beta = beta
        self.loss_fn = nn.MSELoss()
        self.train_epochs = train_epochs
        self.logger = logger or logging.getLogger(__name__)
        self.target_threshold = float(target_threshold)
        self.pred_threshold = float(pred_threshold)
        self.pred_len = self.teacher.prediction_length
        self.context_len = self.teacher.sequence_length

        # K1: calibrated soft labels (group-conditional teacher output shifts)
        self.teacher_calibration_enabled = bool(teacher_calibration_enabled)
        self.teacher_calibration_feature = teacher_calibration_feature
        self.teacher_calibration_group0_offset = float(teacher_calibration_group0_offset)
        self.teacher_calibration_group1_offset = float(teacher_calibration_group1_offset)

        # O2: learned per-group affine calibration head on student outputs
        self.student_calibration_enabled = bool(student_calibration_enabled)
        self.student_calibration_feature = student_calibration_feature
        self.student_calibration_scale = None
        self.student_calibration_bias = None
        if self.student_calibration_enabled:
            self.student_calibration_scale = nn.Parameter(
                torch.ones(2, device=self.device, dtype=torch.float32)
            )
            self.student_calibration_bias = nn.Parameter(
                torch.zeros(2, device=self.device, dtype=torch.float32)
            )
            self.optimizer.add_param_group(
                {
                    "params": [self.student_calibration_scale, self.student_calibration_bias],
                }
            )

        # O1: constrained fairness optimization with projected dual ascent
        self.fairness_constraint_enabled = bool(fairness_constraint_enabled)
        self.fairness_constraint_epsilon = float(fairness_constraint_epsilon)
        self.fairness_dual_lr = float(fairness_dual_lr)
        self.fairness_dual_lambda = float(max(0.0, fairness_dual_init))

        # K3: fairness-aware feature alignment (group-invariant hidden states)
        self.feature_alignment_enabled = bool(feature_alignment_enabled)
        self.feature_alignment_feature = feature_alignment_feature
        self.feature_alignment_weight = float(feature_alignment_weight)
        self.feature_alignment_layer = int(feature_alignment_layer)
        self.feature_alignment_loss_fn = None

        # O3: adversarial group erasure — a gradient-reversed discriminator that
        # tries to predict the group from the student's pooled hidden state,
        # pushing the student toward group-invariant representations.
        self.adv_erasure_enabled = bool(adv_erasure_enabled)
        self.adv_erasure_feature = adv_erasure_feature
        self.adv_erasure_lambda = float(adv_erasure_lambda)
        self.adv_erasure_layer = int(adv_erasure_layer)
        self.adv_erasure_head = None

        # K3 and O3 both consume the student LLM's pooled hidden state, captured
        # by a single shared forward hook.
        self._k3_captured_hidden = None
        self._k3_hook_handle = None
        # The layer index to capture: O3 and K3 share one hook, so they must
        # agree on the layer; if both are on we use K3's setting.
        self._hidden_capture_layer = (
            self.feature_alignment_layer
            if (self.feature_alignment_enabled and self.feature_alignment_weight > 0)
            else self.adv_erasure_layer
        )
        if self.feature_alignment_enabled and self.feature_alignment_weight > 0:
            self.feature_alignment_loss_fn = FeatureAlignmentLoss(
                divergence=feature_alignment_divergence,
            )
        if self.adv_erasure_enabled and self.adv_erasure_lambda > 0:
            self.adv_erasure_head = GroupAdversary(feature_dim=self.student.d_llm)
            self.adv_erasure_head.to(self.device)
            self.optimizer.add_param_group({"params": self.adv_erasure_head.parameters()})
        if (self.feature_alignment_loss_fn is not None) or (self.adv_erasure_head is not None):
            self._register_feature_alignment_hook()

        # K4: selective KD replay — upweight the teacher-matching loss on
        # minority-group hypoglycemic windows so the student is pushed harder to
        # match the teacher on rare events for the underrepresented group.
        self.kd_replay_enabled = bool(kd_replay_enabled)
        self.kd_replay_feature = kd_replay_feature
        self.kd_replay_minority_group = int(kd_replay_minority_group)
        self.kd_replay_factor = float(kd_replay_factor)

        # Fairness-aware loss (disabled when fairness_weight == 0)
        self.fairness_weight = fairness_weight
        if fairness_weight > 0:
            # HypoglycemiaTPREqualityLoss fixes the core issue with EqualizedOddsLoss:
            # rare hypoglycemia windows (~1-5% of batches) caused near-zero gradients.
            # The new loss uses soft-TPR equalization + focal regression upweighting
            # on true hypo timesteps, giving a stable gradient even with rare events.
            self.eo_loss_fn = HypoglycemiaTPREqualityLoss(
                base_loss=None,
                fairness_weight=1.0,
                hypo_threshold=target_threshold,   # 70.0 mg/dL
                focal_gamma=3.0,
                soft_slope=0.1,
            )
        else:
            self.eo_loss_fn = None

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # T2: per-group (multi-)teacher KD. When a second teacher is supplied,
        # the primary teacher serves the non-group0 samples and the second
        # teacher serves the group0 samples; each batch sample is matched to its
        # group's specialized teacher via batch_groups.
        self.second_teacher = second_teacher
        self.multi_teacher_enabled = second_teacher is not None
        self.multi_teacher_group0 = int(multi_teacher_group0)
        if self.second_teacher is not None:
            self.second_teacher.eval()
            for param in self.second_teacher.parameters():
                param.requires_grad = False

    def _compute_soft_hypo_tpr_gap(self, predictions, targets, group_labels):
        """Differentiable EO proxy: |TPR_group0 - TPR_group1| for hypoglycemia."""
        if group_labels is None:
            return torch.tensor(0.0, device=predictions.device)

        unique_groups = torch.unique(group_labels)
        if len(unique_groups) != 2:
            return torch.tensor(0.0, device=predictions.device)

        preds_flat = predictions.reshape(predictions.shape[0], -1)
        tgts_flat = targets.reshape(targets.shape[0], -1)

        true_hypo = (tgts_flat < self.target_threshold).float()
        pred_hypo_soft = torch.sigmoid(
            (self.target_threshold - preds_flat) / (self.target_threshold * 0.1)
        )

        tprs = []
        for group in unique_groups:
            mask = (group_labels == group)
            if mask.sum() == 0:
                continue
            g_true_hypo = true_hypo[mask]
            g_pred_soft = pred_hypo_soft[mask]
            denom = g_true_hypo.sum() + 1e-8
            tpr = (g_true_hypo * g_pred_soft).sum() / denom
            tprs.append(tpr)

        if len(tprs) != 2:
            return torch.tensor(0.0, device=predictions.device)
        return torch.abs(tprs[0] - tprs[1])

    def _apply_teacher_group_calibration(self, y_teacher, batch_groups):
        """Apply group-conditional additive offsets to teacher outputs.

        Group labels follow project convention: group 0 / group 1 (e.g. Female/Male
        for gender). Offsets are scalar glucose shifts in mg/dL.
        """
        if (not self.teacher_calibration_enabled) or (batch_groups is None):
            return y_teacher

        offsets = torch.where(
            batch_groups == 0,
            torch.tensor(self.teacher_calibration_group0_offset, device=y_teacher.device, dtype=y_teacher.dtype),
            torch.tensor(self.teacher_calibration_group1_offset, device=y_teacher.device, dtype=y_teacher.dtype),
        )
        # Broadcast offsets over [pred_len, channels]
        return y_teacher + offsets.view(-1, 1, 1)

    def _apply_student_group_calibration(self, y_student, batch_groups):
        """Apply learned per-group affine calibration to student outputs."""
        if (not self.student_calibration_enabled) or (batch_groups is None):
            return y_student

        scales = torch.where(
            batch_groups == 0,
            self.student_calibration_scale[0].to(dtype=y_student.dtype),
            self.student_calibration_scale[1].to(dtype=y_student.dtype),
        )
        biases = torch.where(
            batch_groups == 0,
            self.student_calibration_bias[0].to(dtype=y_student.dtype),
            self.student_calibration_bias[1].to(dtype=y_student.dtype),
        )
        return y_student * scales.view(-1, 1, 1) + biases.view(-1, 1, 1)

    def _register_feature_alignment_hook(self):
        """Register a forward hook capturing the student LLM's hidden state.

        Shared by K3 (feature alignment) and O3 (adversarial erasure). We hook
        ``self.student.llm_model`` rather than changing the model's forward
        signature, so the rest of the pipeline (which expects the model to return
        a prediction tensor) is unaffected. The captured tensor is consumed and
        cleared once per training step.
        """
        llm_model = getattr(self.student, "llm_model", None)
        if llm_model is None:
            self.logger.warning(
                "K3/O3 enabled but student has no .llm_model attribute; "
                "hidden-state losses will be skipped."
            )
            self.feature_alignment_loss_fn = None
            self.adv_erasure_head = None
            return

        def _hook(module, inputs, output):
            # HF backbones return an object with .hidden_states (tuple) when
            # output_hidden_states=True, else fall back to .last_hidden_state.
            hidden = None
            if self._hidden_capture_layer != -1 and getattr(output, "hidden_states", None):
                layers = output.hidden_states
                idx = self._hidden_capture_layer
                if -len(layers) <= idx < len(layers):
                    hidden = layers[idx]
            if hidden is None:
                hidden = getattr(output, "last_hidden_state", None)
            self._k3_captured_hidden = hidden

        self._k3_hook_handle = llm_model.register_forward_hook(_hook)

    def _pooled_captured_hidden(self, batch_groups):
        """Pool the captured student hidden state to [B, d] aligned with batch_groups.

        Returns None if no hidden state was captured or the batch dim can't be
        reconciled with the group labels.
        """
        if self._k3_captured_hidden is None or batch_groups is None:
            return None
        hidden = self._k3_captured_hidden  # [B*N_vars, seq, d]
        pooled = hidden.mean(dim=1)        # mean-pool over sequence → [B*N_vars, d]
        if pooled.shape[0] != batch_groups.shape[0]:
            n_vars = pooled.shape[0] // batch_groups.shape[0]
            if n_vars * batch_groups.shape[0] == pooled.shape[0]:
                pooled = pooled.reshape(batch_groups.shape[0], n_vars, -1).mean(dim=1)
            else:
                return None
        return pooled

    def _compute_feature_alignment_loss(self, batch_groups):
        """K3: pool the captured student hidden state and align across groups."""
        if self.feature_alignment_loss_fn is None:
            return torch.tensor(0.0, device=self.device)
        pooled = self._pooled_captured_hidden(batch_groups)
        if pooled is None:
            return torch.tensor(0.0, device=self.device)
        return self.feature_alignment_loss_fn(pooled, batch_groups)

    def _compute_adv_erasure_loss(self, batch_groups):
        """O3: discriminator cross-entropy behind a gradient-reversal layer.

        Minimizing this trains the adversary to predict the group while the
        reversed gradient pushes the student toward group-invariant features.
        Needs both groups present in the batch to be meaningful.
        """
        if self.adv_erasure_head is None:
            return torch.tensor(0.0, device=self.device)
        pooled = self._pooled_captured_hidden(batch_groups)
        if pooled is None:
            return torch.tensor(0.0, device=self.device)
        if len(torch.unique(batch_groups)) < 2:
            return torch.tensor(0.0, device=self.device)
        return self.adv_erasure_head(pooled, batch_groups, lambda_=self.adv_erasure_lambda)

    def _kd_replay_teacher_loss(self, y_student, y_teacher, y_true, batch_groups):
        """K4: teacher-matching MSE with minority-group hypo windows upweighted.

        A window gets the replay factor if it belongs to the minority group and
        contains at least one true hypoglycemic timestep; all other windows keep
        weight 1.0. Weights are normalized to mean 1 so the overall loss scale
        (and thus its balance against the GT loss) is preserved.
        """
        per_window_se = ((y_student - y_teacher) ** 2).reshape(y_student.shape[0], -1).mean(dim=1)  # [B]

        if (not self.kd_replay_enabled) or (batch_groups is None):
            return per_window_se.mean()

        tgt_flat = y_true.reshape(y_true.shape[0], -1)
        is_minority = (batch_groups == self.kd_replay_minority_group)
        has_hypo = (tgt_flat < self.target_threshold).any(dim=1)
        upweight = (is_minority & has_hypo).float()

        weights = 1.0 + (self.kd_replay_factor - 1.0) * upweight  # [B]
        weights = weights / weights.mean().clamp_min(1e-8)        # normalize to mean 1
        return (weights * per_window_se).mean()

    def export_student_calibration_metadata(self):
        """Return serializable O2 calibration metadata for checkpoint sidecar save."""
        if not self.student_calibration_enabled:
            return None

        return {
            "feature": self.student_calibration_feature,
            "group_scales": [
                float(self.student_calibration_scale[0].detach().cpu().item()),
                float(self.student_calibration_scale[1].detach().cpu().item()),
            ],
            "group_biases": [
                float(self.student_calibration_bias[0].detach().cpu().item()),
                float(self.student_calibration_bias[1].detach().cpu().item()),
            ],
        }

    def train(self):
        train_loss_l = []
        for epoch in range(self.train_epochs):
            self.student.train()
            total_loss = 0.0
            total_loss_gt = 0.0
            total_loss_teacher = 0.0
            total_loss_fairness = 0.0
            total_o1_gap = 0.0
            total_o1_violation = 0.0
            total_o1_lambda = 0.0
            total_loss_align = 0.0
            total_loss_adv = 0.0

            mse_loss_fn = nn.MSELoss()

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}"):
                # Support 4-element batches (standard) and 5-element batches (with group labels)
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, batch_groups = batch
                    batch_groups = batch_groups.to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    batch_groups = None

                batch_x, batch_y, batch_x_mark, batch_y_mark = [
                    b.float().to(self.device) for b in (batch_x, batch_y, batch_x_mark, batch_y_mark)
                ]
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.context_len, :], dec_inp], dim=1)

                with torch.no_grad():
                    y_teacher = self.teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # T2: route each sample to its group-specialized teacher.
                    if self.multi_teacher_enabled and batch_groups is not None:
                        y_teacher_g0 = self.second_teacher(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        is_g0 = (batch_groups == self.multi_teacher_group0).view(-1, 1, 1)
                        y_teacher = torch.where(is_g0, y_teacher_g0, y_teacher)
                    y_teacher = self._apply_teacher_group_calibration(y_teacher, batch_groups)

                y_student = self.student(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                y_student = self._apply_student_group_calibration(y_student, batch_groups)
                y_true = batch_y[:, -self.pred_len :, :]

                # 1. Ground-truth loss
                loss_gt = mse_loss_fn(y_student, y_true)
                # 2. Teacher distillation loss (match teacher output).
                #    K4 selective KD replay upweights minority-group hypo windows;
                #    when disabled this reduces exactly to plain MSE.
                if self.kd_replay_enabled:
                    loss_teacher = self._kd_replay_teacher_loss(y_student, y_teacher, y_true, batch_groups)
                else:
                    loss_teacher = mse_loss_fn(y_student, y_teacher)

                # Combine losses
                loss = self.alpha * loss_gt + self.beta * loss_teacher

                # 3. Optional fairness regularisation (EO Gap on hypoglycemia detection)
                loss_fairness = torch.tensor(0.0, device=self.device)
                if self.eo_loss_fn is not None and batch_groups is not None:
                    unique_groups = torch.unique(batch_groups)
                    if len(unique_groups) == 2:
                        # EqualizedOddsLoss returns base_loss + fairness_penalty;
                        # here we only want the fairness penalty, so we pass a
                        # zero base_loss by using predictions == targets as base.
                        y_flat = y_student.reshape(y_student.shape[0], -1).mean(dim=1)
                        t_flat = y_true.reshape(y_true.shape[0], -1).mean(dim=1)
                        eo_combined = self.eo_loss_fn(y_flat, t_flat, batch_groups)
                        # eo_combined = base_loss(0) + fairness_penalty; subtract base
                        base_part = mse_loss_fn(y_flat, t_flat)
                        loss_fairness = eo_combined - base_part
                        loss = loss + self.fairness_weight * loss_fairness

                # 4. O1: hard-style fairness constraint via projected dual ascent
                o1_gap = torch.tensor(0.0, device=self.device)
                o1_violation = torch.tensor(0.0, device=self.device)
                if self.fairness_constraint_enabled and batch_groups is not None:
                    o1_gap = self._compute_soft_hypo_tpr_gap(y_student, y_true, batch_groups)
                    o1_violation = torch.relu(o1_gap - self.fairness_constraint_epsilon)
                    loss = loss + self.fairness_dual_lambda * o1_violation

                # 5. K3: fairness-aware feature alignment on student hidden states.
                #    The forward hook captured the student LLM's hidden state during
                #    the self.student(...) call above; align it across groups.
                loss_align = torch.tensor(0.0, device=self.device)
                if self.feature_alignment_loss_fn is not None and batch_groups is not None:
                    loss_align = self._compute_feature_alignment_loss(batch_groups)
                    loss = loss + self.feature_alignment_weight * loss_align

                # 6. O3: adversarial group erasure. The gradient-reversal layer
                #    inside the discriminator means adding this loss both trains
                #    the adversary and de-biases the student in one backward pass.
                loss_adv = torch.tensor(0.0, device=self.device)
                if self.adv_erasure_head is not None and batch_groups is not None:
                    loss_adv = self._compute_adv_erasure_loss(batch_groups)
                    loss = loss + loss_adv

                # Clear the capture so a missed hook on the next step can't reuse stale features.
                self._k3_captured_hidden = None

                self.optimizer.zero_grad()
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                if self.fairness_constraint_enabled and batch_groups is not None:
                    raw_violation = float((o1_gap - self.fairness_constraint_epsilon).detach().item())
                    self.fairness_dual_lambda = max(
                        0.0,
                        self.fairness_dual_lambda + self.fairness_dual_lr * raw_violation,
                    )

                total_loss += loss.item()
                total_loss_gt += loss_gt.item()
                total_loss_teacher += loss_teacher.item()
                total_loss_fairness += loss_fairness.item()
                total_o1_gap += o1_gap.item()
                total_o1_violation += o1_violation.item()
                total_o1_lambda += self.fairness_dual_lambda
                total_loss_align += float(loss_align.item())
                total_loss_adv += float(loss_adv.item())

            avg_loss = total_loss / len(self.dataloader)
            avg_loss_gt = total_loss_gt / len(self.dataloader)
            avg_loss_teacher = total_loss_teacher / len(self.dataloader)
            avg_loss_fairness = total_loss_fairness / len(self.dataloader)
            avg_o1_gap = total_o1_gap / len(self.dataloader)
            avg_o1_violation = total_o1_violation / len(self.dataloader)
            avg_o1_lambda = total_o1_lambda / len(self.dataloader)
            avg_loss_align = total_loss_align / len(self.dataloader)
            avg_loss_adv = total_loss_adv / len(self.dataloader)

            train_loss_l.append(avg_loss)

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1} | Total Loss: {avg_loss:.7f} | GT Loss: {avg_loss_gt:.7f} "
                    f"| Teacher Loss: {avg_loss_teacher:.7f} | Fairness Loss: {avg_loss_fairness:.7f} "
                    f"| O1 Gap: {avg_o1_gap:.7f} | O1 Viol: {avg_o1_violation:.7f} "
                    f"| O1 Lambda: {avg_o1_lambda:.5f} | K3 Align: {avg_loss_align:.7f} "
                    f"| O3 Adv: {avg_loss_adv:.7f}"
                )

            if self.early_stopping:
                # Provide a path to save the best model
                save_path = os.path.join("logs", "best_student.pth")
                self.early_stopping(avg_loss, self.student, save_path)
                if self.early_stopping.early_stop:
                    if self.logger:
                        self.logger.info("Early stopping triggered.")
                    break

        # Remove the K3 forward hook so the student model is left clean after training.
        if self._k3_hook_handle is not None:
            self._k3_hook_handle.remove()
            self._k3_hook_handle = None

        return train_loss_l
