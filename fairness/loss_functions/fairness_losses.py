"""
Fairness-Aware Loss Functions
============================

This module implements various fairness-aware loss functions that can be used
to train models with better fairness properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class FairnessAwareLoss(nn.Module):
    """Base class for fairness-aware loss functions."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0):
        """Initialize fairness-aware loss.
        
        Args:
            base_loss: Base loss function (e.g., MSELoss, CrossEntropyLoss)
            fairness_weight: Weight for the fairness penalty term
        """
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
        self.fairness_weight = fairness_weight
    
    def forward(self, predictions, targets, group_labels):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError


class DemographicParityLoss(FairnessAwareLoss):
    """Loss function that enforces demographic parity."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0):
        super().__init__(base_loss, fairness_weight)
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with demographic parity constraint.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Demographic parity penalty
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) < 2:
            return base_loss_value
        
        group_means = []
        for group in unique_groups:
            group_mask = (group_labels == group)
            if group_mask.sum() > 0:
                group_predictions = predictions[group_mask]
                group_mean = torch.mean(group_predictions)
                group_means.append(group_mean)
        
        if len(group_means) >= 2:
            # Penalty is the variance of group means
            group_means_tensor = torch.stack(group_means)
            fairness_penalty = torch.var(group_means_tensor)
        else:
            fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class EqualizedOddsLoss(FairnessAwareLoss):
    """Loss function that enforces equalized odds."""
    
    def __init__(self, base_loss: nn.Module = None, fairness_weight: float = 1.0,
                 pred_threshold: float = 0.5, target_threshold: Optional[float] = None):
        """Initialize equalized odds loss.
        
        Args:
            base_loss: Base loss function
            fairness_weight: Weight for the fairness penalty term
            pred_threshold: Threshold to binarize predictions. For probability outputs
                           use 0.5; for continuous glucose values use a clinical
                           threshold (e.g. 70.0 for hypoglycemia detection).
            target_threshold: Threshold to binarize targets. If None, uses
                             torch.median(targets) — only appropriate for generic
                             binary splits. For blood glucose set this explicitly
                             (e.g. 70.0 for hypoglycemia, 180.0 for hyperglycemia).
        """
        super().__init__(base_loss, fairness_weight)
        self.pred_threshold = pred_threshold
        self.target_threshold = target_threshold
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with equalized odds constraint.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Binarize using configurable thresholds
        pred_binary = (predictions > self.pred_threshold).float()
        if self.target_threshold is not None:
            target_binary = (targets > self.target_threshold).float()
        else:
            target_binary = (targets > torch.median(targets)).float()
        
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) != 2:
            return base_loss_value
        
        tprs = []
        fprs = []
        
        for group in unique_groups:
            group_mask = (group_labels == group)
            if group_mask.sum() == 0:
                continue
                
            group_pred = pred_binary[group_mask]
            group_target = target_binary[group_mask]
            
            # True Positive Rate
            tp = torch.sum((group_target == 1) & (group_pred == 1)).float()
            fn = torch.sum((group_target == 1) & (group_pred == 0)).float()
            tpr = tp / (tp + fn + 1e-8)
            tprs.append(tpr)
            
            # False Positive Rate
            fp = torch.sum((group_target == 0) & (group_pred == 1)).float()
            tn = torch.sum((group_target == 0) & (group_pred == 0)).float()
            fpr = fp / (fp + tn + 1e-8)
            fprs.append(fpr)
        
        if len(tprs) == 2 and len(fprs) == 2:
            tpr_diff = torch.abs(tprs[0] - tprs[1])
            fpr_diff = torch.abs(fprs[0] - fprs[1])
            fairness_penalty = tpr_diff + fpr_diff
        else:
            fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class GroupRegularizedLoss(FairnessAwareLoss):
    """Loss function with group-wise regularization."""
    
    def __init__(self, base_loss: nn.Module = None, 
                 fairness_weight: float = 1.0,
                 regularization_type: str = 'mse_difference'):
        """Initialize group regularized loss.
        
        Args:
            base_loss: Base loss function
            fairness_weight: Weight for fairness penalty
            regularization_type: Type of regularization ('mse_difference', 'performance_variance')
        """
        super().__init__(base_loss, fairness_weight)
        self.regularization_type = regularization_type
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss with group regularization.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_labels: Group membership indicators [batch_size]
            
        Returns:
            Combined loss with fairness penalty
        """
        # Base loss
        base_loss_value = self.base_loss(predictions, targets)
        
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) < 2:
            return base_loss_value
        
        if self.regularization_type == 'mse_difference':
            group_mses = []
            for group in unique_groups:
                group_mask = (group_labels == group)
                if group_mask.sum() > 0:
                    group_pred = predictions[group_mask]
                    group_target = targets[group_mask]
                    group_mse = F.mse_loss(group_pred, group_target)
                    group_mses.append(group_mse)
            
            if len(group_mses) >= 2:
                # Penalty is the difference between group MSEs
                group_mses_tensor = torch.stack(group_mses)
                fairness_penalty = torch.var(group_mses_tensor)
            else:
                fairness_penalty = torch.tensor(0.0, device=predictions.device)
                
        elif self.regularization_type == 'performance_variance':
            group_performances = []
            for group in unique_groups:
                group_mask = (group_labels == group)
                if group_mask.sum() > 0:
                    group_pred = predictions[group_mask]
                    group_target = targets[group_mask]
                    # Use negative MSE as performance (higher is better)
                    group_performance = -F.mse_loss(group_pred, group_target)
                    group_performances.append(group_performance)
            
            if len(group_performances) >= 2:
                group_performances_tensor = torch.stack(group_performances)
                fairness_penalty = torch.var(group_performances_tensor)
            else:
                fairness_penalty = torch.tensor(0.0, device=predictions.device)
        
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization_type}")
        
        return base_loss_value + self.fairness_weight * fairness_penalty


class HypoglycemiaTPREqualityLoss(FairnessAwareLoss):
    """Fairness loss that directly equalizes hypoglycemia detection TPR across groups.

    The key insight: standard EqualizedOddsLoss fails in practice because
    hypoglycemia windows (target < threshold) are rare (~1-5% of windows),
    so most batches have zero or near-zero hypo samples for one gender,
    making the gradient noisy and ineffective.

    This loss fixes that with two mechanisms:
    1. **Soft TPR**: uses a sigmoid-based soft threshold instead of hard
       binarization, giving a differentiable signal even with rare events.
    2. **Hypo-focal regression**: amplifies the regression loss on true
       hypoglycemia windows by a configurable focal factor, forcing the model
       to pay more attention to getting low-glucose predictions right.

    Result: the model is pushed to predict values below the threshold on
    true hypoglycemia windows for *all* groups equally.
    """

    def __init__(self, base_loss: nn.Module = None,
                 fairness_weight: float = 1.0,
                 hypo_threshold: float = 70.0,
                 focal_gamma: float = 3.0,
                 soft_slope: float = 0.1):
        """
        Args:
            base_loss:        Base regression loss (default MSELoss).
            fairness_weight:  Weight λ applied to the TPR-equalization penalty.
            hypo_threshold:   Glucose threshold for hypoglycemia (mg/dL).
            focal_gamma:      Multiplier on regression loss for true hypo windows.
                              Higher = more focus on not missing hypos.
            soft_slope:       Controls sharpness of the soft-threshold sigmoid.
                              Smaller = smoother gradient; 0.1 works well for
                              glucose-scale values.
        """
        super().__init__(base_loss, fairness_weight)
        self.hypo_threshold = hypo_threshold
        self.focal_gamma = focal_gamma
        self.soft_slope = soft_slope

    def _soft_positive(self, x: torch.Tensor) -> torch.Tensor:
        """Soft indicator: ~1 when x < threshold, ~0 otherwise (differentiable)."""
        return torch.sigmoid((self.hypo_threshold - x) / (self.hypo_threshold * self.soft_slope))

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                group_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions:   [B, T] or [B, T, 1] — denormalized glucose predictions.
            targets:       [B, T] or [B, T, 1] — denormalized glucose targets.
            group_labels:  [B] — integer group index per sample.
        """
        preds_flat = predictions.reshape(predictions.shape[0], -1)  # [B, T]
        tgts_flat  = targets.reshape(targets.shape[0], -1)           # [B, T]

        # ── 1. Focal regression loss ──────────────────────────────────────────
        # Upweight timesteps where the true glucose is hypo
        hypo_mask = (tgts_flat < self.hypo_threshold).float()          # [B, T]
        focal_weights = 1.0 + (self.focal_gamma - 1.0) * hypo_mask    # [B, T]
        sq_err = (preds_flat - tgts_flat) ** 2                         # [B, T]
        base_loss_value = (focal_weights * sq_err).mean()

        # ── 2. Soft-TPR equalization across groups ────────────────────────────
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) < 2:
            return base_loss_value

        soft_tprs = []
        for group in unique_groups:
            mask = (group_labels == group)           # [B]
            if mask.sum() == 0:
                continue

            g_preds  = preds_flat[mask]              # [n_g, T]
            g_tgts   = tgts_flat[mask]               # [n_g, T]

            # Which timesteps are truly hypo?
            true_hypo = (g_tgts < self.hypo_threshold).float()  # [n_g, T]
            n_true_hypo = true_hypo.sum() + 1e-8

            # Soft probability that prediction is also below threshold
            pred_hypo_soft = self._soft_positive(g_preds)        # [n_g, T]

            # Soft TPR = Σ (true_hypo * pred_hypo_soft) / Σ true_hypo
            soft_tpr = (true_hypo * pred_hypo_soft).sum() / n_true_hypo
            soft_tprs.append(soft_tpr)

        if len(soft_tprs) >= 2:
            # Penalty = variance of soft-TPRs across groups (0 when equal)
            tpr_tensor = torch.stack(soft_tprs)
            fairness_penalty = torch.var(tpr_tensor)
        else:
            fairness_penalty = torch.tensor(0.0, device=predictions.device)

        return base_loss_value + self.fairness_weight * fairness_penalty


class FeatureAlignmentLoss(nn.Module):
    """K3: fairness-aware feature alignment across protected groups.

    Penalizes the divergence between the *intermediate hidden-state
    distributions* of two groups (e.g. male vs female), forcing the student to
    learn group-invariant representations. Unlike the output-level losses above,
    this operates on the model's internal features, which makes it a stronger,
    representation-level fairness constraint.

    Two divergence measures are supported:

    - ``coral`` (default): CORAL aligns the second-order statistics (feature
      covariance matrices) of the two groups. It is cheap, has no bandwidth
      hyperparameter, and is stable on small / imbalanced batches — a good fit
      when one group's hypoglycemia windows are rare.
    - ``mmd``: a linear-time, multi-kernel (RBF) Maximum Mean Discrepancy
      between the two groups' pooled features. More sensitive to differences in
      distribution shape, but noisier when a group is underrepresented in a
      batch.

    The loss compares the two groups in the *same* representation space (e.g.
    student-male vs student-female), so no cross-model projection is needed even
    when teacher and student have different hidden dims.
    """

    def __init__(self, divergence: str = "coral",
                 mmd_kernel_muls: Optional[List[float]] = None):
        """
        Args:
            divergence:      'coral' (covariance matching) or 'mmd' (kernel MMD).
            mmd_kernel_muls: Bandwidth multipliers for the multi-kernel MMD,
                             relative to the median pairwise distance. Only used
                             when divergence == 'mmd'.
        """
        super().__init__()
        divergence = divergence.lower()
        if divergence not in ("coral", "mmd"):
            raise ValueError(f"Unknown divergence '{divergence}' (expected 'coral' or 'mmd')")
        self.divergence = divergence
        self.mmd_kernel_muls = mmd_kernel_muls or [0.5, 1.0, 2.0]

    @staticmethod
    def _coral(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scale-invariant CORAL: mean squared difference of *correlation*
        matrices between the two groups.

        source, target: [n_s, d] and [n_t, d] pooled feature matrices.

        We standardize each feature dimension by the pooled per-dim std before
        computing covariances, so the resulting matrices are correlation-like
        (unit diagonal) and the loss is intrinsically O(1) regardless of the
        raw feature magnitude. This is deliberate: the textbook 1/(4 d^2)
        absolute-covariance form produced a raw value of ~1e-4 on the BERT-tiny
        hidden states (covariance gap is tiny in absolute terms), which left the
        penalty ~1e5× too small to affect training even at weight=100. With this
        normalization a weight of ~1-10 is meaningful.
        """
        both = torch.cat([source, target], dim=0)
        # Pooled per-dimension std (detached: it's a normalizing scale, not a
        # parameter we want gradients to flow through as a target).
        std = both.std(dim=0, keepdim=True).clamp_min(1e-6).detach()
        s = source / std
        t = target / std

        def _cov(x):
            x_centered = x - x.mean(dim=0, keepdim=True)
            n = x.shape[0]
            return (x_centered.t() @ x_centered) / max(n - 1, 1)

        cs = _cov(s)
        ct = _cov(t)
        # Mean over the d×d entries keeps magnitude comparable across hidden dims.
        return ((cs - ct) ** 2).mean()

    def _mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-kernel RBF MMD^2 between two pooled feature sets."""
        s_n, t_n = source.shape[0], target.shape[0]
        total = torch.cat([source, target], dim=0)
        # Pairwise squared distances
        dists = torch.cdist(total, total) ** 2
        # Median heuristic for the base bandwidth (detached — it's a scale, not a param)
        with torch.no_grad():
            median = torch.median(dists[dists > 0]) if (dists > 0).any() else torch.tensor(1.0, device=total.device)
            median = torch.clamp(median, min=1e-6)

        kernel = torch.zeros_like(dists)
        for mul in self.mmd_kernel_muls:
            kernel = kernel + torch.exp(-dists / (median * mul + 1e-8))

        k_ss = kernel[:s_n, :s_n].mean()
        k_tt = kernel[s_n:, s_n:].mean()
        k_st = kernel[:s_n, s_n:].mean()
        return k_ss + k_tt - 2.0 * k_st

    def forward(self, features: torch.Tensor,
                group_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:     [B, d] pooled per-sample feature vectors.
            group_labels: [B] integer group index per sample (expects two groups).

        Returns:
            Scalar alignment penalty (0 when a group is absent from the batch).
        """
        unique_groups = torch.unique(group_labels)
        if len(unique_groups) != 2:
            return torch.tensor(0.0, device=features.device)

        g0 = features[group_labels == unique_groups[0]]
        g1 = features[group_labels == unique_groups[1]]
        # CORAL needs ≥2 samples per group for a covariance; MMD needs ≥1.
        min_required = 2 if self.divergence == "coral" else 1
        if g0.shape[0] < min_required or g1.shape[0] < min_required:
            return torch.tensor(0.0, device=features.device)

        if self.divergence == "coral":
            return self._coral(g0, g1)
        return self._mmd(g0, g1)


class _GradientReversalFn(torch.autograd.Function):
    """Identity forward; negates (and scales) the gradient on backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x, lambda_=1.0):
    """Apply a gradient-reversal layer with strength ``lambda_``."""
    return _GradientReversalFn.apply(x, lambda_)


class GroupAdversary(nn.Module):
    """O3: adversarial group-erasure discriminator.

    A small MLP that tries to predict the protected group from the student's
    pooled hidden representation. It is placed behind a gradient-reversal layer,
    so minimizing the discriminator's cross-entropy w.r.t. its own parameters
    trains it to predict the group, while the reversed gradient simultaneously
    pushes the student's representation to be group-*invariant* (group-erased).

    Returns the discriminator cross-entropy loss. With gradient reversal applied
    to the input, the same loss term trains the adversary and de-biases the
    student in a single backward pass.
    """

    def __init__(self, feature_dim: int, num_groups: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor, group_labels: torch.Tensor,
                lambda_: float = 1.0) -> torch.Tensor:
        """
        Args:
            features:     [B, d] pooled per-sample student features.
            group_labels: [B] integer group index per sample.
            lambda_:      gradient-reversal strength (student-side de-bias weight).
        """
        reversed_feats = gradient_reversal(features, lambda_)
        logits = self.net(reversed_feats)
        return self.ce(logits, group_labels.long())


class AdversarialFairnessLoss(nn.Module):
    """Adversarial fairness loss function."""
    
    def __init__(self, base_loss: nn.Module = None, 
                 adversarial_weight: float = 1.0):
        """Initialize adversarial fairness loss.
        
        Args:
            base_loss: Base loss function
            adversarial_weight: Weight for adversarial loss
        """
        super().__init__()
        self.base_loss = base_loss or nn.MSELoss()
        self.adversarial_weight = adversarial_weight
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor,
                group_predictions: torch.Tensor,
                group_labels: torch.Tensor) -> torch.Tensor:
        """Calculate adversarial fairness loss.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: True targets [batch_size, ...]
            group_predictions: Adversary's group predictions [batch_size, num_groups]
            group_labels: True group labels [batch_size]
            
        Returns:
            Combined loss for the main model
        """
        # Base prediction loss
        base_loss_value = self.base_loss(predictions, targets)
        
        # Adversarial loss (we want to fool the adversary)
        # Convert group_labels to one-hot if necessary
        if len(group_labels.shape) == 1:
            num_groups = len(torch.unique(group_labels))
            group_labels_onehot = F.one_hot(group_labels, num_groups).float()
        else:
            group_labels_onehot = group_labels
        
        # We want to minimize the adversary's ability to predict group membership
        # So we maximize the cross-entropy loss of the adversary
        adversarial_loss = -F.cross_entropy(group_predictions, group_labels_onehot.argmax(dim=1))
        
        return base_loss_value + self.adversarial_weight * adversarial_loss


class FairnessLossFactory:
    """Factory class for creating fairness-aware loss functions."""
    
    @staticmethod
    def create_loss(loss_type: str, 
                   base_loss: nn.Module = None,
                   fairness_weight: float = 1.0,
                   **kwargs) -> nn.Module:
        """Create a fairness-aware loss function.
        
        Args:
            loss_type: Type of fairness loss ('demographic_parity', 'equalized_odds', 
                      'group_regularized', 'adversarial')
            base_loss: Base loss function
            fairness_weight: Weight for fairness penalty
            **kwargs: Additional arguments for specific loss types
            
        Returns:
            Fairness-aware loss function
        """
        if loss_type == 'demographic_parity':
            return DemographicParityLoss(base_loss, fairness_weight)
        elif loss_type == 'equalized_odds':
            return EqualizedOddsLoss(base_loss, fairness_weight, **kwargs)
        elif loss_type == 'group_regularized':
            return GroupRegularizedLoss(base_loss, fairness_weight, **kwargs)
        elif loss_type == 'adversarial':
            return AdversarialFairnessLoss(base_loss, fairness_weight)
        else:
            raise ValueError(f"Unknown fairness loss type: {loss_type}")
    
    @staticmethod
    def get_available_losses() -> List[str]:
        """Get list of available fairness loss types."""
        return ['demographic_parity', 'equalized_odds', 'group_regularized', 'adversarial']


# Example usage and testing
if __name__ == "__main__":
    # Test fairness loss functions
    torch.manual_seed(42)
    
    batch_size = 32
    predictions = torch.randn(batch_size, 1)
    targets = torch.randn(batch_size, 1)
    group_labels = torch.randint(0, 2, (batch_size,))
    
    # Test different loss functions
    base_loss = nn.MSELoss()
    
    print("Testing Fairness Loss Functions")
    print("=" * 40)
    
    # Demographic Parity Loss
    dp_loss = DemographicParityLoss(base_loss, fairness_weight=0.5)
    dp_loss_value = dp_loss(predictions, targets, group_labels)
    print(f"Demographic Parity Loss: {dp_loss_value.item():.4f}")
    
    # Equalized Odds Loss
    eo_loss = EqualizedOddsLoss(base_loss, fairness_weight=0.5)
    eo_loss_value = eo_loss(predictions, targets, group_labels)
    print(f"Equalized Odds Loss: {eo_loss_value.item():.4f}")
    
    # Group Regularized Loss
    gr_loss = GroupRegularizedLoss(base_loss, fairness_weight=0.5)
    gr_loss_value = gr_loss(predictions, targets, group_labels)
    print(f"Group Regularized Loss: {gr_loss_value.item():.4f}")
    
    # Base loss for comparison
    base_loss_value = base_loss(predictions, targets)
    print(f"Base Loss (MSE): {base_loss_value.item():.4f}")
    
    print("\nFairness penalties successfully added to base loss!")