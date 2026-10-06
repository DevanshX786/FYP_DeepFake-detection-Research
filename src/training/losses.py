from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for handling severe class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_criterion(loss_type: str = "BCEWithLogitsLoss", pos_weight: Optional[float] = None) -> nn.Module:
    """Get loss function module.

    Args:
        loss_type: 'BCEWithLogitsLoss' or 'FocalLoss'.
        pos_weight: Optional positive weight float for class imbalance.

    Returns:
        PyTorch loss module.
    """
    if loss_type == "BCEWithLogitsLoss":
        pw_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    elif loss_type == "FocalLoss":
        return BinaryFocalLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
