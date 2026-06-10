"""
losses.py (improved)
--------------------
Loss functions for binary segmentation with class imbalance.
Key changes from original:
  - BCEDiceLoss now accepts pos_weight to counter foreground/background imbalance
  - FocalDiceLoss alpha increased to 0.75 (more weight on foreground)
  - New helper: estimate_pos_weight() to auto-compute from the dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss — directly optimizes the Dice/F1 coefficient.
    Robust to class imbalance since it normalises by foreground area.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice with optional pos_weight.

    pos_weight: scalar weight applied to foreground class in BCE.
    For a dataset where neurons = 5% of pixels, a good starting value is
    (1 - 0.05) / 0.05 ≈ 19.  Use estimate_pos_weight() to compute from data.

    Loss = α * BCE(pos_weight) + (1-α) * DiceLoss
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0, pos_weight: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss(smooth=smooth)
        # pos_weight is registered as a buffer so it moves with .to(device)
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_fn   = nn.BCEWithLogitsLoss(pos_weight=self.pw.to(pred.device))
        # BCEWithLogitsLoss expects raw logits, but our model outputs sigmoid.
        # Convert back to logit space to use pos_weight correctly.
        pred_logit = torch.clamp(pred, 1e-7, 1 - 1e-7)
        pred_logit = torch.log(pred_logit / (1 - pred_logit))
        bce_loss  = bce_fn(pred_logit, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss.
    alpha=0.75 strongly upweights foreground (neurons are rare).
    gamma=2.0 down-weights easy background negatives.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice  = DiceLoss(smooth=smooth)

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy(pred, target, reduction="none")
        p_t     = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal   = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(pred, target) + self.dice(pred, target)


def estimate_pos_weight(dataset, n_samples: int = 200) -> float:
    """
    Estimate BCE pos_weight from the dataset.
    Samples n_samples patches and computes mean foreground fraction.
    Returns (1 - fg_frac) / fg_frac — standard class-frequency weighting.

    Usage:
        pw = estimate_pos_weight(full_dataset)
        criterion = get_loss("bce_dice", pos_weight=pw)
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    import random

    idx = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    loader = DataLoader(Subset(dataset, idx), batch_size=16, shuffle=False)

    total_fg, total_px = 0.0, 0.0
    for _, masks in loader:
        total_fg += masks.sum().item()
        total_px += masks.numel()

    fg_frac = total_fg / total_px
    if fg_frac < 1e-6:
        return 10.0  # fallback
    pw = (1.0 - fg_frac) / fg_frac
    print(f"  Foreground fraction: {fg_frac:.4f} → pos_weight = {pw:.1f}")
    return float(pw)


def get_loss(name: str = "focal_dice", pos_weight: float = 10.0) -> nn.Module:
    """
    Loss factory.
    name: "dice" | "bce_dice" | "focal_dice"
    pos_weight: only used for "bce_dice"
    """
    losses = {
        "dice":       DiceLoss(),
        "bce_dice":   BCEDiceLoss(alpha=0.5, pos_weight=pos_weight),
        "focal_dice": FocalDiceLoss(alpha=0.75, gamma=2.0),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(losses.keys())}")
    return losses[name]
