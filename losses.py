"""
losses.py
---------
Loss functions for binary segmentation with class imbalance.

References:
  - Dice Loss:        Milletari et al. (2016) V-Net [3DV]
  - Focal Loss:       Lin et al. (2017) RetinaNet [ICCV]
  - Combined BCE+Dice: Standard practice in medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss.
    Directly optimizes the Dice coefficient (= F1 score).
    More robust to class imbalance than cross-entropy alone.

    Loss = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred   : (B, 1, H, W) — sigmoid probabilities ∈ [0,1]
        target : (B, 1, H, W) — binary mask ∈ {0,1}
        """
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.
    BCE provides stable gradients; Dice handles class imbalance.
    Most commonly used combination in medical segmentation literature.

    Loss = α * BCE + (1-α) * DiceLoss
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.bce   = nn.BCELoss()
        self.dice  = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss.
    Focal loss down-weights easy background negatives (Lin et al., 2017).
    Useful when background pixels heavily outnumber foreground (neurons).

    FL(p) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice  = DiceLoss(smooth=smooth)

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(pred, target) + self.dice(pred, target)


def get_loss(name: str = "bce_dice") -> nn.Module:
    """
    Loss factory.
    name: "dice" | "bce_dice" | "focal_dice"
    """
    losses = {
        "dice":       DiceLoss(),
        "bce_dice":   BCEDiceLoss(alpha=0.5),
        "focal_dice": FocalDiceLoss(alpha=0.25, gamma=2.0),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(losses.keys())}")
    return losses[name]
