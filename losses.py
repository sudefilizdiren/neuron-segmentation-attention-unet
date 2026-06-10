import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_prob(pred):
    return torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = safe_prob(pred)
        target = target.float()

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        pred = safe_prob(pred)
        target = target.float()

        bce_loss = F.binary_cross_entropy(pred, target)
        dice_loss = self.dice(pred, target)

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss(smooth=smooth)

    def focal_loss(self, pred, target):
        pred = safe_prob(pred)
        target = target.float()

        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce

        return focal.mean()

    def forward(self, pred, target):
        return self.focal_loss(pred, target) + self.dice(pred, target)


def get_loss(name="bce_dice"):
    losses = {
        "dice": DiceLoss(),
        "bce_dice": BCEDiceLoss(alpha=0.5),
        "focal_dice": FocalDiceLoss(alpha=0.25, gamma=2.0),
    }

    if name not in losses:
        raise ValueError(f"Unknown loss: {name}")

    return losses[name]