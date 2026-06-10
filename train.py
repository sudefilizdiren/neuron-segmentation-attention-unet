"""
train.py
--------
Full training pipeline for Attention U-Net neuron segmentation.

Usage:
    python train.py --train_path data/trainImage_ph1.tif \
                    --model vanilla \
                    --epochs 100 \
                    --loss bce_dice

Training strategy for single-image setting:
  - Patch-based training (256×256 patches, 128 stride)
  - 80/20 patch-level train/val split
  - Heavy augmentation (elastic, flip, rotate, intensity)
  - ReduceLROnPlateau scheduler
  - Early stopping (patience=20)
  - Best model checkpoint saved by validation Dice
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.dataset import NeuronPatchDataset, get_train_transforms, get_val_transforms
from models.attention_unet import build_model
from utils.losses import get_loss
from utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Train / Validation epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_masks = [], []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        total_loss += loss.item()
        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze()  # (N, H, W)
    all_masks = np.concatenate(all_masks, axis=0).squeeze()

    # Flatten for metric computation
    metrics = compute_all_metrics(all_preds.flatten(), all_masks.flatten())
    return total_loss / len(loader), metrics


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_dice"], label="Val Dice", color="#4CAF50")
    axes[1].plot(history["val_iou"],  label="Val IoU",  color="#FF9800")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → Training curves saved: {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model variant: {args.model}")
    print(f"Loss: {args.loss}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir,    exist_ok=True)

    # ---- Datasets ----
    # Full dataset (all patches, train transforms)
    full_dataset = NeuronPatchDataset(
        image_path=args.train_path,
        patch_size=args.patch_size,
        stride=args.stride,
        transform=get_train_transforms(args.patch_size),
    )

    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    # Validation subset uses val transforms (no augmentation)
    val_dataset = NeuronPatchDataset(
        image_path=args.train_path,
        patch_size=args.patch_size,
        stride=args.stride,
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    print(f"Total patches : {n}")
    print(f"Train patches : {len(train_idx)}")
    print(f"Val patches   : {len(val_idx)}\n")

    # ---- Model, Loss, Optimizer ----
    model     = build_model(args.model, device)
    criterion = get_loss(args.loss)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )

    # ---- Training Loop ----
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(metrics["dice"])
        history["val_iou"].append(metrics["iou"])

        scheduler.step(metrics["dice"])

        # ---- Logging ----
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Rec: {metrics['recall']:.4f}"
        )

        # ---- Checkpoint (best Dice) ----
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            patience_counter = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "dice":        best_dice,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best Dice={best_dice:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ---- Final report ----
    print(f"\n{'='*50}")
    print(f"Training complete. Best Val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    plot_curves(history, args.results_dir)
    return history, best_dice


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Attention U-Net Neuron Segmentation")
    parser.add_argument("--train_path",     type=str,   default="data/trainImage_ph1.tif")
    parser.add_argument("--model",          type=str,   default="vanilla",
                        choices=["vanilla", "pretrained"])
    parser.add_argument("--loss",           type=str,   default="bce_dice",
                        choices=["dice", "bce_dice", "focal_dice"])
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--patch_size",     type=int,   default=256)
    parser.add_argument("--stride",         type=int,   default=128)
    parser.add_argument("--patience",       type=int,   default=20)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--results_dir",    type=str,   default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
