"""
dataset.py
----------
Patch-based dataset for single-image neuron segmentation.
Since we have only ONE training image, we:
  1. Extract overlapping patches to multiply training samples
  2. Apply heavy augmentation at each call (elastic, flip, rotate, intensity)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(patch_size: int = 256) -> A.Compose:
    """
    Heavy augmentation pipeline for training.
    Elastic deformations are the most important augmentation for biomedical
    images (Ronneberger et al., 2015; Simard et al., 2003).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=0,
            p=0.7
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.5
        ),
        A.GridDistortion(p=0.3),
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=0.0, std=1.0),  # z-score normalization
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Minimal transforms for validation/test (only normalize)."""
    return A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuronPatchDataset(Dataset):
    """
    Extracts overlapping patches from a single microscopy image.

    Parameters
    ----------
    image_path   : path to .tif file (2-channel: phase-contrast + mask)
    patch_size   : spatial size of each patch (default 256)
    stride       : stride for patch extraction (default 128 = 50% overlap)
    transform    : albumentations transform pipeline
    """

    def __init__(
        self,
        image_path: str,
        patch_size: int = 256,
        stride: int = 128,
        transform=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # ---- Load .tif (2 channels) ----
        raw = tifffile.imread(image_path)  # shape: (2, H, W) or (H, W, 2)
        raw = self._ensure_chw(raw)        # → (2, H, W)

        self.image = raw[0].astype(np.float32)  # phase-contrast channel
        self.mask  = raw[1].astype(np.float32)  # binary neuron mask

        # Normalize mask to {0, 1}
        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.float32)

        self.H, self.W = self.image.shape
        self.patches = self._compute_patch_coords()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _ensure_chw(arr: np.ndarray) -> np.ndarray:
        """Make sure array is (C, H, W)."""
        if arr.ndim == 2:
            return arr[np.newaxis]          # single-channel edge case
        if arr.shape[0] == 2:
            return arr                      # already (2, H, W)
        if arr.shape[-1] == 2:
            return arr.transpose(2, 0, 1)  # (H, W, 2) → (2, H, W)
        raise ValueError(f"Unexpected tif shape: {arr.shape}")

    def _compute_patch_coords(self) -> list:
        """Return list of (row, col) top-left corners for all patches."""
        coords = []
        for r in range(0, self.H - self.patch_size + 1, self.stride):
            for c in range(0, self.W - self.patch_size + 1, self.stride):
                coords.append((r, c))
        # Edge patches (right/bottom border)
        if (self.H - self.patch_size) % self.stride != 0:
            for c in range(0, self.W - self.patch_size + 1, self.stride):
                coords.append((self.H - self.patch_size, c))
        if (self.W - self.patch_size) % self.stride != 0:
            for r in range(0, self.H - self.patch_size + 1, self.stride):
                coords.append((r, self.W - self.patch_size))
        return list(set(coords))  # deduplicate

    # ------------------------------------------------------------------ Dataset API

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        r, c = self.patches[idx]
        img_patch  = self.image[r:r+self.patch_size, c:c+self.patch_size]
        mask_patch = self.mask [r:r+self.patch_size, c:c+self.patch_size]

        # albumentations expects HWC for image
        img_patch = img_patch[:, :, np.newaxis]   # (H, W, 1)

        if self.transform:
            augmented  = self.transform(image=img_patch, mask=mask_patch)
            img_patch  = augmented["image"]   # Tensor (1, H, W)
            mask_patch = augmented["mask"]    # Tensor (H, W)
        
        return img_patch, mask_patch.unsqueeze(0)  # (1,H,W), (1,H,W)


# ---------------------------------------------------------------------------
# Test-time inference dataset (no mask, full sliding window)
# ---------------------------------------------------------------------------

class NeuronTestDataset(Dataset):
    """
    Sliding-window patch extraction for test image inference.
    No mask required.
    """

    def __init__(self, image_path: str, patch_size: int = 256, stride: int = 64):
        raw = tifffile.imread(image_path)

        # Test image may be single-channel or multi-channel
        if raw.ndim == 2:
            self.image = raw.astype(np.float32)
        elif raw.ndim == 3 and raw.shape[0] == 2:
            self.image = raw[0].astype(np.float32)
        elif raw.ndim == 3 and raw.shape[-1] == 2:
            self.image = raw[:, :, 0].astype(np.float32)
        else:
            self.image = raw.astype(np.float32)
            if self.image.ndim == 3:
                self.image = self.image[0]

        self.H, self.W = self.image.shape
        self.patch_size = patch_size
        self.stride = stride
        self.transform = get_val_transforms()
        self.patches = self._compute_patch_coords()

    def _compute_patch_coords(self):
        coords = []
        for r in range(0, max(1, self.H - self.patch_size + 1), self.stride):
            for c in range(0, max(1, self.W - self.patch_size + 1), self.stride):
                coords.append((r, c))
        return coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        r, c = self.patches[idx]
        patch = self.image[r:r+self.patch_size, c:c+self.patch_size]
        patch = patch[:, :, np.newaxis]
        aug = self.transform(image=patch, mask=np.zeros(patch.shape[:2], dtype=np.float32))
        return aug["image"], (r, c)
