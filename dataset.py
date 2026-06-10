"""
dataset.py (improved)
---------------------
Key changes from original:
  - Default patch_size reduced to 128 (neurons are typically 20-60px wide)
  - Default stride reduced to 64 (75% overlap → ~4× more patches)
  - ElasticTransform alpha/sigma reduced to avoid destroying small cell bodies
  - get_train_transforms now accepts patch_size for correct resize
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(patch_size: int = 128) -> A.Compose:
    """
    Augmentation pipeline for single-image training.
    Elastic params are gentler than original to preserve small neuron bodies.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.6
        ),
        # Reduced alpha/sigma — original (120, 6) was too destructive for small cells
        A.ElasticTransform(
            alpha=40,
            sigma=5,
            alpha_affine=3,
            p=0.4
        ),
        A.GridDistortion(num_steps=3, distort_limit=0.2, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.03, 0.1), p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """No augmentation for validation — normalize only."""
    return A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2(),
    ])


class NeuronPatchDataset(Dataset):
    """
    Extracts overlapping patches from a single microscopy image.
    With patch_size=128, stride=64 you get roughly 4× as many patches
    as the original 256/128 setting, which is critical for single-image training.
    """

    def __init__(
        self,
        image_path: str,
        patch_size: int = 128,    # was 256
        stride: int = 64,          # was 128
        transform=None,
        min_fg_ratio: float = 0.001,  # skip nearly empty patches
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.min_fg_ratio = min_fg_ratio

        raw = tifffile.imread(image_path)
        raw = self._ensure_chw(raw)

        self.image = raw[0].astype(np.float32)
        self.mask  = raw[1].astype(np.float32)

        if self.mask.max() > 1:
            self.mask = (self.mask > 127).astype(np.float32)

        self.H, self.W = self.image.shape
        all_coords = self._compute_patch_coords()

        # Filter patches with almost no foreground (optional but helps focus training)
        self.patches = self._filter_empty(all_coords)
        print(f"  Patch extraction: {len(all_coords)} total → "
              f"{len(self.patches)} kept (fg_ratio > {min_fg_ratio})")

    @staticmethod
    def _ensure_chw(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[np.newaxis]
        if arr.shape[0] == 2:
            return arr
        if arr.shape[-1] == 2:
            return arr.transpose(2, 0, 1)
        raise ValueError(f"Unexpected tif shape: {arr.shape}")

    def _compute_patch_coords(self) -> list:
        coords = []
        for r in range(0, self.H - self.patch_size + 1, self.stride):
            for c in range(0, self.W - self.patch_size + 1, self.stride):
                coords.append((r, c))
        # Edge patches
        if (self.H - self.patch_size) % self.stride != 0:
            for c in range(0, self.W - self.patch_size + 1, self.stride):
                coords.append((self.H - self.patch_size, c))
        if (self.W - self.patch_size) % self.stride != 0:
            for r in range(0, self.H - self.patch_size + 1, self.stride):
                coords.append((r, self.W - self.patch_size))
        return list(set(coords))

    def _filter_empty(self, coords: list) -> list:
        """Remove patches that are almost entirely background."""
        kept = []
        ps = self.patch_size
        for (r, c) in coords:
            patch_mask = self.mask[r:r+ps, c:c+ps]
            if patch_mask.mean() >= self.min_fg_ratio:
                kept.append((r, c))
        return kept if len(kept) > 10 else coords  # fallback if too few kept

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        r, c = self.patches[idx]
        img_patch  = self.image[r:r+self.patch_size, c:c+self.patch_size]
        mask_patch = self.mask [r:r+self.patch_size, c:c+self.patch_size]

        img_patch = img_patch[:, :, np.newaxis]  # (H, W, 1)

        if self.transform:
            augmented  = self.transform(image=img_patch, mask=mask_patch)
            img_patch  = augmented["image"]
            mask_patch = augmented["mask"]

        return img_patch, mask_patch.unsqueeze(0)


class NeuronTestDataset(Dataset):
    """Sliding-window inference dataset (no mask required)."""

    def __init__(self, image_path: str, patch_size: int = 128, stride: int = 32):
        raw = tifffile.imread(image_path)
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
