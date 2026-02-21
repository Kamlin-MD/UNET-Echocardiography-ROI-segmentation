"""
Preprocessing utilities for ultrasound image-mask datasets.

Handles loading LabelMe JSON annotations, generating binary masks,
and preparing image-mask pairs for U-Net training.
"""

import os
import cv2
import numpy as np
from glob import glob
from typing import Tuple, Optional, List
import warnings
import matplotlib.pyplot as plt


class UltrasoundPreprocessor:
    """Preprocess ultrasound images and ROI masks for UNet training."""

    def __init__(self, img_size: Tuple[int, int] = (256, 256)):
        self.img_size = img_size

    def resize_with_padding(self, img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image with padding while preserving aspect ratio."""
        if target_size is None:
            target_size = self.img_size
        h, w = img.shape[:2]
        th, tw = target_size

        scale = min(th / h, tw / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        pad_h, pad_w = (th - nh) // 2, (tw - nw) // 2
        if img.ndim == 3:
            padded = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
            padded[pad_h:pad_h+nh, pad_w:pad_w+nw, :] = resized
        else:
            padded = np.zeros((th, tw), dtype=img.dtype)
            padded[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
        return padded

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single ultrasound image to grayscale [0,1]."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img_resized = self.resize_with_padding(img)
        img_norm = (img_resized.astype(np.float32) / 255.0)[..., None]
        return img_norm

    def preprocess_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess a single ROI mask."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        mask_resized = self.resize_with_padding(mask)
        mask_bin = ((mask_resized > 127).astype(np.float32))[..., None]
        return mask_bin

    def validate_data_paths(self, image_dir: str, mask_dir: str) -> List[str]:
        """Validate dataset directories and return matched pairs."""
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError("Image or mask directory not found.")

        image_files = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))
        mask_files  = sorted(glob(os.path.join(mask_dir, "*.png")) + glob(os.path.join(mask_dir, "*.jpg")))

        if not image_files or not mask_files:
            raise ValueError("No images or masks found.")

        # match by basename
        img_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        mask_names = [os.path.splitext(os.path.basename(f))[0] for f in mask_files]

        matched = [(i, mask_files[mask_names.index(n)])
                   for i, n in zip(image_files, img_names) if n in mask_names]

        if len(matched) == 0:
            raise ValueError("No matching image-mask pairs found.")

        return matched

    def load_dataset(self, image_dir: str, mask_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load entire dataset of image-mask pairs."""
        pairs = self.validate_data_paths(image_dir, mask_dir)
        images, masks = [], []
        for img_path, mask_path in pairs:
            try:
                images.append(self.preprocess_image(img_path))
                masks.append(self.preprocess_mask(mask_path))
            except Exception as e:
                warnings.warn(f"Skipping {img_path}: {e}")
        X, Y = np.array(images, np.float32), np.array(masks, np.float32)
        print(f"Loaded {len(X)} samples → Images {X.shape}, Masks {Y.shape}")
        return X, Y

    def preprocess_for_inference(self, image_path: str) -> np.ndarray:
        """Prepare a single image for inference (batched)."""
        return np.expand_dims(self.preprocess_image(image_path), axis=0)


def print_data_statistics(X: np.ndarray, Y: np.ndarray):
    """Print basic statistics about dataset."""
    print("\n=== DATASET STATS ===")
    print(f"Samples: {len(X)} | Shape: {X.shape[1:]}")
    print(f"Image range: [{X.min():.3f}, {X.max():.3f}]  mean={X.mean():.3f} std={X.std():.3f}")
    roi_cov = np.mean(Y > 0.5)
    print(f"Mask ROI coverage: {roi_cov:.3f} ({roi_cov*100:.1f}%)")
    print("=====================")


def visualize_samples(X: np.ndarray, Y: np.ndarray, n=3):
    """Show random dataset samples with overlays."""
    idxs = np.random.choice(len(X), min(n, len(X)), replace=False)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(10, 4*len(idxs)))
    for row, i in enumerate(idxs):
        img, mask = X[i].squeeze(), Y[i].squeeze()
        overlay = np.dstack([img, img, img])
        overlay[mask > 0.5, 0] = 1.0  # red overlay
        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].set_title("Image"); axes[row, 0].axis("off")
        axes[row, 1].imshow(mask, cmap="gray"); axes[row, 1].set_title("Mask"); axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay); axes[row, 2].set_title("Overlay"); axes[row, 2].axis("off")
    plt.tight_layout(); plt.show()


def create_sample_data(output_dir: str, num_samples: int = 10) -> None:
    """Create synthetic ultrasound-like image/mask pairs for testing.

    Generates grayscale images with a random fan-shaped sector and
    matching binary masks.  Useful for verifying installation and
    running the CLI without real patient data.

    Args:
        output_dir: Root directory; images/ and masks/ subdirs are created.
        num_samples: Number of sample pairs to generate.
    """
    img_dir = os.path.join(output_dir, "images")
    msk_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for i in range(num_samples):
        name = f"sample_{i:03d}.png"
        h, w = 256, 256

        # --- Synthetic image: noise + bright fan sector ---
        img = np.random.randint(10, 40, (h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random fan-shaped polygon (sector apex near top-centre)
        cx = w // 2 + np.random.randint(-20, 20)
        apex_y = np.random.randint(10, 40)
        base_y = np.random.randint(200, 240)
        half_w = np.random.randint(80, 120)

        pts = np.array([
            [cx, apex_y],
            [cx - half_w, base_y],
            [cx + half_w, base_y],
        ], dtype=np.int32)

        cv2.fillConvexPoly(mask, pts, 255)
        # Add bright content inside the sector
        sector_noise = np.random.randint(80, 200, (h, w), dtype=np.uint8)
        img = np.where(mask > 0, sector_noise, img).astype(np.uint8)

        cv2.imwrite(os.path.join(img_dir, name), img)
        cv2.imwrite(os.path.join(msk_dir, name), mask)

    print(f"Created {num_samples} sample image/mask pairs in {output_dir}")