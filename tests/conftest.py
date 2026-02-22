"""Shared pytest fixtures for EchoROI test suite."""

import json
import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synthetic_labelme_dir(tmp_dir):
    """Create a small directory of synthetic LabelMe JSON + PNG pairs.

    Generates 5 fake 256x256 grayscale images with a triangular
    ROI polygon annotation – enough to exercise the dataset loaders.
    """
    for i in range(5):
        stem = f"sample_{i:03d}"
        # --- PNG ---
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, f"{stem}.png"), img)

        # --- JSON (LabelMe format) ---
        polygon = [[64, 20], [200, 128], [64, 236]]  # simple triangle
        annotation = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": "ROI sector",
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            "imagePath": f"{stem}.png",
            "imageData": None,
            "imageHeight": 256,
            "imageWidth": 256,
        }
        with open(os.path.join(tmp_dir, f"{stem}.json"), "w") as f:
            json.dump(annotation, f)

    return tmp_dir


@pytest.fixture
def synthetic_image_mask_dirs(tmp_dir):
    """Create paired image/ and mask/ directories with synthetic PNGs."""
    img_dir = os.path.join(tmp_dir, "images")
    msk_dir = os.path.join(tmp_dir, "masks")
    os.makedirs(img_dir)
    os.makedirs(msk_dir)

    for i in range(5):
        name = f"sample_{i:03d}.png"
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 80, 255, -1)
        cv2.imwrite(os.path.join(img_dir, name), img)
        cv2.imwrite(os.path.join(msk_dir, name), mask)

    return img_dir, msk_dir
