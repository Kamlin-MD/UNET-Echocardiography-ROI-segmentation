#!/usr/bin/env python3
"""One-time script: convert LabelMe JSON annotations to PNG masks.

Reads every *.json in UNET_training_dataset/, rasterises the 'ROI sector'
polygon into a binary mask (0/255 PNG), and copies both the source image
and the generated mask into data/images/ and data/masks/ respectively.

This merges the EchoNet-Dynamic + EchoNet-Paeds annotations with the
existing MIMIC-IV-ECHO / Cardiac-UDC data already in data/.

Usage:
    python scripts/convert_labelme_to_masks.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# --- Configuration -----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LABELME_DIR = REPO_ROOT / "UNET_training_dataset"
IMG_OUT_DIR = REPO_ROOT / "data" / "images"
MASK_OUT_DIR = REPO_ROOT / "data" / "masks"
LABEL = "ROI sector"


def rasterise_mask(json_path: Path) -> np.ndarray:
    """Read a LabelMe JSON and return a uint8 binary mask (0 or 255)."""
    with open(json_path) as f:
        data = json.load(f)
    h = data.get("imageHeight", 256)
    w = data.get("imageWidth", 256)
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data.get("shapes", []):
        if shape.get("label") != LABEL:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def main() -> None:
    if not LABELME_DIR.is_dir():
        print(f"ERROR: {LABELME_DIR} not found.", file=sys.stderr)
        sys.exit(1)

    IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(LABELME_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON annotations in {LABELME_DIR.name}/")

    copied, skipped, errors = 0, 0, 0
    for jf in json_files:
        stem = jf.stem
        src_png = jf.with_suffix(".png")
        dst_img = IMG_OUT_DIR / f"{stem}.png"
        dst_mask = MASK_OUT_DIR / f"{stem}.png"

        # Skip if already converted
        if dst_img.exists() and dst_mask.exists():
            skipped += 1
            continue

        if not src_png.exists():
            print(f"  WARN: no PNG for {jf.name}, skipping")
            errors += 1
            continue

        try:
            # Copy image
            img = cv2.imread(str(src_png), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read {src_png}")
            cv2.imwrite(str(dst_img), img)

            # Generate and save mask
            mask = rasterise_mask(jf)
            cv2.imwrite(str(dst_mask), mask)
            copied += 1
        except Exception as e:
            print(f"  ERROR processing {stem}: {e}")
            errors += 1

    print(f"\nDone: {copied} converted, {skipped} already existed, {errors} errors")
    total_imgs = len(list(IMG_OUT_DIR.glob("*.png")))
    total_masks = len(list(MASK_OUT_DIR.glob("*.png")))
    print(f"Total in data/images/: {total_imgs}")
    print(f"Total in data/masks/:  {total_masks}")


if __name__ == "__main__":
    main()
