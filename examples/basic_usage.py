#!/usr/bin/env python3
"""
Basic Usage Example for UltrasoundROI

This example demonstrates the basic functionality of the UltrasoundROI package
for single image processing and ROI segmentation.

Requirements:
- UltrasoundROI package installed
- Sample ultrasound image
- Trained model weights

Usage:
    python basic_usage.py --image path/to/ultrasound_image.png --model path/to/model.keras
"""

import argparse
import sys
import os
from pathlib import Path

try:
    # Try importing from installed package
    from ultrasound_roi import UNetROISegmenter
except ImportError:
    # If not installed, try importing from local development
    sys.path.append(str(Path(__file__).parent.parent))
    from unet_inference import UltrasoundROISegmentation as UNetROISegmenter

import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Basic UltrasoundROI usage example')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input ultrasound image')
    parser.add_argument('--model', type=str, 
                       default='models/unet_ultrasound_roi.keras',
                       help='Path to trained model weights')
    parser.add_argument('--output', type=str, default='output/',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Segmentation threshold (0.0-1.0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization of results')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        print("💡 Tip: Train a model first or download pre-trained weights")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("🔧 Initializing UltrasoundROI segmentation...")
    
    try:
        # Initialize segmenter
        segmenter = UNetROISegmenter(args.model)
        
        print(f"🖼️  Processing image: {args.image}")
        
        # Process the image
        original, processed, roi_mask = segmenter.predict_single_image(
            args.image, args.threshold
        )
        
        # Extract ROI region
        roi_crop = segmenter.extract_roi(processed, roi_mask)
        
        # Create de-identified version
        deidentified = segmenter.apply_mask_for_deidentification(processed, roi_mask)
        
        # Save results
        base_name = Path(args.image).stem
        
        cv2.imwrite(os.path.join(args.output, f"{base_name}_mask.png"), roi_mask)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_roi_crop.png"), roi_crop)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_deidentified.png"), deidentified)
        
        # Calculate statistics
        total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
        roi_pixels = np.sum(roi_mask > 0)
        roi_coverage = roi_pixels / total_pixels
        
        print(f"✅ Processing complete!")
        print(f"📊 ROI Coverage: {roi_coverage:.1%} ({roi_pixels:,} pixels)")
        print(f"📁 Results saved to: {args.output}")
        print(f"   - ROI mask: {base_name}_mask.png")
        print(f"   - ROI crop: {base_name}_roi_crop.png") 
        print(f"   - De-identified: {base_name}_deidentified.png")
        
        # Visualization
        if args.visualize:
            print("🎨 Displaying results...")
            visualize_results(original, processed, roi_mask, roi_crop, deidentified)
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


def visualize_results(original, processed, roi_mask, roi_crop, deidentified):
    """Visualize processing results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Processed image
    axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Processed Image (Resized)')
    axes[0, 1].axis('off')
    
    # ROI mask
    axes[0, 2].imshow(roi_mask, cmap='gray')
    axes[0, 2].set_title('Predicted ROI Mask')
    axes[0, 2].axis('off')
    
    # ROI crop
    axes[1, 0].imshow(cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Extracted ROI')
    axes[1, 0].axis('off')
    
    # De-identified image
    axes[1, 1].imshow(cv2.cvtColor(deidentified, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('De-identified Image')
    axes[1, 1].axis('off')
    
    # Overlay visualization
    overlay = processed.copy()
    overlay[roi_mask > 0, 0] = np.minimum(overlay[roi_mask > 0, 0] + 50, 255)  # Add red tint
    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('ROI Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('UltrasoundROI Processing Results', fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    print("UltrasoundROI - Basic Usage Example")
    print("=" * 40)
    main()
