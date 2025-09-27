#!/usr/bin/env python3
"""
Example script demonstrating UNET Ultrasound ROI Segmentation usage.

This script shows how to:
1. Load the segmentation model
2. Process individual images
3. Process batches of images
4. Extract ROI regions
5. Create de-identified images

Requirements:
- Trained UNET model (models/unet_ultrasound_roi.keras)
- Test images in a directory
- Dependencies from requirements.txt

Usage:
    python example_usage.py --input_dir data/test_images --output_dir results/
"""

import argparse
import os
import sys
from unet_inference import UltrasoundROISegmentation


def main():
    parser = argparse.ArgumentParser(description='UNET Ultrasound ROI Segmentation Example')
    parser.add_argument('--model_path', type=str, default='models/unet_ultrasound_roi.keras',
                       help='Path to trained UNET model')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='output/',
                       help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for ROI mask prediction (0.0-1.0)')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Process single image instead of batch')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Please ensure you have a trained model or update the model path.")
        sys.exit(1)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    try:
        # Initialize segmentation tool
        print("🔧 Initializing UNET Ultrasound ROI Segmentation...")
        segmenter = UltrasoundROISegmentation(args.model_path)
        
        if args.single_image:
            # Process single image
            print(f"🖼️  Processing single image: {args.single_image}")
            single_image_path = os.path.join(args.input_dir, args.single_image)
            
            if not os.path.exists(single_image_path):
                print(f"❌ Image not found: {single_image_path}")
                sys.exit(1)
            
            # Process the image
            original, processed, roi_mask = segmenter.predict_single_image(
                single_image_path, args.threshold
            )
            
            # Extract ROI and create de-identified version
            roi_crop = segmenter.extract_roi(processed, roi_mask)
            deidentified = segmenter.apply_mask_for_deidentification(processed, roi_mask)
            
            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = os.path.splitext(args.single_image)[0]
            
            import cv2
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_mask.png"), roi_mask)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_roi.png"), roi_crop)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_deident.png"), deidentified)
            
            print(f"✅ Single image processed successfully!")
            print(f"Results saved to: {args.output_dir}")
            
        else:
            # Process batch of images
            print(f"📁 Processing batch of images from: {args.input_dir}")
            
            results = segmenter.process_batch(
                args.input_dir, 
                args.output_dir, 
                args.threshold
            )
            
            print(f"\n📊 Processing Summary:")
            print(f"✅ Successfully processed: {results['processed_count']} images")
            print(f"❌ Failed: {results['failed_count']} images")
            print(f"⏱️  Average processing time: {results.get('avg_processing_time', 0):.3f} seconds")
            print(f"📏 Average ROI coverage: {results.get('avg_roi_coverage', 0)*100:.1f}%")
            
            print(f"\n📂 Output structure:")
            print(f"├── {args.output_dir}/masks/          # ROI masks")
            print(f"├── {args.output_dir}/roi_crops/      # Cropped ROI regions")
            print(f"└── {args.output_dir}/deidentified/   # De-identified images")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("UNET Ultrasound ROI Segmentation - Example Usage")
    print("=" * 50)
    main()
