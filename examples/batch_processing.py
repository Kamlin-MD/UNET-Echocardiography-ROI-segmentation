#!/usr/bin/env python3
"""
Batch Processing Example for UltrasoundROI

This example demonstrates batch processing of multiple ultrasound images
using the UltrasoundROI package.

Usage:
    python batch_processing.py --input_dir path/to/images/ --output_dir results/
"""

import argparse
import sys
import os
import time
from pathlib import Path

try:
    # Try importing from installed package
    from ultrasound_roi import UNetROISegmenter
except ImportError:
    # If not installed, try importing from local development
    sys.path.append(str(Path(__file__).parent.parent))
    from unet_inference import UltrasoundROISegmentation as UNetROISegmenter

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Batch processing example for UltrasoundROI')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input ultrasound images')
    parser.add_argument('--output_dir', type=str, default='batch_output/',
                       help='Directory to save processing results')
    parser.add_argument('--model', type=str, 
                       default='models/unet_ultrasound_roi.keras',
                       help='Path to trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Segmentation threshold (0.0-1.0)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--report', action='store_true',
                       help='Generate processing report with statistics')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    print("🔧 Initializing UltrasoundROI for batch processing...")
    
    try:
        # Initialize segmenter
        segmenter = UNetROISegmenter(args.model)
        
        print(f"📁 Processing images from: {args.input_dir}")
        
        # Process batch
        start_time = time.time()
        results = segmenter.process_batch(
            args.input_dir, 
            args.output_dir, 
            args.threshold
        )
        end_time = time.time()
        
        # Display results
        print(f"\n✅ Batch processing complete!")
        print(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        print(f"📊 Processing Statistics:")
        print(f"   - Successfully processed: {results['processed_count']} images")
        print(f"   - Failed: {results['failed_count']} images")
        print(f"   - Average processing time: {results.get('avg_processing_time', 0):.3f} seconds per image")
        print(f"   - Average ROI coverage: {results.get('avg_roi_coverage', 0)*100:.1f}%")
        
        print(f"\n📂 Output structure:")
        print(f"├── {args.output_dir}/masks/          # ROI masks")
        print(f"├── {args.output_dir}/roi_crops/      # Cropped ROI regions") 
        print(f"└── {args.output_dir}/deidentified/   # De-identified images")
        
        # Generate report if requested
        if args.report and results['processed_count'] > 0:
            generate_report(results, args.output_dir)
            
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        sys.exit(1)


def generate_report(results, output_dir):
    """Generate a processing report with statistics and visualizations."""
    print("\n📊 Generating processing report...")
    
    # Calculate additional statistics
    processing_times = results.get('processing_times', [])
    roi_coverages = results.get('roi_coverage', [])
    
    if not processing_times or not roi_coverages:
        print("⚠️  No detailed statistics available for report")
        return
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Processing time histogram
    axes[0, 0].hist(processing_times, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('Processing Time Distribution')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(processing_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(processing_times):.3f}s')
    axes[0, 0].legend()
    
    # ROI coverage histogram  
    axes[0, 1].hist([c*100 for c in roi_coverages], bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('ROI Coverage Distribution')
    axes[0, 1].set_xlabel('ROI Coverage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(roi_coverages)*100, color='red', linestyle='--',
                       label=f'Mean: {np.mean(roi_coverages)*100:.1f}%')
    axes[0, 1].legend()
    
    # Processing time vs ROI coverage scatter
    axes[1, 0].scatter(roi_coverages, processing_times, alpha=0.6)
    axes[1, 0].set_title('Processing Time vs ROI Coverage')
    axes[1, 0].set_xlabel('ROI Coverage (fraction)')
    axes[1, 0].set_ylabel('Processing Time (seconds)')
    
    # Summary statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
    Processing Summary
    
    Total Images: {results['processed_count']}
    Failed: {results['failed_count']}
    
    Processing Time:
    - Mean: {np.mean(processing_times):.3f}s
    - Std: {np.std(processing_times):.3f}s
    - Min: {np.min(processing_times):.3f}s
    - Max: {np.max(processing_times):.3f}s
    
    ROI Coverage:
    - Mean: {np.mean(roi_coverages)*100:.1f}%
    - Std: {np.std(roi_coverages)*100:.1f}%
    - Min: {np.min(roi_coverages)*100:.1f}%
    - Max: {np.max(roi_coverages)*100:.1f}%
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.suptitle('UltrasoundROI Batch Processing Report', fontsize=16, y=1.02)
    
    # Save report
    report_path = os.path.join(output_dir, 'processing_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print(f"📊 Report saved to: {report_path}")
    
    # Show plot
    plt.show()


if __name__ == "__main__":
    print("UltrasoundROI - Batch Processing Example")
    print("=" * 45)
    main()
