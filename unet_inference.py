"""
UNET Ultrasound ROI Segmentation - Core Functions

This module provides the core functionality for ultrasound ROI segmentation
and de-identification using a trained UNET model.

Usage:
    from unet_inference import UltrasoundROISegmentation
    
    segmenter = UltrasoundROISegmentation('models/unet_model.keras')
    roi_mask = segmenter.predict_single_image('path/to/image.png')
    segmenter.process_batch('input_dir/', 'output_dir/')
"""

import os
import time
from glob import glob
from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
import tensorflow as tf


class UltrasoundROISegmentation:
    """
    UNET-based ultrasound ROI segmentation and de-identification tool.
    """
    
    def __init__(self, model_path: str, img_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the segmentation tool.
        
        Args:
            model_path (str): Path to trained UNET model
            img_size (Tuple[int, int]): Input image size for model
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained UNET model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")
    
    def resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image while preserving aspect ratio using padding.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Resized image with padding
        """
        h, w = img.shape[:2]
        
        # Calculate scaling factor to preserve aspect ratio
        scale = min(self.img_size[1] / w, self.img_size[0] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))

        # Calculate padding values
        top = (self.img_size[0] - new_h) // 2
        bottom = self.img_size[0] - new_h - top
        left = (self.img_size[1] - new_w) // 2
        right = self.img_size[1] - new_w - left

        # Apply padding based on image type
        if img.ndim == 2:  # Grayscale image
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=0)
        else:  # RGB image
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded
    
    def predict_single_image(self, image_path: str, 
                           threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single ultrasound image and generate ROI mask.
        
        Args:
            image_path (str): Path to input image
            threshold (float): Threshold for binary mask prediction
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (original_image, processed_image, roi_mask)
        """
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess for model input
        processed = self.resize_with_padding(original)
        normalized = processed.astype(np.float32) / 255.0
        input_batch = np.expand_dims(normalized, axis=0)
        
        # Generate prediction
        prediction = self.model.predict(input_batch, verbose=0)[0]
        roi_mask = (prediction.squeeze() > threshold).astype(np.uint8) * 255
        
        return original, processed, roi_mask
    
    def extract_roi(self, image: np.ndarray, mask: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Extract ROI region from image based on mask.
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Binary ROI mask
            padding (int): Padding around ROI bounding box
        
        Returns:
            np.ndarray: Cropped ROI region
        """
        # Find ROI bounding box
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image  # Return original if no ROI found
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        return roi
    
    def apply_mask_for_deidentification(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to create de-identified image by blacking out areas outside ROI.
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Binary ROI mask
        
        Returns:
            np.ndarray: De-identified image
        """
        deidentified = image.copy()
        
        # Convert mask to 3-channel if needed
        if len(image.shape) == 3 and len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask
        
        # Apply mask (black out areas outside ROI)
        deidentified = np.where(mask_3ch > 0, image, 0)
        
        return deidentified
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     threshold: float = 0.5) -> Dict:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir (str): Directory containing images to process
            output_dir (str): Directory to save results
            threshold (float): Threshold for binary mask prediction
        
        Returns:
            Dict: Processing results and statistics
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "roi_crops"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "deidentified"), exist_ok=True)
        
        # Get image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(input_dir, ext)))
        
        if not image_files:
            raise ValueError(f"No images found in directory: {input_dir}")
        
        results = {
            'processed_count': 0,
            'failed_count': 0,
            'processing_times': [],
            'roi_coverage': []
        }
        
        print(f"Processing {len(image_files)} images...")
        
        for i, image_path in enumerate(image_files):
            try:
                start_time = time.time()
                
                # Process image
                original, processed, roi_mask = self.predict_single_image(image_path, threshold)
                
                # Extract filename
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save ROI mask
                mask_path = os.path.join(output_dir, "masks", f"{filename}_mask.png")
                cv2.imwrite(mask_path, roi_mask)
                
                # Extract and save ROI crop
                roi_crop = self.extract_roi(processed, roi_mask)
                crop_path = os.path.join(output_dir, "roi_crops", f"{filename}_roi.png")
                cv2.imwrite(crop_path, roi_crop)
                
                # Create and save de-identified version
                deidentified = self.apply_mask_for_deidentification(processed, roi_mask)
                deident_path = os.path.join(output_dir, "deidentified", f"{filename}_deident.png")
                cv2.imwrite(deident_path, deidentified)
                
                # Record statistics
                processing_time = time.time() - start_time
                roi_coverage = np.sum(roi_mask > 0) / (roi_mask.shape[0] * roi_mask.shape[1])
                
                results['processing_times'].append(processing_time)
                results['roi_coverage'].append(roi_coverage)
                results['processed_count'] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                results['failed_count'] += 1
        
        # Calculate final statistics
        results['avg_processing_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
        results['avg_roi_coverage'] = np.mean(results['roi_coverage']) if results['roi_coverage'] else 0
        
        print(f"\n✅ Batch processing complete!")
        print(f"Successfully processed: {results['processed_count']} images")
        print(f"Failed: {results['failed_count']} images")
        print(f"Average processing time: {results['avg_processing_time']:.3f} seconds")
        print(f"Average ROI coverage: {results['avg_roi_coverage']:.3f} ({results['avg_roi_coverage']*100:.1f}%)")
        
        return results


def main():
    """Command-line interface for ultrasound ROI segmentation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UNET-based ultrasound ROI segmentation and de-identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  ultrasound-roi --model models/unet_model.keras --input image.png --output result.png
  
  # Batch process directory
  ultrasound-roi --model models/unet_model.keras --input_dir images/ --output_dir results/
  
  # Extract ROI only
  ultrasound-roi --model models/unet_model.keras --input image.png --roi_only
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained UNET model (.keras or .h5)')
    parser.add_argument('--input', '-i', 
                       help='Input image path (for single image processing)')
    parser.add_argument('--output', '-o',
                       help='Output image path (for single image processing)')
    parser.add_argument('--input_dir', 
                       help='Input directory path (for batch processing)')
    parser.add_argument('--output_dir',
                       help='Output directory path (for batch processing)')
    parser.add_argument('--roi_only', action='store_true',
                       help='Extract ROI crop only (no de-identification mask)')
    parser.add_argument('--img_size', default='256,256',
                       help='Model input size as width,height (default: 256,256)')
    
    args = parser.parse_args()
    
    # Parse image size
    try:
        width, height = map(int, args.img_size.split(','))
        img_size = (width, height)
    except ValueError:
        print("Error: img_size must be in format 'width,height' (e.g., '256,256')")
        return 1
    
    # Initialize segmenter
    try:
        segmenter = UltrasoundROISegmentation(args.model, img_size=img_size)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Single image processing
    if args.input:
        if not args.output:
            args.output = args.input.replace('.', '_segmented.')
        
        try:
            original, processed, roi_mask = segmenter.predict_single_image(args.input)
            
            if args.roi_only:
                # Extract and save ROI crop
                roi_crop = segmenter.extract_roi(processed, roi_mask)
                cv2.imwrite(args.output, roi_crop)
                print(f"✅ ROI extracted and saved to {args.output}")
            else:
                # Apply de-identification mask
                deidentified = segmenter.apply_mask_for_deidentification(processed, roi_mask)
                cv2.imwrite(args.output, deidentified)
                print(f"✅ De-identified image saved to {args.output}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return 1
    
    # Batch processing
    elif args.input_dir and args.output_dir:
        try:
            results = segmenter.process_batch(args.input_dir, args.output_dir)
            print(f"✅ Batch processing completed: {results['total_processed']} images processed")
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return 1
    
    else:
        print("Error: Please specify either --input/--output for single image or --input_dir/--output_dir for batch processing")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
