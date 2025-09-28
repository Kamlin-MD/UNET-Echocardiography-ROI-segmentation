#!/usr/bin/env python3
"""
DICOM Video Showcase Script

This script:
1. Loads a DICOM file with multiple frames
2. Extracts the first frame and generates a mask using the trained model
3. Applies the mask to all frames in the sequence
4. Creates a compiled AVI video showcasing the segmentation results

Usage:
    python dicom_video_showcase.py --dicom-path input.dcm --model-path models/unet_EchoRoi.keras --output showcase_video.avi
"""

import argparse
import os
import sys
import numpy as np
import cv2
import pydicom
from pathlib import Path
import tempfile
import shutil
from typing import Tuple, List, Optional

# Add the package to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from unet_roi.inference import UNetPredictor
    from unet_roi.preprocessing import UltrasoundPreprocessor
except ImportError:
    print("Error: Please install the unet-roi package first with: pip install -e .")
    sys.exit(1)


class DICOMVideoProcessor:
    """Process DICOM files and create showcase videos."""
    
    def __init__(self, model_path: str):
        """Initialize the processor with a trained model.
        
        Args:
            model_path: Path to the trained U-Net model
        """
        self.predictor = UNetPredictor(model_path)
        self.preprocessor = UltrasoundPreprocessor()
        
    def load_dicom_frames(self, dicom_path: str) -> Tuple[np.ndarray, dict]:
        """Load all frames from a DICOM file.
        
        Args:
            dicom_path: Path to the DICOM file
            
        Returns:
            Tuple of (frames_array, metadata)
        """
        try:
            # Load DICOM file
            ds = pydicom.dcmread(dicom_path)
            
            # Extract metadata
            metadata = {
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'rows': ds.Rows,
                'columns': ds.Columns,
                'frames': getattr(ds, 'NumberOfFrames', 1)
            }
            
            # Extract pixel data
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                # Multi-frame DICOM
                pixel_array = ds.pixel_array
                if len(pixel_array.shape) == 3:
                    frames = pixel_array
                else:
                    frames = np.expand_dims(pixel_array, axis=0)
            else:
                # Single frame DICOM
                frames = np.expand_dims(ds.pixel_array, axis=0)
                
            print(f"Loaded DICOM with {frames.shape[0]} frames of size {frames.shape[1]}x{frames.shape[2]}")
            return frames, metadata
            
        except Exception as e:
            raise ValueError(f"Error loading DICOM file {dicom_path}: {str(e)}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame for model input.
        
        Args:
            frame: Raw DICOM frame
            
        Returns:
            Preprocessed frame ready for model
        """
        # Normalize to 0-255 range
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert to RGB (repeat grayscale across 3 channels)
        frame_rgb = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2RGB)
        
        # Resize with padding to model input size
        frame_resized = self.preprocessor.resize_with_padding(frame_rgb)
        
        # Normalize to [0, 1]
        frame_final = frame_resized.astype(np.float32) / 255.0
        
        return frame_final
    
    def generate_reference_mask(self, first_frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Generate mask from the first frame to use as reference.
        
        Args:
            first_frame: First frame of the DICOM sequence
            threshold: Threshold for mask binarization
            
        Returns:
            Binary mask for the frame
        """
        # Preprocess frame
        frame_processed = self.preprocess_frame(first_frame)
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_processed, axis=0)
        
        # Generate mask
        prediction = self.predictor.model.predict(frame_batch, verbose=0)
        mask = prediction[0]  # Remove batch dimension
        
        # Binarize mask
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        binary_mask = binary_mask.squeeze()  # Remove channel dimension if present
        
        return binary_mask
    
    def create_showcase_frame(self, 
                            original_frame: np.ndarray, 
                            mask: np.ndarray, 
                            frame_number: int) -> np.ndarray:
        """Create a showcase frame combining original, mask, and overlay.
        
        Args:
            original_frame: Original DICOM frame
            mask: Binary segmentation mask
            frame_number: Frame number for annotation
            
        Returns:
            Showcase frame with all visualizations
        """
        # Normalize original frame
        original_norm = cv2.normalize(original_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        original_rgb = cv2.cvtColor(original_norm, cv2.COLOR_GRAY2RGB)
        
        # Resize original and mask to same size
        target_size = (256, 256)
        original_resized = cv2.resize(original_rgb, target_size)
        mask_resized = cv2.resize(mask, target_size)
        
        # Create mask visualization (colored)
        mask_colored = np.zeros_like(original_resized)
        mask_colored[:, :, 1] = mask_resized  # Green channel for mask
        
        # Create overlay
        overlay = cv2.addWeighted(original_resized, 0.7, mask_colored, 0.3, 0)
        
        # Create combined visualization
        # Top row: Original | Mask
        # Bottom row: Overlay | ROI extracted
        top_row = np.hstack([original_resized, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)])
        
        # Extract ROI
        roi_extracted = original_resized.copy()
        roi_extracted[mask_resized == 0] = 0
        
        bottom_row = np.hstack([overlay, roi_extracted])
        
        # Combine all
        showcase = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        
        cv2.putText(showcase, "Original", (10, 25), font, font_scale, color, thickness)
        cv2.putText(showcase, "Mask", (266, 25), font, font_scale, color, thickness)
        cv2.putText(showcase, "Overlay", (10, 281), font, font_scale, color, thickness)
        cv2.putText(showcase, "ROI Extracted", (266, 281), font, font_scale, color, thickness)
        
        # Add frame number
        cv2.putText(showcase, f"Frame: {frame_number}", (10, showcase.shape[0] - 10), 
                   font, font_scale, (0, 255, 255), thickness)
        
        return showcase
    
    def process_dicom_to_video(self, 
                             dicom_path: str, 
                             output_path: str, 
                             fps: int = 15,
                             threshold: float = 0.5) -> None:
        """Process entire DICOM file and create showcase video.
        
        Args:
            dicom_path: Path to input DICOM file
            output_path: Path for output AVI file
            fps: Frames per second for output video
            threshold: Threshold for mask generation
        """
        print("🎬 Starting DICOM Video Showcase Processing...")
        
        # Load DICOM frames
        frames, metadata = self.load_dicom_frames(dicom_path)
        num_frames = len(frames)
        
        print(f"📊 DICOM Info:")
        print(f"   Patient ID: {metadata['patient_id']}")
        print(f"   Modality: {metadata['modality']}")
        print(f"   Frames: {num_frames}")
        print(f"   Size: {metadata['rows']}x{metadata['columns']}")
        
        # Generate reference mask from first frame
        print("🎯 Generating reference mask from first frame...")
        reference_mask = self.generate_reference_mask(frames[0], threshold)
        
        # Setup video writer
        # Showcase frame will be 512x512 (2x2 grid of 256x256 images)
        frame_size = (512, 512)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        print(f"🎥 Processing {num_frames} frames...")
        
        try:
            for i, frame in enumerate(frames):
                # Create showcase frame
                showcase_frame = self.create_showcase_frame(frame, reference_mask, i + 1)
                
                # Write frame to video
                out.write(showcase_frame)
                
                # Progress indicator
                if (i + 1) % 10 == 0 or i == num_frames - 1:
                    progress = (i + 1) / num_frames * 100
                    print(f"   Progress: {progress:.1f}% ({i + 1}/{num_frames} frames)")
        
        finally:
            out.release()
        
        print(f"✅ Showcase video created: {output_path}")
        print(f"📹 Video specs: {frame_size[0]}x{frame_size[1]} @ {fps} FPS")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Create showcase video from DICOM ultrasound sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python dicom_video_showcase.py --dicom-path ultrasound.dcm --model-path models/unet_EchoRoi.keras --output showcase.avi
  
  # Custom settings
  python dicom_video_showcase.py --dicom-path data.dcm --model-path model.keras --output demo.avi --fps 30 --threshold 0.6
        """
    )
    
    parser.add_argument('--dicom-path', required=True,
                       help='Path to input DICOM file')
    parser.add_argument('--model-path', required=True,
                       help='Path to trained U-Net model (.keras file)')
    parser.add_argument('--output', required=True,
                       help='Output AVI file path')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second for output video (default: 15)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for mask binarization (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dicom_path):
        print(f"Error: DICOM file not found: {args.dicom_path}")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize processor
        processor = DICOMVideoProcessor(args.model_path)
        
        # Process DICOM to video
        processor.process_dicom_to_video(
            dicom_path=args.dicom_path,
            output_path=args.output,
            fps=args.fps,
            threshold=args.threshold
        )
        
        print(f"\n🎉 Successfully created showcase video!")
        print(f"📁 Output: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
