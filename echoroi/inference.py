"""Inference utilities for the EchoROI U-Net model."""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from .model import load_pretrained_model
from .preprocessing import UltrasoundPreprocessor


class UNetPredictor:
    """Inference class for U-Net model."""
    
    def __init__(self, model_path: str, img_size: Tuple[int, int] = (256, 256)):
        """Initialize predictor.
        
        Args:
            model_path: Path to the trained model
            img_size: Input image size
        """
        self.model_path = model_path
        self.img_size = img_size
        self.preprocessor = UltrasoundPreprocessor(img_size)
        self.model = None
        
        # Load model
        self.load_model()
        
    def load_model(self) -> None:
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading model from: {self.model_path}")
        self.model = load_pretrained_model(self.model_path)
        print("Model loaded successfully!")
        
    def predict_single_image(self, 
                           image_path: str, 
                           threshold: float = 0.5,
                           return_original: bool = False) -> tuple:
        """Predict mask for a single image.
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binarizing the prediction
            return_original: Whether to return the original image
            
        Returns:
            Tuple of (predicted_mask, [original_image if requested])
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Preprocess image
        img_processed = self.preprocessor.preprocess_for_inference(image_path)
        
        # Make prediction
        prediction = self.model.predict(img_processed, verbose=0)
        predicted_mask = prediction[0]  # Remove batch dimension
        
        # Apply threshold
        predicted_binary = (predicted_mask > threshold).astype(np.uint8) * 255
        predicted_binary = predicted_binary.squeeze()  # Remove channel dimension
        
        if return_original:
            original = cv2.imread(image_path, cv2.IMREAD_COLOR)
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            return predicted_binary, original
        
        return predicted_binary
    
    def predict_batch(self, 
                     image_paths: list, 
                     threshold: float = 0.5) -> list:
        """Predict masks for a batch of images.
        
        Args:
            image_paths: List of image file paths
            threshold: Threshold for binarizing predictions
            
        Returns:
            List of predicted binary masks
        """
        predictions = []
        
        for image_path in image_paths:
            try:
                mask = self.predict_single_image(image_path, threshold)
                predictions.append(mask)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                predictions.append(None)
                
        return predictions
    
    def extract_roi(self, 
                   image: np.ndarray, 
                   mask: np.ndarray, 
                   padding: int = 10) -> np.ndarray:
        """Extract ROI from image using predicted mask.
        
        Applies the mask to zero-out non-ROI pixels, then crops to the
        bounding box of the mask contour.
        
        Args:
            image: Original image (grayscale or RGB)
            mask: Binary mask (0/255)
            padding: Padding around the ROI bounding box
            
        Returns:
            Cropped ROI image with non-ROI pixels blacked out
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No contours found in mask")
            return image
            
        # Apply mask — black out everything outside the ROI
        binary = (mask > 127).astype(np.uint8)
        if image.ndim == 3:
            masked = image * binary[:, :, np.newaxis]
        else:
            masked = image * binary
            
        # Get bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(masked.shape[1], x + w + padding)
        y_end = min(masked.shape[0], y + h + padding)
        
        # Crop to bounding box
        roi = masked[y_start:y_end, x_start:x_end]
        
        return roi
    
    def apply_mask_for_deidentification(self, 
                                      image: np.ndarray, 
                                      mask: np.ndarray,
                                      mask_value: int = 0) -> np.ndarray:
        """Apply mask for deidentification (black out non-ROI areas).
        
        Args:
            image: Original image
            mask: Binary mask (ROI = 255, background = 0)
            mask_value: Value to use for masked areas
            
        Returns:
            Deidentified image
        """
        # Ensure mask is binary
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Create deidentified image
        deidentified = image.copy()
        deidentified[binary_mask == 0] = mask_value
        
        return deidentified
    
    def process_image_with_visualization(self, 
                                       image_path: str, 
                                       threshold: float = 0.5,
                                       save_path: Optional[str] = None) -> dict:
        """Process image and create visualization.
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binarizing prediction
            save_path: Optional path to save the visualization
            
        Returns:
            Dictionary containing processed results
        """
        # Get prediction and original image
        predicted_mask, original = self.predict_single_image(image_path, threshold, return_original=True)
        
        # Resize original to match model input size for visualization
        original_resized = self.preprocessor.resize_with_padding(original)
        
        # Extract ROI
        roi_cropped = self.extract_roi(original_resized, predicted_mask)
        
        # Apply deidentification
        deidentified = self.apply_mask_for_deidentification(original_resized, predicted_mask)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(original_resized, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(predicted_mask, cmap='gray')
        axes[0, 1].set_title('Predicted Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(deidentified, cmap='gray')
        axes[1, 0].set_title('Deidentified Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(roi_cropped, cmap='gray')
        axes[1, 1].set_title('Extracted ROI')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'original': original_resized,
            'mask': predicted_mask,
            'deidentified': deidentified,
            'roi': roi_cropped
        }
    
    def benchmark_inference_speed(self, 
                                image_path: str, 
                                num_runs: int = 10) -> dict:
        """Benchmark inference speed.
        
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        # Preprocess image once
        img_processed = self.preprocessor.preprocess_for_inference(image_path)
        
        # Warm up
        _ = self.model.predict(img_processed, verbose=0)
        
        # Time multiple runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.model.predict(img_processed, verbose=0)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        stats = {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)),
            'num_runs': num_runs
        }
        
        print(f"\\nInference Speed Benchmark ({num_runs} runs):")
        print(f"  Mean time: {stats['mean_time']:.4f} ± {stats['std_time']:.4f} seconds")
        print(f"  Min time: {stats['min_time']:.4f} seconds")
        print(f"  Max time: {stats['max_time']:.4f} seconds")
        print(f"  Average FPS: {stats['fps']:.2f}")
        
        return stats


def save_prediction_results(image_path: str, 
                          mask: np.ndarray, 
                          output_dir: str) -> str:
    """Save prediction results to files.
    
    Args:
        image_path: Original image path
        mask: Predicted mask
        output_dir: Directory to save results
        
    Returns:
        Path to saved mask file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_filename = f"{base_name}_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)
    
    # Save mask
    cv2.imwrite(mask_path, mask)
    
    return mask_path
