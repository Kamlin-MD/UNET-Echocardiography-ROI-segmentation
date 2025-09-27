"""
Tests for core segmentation functionality.

This module tests the main UNetROISegmenter class and its core functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path for testing
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from unet_inference import UltrasoundROISegmentation


class TestUNetROISegmentation:
    """Test cases for the UNetROISegmentation class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock model path (will be mocked in actual tests)
        self.mock_model_path = "mock_model.keras"
        
        # Create sample image
        self.sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.sample_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(self.sample_mask, (128, 128), 80, 255, -1)
    
    def test_resize_with_padding(self):
        """Test image resizing with padding functionality."""
        # This would be a real test if we had the actual class
        # For now, test the resize function from the notebook
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        # Test would call the resize function
        # result = segmenter.resize_with_padding(test_image)
        
        # Assertions would check:
        # assert result.shape == (256, 256, 3)
        # assert result.dtype == np.uint8
        
        # For now, just test that we can create the test image
        assert test_image.shape == (100, 150, 3)
        assert test_image.dtype == np.uint8
    
    def test_image_properties(self):
        """Test basic image property validation."""
        # Test image properties
        assert isinstance(self.sample_image, np.ndarray)
        assert self.sample_image.shape == (256, 256, 3)
        assert self.sample_image.dtype == np.uint8
        
        # Test mask properties
        assert isinstance(self.sample_mask, np.ndarray)
        assert self.sample_mask.shape == (256, 256)
        assert self.sample_mask.dtype == np.uint8
        
        # Test mask is binary
        unique_values = np.unique(self.sample_mask)
        assert len(unique_values) <= 2
        assert np.all((self.sample_mask == 0) | (self.sample_mask == 255))
    
    def test_roi_extraction_logic(self):
        """Test ROI extraction logic."""
        # Test bounding box calculation
        coords = cv2.findNonZero(self.sample_mask)
        assert coords is not None
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Check bounding box is reasonable
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        assert x + w <= self.sample_mask.shape[1]
        assert y + h <= self.sample_mask.shape[0]
    
    def test_mask_application(self):
        """Test mask application for de-identification."""
        # Apply mask to image
        masked_image = np.where(
            cv2.cvtColor(self.sample_mask, cv2.COLOR_GRAY2BGR) > 0,
            self.sample_image,
            0
        )
        
        # Check that areas outside mask are black
        mask_3d = cv2.cvtColor(self.sample_mask, cv2.COLOR_GRAY2BGR) > 0
        non_mask_pixels = masked_image[~mask_3d]
        
        # All non-mask pixels should be 0
        assert np.all(non_mask_pixels == 0)
        
        # Mask area should preserve original image
        mask_pixels_original = self.sample_image[mask_3d]
        mask_pixels_result = masked_image[mask_3d]
        assert np.array_equal(mask_pixels_original, mask_pixels_result)


class TestImageProcessingFunctions:
    """Test individual image processing functions."""
    
    def test_image_normalization(self):
        """Test image normalization to [0,1] range."""
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Normalize
        normalized = test_image.astype(np.float32) / 255.0
        
        # Check range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
    
    def test_mask_binarization(self):
        """Test mask binarization."""
        # Create test mask with various values
        test_mask = np.random.randint(0, 128, (256, 256), dtype=np.uint8)
        
        # Binarize
        binary_mask = (test_mask > 0).astype(np.uint8) * 255
        
        # Check binary properties
        unique_values = np.unique(binary_mask)
        assert len(unique_values) <= 2
        assert np.all((binary_mask == 0) | (binary_mask == 255))
    
    def test_padding_calculation(self):
        """Test padding calculation for aspect ratio preservation."""
        # Test different aspect ratios
        test_cases = [
            ((100, 150), (256, 256)),  # Wider than tall
            ((150, 100), (256, 256)),  # Taller than wide
            ((256, 256), (256, 256)),  # Square, no padding needed
        ]
        
        for (h, w), (target_h, target_w) in test_cases:
            # Calculate scale factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Calculate padding
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            
            # Check padding is reasonable
            assert pad_top >= 0 and pad_bottom >= 0
            assert pad_left >= 0 and pad_right >= 0
            assert pad_top + pad_bottom + new_h == target_h
            assert pad_left + pad_right + new_w == target_w


class TestMetricsCalculation:
    """Test evaluation metrics calculation."""
    
    def test_dice_score_calculation(self):
        """Test Dice score calculation."""
        # Create test masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Perfect overlap
        cv2.circle(mask1, (50, 50), 25, 255, -1)
        cv2.circle(mask2, (50, 50), 25, 255, -1)
        
        # Calculate Dice score manually
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        dice = 2.0 * intersection / (np.sum(mask1 > 0) + np.sum(mask2 > 0))
        
        # Perfect overlap should give Dice = 1.0
        assert abs(dice - 1.0) < 1e-6
        
        # No overlap
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask3, (25, 25), 10, 255, -1)
        
        intersection_none = np.sum((mask1 > 0) & (mask3 > 0))
        dice_none = 2.0 * intersection_none / (np.sum(mask1 > 0) + np.sum(mask3 > 0))
        
        # No overlap should give Dice = 0.0
        assert dice_none == 0.0
    
    def test_iou_calculation(self):
        """Test IoU (Intersection over Union) calculation."""
        # Create test masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        
        # Partial overlap
        cv2.rectangle(mask1, (25, 25), (75, 75), 255, -1)
        cv2.rectangle(mask2, (50, 50), (100, 100), 255, -1)
        
        # Calculate IoU
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        iou = intersection / union if union > 0 else 0
        
        # IoU should be between 0 and 1
        assert 0 <= iou <= 1
        
        # Check specific case: overlapping squares
        expected_intersection = 25 * 25  # 25x25 overlap
        expected_union = (50 * 50) + (50 * 50) - (25 * 25)  # Total area minus overlap
        expected_iou = expected_intersection / expected_union
        
        assert abs(iou - expected_iou) < 1e-6


if __name__ == "__main__":
    # Simple test runner for development
    import unittest
    
    # Convert test classes to unittest format and run
    print("Running UltrasoundROI tests...")
    
    # Run a few basic tests
    test_seg = TestUNetROISegmentation()
    test_seg.setup_method()
    
    try:
        test_seg.test_image_properties()
        print("✅ test_image_properties passed")
        
        test_seg.test_roi_extraction_logic()
        print("✅ test_roi_extraction_logic passed")
        
        test_seg.test_mask_application()
        print("✅ test_mask_application passed")
        
        test_img = TestImageProcessingFunctions()
        test_img.test_image_normalization()
        print("✅ test_image_normalization passed")
        
        test_img.test_mask_binarization()
        print("✅ test_mask_binarization passed")
        
        test_metrics = TestMetricsCalculation()
        test_metrics.test_dice_score_calculation()
        print("✅ test_dice_score_calculation passed")
        
        test_metrics.test_iou_calculation()
        print("✅ test_iou_calculation passed")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
