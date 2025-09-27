"""
Test configuration and fixtures for UltrasoundROI tests.

This module provides shared test configuration, fixtures, and utilities
for the UltrasoundROI test suite.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_ultrasound_image():
    """Create a sample ultrasound-like image for testing."""
    # Create a realistic ultrasound-like image
    height, width = 256, 256
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a sector-like shape (common in ultrasound)
    center = (width // 2, height - 20)
    angle_start, angle_end = -60, 60
    radius = 200
    
    # Draw the ultrasound sector
    for angle in range(angle_start, angle_end):
        end_x = int(center[0] + radius * np.cos(np.radians(angle)))
        end_y = int(center[1] - radius * np.sin(np.radians(angle)))
        cv2.line(image, center, (end_x, end_y), (128, 128, 128), 1)
    
    # Add some noise and texture
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Create the sector mask
    mask = cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), 
                       [np.array([(center[0] - 100, center[1]), 
                                 (center[0] + 100, center[1]),
                                 (center[0] + 80, center[1] - 150),
                                 (center[0] - 80, center[1] - 150)])], 
                       255)
    
    return image, mask


@pytest.fixture
def sample_roi_mask():
    """Create a sample ROI mask for testing."""
    height, width = 256, 256
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a simple rectangular ROI
    cv2.rectangle(mask, (50, 50), (200, 200), 255, -1)
    
    return mask


@pytest.fixture
def temp_image_file(sample_ultrasound_image):
    """Create a temporary image file for testing."""
    image, _ = sample_ultrasound_image
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, image)
        yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_batch_images(temp_directory, sample_ultrasound_image):
    """Create a batch of sample images for batch processing tests."""
    image, mask = sample_ultrasound_image
    
    # Create images directory
    images_dir = os.path.join(temp_directory, 'images')
    masks_dir = os.path.join(temp_directory, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Create multiple sample images
    image_paths = []
    mask_paths = []
    
    for i in range(5):
        # Add some variation to each image
        varied_image = image.copy()
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        varied_image = np.clip(varied_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        image_path = os.path.join(images_dir, f'test_image_{i:03d}.png')
        mask_path = os.path.join(masks_dir, f'test_image_{i:03d}.png')
        
        cv2.imwrite(image_path, varied_image)
        cv2.imwrite(mask_path, mask)
        
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    
    return {
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'image_paths': image_paths,
        'mask_paths': mask_paths,
        'temp_dir': temp_directory
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing without requiring actual model weights."""
    class MockModel:
        def __init__(self):
            self.input_shape = (256, 256, 3)
            
        def predict(self, x, verbose=0):
            # Return a simple circular mask as prediction
            batch_size = x.shape[0]
            height, width = 256, 256
            predictions = []
            
            for _ in range(batch_size):
                mask = np.zeros((height, width, 1), dtype=np.float32)
                center = (width // 2, height // 2)
                radius = 80
                
                y, x_coords = np.ogrid[:height, :width]
                mask_circle = (x_coords - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
                mask[mask_circle] = 1.0
                
                predictions.append(mask)
            
            return np.array(predictions)
    
    return MockModel()


# Test utilities
def assert_image_properties(image, expected_shape=None, expected_dtype=None):
    """Assert basic properties of an image array."""
    assert isinstance(image, np.ndarray), "Image should be numpy array"
    
    if expected_shape:
        assert image.shape == expected_shape, f"Expected shape {expected_shape}, got {image.shape}"
    
    if expected_dtype:
        assert image.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {image.dtype}"


def assert_mask_properties(mask, binary=True):
    """Assert properties of a segmentation mask."""
    assert isinstance(mask, np.ndarray), "Mask should be numpy array"
    assert len(mask.shape) == 2, "Mask should be 2D array"
    
    if binary:
        unique_values = np.unique(mask)
        assert len(unique_values) <= 2, "Binary mask should have at most 2 unique values"
        assert np.all((mask == 0) | (mask == 255) | (mask == 1)), "Binary mask values should be 0, 1, or 255"


def create_test_config():
    """Create a test configuration dictionary."""
    return {
        'img_size': (256, 256),
        'batch_size': 2,
        'threshold': 0.5,
        'test_data_size': 5
    }
