"""Tests for the preprocessing module."""

import os
import shutil
import tempfile
import unittest

import numpy as np


class TestUltrasoundPreprocessor(unittest.TestCase):
    """Test cases for UltrasoundPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        from echoroi.preprocessing import UltrasoundPreprocessor
        self.preprocessor = UltrasoundPreprocessor(img_size=(256, 256))

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, "images")
        self.mask_dir = os.path.join(self.temp_dir, "masks")
        os.makedirs(self.image_dir)
        os.makedirs(self.mask_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_resize_with_padding(self):
        """Test image resizing with padding."""
        # Create a simple test image
        test_image = np.ones((100, 150), dtype=np.uint8) * 128

        # Resize with padding
        resized = self.preprocessor.resize_with_padding(test_image, (256, 256))

        # Check output shape
        self.assertEqual(resized.shape, (256, 256))

        # Check that the image is centered with padding
        self.assertGreater(np.sum(resized), 0)  # Should contain non-zero values

    def test_validate_data_paths_missing_dirs(self):
        """Test validation with missing directories."""
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.preprocessor.validate_data_paths("/nonexistent/path", self.mask_dir)

        with self.assertRaises((ValueError, FileNotFoundError)):
            self.preprocessor.validate_data_paths(self.image_dir, "/nonexistent/path")

    def test_validate_data_paths_empty_dirs(self):
        """Test validation with empty directories."""
        with self.assertRaises(ValueError):
            self.preprocessor.validate_data_paths(self.image_dir, self.mask_dir)

    def test_preprocess_image_invalid_path(self):
        """Test preprocessing with invalid image path."""
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess_image("/nonexistent/image.png")


class TestDataStatistics(unittest.TestCase):
    """Test data statistics functions."""

    def test_print_data_statistics(self):
        """Test data statistics printing."""
        from echoroi.preprocessing import print_data_statistics

        # Create sample data
        X = np.random.rand(10, 256, 256, 1).astype(np.float32)
        Y = np.random.randint(0, 2, (10, 256, 256, 1)).astype(np.float32)

        # This should run without error
        try:
            print_data_statistics(X, Y)
        except Exception as e:
            self.fail(f"print_data_statistics raised an exception: {e}")


class TestSampleDataCreation(unittest.TestCase):
    """Test sample data creation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_sample_data(self):
        """Test creating sample synthetic data."""
        from echoroi.preprocessing import create_sample_data

        # Create sample data
        create_sample_data(self.temp_dir, 3)

        # Check that directories were created
        image_dir = os.path.join(self.temp_dir, "images")
        mask_dir = os.path.join(self.temp_dir, "masks")

        self.assertTrue(os.path.exists(image_dir))
        self.assertTrue(os.path.exists(mask_dir))

        # Check that files were created
        import glob
        images = glob.glob(os.path.join(image_dir, "*.png"))
        masks = glob.glob(os.path.join(mask_dir, "*.png"))

        self.assertEqual(len(images), 3)
        self.assertEqual(len(masks), 3)

        # Check file names match
        image_names = [os.path.basename(f) for f in images]
        mask_names = [os.path.basename(f) for f in masks]

        for i in range(3):
            expected_name = f"sample_{i:03d}.png"
            self.assertIn(expected_name, image_names)
            self.assertIn(expected_name, mask_names)


if __name__ == '__main__':
    unittest.main()
