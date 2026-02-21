"""Tests for the model module."""

import unittest
import os
import numpy as np
import tempfile


class TestUNetModel(unittest.TestCase):
    """Test cases for UNetModel class."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from echoroi.model import UNetModel
            self.model_builder = UNetModel(input_shape=(256, 256, 1), num_classes=1)
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model_builder.input_shape, (256, 256, 1))
        self.assertEqual(self.model_builder.num_classes, 1)
        self.assertIsNone(self.model_builder.model)
    
    def test_build_model(self):
        """Test model building."""
        model = self.model_builder.build_model()
        
        # Check that model was created
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.model_builder.model)
        
        # Check input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        self.assertEqual(input_shape[1:], (256, 256, 1))  # Exclude batch dimension
        self.assertEqual(output_shape[1:], (256, 256, 1))  # Exclude batch dimension
    
    def test_compile_model(self):
        """Test model compilation."""
        model = self.model_builder.compile_model(learning_rate=1e-4)
        
        # Check that model was compiled
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.optimizer)
        # Metrics count depends on TF version; just check at least 1 exists
        self.assertGreaterEqual(len(model.metrics), 1)
    
    def test_get_model(self):
        """Test getting model instance."""
        model1 = self.model_builder.get_model()
        model2 = self.model_builder.get_model()
        
        # Should return the same instance
        self.assertIs(model1, model2)
    
    def test_dice_coefficient_metric(self):
        """Test dice coefficient metric calculation."""
        try:
            import tensorflow as tf
            from echoroi.model import dice_coefficient
        except ImportError:
            self.skipTest("TensorFlow not available")
            
        # Create sample data
        y_true = tf.constant([[1.0, 1.0, 0.0, 0.0]])
        y_pred = tf.constant([[0.9, 0.8, 0.1, 0.2]])
        
        dice = dice_coefficient(y_true, y_pred)
        
        # Dice coefficient should be between 0 and 1
        self.assertGreater(dice, 0.0)
        self.assertLess(dice, 1.0)
    
    def test_iou_score_metric(self):
        """Test IoU score metric calculation."""
        try:
            import tensorflow as tf
            from echoroi.model import iou_score
        except ImportError:
            self.skipTest("TensorFlow not available")
            
        # Create sample data
        y_true = tf.constant([[1.0, 1.0, 0.0, 0.0]])
        y_pred = tf.constant([[0.9, 0.8, 0.1, 0.2]])
        
        iou = iou_score(y_true, y_pred)
        
        # IoU score should be between 0 and 1
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestModelSaveLoad(unittest.TestCase):
    """Test model saving and loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.keras")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_model_not_built(self):
        """Test saving model that hasn't been built."""
        try:
            from echoroi.model import UNetModel
            model_builder = UNetModel()
            
            with self.assertRaises(ValueError):
                model_builder.save_model(self.model_path)
        except ImportError:
            self.skipTest("TensorFlow not available")


class TestModelUtils(unittest.TestCase):
    """Test model utility functions."""
    
    def test_load_pretrained_model_invalid_path(self):
        """Test loading model from invalid path."""
        try:
            from echoroi.model import load_pretrained_model
            
            with self.assertRaises(Exception):  # Could be various TF exceptions
                load_pretrained_model("/nonexistent/model.keras")
        except ImportError:
            self.skipTest("TensorFlow not available")


if __name__ == '__main__':
    unittest.main()
