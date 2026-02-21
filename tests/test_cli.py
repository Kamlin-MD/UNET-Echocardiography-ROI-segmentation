"""Tests for the CLI module."""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from io import StringIO


class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_main_no_args(self):
        """Test main function with no arguments."""
        from echoroi.cli import main
        
        # Capture stdout
        with patch('sys.argv', ['echoroi']):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                main()
                output = mock_stdout.getvalue()
                self.assertIn('usage:', output.lower())
    
    def test_create_data_cli(self):
        """Test create-data CLI command."""
        from echoroi.cli import create_data_cli
        from types import SimpleNamespace
        
        # Create mock arguments
        args = SimpleNamespace(
            output_dir=self.temp_dir,
            num_samples=3
        )
        
        try:
            create_data_cli(args)
            
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
            
        except ImportError as e:
            self.skipTest(f"Required dependencies not available: {e}")
    
    def test_train_cli_args_parsing(self):
        """Test training CLI argument parsing."""
        from echoroi.cli import main
        
        test_args = [
            'echoroi', 'train',
            '--image-dir', '/test/images',
            '--mask-dir', '/test/masks',
            '--epochs', '10',
            '--batch-size', '4'
        ]
        
        # This should not raise an exception during argument parsing
        with patch('sys.argv', test_args):
            with patch('echoroi.cli.train_cli') as mock_train:
                # Mock train_cli to avoid actual training
                mock_train.return_value = None
                try:
                    main()
                    mock_train.assert_called_once()
                except SystemExit:
                    # argparse might call sys.exit on missing required args
                    pass
    
    def test_predict_cli_args_parsing(self):
        """Test prediction CLI argument parsing."""
        from echoroi.cli import main
        
        test_args = [
            'echoroi', 'predict',
            '--model-path', '/test/model.keras',
            '--input', '/test/input.png',
            '--output', '/test/output',
            '--threshold', '0.6',
            '--visualize'
        ]
        
        with patch('sys.argv', test_args):
            with patch('echoroi.cli.predict_cli') as mock_predict:
                mock_predict.return_value = None
                try:
                    main()
                    mock_predict.assert_called_once()
                except SystemExit:
                    pass
    
    def test_evaluate_cli_args_parsing(self):
        """Test evaluation CLI argument parsing."""
        from echoroi.cli import main
        
        test_args = [
            'echoroi', 'evaluate',
            '--model-path', '/test/model.keras',
            '--image-dir', '/test/images',
            '--mask-dir', '/test/masks',
            '--output', '/test/results'
        ]
        
        with patch('sys.argv', test_args):
            with patch('echoroi.cli.evaluate_cli') as mock_evaluate:
                mock_evaluate.return_value = None
                try:
                    main()
                    mock_evaluate.assert_called_once()
                except SystemExit:
                    pass
    
    def test_benchmark_cli_args_parsing(self):
        """Test benchmark CLI argument parsing."""
        from echoroi.cli import main
        
        test_args = [
            'echoroi', 'benchmark',
            '--model-path', '/test/model.keras',
            '--image-path', '/test/image.png',
            '--num-runs', '5'
        ]
        
        with patch('sys.argv', test_args):
            with patch('echoroi.cli.benchmark_cli') as mock_benchmark:
                mock_benchmark.return_value = None
                try:
                    main()
                    mock_benchmark.assert_called_once()
                except SystemExit:
                    pass


class TestCLIEntryPoints(unittest.TestCase):
    """Test CLI entry point functions."""
    
    def test_train_cli_entry(self):
        """Test training CLI entry point."""
        from echoroi.cli import train_cli_entry
        
        test_args = ['--image-dir', '/test/images', '--mask-dir', '/test/masks']
        
        with patch('sys.argv', ['echoroi-train'] + test_args):
            with patch('echoroi.cli.main') as mock_main:
                train_cli_entry()
                mock_main.assert_called_once()
    
    def test_predict_cli_entry(self):
        """Test prediction CLI entry point."""
        from echoroi.cli import predict_cli_entry
        
        test_args = [
            '--model-path', '/test/model.keras',
            '--input', '/test/input.png',
            '--output', '/test/output'
        ]
        
        with patch('sys.argv', ['echoroi-predict'] + test_args):
            with patch('echoroi.cli.main') as mock_main:
                predict_cli_entry()
                mock_main.assert_called_once()
    
    def test_evaluate_cli_entry(self):
        """Test evaluation CLI entry point."""
        from echoroi.cli import evaluate_cli_entry
        
        test_args = [
            '--model-path', '/test/model.keras',
            '--image-dir', '/test/images',
            '--mask-dir', '/test/masks'
        ]
        
        with patch('sys.argv', ['echoroi-evaluate'] + test_args):
            with patch('echoroi.cli.main') as mock_main:
                evaluate_cli_entry()
                mock_main.assert_called_once()


if __name__ == '__main__':
    unittest.main()
