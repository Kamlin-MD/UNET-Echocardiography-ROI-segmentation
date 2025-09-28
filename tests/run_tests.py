"""Test runner for the unet_roi package."""

import unittest
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_tests():
    """Run all tests in the tests directory."""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
