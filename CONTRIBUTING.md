# Contributing to UltrasoundROI

We welcome contributions to UltrasoundROI! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

- A clear, descriptive title
- A detailed description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment details (OS, Python version, package versions)
- Minimal code example that demonstrates the problem

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Use cases that would benefit from this enhancement
- Any relevant examples or mockups

### Contributing Code

1. **Fork the repository** and create a new branch from `main`
2. **Set up development environment**:
   ```bash
   git clone https://github.com/yourusername/UNET-Ultrasound-ROI-Segmentation.git
   cd UNET-Ultrasound-ROI-Segmentation
   pip install -e ".[dev]"
   ```

3. **Make your changes**:
   - Write clear, documented code following PEP 8 style guidelines
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

4. **Test your changes**:
   ```bash
   pytest tests/
   python -m flake8 src/
   python -m black src/ tests/
   ```

5. **Submit a pull request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Include test results
   - Request review from maintainers

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/yourusername/UNET-Ultrasound-ROI-Segmentation.git
cd UNET-Ultrasound-ROI-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ultrasound_roi

# Run specific test file
pytest tests/test_segmentation.py

# Run tests with verbose output
pytest -v
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Documentation

- Use clear, descriptive docstrings for all functions and classes
- Follow Google-style docstring format
- Update README.md for user-facing changes
- Add examples for new functionality

Example docstring format:
```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this error occurs.
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names that explain what is being tested
- Test both success and failure cases
- Use appropriate assertions and provide helpful error messages
- Mock external dependencies when appropriate

### Test Structure

```
tests/
├── __init__.py
├── test_segmentation.py      # Core segmentation functionality
├── test_preprocessing.py     # Image preprocessing
├── test_inference.py         # Inference pipeline
├── test_utils.py            # Utility functions
├── fixtures/                # Test data and fixtures
│   ├── sample_images/
│   └── sample_masks/
└── conftest.py              # Shared test configuration
```

### Example Test

```python
import pytest
import numpy as np
from ultrasound_roi import UNetROISegmenter

class TestSegmentation:
    def test_predict_single_image(self):
        """Test single image prediction functionality."""
        segmenter = UNetROISegmenter()
        
        # Create sample input
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test prediction
        result = segmenter.predict(test_image)
        
        # Assertions
        assert result is not None
        assert result.shape == (256, 256)
        assert result.dtype == np.uint8
        assert np.all((result == 0) | (result == 255))  # Binary mask
```

## Documentation Updates

When contributing:

1. Update docstrings for any modified functions/classes
2. Update README.md if user-facing functionality changes
3. Add examples for new features
4. Update the CHANGELOG.md with your changes

## Release Process

For maintainers:

1. Update version numbers in `setup.py` and `__init__.py`
2. Update CHANGELOG.md with release notes
3. Create release tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tags: `git push origin --tags`
5. Create GitHub release with release notes
6. Build and upload to PyPI (automated via GitHub Actions)

## Questions?

If you have questions about contributing:

- Open an issue on GitHub
- Check existing issues and pull requests
- Review the documentation
- Contact the maintainers

## License

By contributing to UltrasoundROI, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to UltrasoundROI! Your contributions help make medical imaging research more accessible and reproducible.
