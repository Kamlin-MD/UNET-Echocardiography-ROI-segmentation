# Examples Directory

This directory contains usage examples for the UltrasoundROI package.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)
Demonstrates single image processing with the UltrasoundROI package.

**Usage:**
```bash
python basic_usage.py --image path/to/ultrasound_image.png --model models/unet_model.keras
```

**Features:**
- Single image ROI segmentation
- ROI extraction and cropping
- De-identification processing
- Result visualization
- Performance statistics

### 2. Batch Processing (`batch_processing.py`)
Shows how to process multiple images efficiently.

**Usage:**
```bash
python batch_processing.py --input_dir path/to/images/ --output_dir results/
```

**Features:**
- Batch processing of image directories
- Processing statistics and reporting
- Performance analysis
- Automated output organization

## Requirements

Before running the examples, ensure you have:

1. **UltrasoundROI package installed**:
   ```bash
   pip install ultrasound-roi
   ```
   
   Or for development installation:
   ```bash
   pip install -e .
   ```

2. **Trained model weights**: 
   - Train your own model using the provided notebook
   - Download pre-trained weights (if available)
   - Place model file in `models/` directory

3. **Sample data**:
   - Ultrasound images in common formats (PNG, JPG, etc.)
   - Images should contain ultrasound sectors to segment

## Sample Data

For testing purposes, you can use the sample data in the `sample_data/` directory (if provided), or use your own ultrasound images.

### Expected Input Format
- **File formats**: PNG, JPG, JPEG, BMP, TIFF
- **Content**: Ultrasound images containing visible ultrasound sectors
- **Size**: Any size (automatically resized to 256×256 for processing)

## Output Structure

Both examples generate organized output:

```
output_directory/
├── masks/              # Binary ROI masks
│   ├── image_001_mask.png
│   └── image_002_mask.png
├── roi_crops/          # Cropped ROI regions
│   ├── image_001_roi.png
│   └── image_002_roi.png
├── deidentified/       # De-identified images
│   ├── image_001_deident.png
│   └── image_002_deident.png
└── processing_report.png  # Statistics report (batch processing only)
```

## Troubleshooting

### Common Issues

1. **Module not found error**:
   ```
   ImportError: No module named 'ultrasound_roi'
   ```
   **Solution**: Install the package or run from the project root directory

2. **Model file not found**:
   ```
   Error: Model file not found: models/unet_model.keras
   ```
   **Solution**: Train a model first or update the model path

3. **No images found**:
   ```
   Error: No images found in directory
   ```
   **Solution**: Check the input directory contains supported image formats

4. **Memory errors with batch processing**:
   **Solution**: Process smaller batches or use `--max_images` parameter

### Getting Help

- Check the main README.md for installation instructions
- Review the notebook for training guidance
- Open an issue on GitHub for specific problems

## Performance Tips

1. **GPU Acceleration**: Use GPU-enabled TensorFlow for faster processing
2. **Batch Size**: Adjust batch size based on available memory
3. **Image Size**: Smaller images process faster but may reduce accuracy
4. **Parallel Processing**: The package handles parallel processing internally

## Example Command Lines

```bash
# Basic single image processing
python basic_usage.py --image sample.png --visualize

# Batch processing with custom threshold
python batch_processing.py --input_dir images/ --threshold 0.6 --report

# Process limited number of images for testing
python batch_processing.py --input_dir large_dataset/ --max_images 10
```
