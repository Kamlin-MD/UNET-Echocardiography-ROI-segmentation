# UNET Echocardiography ROI Segmentation

Automated ultrasound ROI segmentation using U-Net deep learning for medical image de-identification and preprocessing.

## Overview

This notebook-based implementation provides:
- **Automated ROI Detection** in ultrasound images
- **Privacy-preserving de-identification** by masking non-diagnostic areas  
- **Complete pipeline** from training to inference
- **MIMIC-IV-ECHO dataset** integration

## Installation

```bash
git clone https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation.git
cd UNET-Echocardiography-ROI-segmentation
pip install -r requirements.txt
```

## Usage

Open and run the main notebook:

```bash
jupyter notebook "UNET-based ECHO-sector ROI extractor and deidentifier model.ipynb"
```

The notebook contains:
- Complete U-Net implementation
- Training pipeline with MIMIC-IV-ECHO data
- Model evaluation and metrics
- Inference pipeline for new images
- Visualization tools

## Dataset

Designed for **MIMIC-IV-ECHO dataset** from PhysioNet:
- Requires PhysioNet account and training
- Place images in `data/images/` directory
- Place masks in `data/masks/` directory

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- OpenCV, NumPy, Matplotlib, Scikit-learn
- 8GB+ RAM (16GB recommended)
- GPU recommended for training

## Hardware Tested

- **Apple Silicon**: Mac mini M2 Pro with Metal acceleration
- **NVIDIA GPU**: RTX 3090 with CUDA acceleration
- **CPU**: Multi-core processors (slower training)

## Key Features

### 1. Data Processing
- Automatic image resizing with aspect ratio preservation
- Normalization and preprocessing pipeline
- Train/validation splitting with stratification

### 2. U-Net Architecture
- Encoder-decoder structure with skip connections
- Batch normalization and dropout for regularization
- Binary segmentation with sigmoid activation

### 3. Training Pipeline
- Adam optimizer with learning rate scheduling
- Early stopping and model checkpointing
- Comprehensive metrics: Dice, IoU, Accuracy

### 4. Inference Pipeline
- Single image processing
- Batch processing capabilities
- ROI extraction and cropping
- De-identification masking

## Model Performance

Typical results on MIMIC-IV-ECHO validation data:
- **Dice Score**: 0.85+ average
- **IoU Score**: 0.80+ average  
- **Pixel Accuracy**: 0.95+ average
- **Processing Speed**: <1 second per image

## Citation

```bibtex
@software{kambaram2025ultrasound,
  author = {Kambaram, Kamlin},
  title = {UNET Echocardiography ROI Segmentation},
  url = {https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation},
  year = {2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- **MIMIC-IV-ECHO dataset**: PhysioNet collaborative database
- **U-Net architecture**: Ronneberger et al. (2015)
- **TensorFlow/Keras**: Deep learning framework
