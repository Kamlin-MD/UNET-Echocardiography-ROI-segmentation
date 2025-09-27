# UNET Ultrasound ROI Segmentation & De-identification

A deep learning approach for automatic Region of Interest (ROI) segmentation in ultrasound images using UNET architecture. This project enables automated ultrasound sector detection, ROI extraction, and de-identification of medical ultrasound images, designed specifically for use with the **MIMIC-IV-ECHO dataset** from PhysioNet.

![UNET Architecture](docs/images/unet_architecture.png)

## 🔬 Project Overview

This project provides automated ROI segmentation for ultrasound images, enabling:

- **Automated ROI Detection**: Eliminate manual annotation of ultrasound sectors
- **De-identification**: Mask areas outside the diagnostic region for privacy
- **Standardized Processing**: Consistent ROI extraction across different datasets
- **Batch Processing**: Efficient processing of large image collections
- **Research Applications**: Foundation for medical imaging research and analysis

## 📊 Dataset Information

This project is designed to work with the **MIMIC-IV-ECHO: Echocardiogram Matched Dataset** from PhysioNet, though it can be adapted for other ultrasound datasets.

### MIMIC-IV-ECHO Dataset
- **Source**: PhysioNet (https://physionet.org/)
- **Content**: Echocardiogram images with matched clinical data
- **Access**: Requires PhysioNet account and completed training
- **License**: PhysioNet Data Use Agreement required

### Data Requirements
- **Input Format**: Standard image formats (PNG, JPG, etc.)
- **Structure**: Separate directories for images and corresponding masks
- **Resolution**: Any resolution (automatically resized to 256×256 for processing)

## 💻 Development Environment

This project was developed and tested on multiple systems to ensure cross-platform compatibility:

### Primary Development Environment (Apple Silicon)
- **System**: Mac mini (2023)
- **Chip**: Apple M2 Pro
- **Memory**: 16 GB unified memory
- **GPU**: 16-core GPU with hardware-accelerated ML compute
- **Storage**: 512 GB SSD
- **OS**: macOS Sonoma 14.5+
- **TensorFlow**: 2.10+ with Metal GPU acceleration

### Additional Testing Environment (Linux/CUDA)
- **System**: Custom Linux workstation
- **CPU**: AMD Ryzen 9 5900X
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Memory**: 64GB DDR4
- **OS**: Ubuntu 20.04 LTS
- **TensorFlow**: 2.10+ with CUDA 11.8 acceleration
- **Development**: VS Code with Python extension

### Software Environment
- **Python**: 3.8-3.11 (tested on 3.9)
- **Development Tools**: Jupyter Lab, VS Code, GitHub Codespaces
- **Masks**: Binary masks indicating ROI areas (white=ROI, black=background)

### Expected Directory Structure
```
data/
├── images/          # Ultrasound images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/           # ROI masks
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### Data Access
To use the MIMIC-IV-ECHO dataset:
1. Create a PhysioNet account at https://physionet.org/
2. Complete the required CITI Human Research training
3. Sign the Data Use Agreement
4. Request access to MIMIC-IV-ECHO dataset
5. Follow all ethical guidelines and institutional requirements

## 🏗️ Model Architecture

### UNET Overview
- **Encoder**: Contracting path with convolutional blocks and max pooling
- **Bottleneck**: Bridge with highest feature complexity (1024 filters)
- **Decoder**: Expanding path with up-sampling and skip connections
- **Output**: Binary segmentation mask (256×256)

### Key Features
- **Input Size**: 256×256×3 (automatically resized with padding)
- **Skip Connections**: Preserve fine-grained details
- **Batch Normalization**: Training stability
- **Dropout Regularization**: Prevent overfitting
- **Binary Output**: Precise ROI mask generation

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UNET-Ultrasound-ROI-Segmentation.git
cd UNET-Ultrasound-ROI-Segmentation
```

2. Create a virtual environment:
```bash
python -m venv unet_env
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Jupyter Notebook (Training and Analysis)

1. **Prepare Your Data**:
   - Create `data/images/` and `data/masks/` directories
   - Add your ultrasound images and corresponding ROI masks
   - Ensure matching filenames between images and masks

2. **Open the Jupyter Notebook**:
```bash
jupyter notebook "UNET-based ECHO-sector ROI extractor and deidentifier model.ipynb"
```

3. **Configure and Run**:
   - Update data paths in the configuration cell
   - Adjust hyperparameters as needed
   - Execute cells sequentially to train and evaluate the model

#### Option 2: Python Scripts (Inference Only)

For users who just want to apply a pre-trained model:

1. **Single Image Processing**:
```bash
python example_usage.py --input_dir test_images/ --single_image sample.png --output_dir results/
```

2. **Batch Processing**:
```bash
python example_usage.py --input_dir test_images/ --output_dir results/
```

3. **Programmatic Usage**:
```python
from unet_inference import UltrasoundROISegmentation

segmenter = UltrasoundROISegmentation('models/unet_model.keras')
original, processed, roi_mask = segmenter.predict_single_image('image.png')
results = segmenter.process_batch('input_dir/', 'output_dir/')
```

## 📁 Project Structure

```
UNET-Ultrasound-ROI-Segmentation/
├── UNET-based ECHO-sector ROI extractor and deidentifier model.ipynb  # Main training notebook
├── unet_inference.py                      # Core inference functions
├── example_usage.py                       # Example usage script
├── requirements.txt                       # Dependencies
├── README.md                              # This file
├── setup.py                               # Package setup
├── .gitignore                             # Git ignore rules
├── docs/                                  # Documentation
│   ├── images/                           # Architecture diagrams
│   └── examples/                         # Usage examples
├── models/                               # Trained model weights
│   └── unet_ultrasound_roi_v2.keras     # Pre-trained model
├── data/                                 # Sample data (if applicable)
│   ├── images/                          # Sample images
│   └── masks/                           # Sample masks
└── utils/                               # Utility functions
    ├── __init__.py
    ├── preprocessing.py                 # Data preprocessing
    ├── model.py                        # Model architecture
    └── inference.py                    # Inference pipeline
```

## 🔧 Configuration

Update the following paths in the notebook configuration cell:

```python
# Dataset paths
IMAGE_DIR = "/path/to/your/ultrasound/images"
MASK_DIR = "/path/to/your/roi/masks"
MODEL_SAVE_PATH = "/path/to/save/trained/model.keras"

# Model hyperparameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
```

## � Output Structure

When processing images, the tool generates:

```
output_directory/
├── masks/              # Binary ROI masks (PNG format)
│   ├── image_001_mask.png
│   └── image_002_mask.png
├── roi_crops/          # Cropped ROI regions
│   ├── image_001_roi.png
│   └── image_002_roi.png
└── deidentified/       # De-identified images
    ├── image_001_deident.png
    └── image_002_deident.png
```

## �📈 Performance Metrics

The model is evaluated using:
- **Dice Score**: Measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Segmentation accuracy metric
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Precision/Recall**: Classification performance metrics

## 🔄 Complete Pipeline

### 1. Training Pipeline
1. **Data Loading**: Load ultrasound images and ROI masks
2. **Preprocessing**: Resize with padding, normalize pixel values
3. **Model Training**: Train UNET with validation monitoring
4. **Evaluation**: Calculate metrics and visualize results

### 2. Inference Pipeline
1. **Image Loading**: Load new ultrasound images
2. **ROI Prediction**: Generate segmentation masks
3. **Post-processing**: Apply threshold and morphological operations
4. **ROI Extraction**: Crop images based on predicted masks
5. **De-identification**: Mask areas outside ROI for privacy

## 🎯 Applications

This tool enables various use cases:
- **Research Preprocessing**: Standardized ROI extraction for datasets
- **Privacy Protection**: De-identification by masking non-diagnostic areas
- **Quality Control**: Automated filtering and standardization
- **Batch Processing**: Efficient processing of large image collections
- **Clinical Workflow**: Integration into existing ultrasound analysis pipelines

## 📊 Model Training

The notebook includes:
- **Complete Training Pipeline**: From data loading to model evaluation
- **Performance Metrics**: Dice Score, IoU, Pixel Accuracy
- **Visualization Tools**: Training progress and prediction examples
- **Model Saving**: Save trained models for later use

Users can train their own models using their specific ultrasound datasets.

## 🔬 Use Cases

This tool is suitable for:
- **Medical Imaging Research**: Standardized ROI extraction for studies
- **Privacy-Preserving Processing**: De-identification of ultrasound images
- **Quality Assurance**: Automated preprocessing and standardization
- **Educational Applications**: Teaching medical image segmentation
- **Dataset Preparation**: Preprocessing for machine learning projects

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={UNET Model for Automatic ROI Segmentation in Ultrasound Images: A Preprocessing Pipeline for Echo-ViViT},
  author={Your Name},
  journal={Your Journal},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Dataset Citations
- **MIMIC-IV-ECHO**: Gow, B., Pollard, T., Greenbaum, N., Moody, B., Johnson, A., Herbst, E., Waks, J. W., Eslami, P., Chaudhari, A., Carbonati, T., Berkowitz, S., Mark, R., & Horng, S. (2023). MIMIC-IV-ECHO: Echocardiogram Matched Subset (version 0.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/ef48-v217

- **PhysioNet**: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

### Technical References
- **UNET Architecture**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision operations

### Data Ethics
This work demonstrates the use of the MIMIC-IV-ECHO dataset from PhysioNet. All users must comply with PhysioNet's data use agreements and ethical guidelines for medical data research.

## 📧 Contact

For questions or collaboration:
- **Email**: your.email@institution.edu
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Institution**: Your University/Institution

---

**Note**: This is part of ongoing PhD research in cardiac ultrasound analysis. The model and preprocessing pipeline are designed for research purposes and require appropriate validation for clinical applications.
