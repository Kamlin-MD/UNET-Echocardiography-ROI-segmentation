# Technical Specifications and Performance Benchmarks

## Development Environment

### Hardware Configuration
- **System**: Apple Mac mini (2023)
- **Processor**: Apple M2 Pro chip
  - 10-core CPU (6 performance cores + 4 efficiency cores)
  - 16-core GPU with unified memory architecture
  - 16-core Neural Engine for ML acceleration
- **Memory**: 16 GB unified memory (shared between CPU and GPU)
- **Storage**: 512 GB SSD with high-speed I/O
- **Architecture**: ARM64 (Apple Silicon)

### Software Stack
- **Operating System**: macOS Sonoma 14.5+
- **Python**: 3.9.19 (via Homebrew)
- **Deep Learning Framework**: TensorFlow 2.10+ with Metal GPU acceleration
- **Key Dependencies**:
  - OpenCV 4.8+ for image processing
  - NumPy 1.24+ for numerical computations
  - Matplotlib 3.7+ for visualization
  - scikit-image 0.21+ for image analysis

### Additional Testing Environment
The software has also been extensively tested on:
- **System**: Custom Linux workstation
- **OS**: Ubuntu 20.04 LTS
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **CPU**: AMD Ryzen 9 5900X
- **RAM**: 64GB DDR4
- **Development Environment**: VS Code with Python extension
- **CUDA**: 11.8 with cuDNN 8.6
- **TensorFlow**: 2.10.0 with CUDA acceleration

### ML Framework Configuration
```python
# TensorFlow Metal GPU configuration
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Metal GPU Support:", tf.config.list_physical_devices('GPU'))
```

Expected output on Apple Silicon:
```
TensorFlow version: 2.13.0
Built with CUDA: False
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Metal GPU Support: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Performance Benchmarks

### Training Performance
- **Model Architecture**: U-Net with 31M parameters
- **Training Dataset**: MIMIC-IV-ECHO (subset of 10,000 image-mask pairs)
- **Training Hardware**: Mac mini M2 Pro, 16GB RAM
- **Training Time**: ~4 hours for 100 epochs
- **Memory Usage**: Peak 12GB during training (including data loading)
- **GPU Utilization**: 85-95% during training phases

### Inference Performance

#### Single Image Processing
| Image Size | CPU Time (M2 Pro) | GPU Time (Metal) | RTX 3090 (CUDA) | Memory Usage |
|------------|-------------------|------------------|-----------------|--------------|
| 256×256    | 0.45s            | 0.08s           | 0.05s          | 1.2GB        |
| 512×512    | 1.2s             | 0.15s           | 0.09s          | 2.1GB        |
| 1024×1024  | 3.8s             | 0.42s           | 0.18s          | 4.8GB        |

#### Batch Processing Performance
| Batch Size | Images/Second (CPU) | M2 Pro (Metal) | RTX 3090 (CUDA) | Memory Usage |
|------------|---------------------|----------------|-----------------|--------------|
| 1          | 2.2                | 12.5          | 20.1           | 1.2GB        |
| 4          | 6.8                | 35.4          | 68.2           | 3.8GB        |
| 8          | 11.2               | 58.7          | 112.5          | 7.2GB        |
| 16         | 15.1               | 72.3          | 145.8          | 14.1GB       |
| 32         | N/A                | N/A           | 198.4          | 22.5GB       |

### Model Quality Metrics

#### Validation Results (MIMIC-IV-ECHO Test Set)
- **Test Set Size**: 2,000 images
- **Dice Score**: 0.876 ± 0.089
- **IoU Score**: 0.823 ± 0.102
- **Pixel Accuracy**: 0.954 ± 0.023
- **Precision**: 0.891 ± 0.076
- **Recall**: 0.863 ± 0.094

#### Performance by Image Quality
| Quality Level | Dice Score | IoU Score | Sample Count |
|---------------|------------|-----------|--------------|
| High Quality  | 0.923 ± 0.045 | 0.876 ± 0.062 | 800 |
| Medium Quality| 0.861 ± 0.072 | 0.812 ± 0.081 | 900 |
| Low Quality   | 0.798 ± 0.115 | 0.741 ± 0.127 | 300 |

## System Requirements

### Minimum Requirements
- **CPU**: 4-core processor (Intel i5 or Apple M1 equivalent)
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 5 GB free space for models and dependencies
- **GPU**: Optional but recommended for faster inference
- **OS**: macOS 11+, Linux (Ubuntu 18.04+), Windows 10+

### Recommended Configuration
- **CPU**: 8+ core processor (Intel i7/Apple M2 Pro or better)
- **RAM**: 16+ GB
- **GPU**: Dedicated GPU with 4+ GB VRAM or Apple Silicon with 16+ core GPU
- **Storage**: SSD with 10+ GB free space

### Python Environment
```bash
# Recommended Python version
python --version
# Python 3.8.0 to 3.11.x

# Key package versions (tested)
tensorflow>=2.8.0,<2.14.0
opencv-python>=4.5.0
numpy>=1.19.0,<1.25.0
matplotlib>=3.3.0
scikit-image>=0.18.0
```

## Reproducibility Notes

### Random Seed Configuration
For reproducible results, set these seeds before training:
```python
import numpy as np
import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Additional TensorFlow deterministic settings
tf.config.experimental.enable_op_determinism()
```

### Apple Silicon Specific Optimizations
```python
# Enable Metal GPU acceleration on Apple Silicon
import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Verify Metal backend
print("Available devices:", tf.config.list_physical_devices())
```

### Known Platform Differences
- **Apple Silicon (M2 Pro)**: Excellent performance with Metal GPU acceleration, unified memory architecture
- **NVIDIA RTX 3090**: Superior inference speed with CUDA acceleration, larger VRAM for bigger batches
- **Intel Mac**: Good CPU performance, limited GPU acceleration options
- **Linux/Windows**: Full CUDA GPU acceleration available with compatible NVIDIA hardware
- **Memory Usage**: Apple unified memory vs dedicated VRAM affects batch size strategies

### Cross-Platform Performance Comparison
| Platform | Single Image (256×256) | Batch-8 (imgs/sec) | Max Batch Size |
|----------|------------------------|-------------------|----------------|
| M2 Pro (Metal) | 0.08s | 58.7 | 16 (16GB limit) |
| RTX 3090 (CUDA) | 0.05s | 112.5 | 32+ (24GB VRAM) |
| CPU (Ryzen 9) | 0.45s | 11.2 | Limited by RAM |

## Citation and Reproducibility

To reproduce the results reported in this software:

1. Use the exact hardware/software specifications listed above
2. Download the MIMIC-IV-ECHO dataset from PhysioNet
3. Apply the preprocessing steps as documented in the main notebook
4. Use the provided random seeds for deterministic training
5. Train for 100 epochs with the default hyperparameters

For alternative hardware configurations, performance may vary but relative model quality should remain consistent.
