# EchoROI — U-Net ROI Segmentation for Echocardiography

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.x-purple.svg)](https://onnxruntime.ai/)
[![Tests](https://img.shields.io/badge/Tests-23%2F23_passing-brightgreen.svg)](#testing)

A lightweight U-Net model that segments the **region of interest (ROI)** in
echocardiography frames — removing scanner chrome, ECG traces, and text overlays
so that downstream models receive only clinically relevant pixels.

Trained on 99 hand-labelled echocardiographic images spanning four-chamber,
parasternal, and subcostal views, evaluated with Dice coefficient and
visual overlay analysis.

> **Paper:** see [`paper/paper.md`](paper/paper.md) for the full JOSS-style manuscript.

---

## Key Features

| Feature | Detail |
|---|---|
| **Architecture** | Standard U-Net (31 M params, 4 encoder/decoder levels) |
| **Input** | 256 × 256 × 1 grayscale (aspect-ratio preserving, zero-padded) |
| **Output** | 256 × 256 × 1 binary mask (sigmoid, threshold 0.5) |
| **Formats** | Keras (`.keras`, 373 MB) and ONNX (`.onnx`, 124 MB) |
| **Performance** | Mean Dice 0.9587 on held-out test set |
| **ONNX Runtime** | Cross-platform inference — no TensorFlow dependency |

---

## Quick Start

```bash
# Clone
git clone https://github.com/RonSheely/EchoROI.git
cd EchoROI

# Install
pip install -e ".[dev]"

# Run inference on a single image
python -c "
from echoroi import load_model, predict
model = load_model()                       # loads models/echoroi_unified.keras
mask = predict(model, 'path/to/frame.png') # returns (256,256,1) numpy array
"
```

### ONNX Inference (no TensorFlow)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/echoroi_unified.onnx")
# image: (1, 256, 256, 1) float32, normalised [0, 1]
mask = sess.run(None, {"input": image})[0]
```

---

## DICOM Preprocessing Pipeline

**Notebook [`04_dataset_preprocessing.ipynb`](notebooks/04_dataset_preprocessing.ipynb)**
provides a complete, configurable pipeline for batch-processing echocardiography
DICOM datasets using the ONNX model — no TensorFlow required.

### What it does

```
DICOM files (recursive discovery)
  → Extract frames
  → Optional adaptive stride (e.g. normalise all clips to 32 frames)
  → Resize to 256×256 (aspect-ratio preserving, zero-padded)
  → Select representative frame (highest Shannon entropy)
  → ONNX ROI inference → broadcast mask to all frames
  → LV-focused square crop → resize to 112×112
  → Save as compressed NPZ
```

### Key configuration options

```python
CONFIG = {
    'target_frames': 32,    # None = keep original frame count; int = adaptive stride
    'max_files':     None,  # None = process all; int = limit for test runs
    'final_size':    (112, 112),
    'use_gpu':       False, # set True for CUDA acceleration
}
```

### Features

- **Single-file demo** with step-by-step visualisation (input → ROI overlay → cropped output)
- **Batch processor** with progress tracking, error handling, and summary statistics
- **NPZ inspector** to verify saved outputs
- **Representative frame selection** via Shannon entropy (avoids blank/transition frames)
- **Hardware acceleration** — CUDA, CoreML, or CPU via ONNX Runtime providers
- **Optional dependency installer** cell for quick environment setup

---

## Repository Structure

```
EchoROI/
├── data/
│   ├── images/          # 99 training images (PNG)
│   └── masks/           # 99 binary masks (PNG, from LabelMe)
├── models/
│   ├── echoroi_unified.keras   # Trained Keras model (373 MB)
│   └── echoroi_unified.onnx    # ONNX export (124 MB)
├── notebooks/
│   ├── 01_model_training.ipynb         # Training & evaluation
│   ├── 02_onnx_conversion.ipynb        # ONNX export & validation
│   ├── 03_qualitative_evaluation.ipynb # Visual evaluation
│   └── 04_dataset_preprocessing.ipynb  # DICOM preprocessing pipeline
├── echoroi/              # Python package
│   ├── model.py          # U-Net architecture
│   ├── preprocessing.py  # Image preprocessing
│   └── inference.py      # Prediction utilities
├── paper/
│   ├── paper.md          # JOSS manuscript
│   └── paper.bib         # References
├── tests/                # 23 unit tests
└── scripts/              # CLI utilities
```

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [Model Training](notebooks/01_model_training.ipynb) | End-to-end training, augmentation, evaluation |
| 02 | [ONNX Conversion](notebooks/02_onnx_conversion.ipynb) | Export, validation, Keras-vs-ONNX comparison |
| 03 | [Qualitative Evaluation](notebooks/03_qualitative_evaluation.ipynb) | Visual overlays, per-sample Dice, failure analysis |
| 04 | [Dataset Preprocessing](notebooks/04_dataset_preprocessing.ipynb) | DICOM → NPZ pipeline using ONNX model |

---

## Testing

```bash
pytest tests/ -v
```

All 23 tests cover model architecture, preprocessing, inference, and I/O.

---

## How to Cite

If you use EchoROI in your research, please cite:

```bibtex
@article{sheely2025echoroi,
  title     = {{EchoROI}: A U-Net Tool for Automatic Region-of-Interest
               Segmentation in Echocardiography},
  author    = {Sheely, Ron},
  journal   = {Journal of Open Source Software},
  year      = {2025},
  note      = {Manuscript submitted}
}
```

---

## License

MIT — see [LICENSE](LICENSE).