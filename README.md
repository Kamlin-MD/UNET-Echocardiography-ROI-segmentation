# EchoROI — U-Net ROI Segmentation for Echocardiography

[![PyPI version](https://img.shields.io/pypi/v/echoroi.svg)](https://pypi.org/project/echoroi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.x-purple.svg)](https://onnxruntime.ai/)
[![Tests](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation/actions/workflows/ci.yml)

A lightweight U-Net model that segments the **region of interest (ROI)** in
echocardiography frames — removing scanner chrome, ECG traces, and text overlays
so that downstream models receive only clinically relevant pixels.

Trained on 1,355 annotated echocardiographic frames spanning four-chamber,
parasternal, and subcostal views across eight datasets, achieving a Dice
coefficient of 0.9880 on the held-out validation split.

> **Paper:** see [`paper/paper.md`](paper/paper.md) for the full manuscript.

---

## Key Features

| Feature | Detail |
|---|---|
| **Architecture** | Standard U-Net (31 M params, 4 encoder/decoder levels) |
| **Input** | 256 × 256 × 1 grayscale (aspect-ratio preserving, zero-padded) |
| **Output** | 256 × 256 × 1 binary mask (sigmoid, threshold 0.5) |
| **Formats** | Keras (`.keras`, 373 MB) and ONNX (`.onnx`, 124 MB) |
| **Performance** | Mean Dice 0.9880 on held-out validation set |
| **ONNX Runtime** | Cross-platform inference — no TensorFlow dependency |

---

## Model Architecture

EchoROI uses a standard U-Net adapted for scan-sector segmentation with
256 × 256 grayscale input, same-padding convolutions to preserve sector
geometry, and dropout regularisation to reduce overfitting on a small
heterogeneous training set. Additional implementation details, intended use,
and known limitations are summarised in [MODEL_CARD.md](MODEL_CARD.md).

![Reference U-Net architecture used by EchoROI. The model follows a standard
encoder-decoder U-Net layout with same-padding convolutions, dropout
regularisation, and a single-channel sigmoid output for binary scan-sector
segmentation.](paper/figures/figure_2.png)

### Loss Function

The model is trained with a **composite BCE + Dice + Total Variation loss**:

$$\mathcal{L} = w_\text{bce}\,\text{BCE} + w_\text{dice}\,\text{DiceLoss} + \alpha_\text{tv}\,\text{TV}(\hat{y})$$

| Term | Purpose | Weight |
|---|---|---|
| **BCE** | Stable per-pixel classification gradient | 1.0 |
| **Dice** | Region-overlap optimisation; robust to class imbalance | 1.0 |
| **Total Variation** | Penalises high-frequency mask edges → smooth sector boundaries | 1 × 10⁻⁴ |

The **TV regulariser** is the key ingredient for producing the smooth,
fan-shaped sector boundaries typical of ultrasound probes. BCE alone can
produce noisy boundaries; adding a region-based loss (Dice/Jaccard) improves
overlap but does not explicitly enforce spatial smoothness. The TV term fills
this gap by penalising large pixel-to-pixel differences in the predicted mask,
yielding clean, continuous boundaries even on a small heterogeneous training
set. Implementation: [`echoroi/model.py`](echoroi/model.py).

---

## Training Data Summary

The reference model was trained on 1,355 manually annotated echocardiographic
frame-mask pairs drawn from public and institutional sources. Masks were
created in LabelMe by outlining the visible scan sector while excluding
padding, borders, and display graphics. Only one representative frame per cine
loop was used for training because sector geometry is typically static within a
clip.

| Dataset | Frames | Access |
|---|---:|---|
| MIMIC-IV-ECHO | 403 | PhysioNet |
| EchoNet-Dynamic | 145 | Stanford |
| EchoNet-Paediatric | 263 | Stanford |
| CACTUS (A4C subset) | 38 | Open access |
| EchoCP | 60 | Kaggle |
| Private dataset (consented) | 50 | Institutional |
| CardiacUDC | 247 | Kaggle |
| HMC-QU | 149 | By request |
| **Total** | **1,355** | |

The full citation list for these datasets is given in
[paper/paper.md](paper/paper.md).

---

## Quick Start

### Install from PyPI

```bash
pip install echoroi
```

> **Note:** The PyPI package installs the `echoroi` library and CLI but does
> not include model weights (they exceed PyPI size limits). Download the
> pretrained Keras and/or ONNX weights from the
> [GitHub repository `models/` directory](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation/tree/main/models)
> or clone the repository (see below).

### Or install from source (for development)

```bash
git clone https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation.git
cd UNET-Echocardiography-ROI-segmentation
pip install -e ".[dev]"
```

### Run inference

```bash
# Run inference on a single image
python -c "
from echoroi import UNetPredictor
predictor = UNetPredictor('models/echoroi_unified.keras')
mask = predictor.predict_single_image('path/to/frame.png')  # (256,256,1) array
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
│   ├── images/          # 1,355 training images (PNG)
│   └── masks/           # 1,355 binary masks (PNG, from LabelMe)
├── models/
│   ├── echoroi_unified.keras   # Trained Keras model (373 MB)
│   └── echoroi_unified.onnx    # ONNX export (124 MB)
├── notebooks/
│   ├── 01_training_and_evaluation.ipynb # Training & evaluation
│   ├── 02_onnx_conversion.ipynb         # ONNX export & validation
│   ├── 03_inference_demo.ipynb          # Inference & visualisation
│   └── 04_dataset_preprocessing.ipynb   # DICOM preprocessing pipeline
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
| 01 | [Training & Evaluation](notebooks/01_training_and_evaluation.ipynb) | End-to-end training, augmentation, evaluation |
| 02 | [ONNX Conversion](notebooks/02_onnx_conversion.ipynb) | Export, validation, Keras-vs-ONNX comparison |
| 03 | [Inference Demo](notebooks/03_inference_demo.ipynb) | Inference, visualisation, ROI extraction |
| 04 | [Dataset Preprocessing](notebooks/04_dataset_preprocessing.ipynb) | DICOM → NPZ pipeline using ONNX model |

---

## Testing

```bash
pytest tests/ -v
```

All 23 tests cover model architecture, preprocessing, inference, and I/O.

### Note for macOS (Apple Silicon) users

The `tensorflow-metal` GPU plugin can deadlock inside Jupyter kernels on
some Apple Silicon configurations.  The inference notebook
(`03_inference_demo.ipynb`) disables GPU devices automatically so that all
operations run on the CPU.  This has no practical impact — inference on
256 × 256 images takes less than 1 second per frame on CPU.

If you are not using Jupyter (e.g. running via the CLI or a Python script)
the Metal GPU works normally.

---

## How to Cite

If you use EchoROI in your research, please cite:

```bibtex
@article{ekambaram2026echoroi,
  title   = {{EchoROI}: Scan-sector Segmentation and De-identification
             for Echocardiography},
  author  = {Ekambaram, Kamlin and Arnab, Anurag and Herbst, Philip and
             Theart, Rensu},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).