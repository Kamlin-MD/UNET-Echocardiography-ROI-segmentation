# EchoROI

[![CI](https://github.com/Kamlin-MD/echoroi/actions/workflows/ci.yml/badge.svg)](https://github.com/Kamlin-MD/echoroi/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JOSS](https://img.shields.io/badge/JOSS-submitted-green.svg)](paper/paper.md)

**U-Net deep-learning tool for echocardiographic ROI segmentation and de-identification.**

EchoROI provides a pretrained U-Net model that segments the ultrasound
scan sector (region of interest) in echocardiogram frames and masks out
everything else — patient identifiers, ECG traces, vendor overlays, and
other non-diagnostic content. Users can apply the pretrained model
directly or fine-tune on their own annotated data.

---

## Features

- **Pretrained model** — ready-to-use weights trained on 1,355 annotated frames
- **CLI and Python API** — single-image or batch prediction, de-identification, ROI extraction
- **Fine-tuning** — retrain on custom datasets with a single command
- **ONNX export** — convert to ONNX for framework-agnostic deployment
- **Evaluation suite** — Dice, IoU, accuracy, sensitivity, specificity metrics

---

## Installation

```bash
git clone https://github.com/Kamlin-MD/echoroi.git
cd echoroi
pip install -e .
```

For development (linting, tests):

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[notebooks]"   # Jupyter support
pip install -e ".[medical]"     # NIfTI / DICOM loaders
pip install -e ".[export]"      # ONNX conversion (tf2onnx)
```

---

## Quick start

### Command-line interface

```bash
# Predict a mask for a single image
echoroi predict \
    --model-path models/echoroi_unified.keras \
    --input frame.png \
    --output results/

# Predict on a directory of images
echoroi predict \
    --model-path models/echoroi_unified.keras \
    --input video_frames/ \
    --output results/ \
    --visualize

# De-identify: black out everything outside the scan sector
echoroi predict \
    --model-path models/echoroi_unified.keras \
    --input video_frames/ \
    --output clean/ \
    --deidentify

# Extract cropped ROI regions
echoroi predict \
    --model-path models/echoroi_unified.keras \
    --input frame.png \
    --output results/ \
    --extract-roi

# Train a new model from scratch
echoroi train \
    --image-dir data/images \
    --mask-dir data/masks \
    --model-path models/echoroi_unified.keras \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --results-dir training_results

# Evaluate model on a test set
echoroi evaluate \
    --model-path models/echoroi_unified.keras \
    --image-dir data/images \
    --mask-dir data/masks \
    --output evaluation_results

# Benchmark inference speed
echoroi benchmark \
    --model-path models/echoroi_unified.keras \
    --image-path data/images/sample_000.png \
    --num-runs 20
```

### Python API

```python
from echoroi import UNetPredictor

# Load pretrained model
predictor = UNetPredictor("models/echoroi_unified.keras")

# Predict a binary mask
mask = predictor.predict_single_image("frame.png")

# Full pipeline: visualisation + de-identification + ROI extraction
result = predictor.process_image_with_visualization(
    "frame.png", save_path="output.png"
)

# Batch prediction
masks = predictor.predict_batch(["frame1.png", "frame2.png", "frame3.png"])

# Benchmark
stats = predictor.benchmark_inference_speed("frame.png", num_runs=20)
```

#### Fine-tuning from Python

```python
from echoroi import UNetTrainer

trainer = UNetTrainer(
    img_size=(256, 256),
    learning_rate=1e-4,
    batch_size=8,
    epochs=20,
    validation_split=0.2,
)

history = trainer.train(
    image_dir="my_data/images",
    mask_dir="my_data/masks",
    model_save_path="models/echoroi_finetuned.keras",
    results_dir="my_results",
)

# Save training plots and metrics
trainer.save_results("my_results")
```

---

## Model architecture

| Property | Value |
|---|---|
| Architecture | U-Net (4 encoder + 4 decoder blocks) |
| Parameters | 31,031,745 |
| Input | 256 x 256 x 1 (grayscale) |
| Output | 256 x 256 x 1 (binary mask) |
| Loss | Binary cross-entropy |
| Metrics | Dice coefficient, IoU, accuracy |
| Optimizer | Adam (lr = 1e-4) |

The encoder uses 3x3 convolutions with ReLU activation, He-normal
initialisation, spatial dropout (0.1–0.3), and 2x2 max-pooling. The
decoder mirrors this with transposed convolutions and skip connections.

---

## Training datasets

The model was trained on 1,355 manually annotated echocardiogram frames
spanning three public/institutional datasets. Annotations were created
with [LabelMe](https://github.com/wkentaro/labelme).

| Dataset | Samples | Source | Access |
|---|---|---|---|
| MIMIC-IV-ECHO | 947 | PhysioNet | [Credentialed](https://physionet.org/content/mimic-iv-echo/) |
| EchoNet-Dynamic | 145 | Stanford | [Public](https://echonet.github.io/dynamic/) |
| EchoNet-Paediatric | 263 | Institutional | By request |
| **Total** | **1,355** | | |

> **Note:** Training data is **not** included in this repository. The
> pretrained model weights are provided in `models/`. To retrain, obtain
> the original datasets and place matched image/mask pairs in `data/`.

### Dataset limitations

- Training covers common clinical echo machines (GE, Philips, Siemens).
  Handheld / point-of-care (POCUS) devices with very different screen
  layouts may produce lower-quality masks.
- Fine-tuning on 50–100 annotated frames from the target device is
  usually sufficient to adapt the model.

---

## Training results

Pre-computed metrics and visualisations are stored in `training_results/`:

| Metric | Value |
|---|---|
| **Dice coefficient** | 0.9872 |
| **IoU (Jaccard)** | 0.9747 |
| **Accuracy** | 0.9900 |
| **Sensitivity** | 0.9857 |
| **Specificity** | 0.9928 |

Training was early-stopped at epoch 29/50 (patience = 10, monitoring
`val_loss`). Learning rate was reduced on plateau (factor 0.5,
patience 5).

```
training_results/
  training_history.png       Training / validation loss & metric curves
  prediction_samples.png     Sample predictions on held-out validation data
  metrics.json               Final Dice, IoU, accuracy, sensitivity, specificity
  dataset_summary.json       Dataset size, hyperparameters, best metrics
  training_log.csv           Per-epoch metrics
```

---

## Project layout

```
echoroi/                  Installable Python package
  __init__.py               Package entry point and public API
  model.py                  U-Net architecture, registered Dice & IoU metrics
  preprocessing.py          Image/mask loading, resizing, normalisation
  training.py               Training loop, callbacks, result saving
  inference.py              Prediction, ROI extraction, de-identification
  cli.py                    Command-line interface (train, predict, evaluate, benchmark)
models/                   Pretrained model weights
  echoroi_unified.keras     Keras model (355 MB)
  echoroi_unified.onnx      ONNX model (118 MB)
training_results/         Metrics, plots, and training artefacts
notebooks/                Jupyter notebooks for exploration & reproducibility
  01_training_and_evaluation.ipynb
  02_onnx_conversion.ipynb
  03_inference_demo.ipynb
  04_dataset_preprocessing.ipynb
paper/                    JOSS manuscript (paper.md, paper.bib, figures/)
scripts/                  Utility scripts (not part of the package)
  convert_labelme_to_masks.py   One-time: LabelMe JSON to PNG mask conversion
  convert_to_onnx.py            Convert .keras to .onnx with validation
tests/                    pytest suite
```

---

## ONNX export

Convert the Keras model to ONNX for deployment outside TensorFlow:

```bash
pip install -e ".[export]"
python scripts/convert_to_onnx.py
```

Use the ONNX model for inference with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/echoroi_unified.onnx")
# input: float32 [1, 256, 256, 1], output: float32 [1, 256, 256, 1]
pred = sess.run(None, {"input": image_batch})[0]
```

---

## Development

```bash
make dev          # install with dev + notebook extras
make test         # run pytest
make lint         # run ruff linter
make test-cov     # pytest with coverage report
make train        # retrain model on data/
make evaluate     # evaluate model on data/
make onnx         # convert model to ONNX
make clean        # remove build artefacts
```

---

## Citation

If you use EchoROI in your research, please cite:

```bibtex
@article{echoroi2026,
  title   = {EchoROI: A U-Net-based Python Tool for Echocardiographic ROI
             Segmentation and De-identification},
  author  = {Ekambaram, Kamlin},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md)
for guidelines.

## License

[MIT](LICENSE)
