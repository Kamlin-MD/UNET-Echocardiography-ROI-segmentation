---
title: 'EchoROI: A U-Net-based Python Tool for Echocardiographic ROI Segmentation and De-identification'
tags:
  - Python
  - medical imaging
  - ultrasound
  - deep learning
  - segmentation
  - U-Net
  - de-identification
  - echocardiography
authors:
  - name: Kamlin Ekambaram
    orcid: 0000-0002-1366-8451
    affiliation: "1"
  - name: Rensu Theart
    orcid:
    afflication: "2"
affiliations:
  - name: University of Stellenbosch, Institute of Biomedical Engineering, South Africa
    index: 1
  - name: University of Stellenbosch, Department of Electrical Engineering, South Africa
    index: 2
date: 21 February 2026
bibliography: paper.bib
---

# Summary

Echocardiography is the most widely performed cardiac imaging modality, yet
large-scale computational analysis of echocardiogram videos is hindered by
protected health information (PHI) and vendor-specific overlays that are often
burned into the pixel data. `EchoROI` is an open-source Python package that uses
a U-Net convolutional neural network [@ronneberger2015unet] to segment the
fan-shaped ultrasound scan sector---the region of interest (ROI)---in each
frame, and to mask out non-ROI content (e.g., identifiers, ECG traces, calipers,
measurement readouts, and vendor logos). The resulting frames preserve the
native scan-sector resolution while removing distracting background elements,
facilitating dataset preprocessing for machine learning and enabling safer
sharing of examples for teaching.

EchoROI is distributed as an installable Python package with both a command-line
interface (CLI) and a Python API. Users can apply the pretrained model, fine-
tune it on site-specific annotations, and export models to ONNX for deployment
outside TensorFlow.

![EchoROI processing pipeline: raw input, predicted scan-sector mask,
masked/de-identified output, and optional ROI extraction.](pipeline_overview.png)

# Statement of Need

Many echocardiography files contain PHI and vendor overlays rendered directly
into the image, which restricts data sharing and introduces confounders for
computer-vision models. Public datasets such as MIMIC-IV-ECHO [@gow2023mimic]
contain identifiers and overlays that must be removed to support privacy-
preserving research workflows; manual anonymisation is impractical for modern-
scale collections.

Existing tools address related subproblems. OCR-based pipelines remove text but
may fail when overlays vary across vendors or appear in low-contrast regions
[@monteiro2017deid]. Heuristic ROI detection approaches can perform well on
limited layouts (e.g., fixed fan angle and orientation), but may not generalise
across vendors, zoom levels, and probe tilts [@kline2023pylogik]. The
EchoNet-Dynamic dataset distributed pre-cropped videos [@ouyang2020echonet],
which simplifies modelling but discards the original pixel geometry and assumes
stable display conventions.

`EchoROI` provides an end-to-end, open-source workflow for learning the true
scan-sector boundary with deep segmentation and using that boundary to standardise
frames for downstream analysis. By explicitly modelling the curved sector edges,
EchoROI avoids brittle cropping heuristics and supports diverse acquisition
layouts. Standardising the field of view also reduces wasted model capacity on
static overlays, which is particularly relevant for representation-learning
approaches such as masked autoencoders [@he2022mae].

# Usage

EchoROI can be used from the command line for batch preprocessing or as a Python
library within research pipelines.

```bash
# De-identify: black out everything outside the scan sector
echoroi predict \
  --model-path models/echoroi_unified.keras \
  --input video_frames/ \
  --output clean/ \
  --deidentify

# Fine-tune on site-specific annotations
echoroi train \
  --image-dir data/images \
  --mask-dir data/masks \
  --model-path models/echoroi_finetuned.keras \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --results-dir training_results
```

```python
from echoroi import UNetPredictor

predictor = UNetPredictor("models/echoroi_unified.keras")
mask = predictor.predict_single_image("frame.png")
result = predictor.process_image_with_visualization(
    "frame.png", save_path="output.png"
)
```

# Implementation

## Architecture

EchoROI uses a standard U-Net [@ronneberger2015unet] with four encoder and four
decoder blocks (31 million parameters). Encoder blocks apply 3x3 convolutions
with ReLU activation, He-normal initialisation, batch normalisation, and spatial
dropout (0.1--0.3), followed by 2x2 max-pooling. The decoder mirrors this
structure using transposed convolutions and skip connections. A final 1x1
convolution with sigmoid activation produces a single-channel binary mask. Input
and output are 256x256x1 grayscale images.

Training uses binary cross-entropy loss and is monitored with Dice and
intersection-over-union (IoU). Optimisation uses Adam with an initial learning
rate of $1 \times 10^{-4}$ and a reduce-on-plateau schedule (factor 0.5,
patience 5 epochs). The reference implementation is built in TensorFlow/Keras
and was trained and evaluated on an Apple Mac mini with an M2 Pro (CPU/GPU).

<!-- FIGURE 2 (placeholder) — Model diagram
     Simple U-Net schematic with feature-map sizes and skip connections.
     Save as: figures/unet_architecture.png
-->
![U-Net architecture used in EchoROI.](figures/unet_architecture.png)

## Training Data

The model was trained on 1,356 manually annotated echocardiogram
frame/mask pairs drawn from multiple sources:

| Dataset                         | Frames | Source |
|:--------------------------------|-------:|:-------|
| MIMIC-IV-ECHO                   |    403 | PhysioNet [@gow2023mimic; @goldberger2000physionet] |
| EchoNet-Dynamic                 |    145 | Stanford [@ouyang2020echonet] |
| EchoNet-Paediatric              |    263 | Institutional |
| A4C Cactus dataset              |     38 | Public dataset |
| echoCP                          |     60 | Public dataset |
| Private dataset (consented)     |     50 | Institutional (Mindray/Samsung) |
| CardiacUDC + HMC-QU             |    397 | Public datasets |
| **Total**                       | **1,356** | |

Ground-truth masks were created in LabelMe by outlining the scan-sector
boundary. Annotations included all visible diagnostic content while excluding
padding, borders, and overlay graphics. Training used a single 80/20
train--validation split created once at training start by randomly shuffling the
full dataset with a fixed seed. The split was not stratified by dataset source.
Batch size was 16 and training ran for 50 epochs.

## Software Design

EchoROI is distributed as a pip-installable package (`echoroi`) with modular
components for preprocessing, model definition, training, and inference. The CLI
provides subcommands for training, prediction, evaluation, and benchmarking.
Models can also be exported to ONNX for framework-agnostic deployment.

# Validation

On the validation split (20% of the 1,356 annotated frames), EchoROI achieves:

| Metric            | Value  |
|:------------------|-------:|
| Dice coefficient  | 0.9880 |
| IoU (Jaccard)     | 0.9763 |
| Pixel accuracy    | 0.9906 |
| Sensitivity       | 0.9894 |
| Specificity       | 0.9914 |

These segmentation results meet or exceed prior open approaches (e.g., PyLogik
reported 0.976 Dice on 50 images [@kline2023pylogik]). No separate held-out test
set was used; the final metrics are reported on the same validation split used
for model selection (best checkpoint by `val_dice_coefficient`), and may modestly
overestimate generalisation performance.

De-identification quality was assessed by manual spot-checking of model outputs
on the validation split. During LabelMe annotation, frames were also reviewed to
ensure that PHI was not present within the scan-sector ground-truth masks used
for training.

On a consumer Apple M2 Pro (CPU/GPU), inference takes approximately 25 ms per
256x256 frame, enabling real-time preprocessing of short echo clips.

<!-- FIGURE 3 — Qualitative segmentation results
     Grid: input frame, ground-truth mask, predicted mask, masked output.
     Include examples across vendors/views.

     JOSS expects figures under paper/figures/. In this Overleaf preview
     project the image is stored at the project root.
     Target path for GitHub/JOSS: figures/prediction_samples.png
-->
![Sample predictions on held-out validation frames.](prediction_samples.png)

# Limitations

EchoROI should be treated as a preprocessing and de-identification *aid* rather
than a guarantee of complete PHI removal. Masking may fail for atypical layouts,
low contrast, extreme zoom, handheld/POCUS devices, or when identifiers overlap
or traverse the scan sector. Because PHI can appear inside the ROI, users should
apply human-in-the-loop review and follow local governance procedures before
external sharing of derived images or cine loops.

<!-- FIGURE 4 (placeholder) — Failure cases and safety checks
     Show 3--4 representative failure modes:
       (a) text intersecting ROI,
       (b) atypical layout,
       (c) low-contrast borders,
       (d) incorrect mask extent.
     Optionally add recommended workflow: run model -> detect failures -> human review.
     Save as: figures/failure_cases.png
-->

# Reproducibility

A minimal reproduction of the reported segmentation metrics can be performed
using the built-in evaluation command on a directory of images and masks:

```bash
echoroi evaluate \
  --model-path models/echoroi_unified.keras \
  --image-dir <IMAGE_DIR> \
  --mask-dir <MASK_DIR> \
  --output <EVAL_DIR>
```

The repository includes scripts and notebooks for training, evaluation, and
ONNX export. Training data are not redistributed; users can retrain by providing
matched image/mask pairs derived from their permitted datasets and/or
site-specific LabelMe annotations.

# Availability and Reuse

EchoROI is released under the MIT licence. Source code, pretrained weights,
example notebooks, and a test suite are available at
[https://github.com/Kamlin-MD/echoroi](https://github.com/Kamlin-MD/echoroi).

The primary use case is research preprocessing of large echocardiography
collections: standardising frames by removing non-diagnostic background content
while preserving scan-sector resolution. A secondary use case is education
(e.g., sharing de-identified stills or cine loops for teaching and FOAMed),
subject to user responsibility and local policy.

Users working with devices or layouts not represented in the training set can
fine-tune the model on 50--100 annotated frames using the CLI or Python API.
EchoROI is not a substitute for institutional de-identification procedures;
governance review and human-in-the-loop workflows are recommended before any
external data sharing.

# Acknowledgements

This work was supported by the University of Stellenbosch Institute of Biomedical
Engineering. We thank the PhysioNet team [@goldberger2000physionet] for
providing the MIMIC-IV-ECHO dataset [@gow2023mimic]. EchoROI is built on
TensorFlow/Keras, OpenCV, and the Python scientific computing ecosystem.

# References
