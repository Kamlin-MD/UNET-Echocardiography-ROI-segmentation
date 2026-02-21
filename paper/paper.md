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
    affiliation: "1, 2"
affiliations:
  - name: University of Stellenbosch, Institute of Biomedical Engineering, South Africa
    index: 1
  - name: University of KwaZulu-Natal, School of Clinical Medicine, Division of Emergency Medicine, South Africa
    index: 2
date: 21 February 2026
bibliography: paper.bib
---

# Summary

Echocardiography is the most widely performed cardiac imaging modality,
yet large-scale computational analysis of echocardiogram videos is
hindered by protected health information (PHI) and vendor-specific
overlays embedded directly in the pixel data.  `EchoROI` is an
open-source Python package that uses a U-Net convolutional neural
network [@ronneberger2015unet] to segment the fan-shaped ultrasound
scan sector---the region of interest (ROI)---in each echocardiogram
frame and masks out everything else: patient identifiers, ECG traces,
calipers, measurement readouts, and vendor logos.  The result is a
clean frame containing only the cardiac image on a black background,
suitable for privacy-compliant data sharing and downstream machine
learning.

`EchoROI` is distributed as an installable Python package with both a
command-line interface (CLI) and a Python API.  Users can apply the
pretrained model directly, fine-tune it on site-specific annotated
frames, and export to ONNX for deployment outside TensorFlow.

<!-- FIGURE 1 — Pipeline overview (4-panel composite)
     Suggested content: a single horizontal strip or 2x2 grid showing
       (a) Raw echocardiogram frame with PHI/overlays visible
       (b) U-Net predicted binary mask
       (c) De-identified frame (non-ROI blacked out)
       (d) Extracted ROI crop
     Source: output of `echoroi predict --visualize` on a non-PHI sample.
     Save as: figures/pipeline_overview.png (300 dpi, ~1200 px wide)
-->
![EchoROI processing pipeline.  (a) Raw echocardiogram frame with
vendor overlays and patient identifiers.  (b) U-Net predicted binary
mask of the scan sector.  (c) De-identified frame with non-ROI content
masked to black.  (d) Extracted ROI
crop.\label{fig:pipeline}](figures/pipeline_overview.png)

# Statement of Need

Public echocardiography datasets such as MIMIC-IV-ECHO
[@gow2023mimic]---comprising over 500,000 sequences from 4,579
patients---contain PHI burned into the image pixels.  Before these data
can be shared or used to train models, all identifiers and extraneous
overlays must be removed to comply with privacy regulations.  Manual
anonymisation is impractical at scale.

Existing automated approaches address parts of the problem.  Monteiro
et al. combined optical character recognition with filtering to remove
text from ultrasound DICOM images, achieving approximately 89% success
on 500 test images [@monteiro2017deid].  Kline et al. released
PyLogik, a Python library that uses morphological image operations to
detect the ROI, reporting an average Dice coefficient of 0.976 on 50
cardiac echo images [@kline2023pylogik].  The EchoNet-Dynamic dataset
[@ouyang2020echonet] distributed pre-cropped 112x112 videos, but used
fixed heuristic cropping that assumes a constant fan angle, straight
sector edges, and an apex-up orientation---assumptions that do not
generalise across vendors, zoom levels, or probe tilts.

No open-source tool combines deep-learning segmentation with
de-identification for echocardiography.  `EchoROI` fills this gap with
a U-Net trained on diverse annotated frames that learns the true curved
sector boundary rather than relying on heuristics.  The model
generalises across vendors (GE, Philips, Siemens), views (apical
four-chamber, two-chamber, parasternal long-axis), and display layouts.
Downstream vision models trained on EchoROI-processed data benefit from
a standardised appearance free of distractors---a consideration
highlighted by self-supervised methods such as masked autoencoders
[@he2022mae], which waste capacity reconstructing static text or
borders when these are not removed.

# Implementation

## Architecture

`EchoROI` is built on a standard U-Net [@ronneberger2015unet] with
four encoder and four decoder blocks (31 million parameters).  Each
encoder block applies two 3x3 convolutions with ReLU activation,
He-normal initialisation, batch normalisation, and spatial dropout
(0.1--0.3), followed by 2x2 max-pooling.  The decoder mirrors this
structure with 2x2 transposed convolutions and skip connections from
the corresponding encoder level.  The final layer is a 1x1 convolution
with sigmoid activation producing a single-channel mask.  Input and
output are 256x256x1 grayscale images.

The model is trained with binary cross-entropy loss and monitored using
the Dice coefficient and intersection-over-union (IoU).  The Adam
optimiser is used with an initial learning rate of $1 \times 10^{-4}$
and a reduce-on-plateau schedule (factor 0.5, patience 5 epochs).

<!-- FIGURE 2 — U-Net architecture diagram
     Suggested content: block diagram of the 4-level encoder-decoder
     with skip connections, showing layer sizes:
       Encoder: 64→128→256→512, bottleneck 1024
       Decoder: 512→256→128→64, output 1 (sigmoid)
     Options:
       - Draw in draw.io / Inkscape / TikZ and export as PDF or PNG
       - Or use a clean schematic similar to the original U-Net paper
     Save as: figures/unet_architecture.png (or .pdf)
-->
![U-Net architecture used in EchoROI.  The encoder (left) contracts the
256x256 input through four pooling stages; the decoder (right) recovers
spatial resolution via transposed convolutions and skip connections.
Numbers indicate feature-map
channels.\label{fig:architecture}](figures/unet_architecture.png)

## Training Data

The model was trained on 1,206 manually annotated echocardiogram
frames drawn from three sources:

| Dataset              | Frames | Source      |
|:---------------------|-------:|:------------|
| MIMIC-IV-ECHO        |    947 | PhysioNet [@gow2023mimic; @goldberger2000physionet] |
| EchoNet-Dynamic      |    145 | Stanford [@ouyang2020echonet]  |
| EchoNet-Paediatric   |    263 | Institutional |
| **Total**            | **1,206** |          |

Ground-truth masks were created with LabelMe by outlining the scan
sector boundary on each frame.  Annotations included all visible
cardiac content while excluding padding, borders, and overlay graphics.
Training used an 80/20 train--validation split with a batch size of 16
for 50 epochs.

## Software Design

`EchoROI` is structured as a pip-installable package (`echoroi`) with
five modules:

- **`model`** -- U-Net architecture with Keras-serialisable Dice and IoU
  metrics.
- **`preprocessing`** -- Aspect-ratio-preserving resize with zero-padding.
- **`training`** -- Training loop with checkpointing, learning-rate
  scheduling, and result export (plots, CSV logs, JSON metrics).
- **`inference`** -- Single-image and batch prediction, ROI extraction,
  de-identification, and inference benchmarking.
- **`cli`** -- Subcommands: `train`, `predict`, `evaluate`, `benchmark`,
  and `create-data`.

Models can be exported to ONNX via an included conversion script for
framework-agnostic deployment with ONNX Runtime.

# Validation

On a held-out validation set (20% of the 1,206 annotated frames) the
model achieves:

| Metric        | Value  |
|:--------------|-------:|
| Dice coefficient | 0.9880 |
| IoU (Jaccard)    | 0.9763 |
| Pixel accuracy   | 0.9906 |
| Sensitivity      | 0.9894 |
| Specificity      | 0.9914 |

These results meet or exceed the accuracy of prior tools: PyLogik
reported 0.976 Dice on 50 images [@kline2023pylogik], and the Monteiro
et al. pipeline achieved 89% success on 500 images
[@monteiro2017deid].  Visual inspection of de-identified outputs
confirmed that no patient names, medical record numbers, or ECG traces
remained visible in the masked frames.

<!-- FIGURE 3 — Prediction samples on held-out validation data
     Suggested content: 3–4 columns, each showing:
       Row 1: Input frame
       Row 2: Ground-truth mask
       Row 3: Predicted mask
     Pick examples from different datasets / vendors if possible.
     Source: training_results/prediction_samples.png or re-generate
             with diverse samples.
     Save as: figures/prediction_samples.png (300 dpi)
-->
![Sample predictions on held-out validation frames.  Each column shows
the input frame (top), ground-truth mask (middle), and U-Net predicted
mask (bottom).  Frames span multiple vendors and
views.\label{fig:predictions}](figures/prediction_samples.png)

On a consumer Apple M2 Pro (CPU/GPU), inference takes approximately
25 ms per 256x256 frame, enabling real-time processing of short echo
clips.

# Availability and Reuse

`EchoROI` is released under the MIT licence.  Source code, pretrained
weights, example notebooks, and a test suite are available at
[https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation).
Users who work with devices or views not represented in the training
set can fine-tune the model on 50--100 site-specific annotated frames
using the built-in CLI or Python API.  Beyond cardiology, the same
architecture can be adapted to any ultrasound modality with a distinct
scan-sector shape (e.g., lung, abdominal, or vascular imaging) via
transfer learning.

# Acknowledgements

This work was supported by the University of Stellenbosch Institute of
Biomedical Engineering.  We thank the PhysioNet team
[@goldberger2000physionet] for providing the MIMIC-IV-ECHO dataset
[@gow2023mimic].  `EchoROI` is built on TensorFlow/Keras, OpenCV, and
the Python scientific computing ecosystem.

# References
