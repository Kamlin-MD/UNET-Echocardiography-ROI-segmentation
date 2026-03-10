---
title: 'EchoROI: U-Net-based ROI Segmentation and De-identification for Echocardiography'
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
    affiliation: 1
    corresponding: true
  - name: Anurag Arnab
    affiliation: 2
  - name: Philip Herbst
    affiliation: 3
  - name: Rensu Theart
    affiliation: 4
affiliations:
  - name: University of Stellenbosch, Institute of Biomedical Engineering, South Africa
    index: 1
  - name: Google DeepMind, United Kingdom
    index: 2
  - name: University of Stellenbosch, Division of Cardiology, South Africa
    index: 3
  - name: University of Stellenbosch, Department of Electrical Engineering, South Africa
    index: 4
date: 10 March 2026
bibliography: paper.bib
---

# Summary

Echocardiography is the most widely used cardiac imaging modality, but secondary
analysis of echo videos is often limited by protected health information (PHI),
electrocardiographic traces, measurement overlays, and vendor-specific graphics
burned directly into pixel data [@madani2018fast-088]. `EchoROI` is an
open-source Python package that segments the fan-shaped ultrasound scan sector
and masks non-diagnostic image content using a U-Net-based model
[@ronneberger2015unet]. The resulting outputs preserve the native scan-sector
geometry while removing surrounding display content, enabling preprocessing for
machine learning, privacy-aware sharing of teaching examples, and more
consistent inputs for downstream analysis.

`EchoROI` is distributed as an installable package with a command-line
interface and a Python API. The repository includes pretrained weights,
training and evaluation workflows, ONNX export utilities, notebooks, and tests.
Users can apply the reference model to new image collections, fine-tune on
local annotations, evaluate image-mask pairs, and integrate the exported model
into non-TensorFlow pipelines.

![EchoROI processing pipeline showing a raw frame, predicted scan-sector mask,
de-identified output, and optional ROI crop. The annotations highlight both
successful removal of burned-in content outside the scan sector and a common
limitation of sector-based masking: a partial ECG trace traversing the scan
sector is retained.
\label{fig:pipeline}](figures/figure_1.png)

# Statement of Need

Echocardiography files commonly contain identifiers and vendor overlays
rendered directly into image pixels. In research settings these artifacts can
restrict data sharing and introduce shortcut signals for computer-vision models
[@panhuis2014systematic-d31]. Large collections such as MIMIC-IV-ECHO
[@gow2023mimic] therefore require preprocessing before they can be used safely
for model development or shared for secondary analysis. Manual removal is
feasible for small studies but becomes impractical at modern dataset scale. In
prior work, the authors manually segmented and cropped approximately 1,000
apical four-chamber studies from MIMIC-IV-ECHO using LabelMe
[@ekambaram2026mimicext; @russell2008labelme-d8b], which motivated the
development of a reusable automated workflow.

EchoROI was developed for researchers who need a scriptable and adaptable way
to standardise echocardiography inputs across heterogeneous sources. Rather
than relying on fixed crop templates, it learns the sector boundary directly
and uses the predicted mask to remove non-diagnostic content outside the scan
sector. This is useful for privacy-preserving preprocessing, multi-source
dataset harmonisation, and workflows where background overlays would otherwise
consume model capacity or bias evaluation. The package targets users preparing
echocardiography datasets for machine learning, benchmarking, or educational
reuse, especially when acquisition layouts vary across vendors, probe settings,
or sites.

# State of the Field

Existing approaches address related needs but not the full workflow targeted by
EchoROI. OCR-based de-identification pipelines remove text from ultrasound
files but do not explicitly recover the scan sector and may struggle with
graphical overlays or variable vendor layouts [@monteiro2017deid]. Heuristic
cleaning tools such as `PyLogik` can work well on stable layouts, but
layout-specific rules may require retuning when fan angle, depth, zoom, or
surrounding display content changes [@kline2023pylogik]. General medical
imaging toolboxes such as `ivadomed` support segmentation model development
across modalities [@gros2021ivadomed], but they do not provide
echocardiography-specific sector masking, representative-frame cine
preprocessing, or de-identification-oriented outputs out of the box. Public
datasets such as EchoNet-Dynamic also distribute pre-cropped videos
[@ouyang2020echonet], which simplifies downstream modelling but removes the
original scan-sector geometry and assumes consistent cropping upstream.

EchoROI therefore occupies a distinct niche as reusable research software for
learned scan-sector segmentation in heterogeneous echocardiography data. Its
main contribution is not a new segmentation architecture, but a packaged
workflow that combines pretrained inference, fine-tuning, evaluation, and
deployment-friendly export around the practical problem of sector masking and
de-identification.

# Software Design

EchoROI is packaged as a modular Python library with components for
preprocessing, model definition, training, inference, evaluation, and export.
The command-line interface exposes the main workflows for prediction,
fine-tuning, evaluation, and benchmarking, while the Python API supports
integration into larger research pipelines. A typical de-identification
workflow is:

```bash
echoroi predict \
  --model-path models/echoroi_unified.keras \
  --input video_frames/ \
  --output clean/ \
  --deidentify
```

For most echocardiography clips, sector geometry remains fixed throughout the
cine loop. EchoROI therefore predicts the mask from a single representative
frame and applies the same mask and crop logic across the full sequence. By
default, the representative frame is chosen from the first few frames using
grayscale Shannon entropy, reducing compute cost and helping avoid blank or
transition frames. EchoROI exposes the predicted binary mask as part of its
outputs, so users can generate masked ROI images directly and adapt downstream
code to retain complementary unmasked pixels when burned-in traces such as ECG
are needed for temporal interpretation. The model is trained for sector
segmentation only and does not attempt OCR or explicit extraction of overlay
elements.

The reference model uses a lightly modified U-Net for binary scan-sector
segmentation [@ronneberger2015unet]. The main design choices are
$256 \times 256$ grayscale inputs, same-padding convolutions to preserve the
echocardiographic sector geometry, and dropout regularisation
[@srivastava2014dropout] to reduce overfitting on a comparatively small and
heterogeneous annotation set. This is a pragmatic adaptation of a standard
U-Net rather than a novel architecture, favouring robust segmentation and
modest hardware requirements. The software is implemented in TensorFlow/Keras
and can export trained models to ONNX for deployment outside TensorFlow.

The pretrained model was developed on 1,355 manually annotated frame-mask pairs
collected from multiple public and institutional sources, including
MIMIC-IV-ECHO [@gow2023mimic], EchoNet-Dynamic [@ouyang2020echonet],
EchoNet-Paediatric [@reddy2022echonetpeds], CACTUS [@elmekki2025cactus],
EchoCP [@wang2021echocp], CardiacUDC [@yang2023graphecho], HMC-QU
[@degerli2024hmcqu], and a small consented institutional set. Masks were
annotated in LabelMe by outlining the visible scan sector while excluding
padding, borders, and display graphics [@russell2008labelme-d8b]. Only one
representative frame per cine sequence was used during training because the
sector boundary is generally static within a clip and this maximised diversity
across acquisitions.

# Research Impact Statement

EchoROI provides directly reproducible evidence that the software works as
intended on heterogeneous echocardiography data. On the validation split of the
annotated dataset, the reference model achieved:

| Metric           | Value  |
|:-----------------|-------:|
| Dice coefficient | 0.9880 |
| IoU              | 0.9763 |
| Pixel accuracy   | 0.9906 |
| Sensitivity      | 0.9894 |
| Specificity      | 0.9914 |

These figures come from the validation split used for model selection, so they
should be interpreted as software validation rather than a definitive external
benchmark. On a consumer Apple Mac mini with an M2 Pro, TensorFlow/Keras
inference averaged approximately 56 ms per $256 \times 256$ frame, making the
package practical for routine preprocessing of short clips on modest hardware.

The repository is intended for immediate reuse in echocardiography research
workflows. It includes pretrained Keras and ONNX models, example notebooks,
evaluation outputs, training logs, and tests, so users can reproduce the
reported workflow or adapt it to site-specific data. Source code and artifacts
are available at
[https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation).
The included evaluation command and DICOM-to-NPZ notebook support end-to-end
preprocessing pipelines in which clips are loaded, optionally resampled,
masked using EchoROI, cropped using the predicted sector geometry, and saved
for downstream modelling. This combination of usable software, pretrained
artifacts, and reproducible examples gives EchoROI credible near-term impact as
a preprocessing component for privacy-aware dataset preparation and
standardised input generation in echocardiography research.

EchoROI should nonetheless be used as a preprocessing aid rather than a
guarantee of complete PHI removal. Still frames are more likely than moving
cine clips to contain measurements or text annotations within the scan sector,
some of which may contain PHI. By contrast, cine clips are typically saved
without diagnostic-sector annotations; the more common residual failure mode is
temporal overlays such as ECG or respiratory traces that cross the sector
boundary and therefore cannot be removed by masking alone, as illustrated in
\autoref{fig:pipeline}. These traces are unlikely to contain PHI, but human
review and local governance procedures remain necessary before external data
sharing.

# AI Usage Disclosure

GitHub Copilot in Visual Studio Code was used for limited repository
maintenance and manuscript revision. All AI-assisted changes were reviewed,
edited, and validated by the authors, who take responsibility for the final
software and manuscript.

# Acknowledgements

This work was supported by the University of Stellenbosch Institute of
Biomedical Engineering. The manual A4C segmentation effort described in
[@ekambaram2026mimicext] directly motivated the development of EchoROI. We
gratefully acknowledge the providers of the datasets used for training and
evaluation, including PhysioNet and Beth Israel Deaconess Medical Center for
MIMIC-IV-ECHO [@gow2023mimic; @goldberger2000physionet], the Stanford AIMI
Center for EchoNet-Dynamic [@ouyang2020echonet] and EchoNet-Paediatric
[@reddy2022echonetpeds], and the authors of CACTUS [@elmekki2025cactus],
EchoCP [@wang2021echocp], CardiacUDC [@yang2023graphecho], and HMC-QU
[@degerli2024hmcqu]. EchoROI is built on TensorFlow/Keras, OpenCV, and the
Python scientific computing ecosystem.

# References