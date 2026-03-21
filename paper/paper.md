---
title: 'EchoROI: Scan-sector Segmentation and De-identification for Echocardiography'
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
  - name: Institute for Biomedical Engineering, Faculty of Engineering, Stellenbosch University, South Africa
    index: 1
  - name: Independent Researcher
    index: 2
  - name: Division of Cardiology, Department of Medicine, Faculty of Medicine and Health Sciences, Stellenbosch University and Tygerberg Hospital, Cape Town, South Africa
    index: 3
  - name: Department of Electrical Engineering, Faculty of Engineering, Stellenbosch University, South Africa
    index: 4
date: 10 March 2026
bibliography: paper.bib
header-includes:
  - \usepackage{graphicx}
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
consistent inputs for downstream analysis. The pretrained model targets the
fan-shaped scan sector produced by phased-array cardiac probes and may
generalise to similar curvilinear sector geometries, but is unlikely to
segment the rectangular field of view of linear-array probes without
fine-tuning or retraining on appropriate annotations.

`EchoROI` is distributed as an installable package via
[PyPI](https://pypi.org/project/echoroi/) (`pip install echoroi`) and provides
a command-line interface and a Python API. The repository includes pretrained
weights, training and evaluation workflows, ONNX export utilities, notebooks,
and tests.
Users can apply the reference model to new image collections, fine-tune on
local annotations, evaluate image-mask pairs, and integrate the exported model
into non-TensorFlow pipelines.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{figures/figure_1a.png}\hfill
\includegraphics[width=0.48\textwidth]{figures/figure_1b.png}\\[0.5em]
\includegraphics[width=0.48\textwidth]{figures/figure_1c.png}\hfill
\includegraphics[width=0.48\textwidth]{figures/figure_1d.png}
\caption{\texttt{EchoROI} preprocessing workflow. (a) Raw frame, with examples of burned-in
overlays targeted for removal highlighted in red; (b) predicted scan-sector
mask; (c) de-identified output, where overlays outside the scan sector are
removed but a partial ECG trace and partial scale bar are retained where they
traverse the scan sector; and (d) ROI crop.}
\label{fig:pipeline}
\end{figure}

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

`EchoROI` was developed for researchers who need a scriptable and adaptable way
to standardise echocardiography inputs across heterogeneous sources. Rather
than relying on fixed crop templates, the provided pretrained model has learnt
the sector boundary directly and uses the predicted mask to remove
non-diagnostic content outside the scan
sector. This is useful for privacy-preserving preprocessing, multi-source
dataset harmonisation, and workflows where background overlays would otherwise
consume model capacity or bias evaluation. The package targets users preparing
echocardiography datasets for machine learning, benchmarking, or educational
reuse, especially when acquisition layouts vary across vendors, probe settings,
or sites.

# State of the Field

Existing approaches address related needs but not the full workflow targeted by
`EchoROI`. OCR-based de-identification pipelines remove text from ultrasound
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

`EchoROI` therefore occupies a distinct niche as reusable research software for
learned scan-sector segmentation in heterogeneous echocardiography data. Its
main contribution is not a new segmentation architecture, but a packaged
workflow that combines pretrained inference, fine-tuning, evaluation, and
deployment-friendly export around the practical problem of sector masking and
de-identification.

# Software Design

`EchoROI` is packaged as a modular Python library with components for
preprocessing, model definition, training, inference, evaluation, and export.
The command-line interface exposes the main workflows for prediction,
fine-tuning, evaluation, and benchmarking, while the Python API supports
integration into larger research pipelines. The core operation is mask
prediction: the model segments the scan sector and outputs a binary mask for
each input. When the optional `--deidentify` flag is set, the predicted mask
is applied to zero out all pixels outside the scan sector, producing a
de-identified output in a single step. A typical workflow is:

```bash
echoroi predict \
  --model-path models/echoroi_unified.keras \
  --input video_frames/ \
  --output clean/ \
  --deidentify
```

For most echocardiography clips, sector geometry remains fixed throughout the
cine loop. `EchoROI` therefore predicts the mask from a single representative
frame and applies the same mask and crop logic across the full sequence. By
default, the representative frame is chosen from the first few frames using
grayscale Shannon entropy [@shannon1948], reducing compute cost and helping
avoid blank or transition frames. Without `--deidentify`, only the predicted
binary mask is saved, allowing users to apply it in custom downstream
workflows or inspect residual traces such as ECG that may cross the sector
boundary.
The model performs sector segmentation only and does not attempt OCR or
explicit overlay extraction.

The reference model uses a standard U-Net [@ronneberger2015unet] for binary
scan-sector segmentation, adapted to this application with $256 \times 256$
grayscale inputs, same-padding convolutions to preserve sector geometry, and
dropout regularisation [@srivastava2014dropout] to reduce overfitting on a
small heterogeneous annotation set. Training uses a composite loss that sums
binary cross-entropy, Dice loss [@milletari2016vnet], and a total-variation
(TV) regularisation term [@rudin1992tv]. The Dice component handles
foreground/background class imbalance, while the TV term penalises
high-frequency mask gradients and encourages the smooth, fan-shaped sector
boundaries characteristic of phased-array ultrasound probes.  The
train/validation split (80/20) is stratified by source dataset so that every
contributing collection is proportionally represented.  The TensorFlow/Keras
implementation prioritises robust segmentation and modest hardware
requirements, and supports ONNX export for deployment outside TensorFlow.

The pretrained model was developed on 1,355 annotated frame-mask pairs from
public and institutional echocardiography datasets, including
MIMIC-IV-ECHO [@gow2023mimic], EchoNet-Dynamic [@ouyang2020echonet],
EchoNet-Paediatric [@reddy2022echonetpeds], CACTUS [@elmekki2025cactus],
EchoCP [@wang2021echocp], CardiacUDC [@yang2023graphecho], HMC-QU
[@degerli2024hmcqu], and a small consented institutional set. Sector masks
were created in LabelMe [@russell2008labelme-d8b] by outlining the visible
scan sector while excluding padding and display graphics. One representative
frame per cine loop was used because sector geometry is typically static
within a clip; a per-source breakdown is provided in the repository
documentation.

# Research Impact Statement

`EchoROI` provides directly reproducible evidence that the software works as
intended on heterogeneous echocardiography data. On the validation split of the
annotated dataset, the reference model achieved:

| Metric           | Value  |
|:-----------------|-------:|
| Dice coefficient | 0.9884 |
| IoU              | 0.9770 |
| Pixel accuracy   | 0.9907 |
| Sensitivity      | 0.9891 |
| Specificity      | 0.9918 |

These figures come from the validation split used for model selection, so they
should be interpreted as software validation rather than a definitive external
benchmark. The software is suitable for routine preprocessing workflows, and
implementation-specific benchmarking examples are provided in the repository
notebooks.

The repository supports immediate reuse in echocardiography research
workflows. It includes pretrained Keras and ONNX models, example notebooks,
evaluation outputs, training logs, and tests, so users can reproduce the
reported workflow or adapt it to site-specific data. Source code and artifacts
are available at
[https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation).
The included evaluation command and DICOM-to-NPZ notebook support end-to-end
preprocessing pipelines in which clips are loaded, optionally resampled,
masked using `EchoROI`, cropped using the predicted sector geometry, and saved
for downstream modelling. This combination of usable software, pretrained
artifacts, and reproducible examples gives `EchoROI` credible near-term impact as
a preprocessing component for privacy-aware dataset preparation and
standardised input generation in echocardiography research.

`EchoROI` should nonetheless be used as a preprocessing aid rather than a
guarantee of complete PHI removal. On still frames, clinicians may place
measurement callipers, annotations, or text labels at arbitrary positions
within the scan sector; because this placement is user-dependent, such
overlays can include patient identifiers or clinical notes that the sector
mask cannot remove. Cine clips are typically saved without user-placed
in-sector annotations, but machine-generated temporal overlays — such as ECG
or respiratory traces — may traverse the sector boundary and therefore remain
after masking, as illustrated in \autoref{fig:pipeline}. These traces are
unlikely to contain PHI, but human review and local governance procedures
remain necessary before external data sharing.

# AI Usage Disclosure

GitHub Copilot in Visual Studio Code was used for limited repository
maintenance and manuscript revision. All AI-assisted changes were reviewed,
edited, and validated by the authors, who take responsibility for the final
software and manuscript.

# Acknowledgements

This work was supported by the University of Stellenbosch Institute of
Biomedical Engineering and in part by the National Research Foundation of
South Africa (Grant Number: TTK240321210363). The manual A4C segmentation effort described in
[@ekambaram2026mimicext] directly motivated the development of `EchoROI`. We
gratefully acknowledge the providers of the datasets used for training and
evaluation, including PhysioNet and Beth Israel Deaconess Medical Center for
MIMIC-IV-ECHO [@gow2023mimic; @goldberger2000physionet], the Stanford AIMI
Center for EchoNet-Dynamic [@ouyang2020echonet] and EchoNet-Paediatric
[@reddy2022echonetpeds], and the authors of CACTUS [@elmekki2025cactus],
EchoCP [@wang2021echocp], CardiacUDC [@yang2023graphecho], and HMC-QU
[@degerli2024hmcqu]. `EchoROI` is built on TensorFlow/Keras, OpenCV, and the
Python scientific computing ecosystem.

# References