---
title: 'EchoROI: Scan-sector Segmentation and De-identification for Echocardiography'
keywords:
  - Python
  - medical imaging
  - ultrasound
  - deep learning
  - segmentation
  - U-Net
  - de-identification
  - echocardiography
author:
  - name: Kamlin Ekambaram
    orcid: 0000-0002-1366-8451
    affiliation: Institute for Biomedical Engineering, Faculty of Engineering, Stellenbosch University, South Africa
  - name: Anurag Arnab
    affiliation: Independent Researcher
  - name: Philip Herbst
    affiliation: Division of Cardiology, Department of Medicine, Faculty of Medicine and Health Sciences, Stellenbosch University and Tygerberg Hospital, Cape Town, South Africa
  - name: Rensu Theart
    affiliation: Department of Electrical Engineering, Faculty of Engineering, Stellenbosch University, South Africa
date: 10 March 2026
bibliography: paper.bib
---

*Unrefereed technical manuscript; retained as material for a Biomedical Engineering doctoral thesis chapter at Stellenbosch University.*

## Author affiliations

- **Kamlin Ekambaram** (corresponding author): Institute for Biomedical
  Engineering, Faculty of Engineering, Stellenbosch University, South Africa.
  ORCID: [0000-0002-1366-8451](https://orcid.org/0000-0002-1366-8451).
- **Anurag Arnab**: Independent Researcher.
- **Philip Herbst**: Division of Cardiology, Department of Medicine, Faculty
  of Medicine and Health Sciences, Stellenbosch University and Tygerberg
  Hospital, Cape Town, South Africa.
- **Rensu Theart**: Department of Electrical Engineering, Faculty of
  Engineering, Stellenbosch University, South Africa.

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
a command-line interface and a Python API. The repository includes training
and evaluation workflows, Keras and ONNX support, export utilities, notebooks,
aggregate reference-run records, and tests, but not pretrained weights or
source datasets. Users must train locally or otherwise obtain a checkpoint
lawfully before applying the inference workflows.

Pretrained weights are not distributed with this release because the contributing datasets have differing access and reuse conditions. Researchers must obtain source datasets independently from their providers. Requests for scan-sector annotations should be directed to the EchoROI authors; availability is assessed for each dataset under the applicable agreements and institutional permissions.

![Raw echocardiographic frame](figures/figure_1a.png){width=48%}
![Predicted scan-sector mask](figures/figure_1b.png){width=48%}

![Sector-masked output](figures/figure_1c.png){width=48%}
![ROI crop](figures/figure_1d.png){width=48%}

::: {#fig-pipeline}
**Figure 1.** `EchoROI` preprocessing workflow. (a) Raw frame, with examples of burned-in
overlays targeted for removal highlighted in red; (b) predicted scan-sector
mask; (c) sector-masked output, where overlays outside the scan sector are
removed but a partial ECG trace and partial scale bar are retained where they
traverse the scan sector; and (d) ROI crop.
:::

# Motivation

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

# Related Work

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
workflow that combines inference with a user-supplied checkpoint, training, evaluation, and
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
sector-masked output in a single step. This operation does not remove
identifiers or overlays within the predicted ROI and does not itself
de-identify DICOM metadata. With a locally trained or otherwise lawfully obtained
checkpoint at `path/to/model.keras`, a typical workflow is:

```bash
echoroi predict \
  --model-path path/to/model.keras \
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

The documented annotation collection contains 1,355 frame-mask pairs from
public and institutional echocardiography datasets, including
MIMIC-IV-ECHO [@gow2023mimic], EchoNet-Dynamic [@ouyang2020echonet],
EchoNet-Paediatric [@reddy2022echonetpeds], CACTUS [@elmekki2025cactus],
EchoCP [@wang2021echocp], CardiacUDC [@yang2023graphecho], HMC-QU
[@degerli2024hmcqu], and a small consented institutional set. Sector masks
were created in LabelMe [@russell2008labelme-d8b] by outlining the visible
scan sector while excluding padding and display graphics. One representative
frame per cine loop was used because sector geometry is typically static
within a clip; a per-source breakdown is provided in the repository
documentation. This collection inventory must be distinguished from the saved
reference-run records, which report 1,206 loaded pairs (964 training and 242
validation), batch size 16, 50 epochs, and learning rate $10^{-4}$.
The saved run's source code selects checkpoints by validation IoU; batch size
8 is the current command-line default, not the recorded run setting.
There is no saved per-file manifest establishing which collection members
were used or excluded, so the 149-pair difference cannot be attributed to a
particular source from these records. A historical summary edit at commit
`1a36462` changed counts from 1,206 to 1,355 without changing other summary
fields; the later reference-run update at `67d6ecc` records 1,206 pairs
alongside new logs and metrics. The intermediate summary is not independent
evidence of training on the entire inventory.

# Validation and Reuse

The authors' saved reference-model evaluation reports the following results
on the 242-sample validation split described in the evaluation notebook:

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

The values above are thresholded, pooled-pixel metrics from
`training_results/metrics.json`. The training log instead records best soft
validation Dice 0.98726451397 and IoU 0.97487163544, both at zero-based epoch 47.
The model card also retains previously reported values (Dice 0.9880 and IoU
0.9763), matching the older `metrics.json` at `5e10245`, whose summary records
1,206 pairs and batch size 16. That earlier source selected by validation
Dice; the later source at `61e447b` introduced IoU selection and source
stratification. These experiments and metric summaries should not be treated
as interchangeable; neither provides a released per-file split and
evaluation-checkpoint hash.

The repository supports reuse of the software once researchers have obtained
authorised data, annotations, and a local checkpoint. It includes example
notebooks, aggregate evaluation metrics, training logs, and tests.
The access and retraining documentation describes a pathway for new training
runs, not a guarantee of immediate or exact reproduction of the reference
checkpoint. Exact reproduction would additionally require the original
per-file membership, extraction recipes, split assignments, software environment,
and random training state. Source code and aggregate records
are available at
[https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation).
The included evaluation command and DICOM-to-NPZ notebook support end-to-end
preprocessing pipelines in which clips are loaded, optionally resampled,
masked using `EchoROI`, cropped using the predicted sector geometry, and saved
for downstream modelling using a user-supplied model. This combination of software,
documented reference results, and adaptable workflows supports `EchoROI`'s use as
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
after masking, as illustrated in [Figure 1](#fig-pipeline). These traces are
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