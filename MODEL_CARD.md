# Model Card: EchoROI U-Net

## Model Details

| Field | Value |
|:------|:------|
| **Model name** | `echoroi_unified` |
| **Architecture** | U-Net (4 encoder + 4 decoder blocks, 1024-channel bottleneck) |
| **Parameters** | ~31 million |
| **Input** | 256×256×1 grayscale image (float32, pixel values in [0, 1]) |
| **Output** | 256×256×1 binary mask (sigmoid activation) |
| **Framework** | TensorFlow/Keras 2.x; ONNX export also provided |
| **License** | MIT |
| **Version** | 1.0 |
| **Release date** | March 2026 |
| **Repository** | [github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation) |

## Intended Use

### Primary use cases

1. **Dataset preprocessing** — Segment the ultrasound scan sector (ROI) in
   echocardiogram frames and mask out non-ROI content (PHI, vendor overlays,
   ECG traces, calipers, logos) to produce clean inputs for downstream ML
   pipelines.
2. **De-identification aid** — Black out regions outside the scan sector to
   reduce the risk of exposing protected health information burned into the
   pixel data. **This is not a guarantee of complete PHI removal.**
3. **Education** — Generate de-identified stills or clips for teaching,
   FOAMed, and collaboration, subject to institutional governance.

### Intended users

- Medical imaging researchers preprocessing echocardiography datasets.
- Machine learning practitioners building cardiac analysis pipelines.
- Clinical educators sharing de-identified echo examples.

### Out-of-scope uses

- **Clinical diagnosis** — This model is not intended for, and has not been
  validated for, any diagnostic or clinical decision-making purpose.
- **Guaranteed de-identification** — The model should not be used as the sole
  means of PHI removal. Human-in-the-loop review is required.
- **Non-echocardiography ultrasound** — The model was trained exclusively on
  cardiac ultrasound and is not expected to generalise to other ultrasound
  modalities (e.g., obstetric, vascular, musculoskeletal).

## Training Data

The model was trained on 1,355 manually annotated echocardiogram frame/mask
pairs (80/20 train/validation split, fixed random seed) drawn from:

| Dataset | Frames | Access |
|:--------|-------:|:-------|
| MIMIC-IV-ECHO | 403 | PhysioNet (credentialed) |
| EchoNet-Dynamic | 145 | Stanford (open) |
| EchoNet-Paediatric | 263 | Stanford |
| CACTUS (A4C subset) | 38 | Open access |
| EchoCP | 60 | Kaggle |
| Private dataset (consented) | 50 | Institutional (Mindray/Samsung) |
| CardiacUDC | 247 | Kaggle |
| HMC-QU | 149 | By request to authors |
| **Total** | **1,355** | |

### Annotation protocol

Ground-truth masks were created in LabelMe by outlining the scan-sector
boundary as polygons. Sector ROIs were annotated with a virtual apex
(triangular sector) even for curved-probe images. Only one frame per video
sequence was included.

### Data demographics and diversity

- **Modality**: 2D transthoracic echocardiography (sector/phased-array probes).
- **Vendors**: GE, Philips, Siemens, Mindray, Samsung (across sources).
- **Views**: Predominantly apical four-chamber (A4C), parasternal long-axis
  (PLAX), and other standard views.
- **Patient populations**: Adult and paediatric; datasets span multiple
  institutions and geographies.
- **Not represented**: Linear-probe (rectangular) layouts, 3D echo, TEE,
  handheld/POCUS-specific devices, non-cardiac ultrasound.

## Performance

### Validation split metrics (20% of 1,355 frames)

| Metric | Value |
|:-------|------:|
| Dice coefficient | 0.9880 |
| IoU (Jaccard) | 0.9763 |
| Pixel accuracy | 0.9906 |
| Sensitivity | 0.9894 |
| Specificity | 0.9914 |

### Training configuration

| Parameter | Value |
|:----------|:------|
| Optimiser | Adam |
| Learning rate | 1×10⁻⁴ (reduce-on-plateau: factor 0.5, patience 5) |
| Loss | Binary cross-entropy |
| Batch size | 16 |
| Epochs | 50 |
| Hardware | Apple Mac mini, M2 Pro (CPU/GPU) |
| Inference speed | ~25 ms per 256×256 frame (M2 Pro) |

### Caveats

- No separate held-out test set was used. Validation metrics are from the same
  split used for model selection (best `val_dice_coefficient` checkpoint) and
  may modestly overestimate generalisation performance.
- Performance was not stratified by dataset source, vendor, or view type.

## Known Failure Modes

Users should treat the following as high-risk scenarios where masking quality
may degrade:

1. **Atypical display layouts** — Non-standard screen arrangements, split-screen
   views, or colour Doppler overlays that significantly alter the visual
   appearance of the scan sector boundary.
2. **Extreme zoom or depth** — Very deep or very shallow acquisitions that
   produce sector shapes substantially different from training examples.
3. **Low contrast** — Frames where the sector boundary blends with the
   background (e.g., very bright or very dark overall images).
4. **Handheld/POCUS devices** — Acquisitions from handheld ultrasound devices
   with non-standard display formats were not represented in training.
5. **Linear-probe layouts** — Rectangular ultrasound fields were not included
   in training; performance is not expected to be reliable.
6. **PHI inside the ROI** — Text or identifiers rendered within the scan sector
   itself will not be masked, as the model preserves all content within the
   predicted ROI.
7. **Curved-probe boundary error** — For curved-probe acquisitions, the
   virtual-apex (triangular) annotation convention may introduce small boundary
   errors near the probe face.

### Indicators of potential failure

- **Low mask confidence**: Peak predicted probability below 0.5 in the output
  mask suggests the model is uncertain about the sector location.
- **Atypical aspect ratio**: Input frames with aspect ratios far from 1:1 after
  preprocessing may produce distorted masks.
- **Implausible ROI geometry**: Predicted masks with multiple disconnected
  regions, very small area, or non-convex shapes inconsistent with a sector
  probe may indicate a failure case.

## Ethical Considerations

- The model processes medical images that may contain protected health
  information. Users are responsible for compliance with local data governance
  regulations (e.g., HIPAA, POPIA, GDPR).
- De-identification via ROI masking is a risk-reduction measure, not a
  guarantee. Human review is required before sharing derived data externally.
- The training data spans multiple institutions but is not demographically
  balanced or stratified. Performance disparities across patient populations,
  device manufacturers, or clinical settings have not been characterised.
- The model should not be used for clinical decision-making.

## Citation

If you use EchoROI in your research, please cite:

```bibtex
@article{ekambaram2026echoroi,
  title   = {{EchoROI}: A {U-Net}-based Python Tool for Echocardiographic
             {ROI} Segmentation and De-identification},
  author  = {Ekambaram, Kamlin and Arnab, Anurag and Herbst, Philip and
             Theart, Rensu},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the
[GitHub repository](https://github.com/Kamlin-MD/UNET-Echocardiography-ROI-segmentation).
