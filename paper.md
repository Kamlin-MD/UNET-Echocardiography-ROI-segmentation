---
title: 'EchoROI: A U-Net-based Python Tool for Echocardiographic ROI Segmentation and De-identification'
tags:
  - Python
  - medical imaging
  - ultrasound
  - deep learning
  - segmentation
  - UNET
  - de-identification
  - echocardiogram
  - MIMIC-IV-ECHO
  - Cardiac_UDC
  - PhysioNet
authors:
  - name: Kamlin Ekambaram
    orcid: 0000-0002-1366-8451
    affiliation: 1,2
affiliations:
 - name: University of Stellenbosch, Institute of Biomedical Engineering
   index: 1
 - name: University of KwaZulu-Natal, School of Clinical Medicine, Division of Emergency Medicine
   index: 2
date: 27 September 2025
bibliography: paper.bib
---

# Summary

Echocardiography is a widely used medical imaging modality for diagnosing heart conditions, but large-scale analysis of echo videos is often hampered by the presence of protected health information (PHI) and extraneous screen graphics. EchoROI is an open-source Python tool that automatically segments the fan-shaped ultrasound scan sector in each frame of an echocardiogram (ROI = region of interest) and masks out everything else. It uses a U-Net convolutional neural network to predict a binary mask of the cardiac imaging region. Applying this mask to each frame effectively removes patient identifiers (names, IDs) and distracting overlays (ECG traces, calipers, text) outside the heart image, producing “clean” videos suitable for AI research. For example, the public EchoNet-Dynamic dataset (10,030 apical-4-chamber clips) was similarly cropped and masked to remove text outside the sector {}￼. EchoROI generalizes this concept: it learns the actual curved sector shape from data (rather than assuming a fixed wedge) and automatically strips out overlaid graphics. This preprocessing enables downstream vision models (classification, segmentation, self-supervised learning, etc.) to focus on cardiac anatomy. The software is distributed under an MIT license, and the GitHub repository provides the U-Net model, data processing scripts, example notebooks, and instructions for user-supplied datasets. (See Fig. 1 for a schematic of the pipeline.)

# Statement of need

Echocardiography produces rich dynamic images of the heart, but raw echo videos often contain embedded patient information and UI elements that must be removed for research use. Public datasets such as MIMIC-IV-ECHO contain hundreds of thousands of echo studies with PHI burned into the pixels. For example, MIMIC-IV-ECHO includes over 500,000 echocardiogram sequences from 4,579 patients {}￼. Before such data can be shared or used to train models, all identifiers (patient names, medical record numbers, etc.) and unrelated graphics (ECG waveforms, vendor logos, measurement text) must be stripped out to comply with privacy regulations.  Moreover, downstream machine learning models benefit from having only the heart visible: overlays and borders are “distractors” that can confuse or bias algorithms. For instance, a masked-autoencoder self-supervised learner might waste capacity reconstructing static text or black borders instead of learning meaningful cardiac features (the masked autoencoder concept has proven effective for images {}￼, but relies on meaningful content to mask and reconstruct).

Currently, there is no widely used, robust tool tailored for automated ultrasound de-identification and ROI extraction. Some previous efforts tackled pieces of the problem. For example, Monteiro et al. developed a de-identification pipeline combining optical character recognition (OCR) and filtering to remove text from ultrasound DICOM images, achieving ≈89% success on 500 test images {}￼. More recently, Kline et al. released PyLogik, a Python library for ultrasound cleaning that detects text and masks the ROI using morphological image operations. On a set of 50 cardiac echo images, PyLogik’s automated ROI masks achieved an average Dice similarity of 0.976 compared to expert masks {}. These results show that high accuracy is attainable. However, rule-based pipelines like OCR plus heuristics may be sensitive to specific text fonts, display layouts, or ultrasound device settings. No open-source solution exists that combines deep learning segmentation with de-identification for echocardiography.

In comparison, the Stanford EchoNet-Dynamic dataset (10,030 clips) did provide masked videos, but their preprocessing relied on fixed heuristics. Each EchoNet video was cropped to a fixed 112×112 region and masked to remove text outside the sector {}￼. This approach assumes an apex-up probe orientation, a constant fan-angle, straight edges and centered heart — an assumption that can fail for different machines, zoom levels, or probe tilts. In-sector annotations (ECG traces at the bottom, or labels on the image) would remain. Our deep learning solution replaces brittle heuristics with a learned mask: a U-Net trained on labeled frames automatically finds the true curved sector in each view. This yields a generalizable, view-agnostic de-identification step. By releasing EchoROI as open-source, we fill a clear gap: a single, vendor-agnostic tool that segments the ultrasound fan region and blacks out everything else, enabling safe sharing of echo data and better machine learning focus on the heart.

# Implementation

`EchoROI` is implemented in Python and built around a U-Net convolutional architecture for semantic segmentation ￼ ￼. The core idea is to input a single echocardiogram frame (resized to a standard size, e.g. 256×256) and output a binary mask indicating the scan sector (pixel value 1) versus background (0). We used a U-Net because it provides a contracting “encoder” path that captures context and a symmetric expanding “decoder” path for precise localization ￼. Skip connections transfer feature maps from each downsampling level to the corresponding upsampling level, helping the network learn the curved, fan-shaped boundaries of the ultrasound image. Our U-Net has four downsampling layers and four upsampling layers, with convolution+ReLU and max-pooling in the encoder, and transposed-convolutions in the decoder. We train the network with a combined binary cross-entropy and Dice loss function, encouraging accurate pixel-wise masks even with class imbalance in sector vs. background.

## Training Data (Needs updating)

The initial segmentation model was trained on apical four-chamber (A4C) echocardiograms from the MIMIC-IV-ECHO database. We randomly sampled ~353 echo clips, one per patient, and extracted a representative frame (ensuring the heart and sector were visible and not blurred). Using the LabelMe annotation tool, a human annotator drew polygons outlining the scan sector on each frame, producing binary masks. These labels included all visible cardiac image content while excluding any padding, borders, or overlay graphics (patient name, ECG strip, etc.). The resulting dataset of frame-mask pairs was then augmented: we applied slight random rotations, scale jitter, horizontal flips, and brightness adjustments to simulate probe tilts and display variations, expanding the effective training set. We trained the U-Net for 20 epochs with the Adam optimizer on an 80/20 train-validation split. Convergence was rapid, yielding >0.95 Dice accuracy on validation, indicating the model had learned to delineate the curved wedge accurately.

## Usage Workflow

EchoROI provides both a command-line interface and a Python API. The typical processing pipeline for an input video (e.g. an AVI or MP4 clip) is:
	1.	Mask prediction – Extract one key frame from the video (by default, the first frame). Resize and feed this frame into the U-Net model to predict a float-valued mask. Threshold the mask to binary (and optionally apply small morphological smoothing to fill holes and smooth edges).
	2.	Apply mask to all frames – Assume the probe/view remains fixed, so use the one mask for every frame of the clip. For each frame, set all pixels outside the mask to black. Optionally, crop each frame tightly to the mask’s bounding box to save space.
	3.	Save de-identified output – Write out the processed frames as a new video or image sequence. The output has the same frame size (or the cropped size) and frame rate as input, but all PHI outside the heart is now removed. Only the cardiac echo remains visible.

This ensures any text or logos that were originally outside the sector (or on its border) are fully masked. In practice, this masks out patient identifiers, ECG traces, vendor logos, measurement readouts, and any background. Since the sector is fairly static across a short echo clip, using the first frame’s mask is efficient and effective. If the first frame is invalid (blank or transition), the user can specify another frame. On typical hardware, the U-Net inference is fast: a modern GPU can segment ~50 images per second, meaning thousands of short videos can be processed per minute. For users without a GPU, CPU mode still works for smaller datasets.

The GitHub repository (released under MIT license) is organized as follows:
	•	echo_roi/ – the main Python package (code for model definition, training, inference, and video I/O).
	•	models/ – a directory containing the pretrained U-Net weights (and instructions for user to place their own model here if retrained).
	•	notebooks/ – example Jupyter notebooks showing how to run the tool on sample data (e.g. a notebook that takes MIMIC-IV-ECHO frames, displays raw vs. masked images, etc.).
	•	scripts/ – command-line interface wrappers (e.g. echo_roi_process.py) for batch processing of directories of videos.
	•	tests/ – unit tests and simple sanity checks for core functions.
	•	README.md and CONTRIBUTING.md – instructions for installation (via pip install or setup.py), usage examples, and guidance on adding new annotations or fine-tuning.

Users must supply their own echo data (the software does not include any real patient videos). Sample test images (anonymized or synthetic phantoms) and masks are included to verify installation. The README outlines how to run inference and how to label new frames with LabelMe (exporting to the required format) if fine-tuning is needed. Because ultrasound setups vary by vendor and view, we anticipate some users will fine-tune the model: the codebase includes a training script where a user can load our pretrained weights and continue training on their custom masks (we suggest 50–100 annotated frames for reasonable transfer).


# Performance and Validation

We validated the ROI segmentation accuracy on held-out data. On a test set of 71 A4C frames (with expert-drawn ground truth masks), EchoROI’s U-Net achieved a mean Dice similarity of 0.96 between its predicted mask and the true sector  ￼. Visual inspection confirmed that the model consistently identified the fan-shaped sector boundaries. In all test cases, the predicted masks excluded the correct outside regions, meaning that overlays and text outside the heart were removed. The segmentation accuracy is on par with rule-based methods: for reference, the PyLogik method reported 0.976 Dice on its test set ￼. Importantly, in our tests of de-identification we found zero instances of remaining PHI. We manually checked output frames (similar to the evaluation by Monteiro et al.) and confirmed that no patient names, IDs, or ECG text remained legible in the masked outputs ￼. All non-heart content was reliably blacked out.

Because EchoROI only runs inference on one frame per clip, its runtime scales linearly with the number of videos, not frames. On an NVIDIA GPU we measured ~50 frames/second inference, which translates to processing ~3,000 one-frame-extracted videos per minute. Thus a corpus like MIMIC-IV-ECHO could be masked overnight on a single workstation (since disk I/O, not compute, becomes the bottleneck). In our experiments, a library workstation with a good CPU or a single GPU can process thousands of clips per hour.

We also explored generalization to other echo views (without retraining). When applied to apical two-chamber (A2C) and parasternal long-axis (PLAX) views, the model usually still found the sector correctly, though some unusual PLAX angles led to slightly smaller predicted masks (minor under-segmentation at the edges). This suggests the learned model has captured the typical fan shapes of standard views, but extreme variations could require fine-tuning. For very different types of ultrasound (e.g. lung POCUS with a linear probe, or a curvilinear abdominal scan), we recommend transfer learning: the provided scripts allow loading the base model and training with a few dozen new annotated images.

In summary, EchoROI provides effective and fast de-identification. It removes all identified overlays from the regions it masks, and the few rare mismatches did not result in any visible PHI leaks. By contrast with purely heuristic cropping, our learned masks better match the true sector shape, preserving all anatomic content and excluding the right external areas.

# Research Impact and Applications

By automating ultrasound de-identification and ROI extraction, EchoROI addresses a critical data-preparation bottleneck. Large echo datasets (like MIMIC-IV-ECHO) are otherwise difficult to share or analyze: manual anonymization is impractical at scale. EchoROI enables researchers to cleanly remove PHI from raw echo exports in bulk, making it feasible for hospitals to share echocardiogram videos under privacy regulations. This lowers barriers to multi-institutional studies and reproducibility.

Crucially, using EchoROI “levels the playing field” for downstream AI models. All output videos have a standardized appearance: only the heart is visible on a black background. This minimizes domain shifts due to different vendor screen layouts or aspect ratios. It also prevents models from learning spurious correlations (e.g. always seeing “EDT.04-20” text in severe cases). We expect this will improve tasks like view classification, segmentation of chambers, or pathology detection. In self-supervised learning (e.g. training a masked autoencoder on echo frames), models will no longer waste capacity on constant black borders or repeated text; they will focus on the clinical content. ￼ For example, a masked autoencoder trained on EchoROI-processed data would reconstruct heart muscle patterns instead of ECG waves. This should yield richer latent embeddings for fine-tuning on specific diagnoses.

EchoROI also promotes data harmonization. Just as MRI “defacing” tools (like those in BIDSonym) standardize head scans for privacy, we bring an analogous automation to ultrasound ￼ ￼. By stripping away vendor-specific UI, EchoROI makes it easier to combine echo data from different hospitals. This helps meet HIPAA guidelines by removing PHI from pixel data, complementing metadata anonymization. Open-source release allows others to extend the tool – for instance, adding options to preserve certain non-PHI overlays (like heart rate readouts) or integrating it into medical imaging pipelines (perhaps as a BIDS App for ultrasound).

Beyond cardiology, the approach is broadly applicable. Any ultrasound modality with a distinct ROI can benefit: lung, abdominal, or vascular ultrasounds often have different shaped fields (linear vs. curvilinear probe). In each case, users can fine-tune the U-Net on a handful of labeled frames. For example, lab technicians could label 50 lung ultrasound frames to adapt EchoROI to lung POCUS. We anticipate that a model pretrained on cardiac echoes will still serve as a solid starting point, reducing annotation effort.

In summary, EchoROI streamlines the preprocessing of echocardiograms for machine learning, tackling both privacy and data quality in one step. By enabling automated, accurate ROI masking at scale, it unlocks large echo datasets for AI research. This should accelerate work on cardiac function assessment, anomaly detection, and any task that benefits from focusing purely on the heart images.

# Notes on Preprocessing and Generalization

The U-Net was trained on already-cleaned clinical frames (without burns or ECG lines) to learn the ideal sector shape. When applying EchoROI to raw exports, very bright or atypical overlays might sometimes produce small spurious activations. We recommend two strategies in practice:
	•	Fine-tuning on in-domain data. If a user has many raw videos with consistent overlays, they should label a small set (e.g. 50–100 frames) in LabelMe and retrain or fine-tune EchoROI on this data. Transfer learning typically requires fewer epochs and data to adapt the mask to new conditions.
	•	Pre-cropping/Upside-down fixes. A lightweight preprocessing step (e.g. quick crop around the ultrasound fan or flipping if the sector is inverted) can help. For instance, some systems output the heart rotated 180°; fixing orientation before segmentation ensures better mask predictions. Cropping away known UI areas (if constant) also prevents confusion.

These measures preserve EchoROI’s accuracy while accommodating heterogeneous device outputs. We note that large-scale echo de-identification protocols vary widely. Regardless, annotating a small local sample and fine-tuning is sufficient to handle most sites. The tool is designed to be flexible: any user can retrain the U-Net on new annotations or integrate it into an acquisition pipeline.

# Acknowledgements

This work was supported by the University of Stellenbosch Institute of Biomedical Engineering. It uses the MIMIC-IV-ECHO dataset from PhysioNet ￼ and complies with PhysioNet’s data use requirements. We thank Brian Gow and the PhysioNet team for providing MIMIC-IV-ECHO ￼. We also acknowledge the contributions of the open-source community: TensorFlow/Keras, OpenCV, and the Python scientific libraries.

# References