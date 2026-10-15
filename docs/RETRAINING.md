# Retraining with authorised data

This is a pathway for a **new, locally trained model**, not a guarantee of
reproducing the authors' reference checkpoint. This release includes neither
pretrained weights nor source datasets. Researchers must obtain source
datasets independently from their providers and may request corresponding
sector masks from the authors; annotation availability depends on applicable
agreements and institutional permissions.

The documented collection contains 1,355 frame-mask pairs, while saved
reference-run records report 1,206 loaded pairs and do not identify their
complete file membership. The complete original training membership is
unresolved and the private subset may be unavailable, so retraining does not
guarantee the identical checkpoint or reported metrics. See the
[reference-run records](../training_results/README.md).

## 1. Acquire source datasets independently

Follow the [per-provider access routes](ANNOTATION_ACCESS.md). Record versions,
download dates, original file hashes, agreements, and permitted uses in your
authorised workspace. The private institutional subset may not be obtainable.
Report which sources are included in your new experiment.

## 2. Request annotations

Contact **kamlin.ekambaram@gmail.com** with the source/version and authorisation
details. Source-data access does not guarantee annotation access. Do not start
from a public clone assuming masks, frame identities, or checkpoints are
included. If annotations cannot be supplied, create and validate local
scan-sector annotations under your institution's governance and document that
they are new annotations.

## 3. Reconstruct the exact annotated frames

Use the authorised package's manifest and per-frame decoding recipes.
[Matching conventions](../annotations/README.md) specify presentation-order
indexing, explicit 0/1 index base, native orientation, no pre-annotation
resize/crop, pixel coordinates, decoding versions, and original/decoded hashes.
Do not infer frame indices from arbitrary extracted filenames. Resolve any
legacy transform or ambiguous source mapping with the authors.

The local validator expects reconstructed PNG frames, **not** videos or DICOM
files in `source_relative_path`. It verifies their dimensions and hashes
against the agreed recipe, not the correctness of decoding the original clip.

## 4. Validate and prepare training pairs

```bash
pip install -e ".[dev]"
python scripts/validate_annotation_manifest.py authorised/manifest.csv \
  --source-root local/decoded_frames \
  --original-root local/provider_downloads \
  --annotation-root authorised
```

Resolve every reported failure; do not bypass hash or dimension mismatches.
Packages must contain annotations only, with no embedded LabelMe `imageData`.
If approved annotations are LabelMe polygons, rasterise `ROI sector` polygons
at native dimensions following the documented integer-coordinate convention.
`scripts/convert_labelme_to_masks.py` is a legacy local helper with fixed
paths; it does not decode media, validate manifests, or establish permissions.
Do not run it on an unreviewed collection.

Place the authorised decoded images in local `data/images/` and binary
masks in local `data/masks/`, using matching, unique basenames and PNG format.
Keep a private mapping from those basenames to manifest rows. Verify a
one-to-one image/mask set with identical native dimensions before training;
the existing loader intersects basenames and can warn/skip unreadable files,
so a successful training command alone is not evidence of a complete pairing.
The source-stratification helper infers source names from filename prefixes
(`dataset_label_from_filename`); verify labels for every prepared filename.
Arbitrary renaming can silently place files in the wrong stratum. Preserve
the expected source prefixes or use a separately reviewed training workflow
with explicit source labels.

Training preprocessing reads grayscale images with OpenCV, scales them by
`1/255`, and preserves aspect ratio at 256 x 256 with zero padding. For native
height `h`, width `w`, scale is `min(256/h, 256/w)`; resized dimensions are
`int(h*scale), int(w*scale)` with top/left padding equal to half the remaining
space rounded down. The current implementation uses `INTER_AREA` for both
images and masks, then thresholds masks at `>127`. This transform is applied
by the loader, not before native-frame annotation matching. Keep input and
mask coordinates aligned and inspect edge effects.

## 5. Train a new model

```bash
mkdir -p models
echoroi train --image-dir data/images --mask-dir data/masks \
  --model-path models/my_model.keras --epochs 50 --batch-size 16 \
  --learning-rate 1e-4 --validation-split 0.2 \
  --results-dir results/retraining-run
```

Batch size 16 reflects the saved run configuration, while the runtime default
is 8. Current training uses a source-stratified 80/20 split with
`random_state=42`, composite BCE + Dice + TV loss, and best `val_iou_score`
checkpoint selection. It does not fix all initialisation/device randomness.
Record file membership, actual loaded counts, split assignments, environment,
hardware, seeds, and checkpoint hashes for your run. Use a separate results
directory for the new run.

## 6. Evaluate the saved checkpoint

```bash
echoroi evaluate --model-path models/my_model.keras \
  --image-dir local/validation/images --mask-dir local/validation/masks \
  --output results/retraining-evaluation
```

The validation directories must contain **only** your recorded held-out
validation pairs. The CLI does not recover the training split for you; do not
evaluate the entire training collection and label it held-out validation.
Export the actual split using the Python training workflow and save its
membership privately when running training; the CLI does not persist it.
Notebook 01 illustrates split recreation but depends on unchanged filenames,
file ordering, source labels, library versions, and load success. Compare
thresholded pooled evaluation metrics separately from logged soft batch metrics.
Report new results as your own run, not as a reproduction of the published numbers.

## 7. Optional ONNX export and inference

Install the optional export dependencies in a compatible local environment
(`pip install -e ".[export]"`; ONNX Runtime is also needed for parity checks).
Notebook 02 and `scripts/convert_to_onnx.py` demonstrate conversion and parity
checks using a **local** `models/echoroi_unified.keras` path. Set the notebook's
paths to your own model, or place a lawfully obtained local checkpoint at the
legacy path before using the fixed-path script. Check export numerical parity
on synthetic and authorised validation inputs. Export does not change data
licensing or authorise model redistribution.

```bash
make predict MODEL_PATH=path/to/model.keras INPUT=local/frame.png
make evaluate MODEL_PATH=path/to/model.keras \
  IMAGE_DIR=local/validation/images MASK_DIR=local/validation/masks
```

Notebooks 03/04 also require local checkpoints and independently obtained input
images/DICOM data. Clear all outputs before sharing notebooks; review saved
plots separately. ROI masking is not complete de-identification.
