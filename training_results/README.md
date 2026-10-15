# Reference-run records and provenance

These aggregate records describe the authors' documented reference
experiment. They do not include a checkpoint, source data, or a per-file
training/split manifest. This release includes neither pretrained weights nor
source datasets. See the [annotation-access policy](../docs/ANNOTATION_ACCESS.md)
and [retraining instructions](../docs/RETRAINING.md) for obtaining data and
conducting a new local run.

## What the saved records establish

| Evidence | Recorded value / interpretation |
|---|---|
| `dataset_summary.json` | 1,206 loaded pairs; 964 training / 242 validation; 256 x 256 inputs |
| Configuration in that summary | 50 configured and completed epochs; batch size 16; learning rate 0.0001; validation fraction 0.2 |
| `training_log.csv` | 50 rows, epochs 0-49; best validation Dice 0.9872645139694214 and IoU 0.9748716354370117, both at epoch 47 |
| `metrics.json` | Thresholded evaluation: Dice 0.9883648525867593; IoU 0.9769973444616058; accuracy 0.9907418243155992; sensitivity 0.9891411355779931; specificity 0.9917980637410437 |
| Notebook 01's recorded evaluation | Reports 1,206 loaded pairs and a 242-sample validation split, with the rounded values in `metrics.json` |
| Reference-run training workflow | Checkpoint selection monitors validation IoU; the split is source-stratified with `random_state=42` |

## Data and metric limitations

- **Collection inventory versus run membership:** the documented source
  collection contains 1,355 frame-mask pairs, while the saved run records
  1,206 loaded pairs. The complete original training membership is unresolved
  and the private subset may be unavailable, so retraining does not guarantee
  the identical checkpoint or reported metrics. The 1,355-pair inventory
  must not be treated as the verified membership of the reference run.
- **Run configuration:** the saved run records batch size 16; the current
  CLI, trainer, and Makefile default is 8. The retraining instructions pass
  16 explicitly to match the recorded configuration.
- **Checkpoint selection:** the documented reference workflow selects by
  validation IoU. Both logged metrics peak at epoch 47, but the CSV does not
  identify the exact checkpoint bytes used for evaluation.
- **Metric definitions:** the model's logged Dice/IoU are soft, batch-based
  Keras metrics. `calculate_metrics` thresholds predictions at `> 0.5`
  and pools pixels over the supplied validation arrays. Thus the
  `metrics.json` evaluation is not the log's best soft metric. A separate
  validation summary is also reported in the model card; available records
  do not establish that it used the same checkpoint or split.
- There was no separate external test set. The reported values are validation
  results from the documented reference experiment, not an external
  benchmark. Exact file membership, decoded-frame identities, split
  assignments, environment, random training state, and evaluation-checkpoint
  hash are not available.

## Files

| File | Description |
|---|---|
| `training_history.png` | Training and validation curves |
| `prediction_samples.png` | Reference validation prediction grid; requires distribution review |
| `metrics.json` | Saved thresholded validation evaluation |
| `training_log.csv` | Per-epoch metrics |
| `dataset_summary.json` | Saved configuration and loaded-pair counts |

Do not overwrite these records with a new run. Use a separate local results
directory, as shown in the retraining instructions. Image-bearing artifacts
are not automatically cleared for redistribution by this documentation.
