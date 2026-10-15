# Reference-run records and provenance

These saved records were updated together in commit `67d6ecc` (19 March
2026). They describe the authors' reference experiment, not a model supplied
by this release. No new training or evaluation has been performed for this
documentation change. See the [full retraining instructions](../docs/RETRAINING.md).

## What the saved records establish

| Evidence | Recorded value / interpretation |
|---|---|
| `dataset_summary.json` | 1,206 loaded pairs; 964 training / 242 validation; 256 x 256 inputs |
| Configuration in that summary | 50 configured and completed epochs; batch size 16; learning rate 0.0001; validation fraction 0.2 |
| `training_log.csv` | 50 rows, epochs 0-49; best validation Dice 0.9872645139694214 and IoU 0.9748716354370117, both at epoch 47 |
| `metrics.json` | Thresholded evaluation: Dice 0.9883648525867593; IoU 0.9769973444616058; accuracy 0.9907418243155992; sensitivity 0.9891411355779931; specificity 0.9917980637410437 |
| Notebook 01's recorded evaluation | Reports 1,206 loaded pairs and a 242-sample validation split, with the rounded values in `metrics.json` |
| `echoroi/training.py` at `67d6ecc` | `ModelCheckpoint` and `EarlyStopping` monitor `val_iou_score`, mode `max`; split is source-stratified with `random_state=42` |

## Reconciliation and limits

- **1,355 versus 1,206:** the source inventory in the manuscript, model card,
  and notebook introduction sums to 1,355 annotated frames across eight
  sources. The saved run summary and notebook output instead record 1,206
  loaded pairs. There is no saved per-file membership/split manifest to
  explain the 149-pair difference. Its equality to one inventory subtotal
  is not evidence of that source's exclusion. The inventory must not be
  presented as this run's verified training membership. History also shows
  a count-only edit to the summary at `1a36462` (4 March 2026), changing
  1,206 / 964 / 242 to 1,355 / 1,084 / 271. The later `67d6ecc` records
  restore 1,206 / 964 / 242 with new logs and metrics. The intermediate
  summary counts are not independent proof of a 1,355-pair training run.
- **Batch size 8 versus 16:** 8 is the CLI, trainer, and Makefile default;
  the saved run summary explicitly records 16. The retraining example
  passes 16 explicitly without altering runtime defaults.
- **Dice versus IoU selection:** the saved-run source and notebook summary
  select by validation IoU, not Dice. Both logged metrics peak at epoch 47,
  but that coincidence does not change the selection criterion. The CSV
  alone does not identify the exact checkpoint bytes used for evaluation.
  The earlier source at `5e10245` (21 February 2026) selected by validation
  Dice; `61e447b` (19 March) changed selection to IoU and introduced
  source-stratified splitting. These are different experiment configurations.
- **Different metrics:** the model's logged Dice/IoU are soft, batch-based
  Keras metrics. `calculate_metrics` thresholds predictions at `> 0.5`
  and pools pixels over the supplied validation arrays. Thus the
  `metrics.json` evaluation is not the log's best soft metric. The model
  card's older table (Dice 0.9880, IoU 0.9763, accuracy 0.9906,
  sensitivity 0.9894, specificity 0.9914) matches the rounded
  `metrics.json` at `5e10245`, whose summary recorded 1,206 pairs,
  batch size 16, and 964 / 242 splitting. It is preserved as that earlier
  report, not silently relabelled as the March IoU-selected run. Exact
  per-file membership and evaluation-checkpoint hashes remain unavailable.

The checkpoint, source data, exact decoded-frame list, split assignments,
full environment, and random training state are not distributed. These
records support transparent inspection of reported results, not a claim
of exact checkpoint reproduction or an external test benchmark.

## Files

| File | Description |
|---|---|
| `training_history.png` | Training and validation curves |
| `prediction_samples.png` | Historical reference validation prediction grid; requires distribution review |
| `metrics.json` | Saved thresholded validation evaluation |
| `training_log.csv` | Per-epoch metrics |
| `dataset_summary.json` | Saved configuration and loaded-pair counts |

Do not overwrite these records with a new run. Use a separate local results
directory, as shown in the retraining instructions. Image-bearing artifacts
are not automatically cleared for redistribution by this documentation.
