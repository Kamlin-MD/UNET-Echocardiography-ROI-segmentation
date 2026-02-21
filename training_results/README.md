# Training Results

This directory contains the training metrics and visualisations for the
EchoROI U-Net model. These are committed to the repository as evidence
for the JOSS paper.

## Contents (populated after training)

| File | Description |
|------|-------------|
| `training_history.png` | Loss and Dice curves (train + validation) |
| `prediction_samples.png` | Side-by-side predictions on held-out test images |
| `metrics.json` | Final evaluation metrics (Dice, IoU, accuracy, etc.) |
| `training_log.csv` | Per-epoch training metrics |
| `dataset_summary.json` | Breakdown of training data by source |

## Reproducing

```bash
# With the training data in data/images and data/masks:
echoroi train --image-dir data/images --mask-dir data/masks \
    --model-path models/echoroi.keras --epochs 50
```
