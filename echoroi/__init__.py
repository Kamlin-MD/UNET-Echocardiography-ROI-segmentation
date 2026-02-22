"""EchoROI – U-Net echocardiographic ROI segmentation & de-identification.

This package provides a complete pipeline for training, evaluating, and
deploying a U-Net model that segments the ultrasound scan-sector in
echocardiogram frames, enabling automated de-identification and
standardised preprocessing for downstream AI research.
"""

__version__ = "0.1.0"
__author__ = "Kamlin Ekambaram"
__email__ = "kamlin.ekambaram@gmail.com"

from .inference import UNetPredictor
from .model import UNetModel, dice_coefficient, iou_score
from .preprocessing import UltrasoundPreprocessor
from .training import UNetTrainer

__all__ = [
    "UNetModel",
    "UltrasoundPreprocessor",
    "UNetTrainer",
    "UNetPredictor",
    "dice_coefficient",
    "iou_score",
]
