"""Training utilities for the EchoROI U-Net model."""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .model import UNetModel
from .preprocessing import UltrasoundPreprocessor


def dataset_label_from_filename(filename: str) -> str:
    """Infer the source dataset from an image filename.

    Used to build a stratification vector so that train/validation
    splits contain proportional representation from every dataset.
    """
    name = os.path.basename(filename)
    if name.startswith("0X1A"):
        return "echonet_dynamic"
    if name.startswith("0X1B") or name.startswith("CARD") or name.startswith("CR"):
        return "echonet_peds"
    if name.startswith("Site_"):
        return "cardiac_udc"
    if name.startswith("EchoCP"):
        return "echocp"
    if name.startswith("label_all_frame"):
        return "hmc_qu"
    if name.startswith("Temp_"):
        return "private"
    if "_D1" in name[:8] and "_frame" in name:
        return "cactus"
    # Remaining numeric-prefix files are MIMIC-IV-ECHO
    return "mimic"


class UNetTrainer:
    """Training class for U-Net model."""

    def __init__(self,
                 img_size: tuple[int, int] = (256, 256),
                 learning_rate: float = 1e-4,
                 batch_size: int = 8,
                 epochs: int = 50,
                 validation_split: float = 0.2):
        """Initialize trainer.

        Args:
            img_size: Input image size
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
        """
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

        # Initialize components
        self.preprocessor = UltrasoundPreprocessor(img_size)
        self.model_builder = UNetModel(input_shape=(*img_size, 1))
        self.model = None
        self.history = None

    def prepare_data(self, image_dir: str, mask_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data.

        The split is stratified by source dataset (inferred from
        filenames) so that the validation set contains proportional
        representation from each contributing dataset.

        Args:
            image_dir: Directory containing training images
            mask_dir: Directory containing training masks

        Returns:
            Tuple of (X_train, X_val, Y_train, Y_val)
        """
        print("Loading and preprocessing data...")
        X, Y = self.preprocessor.load_dataset(image_dir, mask_dir)

        # Build stratification labels from filenames
        pairs = self.preprocessor.validate_data_paths(image_dir, mask_dir)
        labels = [dataset_label_from_filename(img_path) for img_path, _ in pairs]

        # Report per-dataset counts
        from collections import Counter
        counts = Counter(labels)
        print("\\nDataset composition:")
        for ds, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {ds:20s}: {n}")

        # Stratified split
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=self.validation_split, random_state=42,
            stratify=labels,
        )

        print("\\nStratified data split:")
        print(f"  Training:   {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")

        return X_train, X_val, Y_train, Y_val

    def create_callbacks(self, model_save_path: str, log_dir: str | None = None) -> list[tf.keras.callbacks.Callback]:
        """Create training callbacks.

        Args:
            model_save_path: Path to save the best model
            log_dir: Optional directory for CSVLogger output

        Returns:
            List of Keras callbacks
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_iou_score',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_iou_score',
                patience=50,
                verbose=1,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            csv_path = os.path.join(log_dir, "training_log.csv")
            callbacks.append(
                tf.keras.callbacks.CSVLogger(csv_path, separator=',', append=False)
            )

        return callbacks

    def train(self,
              image_dir: str,
              mask_dir: str,
              model_save_path: str = "models/unet_model.keras",
              results_dir: str | None = None) -> tf.keras.callbacks.History:
        """Train the U-Net model.

        Args:
            image_dir: Directory containing training images
            mask_dir: Directory containing training masks
            model_save_path: Path to save the trained model
            results_dir: Optional directory to save training artifacts

        Returns:
            Training history
        """
        # Prepare data
        X_train, X_val, Y_train, Y_val = self.prepare_data(image_dir, mask_dir)
        self._X_val = X_val
        self._Y_val = Y_val
        self._n_total = X_train.shape[0] + X_val.shape[0]

        # Build and compile model
        print("\\nBuilding and compiling model...")
        self.model = self.model_builder.compile_model(self.learning_rate)

        # Print model summary
        print("\\nModel Summary:")
        self.model.summary()

        # Create callbacks
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        callbacks = self.create_callbacks(model_save_path, log_dir=results_dir)

        # Train model
        print(f"\\nStarting training for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")

        self.history = self.model.fit(
            X_train, Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
            verbose=1
        )

        print("\\nTraining completed!")
        return self.history

    def plot_training_history(self, save_path: str | None = None) -> None:
        """Plot training history.

        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot training & validation loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot training & validation accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot IoU (Jaccard) score — primary metric
        iou_key = next((k for k in self.history.history
                        if 'iou' in k and not k.startswith('val')), None)
        if iou_key:
            axes[1, 0].plot(self.history.history[iou_key], label='Training IoU')
            axes[1, 0].plot(self.history.history[f'val_{iou_key}'], label='Validation IoU')
            axes[1, 0].set_title('IoU (Jaccard Index)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot Dice coefficient
        dice_key = next((k for k in self.history.history
                         if 'dice' in k and not k.startswith('val')), None)
        if dice_key:
            axes[1, 1].plot(self.history.history[dice_key], label='Training Dice')
            axes[1, 1].plot(self.history.history[f'val_{dice_key}'], label='Validation Dice')
            axes[1, 1].set_title('Dice Coefficient')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Dice Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")

        plt.show()

    def get_model(self) -> tf.keras.Model:
        """Get the trained model.

        Returns:
            The trained Keras model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model

    def save_prediction_samples(self, save_path: str, n_samples: int = 6) -> None:
        """Save a grid of prediction samples (input / ground truth / prediction).

        Args:
            save_path: Path to save the figure
            n_samples: Number of samples to visualise
        """
        if self.model is None or not hasattr(self, '_X_val'):
            raise ValueError("Train the model first.")

        n_samples = min(n_samples, self._X_val.shape[0])
        indices = np.random.RandomState(42).choice(
            self._X_val.shape[0], n_samples, replace=False
        )

        preds = self.model.predict(self._X_val[indices], verbose=0)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i, idx in enumerate(range(n_samples)):
            img = self._X_val[indices[idx]].squeeze()
            gt = self._Y_val[indices[idx]].squeeze()
            pred = (preds[idx].squeeze() > 0.5).astype(np.float32)

            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Prediction samples saved to: {save_path}")

    def save_results(self, results_dir: str) -> dict:
        """Save all training artifacts to a directory.

        Saves: training_history.png, prediction_samples.png,
               metrics.json, dataset_summary.json

        Args:
            results_dir: Directory to write results into

        Returns:
            Dictionary of final validation metrics
        """
        import json

        os.makedirs(results_dir, exist_ok=True)

        # 1. Training history plot
        self.plot_training_history(
            os.path.join(results_dir, "training_history.png")
        )

        # 2. Prediction sample grid
        self.save_prediction_samples(
            os.path.join(results_dir, "prediction_samples.png")
        )

        # 3. Validation metrics
        metrics = calculate_metrics(self._Y_val,
                                    self.model.predict(self._X_val, verbose=0))
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

        # 4. Dataset summary
        summary = {
            "total_samples": int(self._n_total),
            "training_samples": int(self._n_total - self._X_val.shape[0]),
            "validation_samples": int(self._X_val.shape[0]),
            "image_size": list(self.img_size),
            "epochs_configured": self.epochs,
            "epochs_completed": len(self.history.history['loss']),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "best_val_loss": float(min(self.history.history['val_loss'])),
            "best_val_dice": float(max(
                self.history.history.get('val_dice_coefficient', [0]))),
            "best_val_iou": float(max(
                self.history.history.get('val_iou_score', [0]))),
        }
        summary_path = os.path.join(results_dir, "dataset_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Dataset summary saved to: {summary_path}")

        return metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate segmentation metrics.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary of metrics
    """
    # Binarize predictions
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_bin = (y_true > 0.5).astype(np.float32)

    # Flatten arrays
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()

    # Calculate metrics
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection

    # Dice coefficient
    dice = (2.0 * intersection + 1e-6) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-6)

    # IoU (Jaccard index)
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Accuracy
    accuracy = np.mean(y_true_flat == y_pred_flat)

    # Sensitivity (Recall)
    true_positives = intersection
    false_negatives = np.sum(y_true_flat) - true_positives
    sensitivity = (true_positives + 1e-6) / (true_positives + false_negatives + 1e-6)

    # Specificity
    true_negatives = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    false_positives = np.sum(y_pred_flat) - true_positives
    specificity = (true_negatives + 1e-6) / (true_negatives + false_positives + 1e-6)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'intersection': float(intersection),
        'union': float(union)
    }


def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, Y_val: np.ndarray) -> dict:
    """Evaluate model performance.

    Args:
        model: Trained Keras model
        X_val: Validation images
        Y_val: Validation masks

    Returns:
        Dictionary of evaluation metrics
    """
    print("\\nEvaluating model performance...")

    # Get predictions
    Y_pred = model.predict(X_val, verbose=1)

    # Calculate metrics
    metrics = calculate_metrics(Y_val, Y_pred)

    print("\\nValidation Metrics:")
    print(f"  Dice Score: {metrics['dice']:.4f}")
    print(f"  IoU Score: {metrics['iou']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    return metrics
