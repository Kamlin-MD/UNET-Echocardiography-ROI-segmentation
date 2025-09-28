"""Training utilities for U-Net model."""

import os
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .model import UNetModel
from .preprocessing import UltrasoundPreprocessor


class UNetTrainer:
    """Training class for U-Net model."""
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (256, 256),
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
        
    def prepare_data(self, image_dir: str, mask_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data.
        
        Args:
            image_dir: Directory containing training images
            mask_dir: Directory containing training masks
            
        Returns:
            Tuple of (X_train, X_val, Y_train, Y_val)
        """
        print("Loading and preprocessing data...")
        X, Y = self.preprocessor.load_dataset(image_dir, mask_dir)
        
        # Split data
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=self.validation_split, random_state=42, stratify=None
        )
        
        print(f"\\nData split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        
        return X_train, X_val, Y_train, Y_val
    
    def create_callbacks(self, model_save_path: str) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks.
        
        Args:
            model_save_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
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
        
        return callbacks
    
    def train(self, 
              image_dir: str, 
              mask_dir: str, 
              model_save_path: str = "models/unet_model.keras") -> tf.keras.callbacks.History:
        """Train the U-Net model.
        
        Args:
            image_dir: Directory containing training images
            mask_dir: Directory containing training masks
            model_save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        # Prepare data
        X_train, X_val, Y_train, Y_val = self.prepare_data(image_dir, mask_dir)
        
        # Build and compile model
        print("\\nBuilding and compiling model...")
        self.model = self.model_builder.compile_model(self.learning_rate)
        
        # Print model summary
        print("\\nModel Summary:")
        self.model.summary()
        
        # Create callbacks
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        callbacks = self.create_callbacks(model_save_path)
        
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
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
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
        
        # Plot Dice coefficient
        if 'dice_coefficient' in self.history.history:
            axes[1, 0].plot(self.history.history['dice_coefficient'], label='Training Dice')
            axes[1, 0].plot(self.history.history['val_dice_coefficient'], label='Validation Dice')
            axes[1, 0].set_title('Dice Coefficient')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Dice Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot IoU score
        if 'iou_score' in self.history.history:
            axes[1, 1].plot(self.history.history['iou_score'], label='Training IoU')
            axes[1, 1].plot(self.history.history['val_iou_score'], label='Validation IoU')
            axes[1, 1].set_title('IoU Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('IoU Score')
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
