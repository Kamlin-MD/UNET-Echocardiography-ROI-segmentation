"""U-Net model implementation for ultrasound ROI segmentation."""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


class UNetModel:
    """U-Net model class for ultrasound ROI segmentation."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 1), num_classes: int = 1):
        """Initialize U-Net model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (1 for binary segmentation)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self) -> tf.keras.Model:
        """Build U-Net architecture with encoder-decoder structure.
        
        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder path
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = layers.Dropout(0.1)(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = layers.Dropout(0.2)(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = layers.Dropout(0.2)(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
        
        # Bottleneck
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        
        # Decoder path
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', name="segmentation_output")(c9)
        
        self.model = models.Model(inputs=[inputs], outputs=[outputs])
        return self.model
    
    def compile_model(self, learning_rate: float = 1e-4) -> tf.keras.Model:
        """Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Compiled model
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', self._dice_coefficient, self._iou_score]
        )
        return self.model
    
    def get_model(self) -> tf.keras.Model:
        """Get the model instance.
        
        Returns:
            The Keras model instance
        """
        if self.model is None:
            self.compile_model()
        return self.model
    
    def load_weights(self, weights_path: str) -> None:
        """Load pre-trained weights.
        
        Args:
            weights_path: Path to the weights file
        """
        if self.model is None:
            self.build_model()
        self.model.load_weights(weights_path)
    
    def save_model(self, model_path: str) -> None:
        """Save the complete model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(model_path)
    
    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric for segmentation.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient score
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    @staticmethod
    def _iou_score(y_true, y_pred, smooth=1e-6):
        """IoU (Intersection over Union) metric for segmentation.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            smooth: Smoothing factor
            
        Returns:
            IoU score
        """
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)


def load_pretrained_model(model_path: str) -> tf.keras.Model:
    """Load a pre-trained U-Net model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            'dice_coefficient': UNetModel._dice_coefficient,
            'iou_score': UNetModel._iou_score
        }
    )
