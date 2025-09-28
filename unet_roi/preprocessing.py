"""Preprocessing utilities for ultrasound images."""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional
from glob import glob
import warnings


class UltrasoundPreprocessor:
    """Preprocessing class for ultrasound images."""
    
    def __init__(self, img_size: Tuple[int, int] = (256, 256)):
        """Initialize preprocessor.
        
        Args:
            img_size: Target image size (height, width)
        """
        self.img_size = img_size
        
    def resize_with_padding(self, img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio.
        
        Args:
            img: Input image array
            target_size: Target size (height, width). If None, uses self.img_size
            
        Returns:
            Resized image with padding
        """
        if target_size is None:
            target_size = self.img_size
            
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        if len(img.shape) == 3:
            padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=img.dtype)
            
        # Calculate padding offsets
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place resized image in padded array
        if len(img.shape) == 3:
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized
        else:
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
            
        return padded
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load image as RGB to match model input
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with padding
        img_resized = self.resize_with_padding(img)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def preprocess_mask(self, mask_path: str) -> np.ndarray:
        """Preprocess a single mask.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Preprocessed mask array
        """
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
            
        # Resize with padding
        mask_resized = self.resize_with_padding(mask)
        
        # Binarize mask
        mask_binary = (mask_resized > 127).astype(np.float32)
        
        # Add channel dimension
        mask_final = np.expand_dims(mask_binary, axis=-1)
        
        return mask_final
    
    def validate_data_paths(self, image_dir: str, mask_dir: str) -> bool:
        """Validate that image and mask directories exist and contain files.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            
        Returns:
            True if validation passes
        """
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
            
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")
            
        # Get image and mask files
        image_files = glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg"))
        mask_files = glob(os.path.join(mask_dir, "*.png")) + glob(os.path.join(mask_dir, "*.jpg"))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in: {image_dir}")
            
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in: {mask_dir}")
            
        print(f"Found {len(image_files)} images and {len(mask_files)} masks")
        return True
    
    def load_dataset(self, image_dir: str, mask_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess entire dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            
        Returns:
            Tuple of (images, masks) arrays
        """
        # Validate paths
        self.validate_data_paths(image_dir, mask_dir)
        
        # Get sorted file lists
        image_files = sorted(glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg")))
        mask_files = sorted(glob(os.path.join(mask_dir, "*.png")) + glob(os.path.join(mask_dir, "*.jpg")))
        
        # Match files by name
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        mask_names = [os.path.splitext(os.path.basename(f))[0] for f in mask_files]
        
        # Find matching pairs
        matched_images = []
        matched_masks = []
        
        for img_file, img_name in zip(image_files, image_names):
            if img_name in mask_names:
                mask_idx = mask_names.index(img_name)
                matched_images.append(img_file)
                matched_masks.append(mask_files[mask_idx])
            else:
                warnings.warn(f"No matching mask found for image: {img_file}")
        
        if len(matched_images) == 0:
            raise ValueError("No matching image-mask pairs found")
            
        print(f"Loading {len(matched_images)} image-mask pairs...")
        
        # Load and preprocess all images and masks
        images = []
        masks = []
        
        for img_path, mask_path in zip(matched_images, matched_masks):
            try:
                img = self.preprocess_image(img_path)
                mask = self.preprocess_mask(mask_path)
                images.append(img)
                masks.append(mask)
            except Exception as e:
                warnings.warn(f"Failed to process {img_path}: {str(e)}")
                continue
        
        if len(images) == 0:
            raise ValueError("No images could be successfully processed")
            
        # Convert to numpy arrays
        X = np.array(images)
        Y = np.array(masks)
        
        print(f"Dataset loaded successfully:")
        print(f"  Images shape: {X.shape}")
        print(f"  Masks shape: {Y.shape}")
        print(f"  Image range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  Mask range: [{Y.min():.3f}, {Y.max():.3f}]")
        
        return X, Y
    
    def preprocess_for_inference(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image ready for model prediction (with batch dimension)
        """
        img = self.preprocess_image(image_path)
        return np.expand_dims(img, axis=0)  # Add batch dimension


def print_data_statistics(X: np.ndarray, Y: np.ndarray) -> None:
    """Print comprehensive statistics about the dataset.
    
    Args:
        X: Images array
        Y: Masks array
    """
    print("\\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Dataset size: {X.shape[0]} samples")
    print(f"Image shape: {X.shape[1:]} (H x W x C)")
    print(f"Mask shape: {Y.shape[1:]} (H x W x C)")
    print(f"Total parameters per image: {np.prod(X.shape[1:]):,}")
    
    print(f"\\nImage statistics:")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    
    print(f"\\nMask statistics:")
    print(f"  Min: {Y.min():.6f}")
    print(f"  Max: {Y.max():.6f}")
    print(f"  Mean: {Y.mean():.6f} (ROI coverage)")
    print(f"  Std: {Y.std():.6f}")
    
    # Class distribution
    total_pixels = Y.size
    roi_pixels = np.sum(Y > 0.5)
    background_pixels = total_pixels - roi_pixels
    
    print(f"\\nClass distribution:")
    print(f"  Background: {background_pixels:,} pixels ({100*background_pixels/total_pixels:.1f}%)")
    print(f"  ROI: {roi_pixels:,} pixels ({100*roi_pixels/total_pixels:.1f}%)")
    
    print("="*50)


def create_sample_data(data_dir: str = "data", num_samples: int = 5) -> None:
    """Create sample synthetic data for testing.
    
    Args:
        data_dir: Directory to create sample data in
        num_samples: Number of sample image-mask pairs to create
    """
    import os
    
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create synthetic ultrasound-like image
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Add some structure to mimic ultrasound appearance
        center_x, center_y = 128, 128
        radius = np.random.randint(60, 100)
        
        # Create circular ROI mask
        y, x = np.ogrid[:256, :256]
        mask_condition = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[mask_condition] = 255
        
        # Add some realistic structure to the image within ROI
        img[mask_condition] = np.clip(img[mask_condition] + 50, 0, 255)
        
        # Save files
        img_path = os.path.join(image_dir, f"sample_{i:03d}.png")
        mask_path = os.path.join(mask_dir, f"sample_{i:03d}.png")
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
    
    print(f"Created {num_samples} sample image-mask pairs in {data_dir}/")
    print(f"  Images: {image_dir}")
    print(f"  Masks: {mask_dir}")
