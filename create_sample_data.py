#!/usr/bin/env python
"""Standalone script to create sample synthetic ultrasound data."""

import os
import cv2
import numpy as np

def create_sample_data(data_dir="data", num_samples=15):
    """Create sample synthetic ultrasound data for testing."""
    
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    
    # Create directories
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print("Creating {} synthetic ultrasound samples...".format(num_samples))
    
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
        
        # Add some ultrasound-like texture
        # Simulate fan-shaped ultrasound beam
        height, width = img.shape
        for row in range(height):
            for col in range(width):
                # Create fan effect from top center
                angle = np.arctan2(row - 0, col - width//2)
                distance = np.sqrt((row - 0)**2 + (col - width//2)**2)
                
                # Add some beam-like intensity variation
                beam_intensity = max(0, 1 - distance / (height * 0.8))
                if abs(angle) < np.pi/3:  # 60-degree fan
                    img[row, col] = np.clip(img[row, col] * (0.7 + 0.3 * beam_intensity), 0, 255)
        
        # Save files
        img_path = os.path.join(image_dir, "sample_{:03d}.png".format(i))
        mask_path = os.path.join(mask_dir, "sample_{:03d}.png".format(i))
        
        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)
        
        if (i + 1) % 5 == 0:
            print("  Created {}/{} samples...".format(i + 1, num_samples))
    
    print("\nSample data creation completed!")
    print("  Images saved to: {}".format(image_dir))
    print("  Masks saved to: {}".format(mask_dir))
    
    # Verify creation
    import glob
    images = glob.glob(os.path.join(image_dir, "*.png"))
    masks = glob.glob(os.path.join(mask_dir, "*.png"))
    print("  Total files created: {} images, {} masks".format(len(images), len(masks)))
    
    return len(images), len(masks)

if __name__ == "__main__":
    create_sample_data()
