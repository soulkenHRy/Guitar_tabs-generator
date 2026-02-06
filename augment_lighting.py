#!/usr/bin/env python3
"""
Augment training images with different lighting effects.
Creates copies of images with various brightness/contrast adjustments,
while keeping the same labels.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Paths
DATASET_PATH = "/home/shaken/fun_guitar/guitar.v1i.yolov8"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_PATH, "train", "labels")

def adjust_brightness(image, factor):
    """Adjust brightness by multiplying pixel values."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor):
    """Adjust contrast around the mean."""
    mean = np.mean(image)
    adjusted = (image - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def adjust_gamma(image, gamma):
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def add_shadow(image):
    """Add random shadow effect."""
    h, w = image.shape[:2]
    # Create a gradient shadow
    shadow = np.ones((h, w), dtype=np.float32)
    # Random shadow direction
    x1, x2 = np.random.randint(0, w, 2)
    shadow_intensity = np.random.uniform(0.3, 0.7)
    
    for i in range(w):
        shadow[:, i] = 1.0 - (shadow_intensity * (i / w))
    
    # Apply shadow to image
    result = image.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * shadow
    return np.clip(result, 0, 255).astype(np.uint8)

def warm_filter(image):
    """Apply warm (yellowish) lighting filter."""
    result = image.copy().astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)  # Green
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)   # Blue
    return result.astype(np.uint8)

def cool_filter(image):
    """Apply cool (bluish) lighting filter."""
    result = image.copy().astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.9, 0, 255)   # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 0.95, 0, 255)  # Green
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.1, 0, 255)   # Blue
    return result.astype(np.uint8)

# Define lighting augmentations: (name_suffix, function)
AUGMENTATIONS = [
    ("_bright", lambda img: adjust_brightness(img, 1.3)),
    ("_dark", lambda img: adjust_brightness(img, 0.7)),
    ("_highcontrast", lambda img: adjust_contrast(img, 1.3)),
    ("_lowcontrast", lambda img: adjust_contrast(img, 0.7)),
    ("_gamma_high", lambda img: adjust_gamma(img, 1.5)),
    ("_gamma_low", lambda img: adjust_gamma(img, 0.7)),
    ("_warm", warm_filter),
    ("_cool", cool_filter),
]

def get_base_name(filename):
    """Get base name without extension."""
    return os.path.splitext(filename)[0]

def main():
    # Get list of all training images
    image_files = [f for f in os.listdir(TRAIN_IMAGES) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_images = len(image_files)
    print(f"Found {total_images} training images")
    print(f"Will create {len(AUGMENTATIONS)} augmented versions per image")
    print(f"Total new images to create: {total_images * len(AUGMENTATIONS)}")
    print("-" * 50)
    
    created_count = 0
    errors = []
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(TRAIN_IMAGES, img_file)
        base_name = get_base_name(img_file)
        ext = os.path.splitext(img_file)[1]
        
        # Corresponding label file
        label_file = base_name + ".txt"
        label_path = os.path.join(TRAIN_LABELS, label_file)
        
        # Check if label exists
        if not os.path.exists(label_path):
            errors.append(f"Label not found for: {img_file}")
            continue
        
        # Read original image
        image = cv2.imread(img_path)
        if image is None:
            errors.append(f"Could not read image: {img_file}")
            continue
        
        # Apply each augmentation
        for suffix, aug_func in AUGMENTATIONS:
            try:
                # Create augmented image
                aug_image = aug_func(image)
                
                # New file names
                new_img_name = base_name + suffix + ext
                new_label_name = base_name + suffix + ".txt"
                
                new_img_path = os.path.join(TRAIN_IMAGES, new_img_name)
                new_label_path = os.path.join(TRAIN_LABELS, new_label_name)
                
                # Save augmented image
                cv2.imwrite(new_img_path, aug_image)
                
                # Copy label file (same annotations)
                shutil.copy2(label_path, new_label_path)
                
                created_count += 1
                
            except Exception as e:
                errors.append(f"Error augmenting {img_file} with {suffix}: {str(e)}")
        
        # Progress update
        if idx % 50 == 0 or idx == total_images:
            print(f"Processed {idx}/{total_images} images ({created_count} augmented images created)")
    
    print("-" * 50)
    print(f"Completed! Created {created_count} augmented images with labels")
    
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    main()
