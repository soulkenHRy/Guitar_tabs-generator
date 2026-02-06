#!/usr/bin/env python3
"""
Train a YOLO model to detect guitar neck and nut
Converts COCO format dataset and trains the model
"""

import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def convert_coco_to_yolo(coco_dataset_path, output_path):
    """Convert COCO format annotations to YOLO format."""
    
    coco_path = Path(coco_dataset_path)
    output_path = Path(output_path)
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Map COCO category IDs to YOLO class indices
    # We'll use: 0 = neck, 1 = nut
    class_mapping = {}
    
    for split in ['train', 'valid', 'test']:
        split_path = coco_path / split
        if not split_path.exists():
            print(f"Skipping {split} - directory not found")
            continue
            
        annotations_file = split_path / '_annotations.coco.json'
        if not annotations_file.exists():
            print(f"Skipping {split} - no annotations found")
            continue
        
        print(f"Processing {split}...")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping (neck -> 0, nut -> 1)
        for cat in coco_data['categories']:
            cat_name = cat['name'].lower()
            if cat_name == 'neck':
                class_mapping[cat['id']] = 0
            elif cat_name == 'nut':
                class_mapping[cat['id']] = 1
        
        print(f"  Category mapping: {class_mapping}")
        
        # Build image ID to filename mapping
        image_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Process each image
        for img_id, img_data in image_info.items():
            filename = img_data['file_name']
            img_width = img_data['width']
            img_height = img_data['height']
            
            # Copy image
            src_img = split_path / filename
            if src_img.exists():
                dst_img = output_path / 'images' / split / filename
                shutil.copy(src_img, dst_img)
            
            # Create YOLO label file
            label_filename = Path(filename).stem + '.txt'
            label_path = output_path / 'labels' / split / label_filename
            
            yolo_annotations = []
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in class_mapping:
                        continue
                    
                    yolo_class = class_mapping[cat_id]
                    
                    # COCO bbox format: [x, y, width, height] (top-left corner)
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    
                    # Convert to YOLO format: [class, x_center, y_center, width, height] (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Clamp values to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    yolo_annotations.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"  Processed {len(image_info)} images")
    
    # Create data.yaml
    data_yaml = f"""# Guitar Neck and Nut Detection Dataset
path: {output_path.absolute()}
train: images/train
val: images/valid
test: images/test

# Classes
names:
  0: neck
  1: nut

nc: 2
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml)
    
    print(f"\nDataset converted successfully!")
    print(f"  Output path: {output_path}")
    print(f"  data.yaml: {yaml_path}")
    
    return yaml_path


def train_model(data_yaml_path, epochs=100, imgsz=640, batch=16):
    """Train YOLO model for neck and nut detection."""
    
    print("\n" + "="*60)
    print("TRAINING YOLO MODEL FOR NECK AND NUT DETECTION")
    print("="*60)
    
    # Start with pretrained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/neck_nut_detection',
        name='train',
        patience=20,
        save=True,
        device='cpu',  # Use CPU since CUDA is not available
        verbose=True
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Find the best weights
    best_weights = Path('runs/neck_nut_detection/train/weights/best.pt')
    if best_weights.exists():
        print(f"Best weights saved to: {best_weights}")
        return best_weights
    else:
        # Try to find it
        import glob
        weights = glob.glob('runs/neck_nut_detection/*/weights/best.pt')
        if weights:
            print(f"Best weights saved to: {weights[0]}")
            return Path(weights[0])
    
    return None


def main():
    # Paths
    coco_dataset = "/home/shaken/fun_guitar/Guitar nut and neck.v1i.coco"
    yolo_dataset = "/home/shaken/fun_guitar/guitar_neck_nut_yolo"
    
    # Step 1: Convert COCO to YOLO format
    print("="*60)
    print("STEP 1: Converting COCO dataset to YOLO format")
    print("="*60)
    data_yaml = convert_coco_to_yolo(coco_dataset, yolo_dataset)
    
    # Step 2: Train the model
    print("\n" + "="*60)
    print("STEP 2: Training model")
    print("="*60)
    best_weights = train_model(data_yaml, epochs=100, imgsz=640, batch=16)
    
    if best_weights:
        print(f"\n✓ Training complete! Best model: {best_weights}")
        print(f"\nTo use this model, update MODEL_PATH in guitar_neck_detector_simple.py to:")
        print(f'  MODEL_PATH = "{best_weights}"')


if __name__ == "__main__":
    main()
