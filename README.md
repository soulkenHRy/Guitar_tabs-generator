# Guitar Detection System

A Python-based computer vision system that detects guitars in images and video streams using YOLO (You Only Look Once) object detection.

## Features

- Detect guitars in static images
- Real-time guitar detection from webcam
- Uses YOLOv3 pre-trained model
- Automatic model download
- Bounding box visualization with confidence scores

## Installation

1. Activate your virtual environment:
```bash
source venv/bin/activate  # or your virtual environment path
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Detect in an Image

```bash
python guitar_detection.py <image_path> [output_path]
```

Example:
```bash
python guitar_detection.py guitar.jpg detected_guitar.jpg
```

### Real-time Webcam Detection

```bash
python guitar_detection.py webcam
```

Or simply run without arguments:
```bash
python guitar_detection.py
```

Press 'q' to quit the webcam view.

## How It Works

1. **Model**: Uses YOLOv3 trained on COCO dataset
2. **Detection**: Processes images through the neural network
3. **Visualization**: Draws bounding boxes around detected objects
4. **Real-time**: Can process webcam feed for live detection

## Notes

- The model will automatically download on first run (~240MB)
- Works best with clear images of guitars
- Adjustable confidence threshold for detection sensitivity
- COCO dataset has general object categories, so detection may include various objects

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Internet connection (for initial model download)
