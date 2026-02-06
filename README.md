# Guitar Detector GUI

A computer vision application for real-time guitar playing detection and tablature generation. This application uses YOLOv8 for object detection, hand tracking, and audio analysis to detect guitar notes and generate guitar tabs from video input.

## Features

- Real-time guitar neck and fret detection using YOLOv8
- Hand position tracking for finger placement detection
- Audio-based pitch detection for accurate note identification
- Automatic tablature generation
- GUI interface for easy video processing
- Support for various video formats

## System Requirements

- Python 3.8 or higher
- ffmpeg (required for audio processing)
- Webcam or video file input
- Minimum 4GB RAM recommended

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/guitar-detector-gui.git
cd guitar-detector-gui
```

### 2. Install system dependencies

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-tk ffmpeg
```

#### macOS
```bash
brew install python-tk ffmpeg
```

#### Windows
Download and install ffmpeg from https://ffmpeg.org/download.html
Ensure ffmpeg is added to your system PATH.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Note: The installation may take several minutes as it includes deep learning libraries like PyTorch and Ultralytics YOLO.

## Usage

### GUI Application

Launch the graphical interface:

```bash
python3 guitar_detector_gui.py
```

The GUI allows you to:
1. Select a video file containing guitar playing
2. Configure detection parameters
3. Run detection and view results
4. Export generated tablature

### Shell Scripts

Quick start scripts are provided for convenience:

```bash
./quick_start_gui.sh        # Launch GUI application
./run_webcam_detection.sh   # Run detection from webcam
```

Make sure scripts are executable:
```bash
chmod +x *.sh
```

## Project Structure

```
guitar-detector-gui/
├── guitar_detector_gui.py       # Main GUI application
├── guitar_tuner.py              # Audio pitch detection module
├── audio_preprocessor.py        # Audio isolation using Demucs/Spleeter
├── guitar_complete_detector.py  # Complete detection pipeline
├── guitar_neck_detector_simple.py  # Simplified neck detection
├── guitar_hertz_detector.py     # Frequency analysis
├── requirements.txt             # Python dependencies
├── quick_start_gui.sh          # GUI launcher script
└── run_webcam_detection.sh     # Webcam detection script
```

## Dependencies

### Core Libraries

- opencv-python >= 4.8.0 - Computer vision and image processing
- numpy >= 1.24.0 - Numerical computations
- ultralytics >= 8.0.0 - YOLOv8 object detection
- cvzone >= 1.6.0 - Hand tracking module
- librosa >= 0.10.0 - Audio analysis
- soundfile >= 0.12.0 - Audio file I/O

### Audio Processing

- demucs >= 4.0.0 - Audio source separation (recommended)
- torch >= 2.0.0 - Deep learning framework (required by Demucs)
- torchaudio >= 2.0.0 - Audio processing for PyTorch

### GUI

- tkinter - Standard Python GUI library (included with Python)
- Pillow >= 10.0.0 - Image handling for GUI

## Configuration

The application can be configured by modifying constants in `guitar_detector_gui.py`:

- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (default: 0.5)
- `IOU_THRESHOLD`: Intersection over Union threshold (default: 0.45)
- `SAMPLE_RATE`: Audio sampling rate (default: 44100 Hz)

## Model Information

This application requires a trained YOLOv8 model file named `guitar_detector_best.pt` which should detect:
- Guitar frets
- Guitar neck
- Guitar nut

Due to file size limitations, the model is not included in this repository. You can train your own model using the Ultralytics YOLO framework or contact the repository maintainer for access to a pre-trained model.

## Audio Preprocessing

The audio preprocessor can use either:
- **Demucs** (recommended): Higher quality audio separation
- **Spleeter** (alternative): Faster processing with slightly lower quality

Both tools isolate guitar audio from background music, improving pitch detection accuracy.

## Troubleshooting

### Common Issues

**ModuleNotFoundError for tkinter:**
```bash
# Linux
sudo apt-get install python3-tk

# macOS
brew install python-tk
```

**ffmpeg not found:**
Ensure ffmpeg is installed and accessible in your system PATH.

**CUDA/GPU errors:**
The application will automatically fall back to CPU if GPU is not available. For faster processing, ensure you have CUDA-compatible PyTorch installed.

**Model file missing:**
Ensure `guitar_detector_best.pt` is present in the project directory.

## Performance Notes

- Processing speed depends on video resolution and hardware capabilities
- GPU acceleration significantly improves performance
- For real-time applications, consider reducing video resolution
- Audio preprocessing may take additional time for long videos

## License

This project is available for educational and research purposes.

## Contributing

Contributions are welcome. Please ensure code follows PEP 8 style guidelines and includes appropriate documentation.

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

- YOLOv8 by Ultralytics
- cvzone by cvzone
- Demucs by Facebook Research
- Hand tracking implementation using MediaPipe
