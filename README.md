# 🎸 Fun Guitar — AI Guitar Detection & Tab Generator

> Computer vision meets music — detect guitar notes from video and generate tablature automatically.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/fun-guitar)

## Live Demo

**Web App**: Browser-based guitar tuner & project showcase

**Desktop App**: Full detection pipeline with YOLO model inference (see [Desktop Setup](#desktop-setup) below)

---

## Features

### Web Application (Vercel)
- **Browser Guitar Tuner** — Real-time pitch detection using Web Audio API
- **Reference Tone Player** — Click any string to hear its target frequency
- **Note/Frequency API** — REST endpoints for guitar frequency lookups
- **Tab Formatter API** — Convert note data to guitar tablature format

### Desktop Application (Python)
- **Real-time guitar detection** — YOLOv8 custom model detects frets, neck, and nut
- **Hand tracking** — MediaPipe-powered finger position detection on fretboard
- **Audio pitch detection** — FFT + autocorrelation for accurate frequency analysis
- **Audio preprocessing** — Demucs source separation to isolate guitar audio
- **Automatic tab generation** — Multi-modal fusion of visual and audio data
- **Fret stabilization** — EMA smoothing prevents detection jitter
- **Gap tracking** — Intelligent occlusion detection for hidden frets

---

## Web Deployment (Vercel)

### Quick Deploy

1. **Fork/clone** this repository
2. **Import** into [Vercel](https://vercel.com/new)
3. Vercel auto-detects the configuration — click **Deploy**

That's it! The web app includes:
- Static frontend served from `public/`
- Python serverless API routes in `api/`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service health check |
| `/api/notes?action=identify&freq=440` | GET | Identify frequency to note |
| `/api/notes?action=fret&string=1&fret=0` | GET | Get note for string/fret |
| `/api/notes?action=fretboard` | GET | Full fretboard frequency map |
| `/api/notes?action=tuning` | GET | Standard tuning reference |
| `/api/tab` | GET | Demo tab output |
| `/api/tab` | POST | Generate tab from note array |

### API Usage Example

```bash
# Identify a frequency
curl https://your-app.vercel.app/api/notes?action=identify&freq=440

# Get note for string 1, fret 5
curl https://your-app.vercel.app/api/notes?action=fret&string=1&fret=5

# Generate tab from notes
curl -X POST https://your-app.vercel.app/api/tab \
  -H "Content-Type: application/json" \
  -d '{"notes": [{"string": 1, "fret": 0}, {"string": 2, "fret": 1}]}'
```

---

## Desktop Setup

The full detection pipeline requires Python and ML dependencies that run locally.

### Requirements
- Python 3.8+
- ffmpeg
- GPU recommended (CPU fallback available)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/fun-guitar.git
cd fun-guitar

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Desktop GUI

```bash
python3 guitar_detector_gui.py
```

### Desktop Dependencies
- `opencv-python` >= 4.8.0 — Computer vision
- `ultralytics` >= 8.0.0 — YOLOv8 object detection
- `cvzone` >= 1.6.0 — Hand tracking (MediaPipe)
- `numpy` >= 1.24.0 — Numerical computation
- `scipy` — Signal processing and audio I/O
- `torch` >= 2.0.0 — Deep learning backend
- `demucs` >= 4.0.0 — Audio source separation
- `Pillow` >= 10.0.0 — Image handling

### Model File

The desktop app requires `guitar_detector_best.pt` (custom YOLOv8 model trained to detect frets, neck, and nut). Due to file size, it is not included in the repository. Train your own using the provided dataset structure or contact the maintainer.

---

## Project Structure

```
fun-guitar/
├── public/                      # Web frontend (served by Vercel)
│   ├── index.html              # Main web page
│   ├── style.css               # Styles
│   ├── app.js                  # UI interactions and animations
│   └── tuner.js                # Browser guitar tuner (Web Audio API)
├── api/                         # Vercel Python serverless functions
│   ├── health.py               # Health check endpoint
│   ├── notes.py                # Note/frequency lookup API
│   └── tab.py                  # Tab generation API
├── guitar_detector_gui.py       # Desktop GUI (tkinter)
├── guitar_tuner.py              # Audio pitch detection module
├── audio_preprocessor.py        # Audio isolation (Demucs/Spleeter)
├── guitar_complete_detector.py  # Complete detection pipeline
├── requirements.txt             # Python dependencies (desktop)
├── vercel.json                  # Vercel deployment configuration
├── package.json                 # Project metadata
└── .gitignore                   # Git ignore rules
```

---

## How Detection Works

```
Video Input -> Frame Extraction -> YOLOv8 Detection (fret/neck/nut)
                                        |
Audio Input -> FFT Pitch Detection -> Frequency -> Note Identification
                                        |
Hand Tracking -> Finger Position -> Fret Matching -> Tab Generation
```

1. **Frame Processing**: Video frames resized to 640x480 for YOLO inference
2. **YOLO Detection**: Custom model identifies frets, neck, and nut bounding boxes
3. **Fret Stabilization**: EMA smoothing locks detected fret positions
4. **Gap Tracking**: Detects hidden frets by analyzing spacing patterns
5. **Hand Detection**: MediaPipe tracks 21 hand landmarks per hand
6. **Audio Analysis**: FFT + autocorrelation identifies played frequencies
7. **Multi-Modal Fusion**: Visual (hand on fret) + audio (detected pitch) = confirmed note
8. **Tab Output**: Notes accumulated and formatted as standard guitar tablature

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Hand Tracking | MediaPipe via cvzone |
| Video Processing | OpenCV |
| Pitch Detection | FFT + Autocorrelation (scipy/numpy) |
| Audio Separation | Demucs (Facebook Research) |
| Desktop GUI | tkinter + Pillow |
| Web Frontend | Vanilla HTML/CSS/JS |
| Web Tuner | Web Audio API |
| API Routes | Python (Vercel Serverless) |
| Deployment | Vercel |

---

## License

MIT

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google/mediapipe) via [cvzone](https://github.com/cvzone/cvzone)
- [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research
- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
